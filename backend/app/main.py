from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
import asyncio
import json
from pathlib import Path
from typing import Annotated, Literal
from xml.sax.saxutils import escape

from bson import ObjectId
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import Response, StreamingResponse

from .db import MongoStore
from .llm_evaluator import LLMEvaluationError
from .refinement_agent import RefinementAgent, format_refinement_response
from .settings import Settings
from .workflow import ChartWorkflow


settings = Settings()
store = MongoStore(settings)
workflow = ChartWorkflow(store, settings)
refinement_agent = RefinementAgent(store, settings)

CLAIM_EXTENSIONS = {".xlsx", ".pdf"}
SUPPORTING_EXTENSIONS = {".docx", ".pdf"}
MAX_CLAIM_BYTES = 5 * 1024 * 1024
MAX_SUPPORTING_BYTES = 15 * 1024 * 1024


class ChatCreateResponse(BaseModel):
    chat_id: str
    run_id: str
    title: str
    claim_chart_file_ids: list[str]
    supporting_file_ids: list[str]
    chart_revision_id: str | None
    processing_engine: str
    verification_mode: str
    processing_status: str
    chart_rows: list[dict]


class ChatSummary(BaseModel):
    id: str
    title: str


class ChatMessage(BaseModel):
    id: str
    role: str
    content: str
    status: str | None = None
    file_ids: list[str] = Field(default_factory=list)
    kind: str = "message"
    action_id: str | None = None
    action: dict | None = None
    sources: list[dict] = Field(default_factory=list)


class WorkspaceAttachment(BaseModel):
    id: str
    filename: str
    category: str


class ChatWorkspaceResponse(BaseModel):
    chat_id: str
    run_id: str
    processing_status: str
    confidence_status: str = "idle"
    chart_revision_id: str | None
    error: str | None = None
    chart_rows: list[dict]
    messages: list[ChatMessage]
    attachments: list[WorkspaceAttachment]


class ConfidenceGenerationRequest(BaseModel):
    run_id: str


class ConfidenceGenerationResponse(BaseModel):
    chat_id: str
    run_id: str
    chart_revision_id: str
    chart_rows: list[dict]


class ChartTableResponse(BaseModel):
    chat_id: str
    run_id: str
    processing_status: str
    chart_revision_id: str | None
    error: str | None = None
    chart_rows: list[dict]


class AttachmentResponse(BaseModel):
    file_ids: list[str]
    filenames: list[str]


class MessageResponse(BaseModel):
    message_id: str
    status: str
    content: str
    file_ids: list[str] = Field(default_factory=list)


class PromptMessageRequest(BaseModel):
    run_id: str
    content: str
    file_ids: list[str] = Field(default_factory=list)


class AssistantActionDecisionRequest(BaseModel):
    run_id: str
    decision: Literal["approve", "reject"]


class AssistantActionResponse(BaseModel):
    action_id: str
    status: str
    message: str
    chart_revision_id: str | None = None
    chart_rows: list[dict] = Field(default_factory=list)


class EvidenceEditRequest(BaseModel):
    run_id: str
    old_value: str
    value: str


class EvidenceEditResponse(BaseModel):
    chat_id: str
    run_id: str
    chart_revision_id: str
    chart_rows: list[dict] = Field(default_factory=list)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def require_database() -> None:
    if not store.configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MongoDB is not configured. Set MONGODB_URI before creating a chat.",
        )


def serialize_chart_row(row: dict) -> dict:
    return {
        "rowId": str(row["_id"]) if isinstance(row.get("_id"), ObjectId) else str(row.get("rowId", "")),
        "rowIndex": row["rowIndex"],
        "serialNumber": row["rowIndex"],
        "patentClaim": row["patentClaim"],
        "evidenceOfInfringement": row["evidenceOfInfringement"],
        "aiEvidence": row.get("aiEvidence", ""),
        "confidenceScore": row.get("confidenceScore"),
        "confidenceRemark": row.get("confidenceRemark"),
        "source": row.get("source", {}),
        "imageRefs": row.get("imageRefs", []),
    }


def serialize_message(message: dict) -> ChatMessage:
    return ChatMessage(
        id=str(message["_id"]),
        role=message.get("role", "user"),
        content=message.get("content", ""),
        status=message.get("status"),
        file_ids=[str(file_id) for file_id in message.get("fileIds", [])],
        kind=message.get("kind", "message"),
        action_id=str(message["actionId"]) if isinstance(message.get("actionId"), ObjectId) else message.get("actionId"),
        action=message.get("action"),
        sources=message.get("sources", []),
    )


def serialize_attachment(file_record: dict) -> WorkspaceAttachment:
    return WorkspaceAttachment(
        id=str(file_record["_id"]),
        filename=file_record.get("filename", ""),
        category=file_record.get("category", "supporting_document"),
    )


async def get_workspace_response(chat_id: str, run_id: str | None = None) -> ChatWorkspaceResponse:
    require_database()
    try:
        chat_object_id = ObjectId(chat_id)
        run_object_id = ObjectId(run_id) if run_id else None
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid chat or run identifier.") from error

    run = (
        await store.find_one("analysis_runs", {"_id": run_object_id, "chatId": chat_object_id})
        if run_object_id is not None
        else await store.find_latest_run(chat_object_id)
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Chat run not found.")

    revision_id = run.get("revisionId")
    rows = []
    if run.get("status") == "completed" and isinstance(revision_id, ObjectId):
        rows = sorted(
            await store.find_many("chart_rows", {"revisionId": revision_id}),
            key=lambda item: item["rowIndex"],
        )
    messages = await store.find_many(
        "messages",
        {"chatId": chat_object_id, "runId": run["_id"]},
    )
    messages.sort(key=lambda item: item.get("createdAt", datetime.min.replace(tzinfo=timezone.utc)))
    attachments = await store.find_many(
        "files",
        {"chatId": chat_object_id, "runId": run["_id"], "category": "supporting_document"},
    )
    attachments.sort(key=lambda item: item.get("createdAt", datetime.min.replace(tzinfo=timezone.utc)))
    return ChatWorkspaceResponse(
        chat_id=chat_id,
        run_id=str(run["_id"]),
        processing_status=run.get("status", "processing"),
        confidence_status=run.get("confidenceStatus", "idle"),
        chart_revision_id=str(revision_id) if isinstance(revision_id, ObjectId) else None,
        error=run.get("error"),
        chart_rows=[serialize_chart_row(row) for row in rows],
        messages=[serialize_message(message) for message in messages if message.get("content")],
        attachments=[serialize_attachment(file_record) for file_record in attachments],
    )


async def claim_confidence_generation(run_id: ObjectId) -> bool:
    """Claim one confidence run atomically; failed runs remain retryable."""
    return await store.compare_and_set(
        "analysis_runs",
        {
            "_id": run_id,
            "status": "completed",
            "confidenceStatus": {"$in": [None, "pending", "failed"]},
        },
        {"$set": {"confidenceStatus": "generating", "updatedAt": now_utc()}},
    )


async def process_chart_import(
    chat_id: ObjectId,
    run_id: ObjectId,
    file_records: list[dict],
) -> None:
    try:
        workflow_state = await workflow.run(
            chat_id=chat_id,
            run_id=run_id,
            claim_files=[record for record in file_records if record["category"] == "claim_chart"],
            supporting_files=[],
        )
        await store.update(
            "analysis_runs",
            {"_id": run_id},
            {"$set": {"status": "completed", "updatedAt": now_utc()}},
        )
    except Exception as error:
        await store.update(
            "analysis_runs",
            {"_id": run_id},
            {"$set": {"status": "failed", "error": str(error), "updatedAt": now_utc()}},
        )
        await store.insert(
            "audit_events",
            {
                "_id": ObjectId(),
                "chatId": chat_id,
                "runId": run_id,
                "event": "chart_processing_failed",
                "error": str(error),
                "createdAt": now_utc(),
            },
        )


async def process_refinement_message(
    chat_id: ObjectId,
    run_id: ObjectId,
    message_id: ObjectId,
    prompt: str,
    file_ids: list[str],
) -> None:
    """Run the isolated table-refinement graph for one persisted user turn."""
    try:
        await store.update(
            "messages",
            {"_id": message_id, "chatId": chat_id, "runId": run_id},
            {"$set": {"status": "processing", "updatedAt": now_utc()}},
        )
        # A prompt can be queued while the import is still running. The
        # refinement graph must only receive a completed table snapshot.
        for _ in range(600):
            current_run = await store.find_one(
                "analysis_runs", {"_id": run_id, "chatId": chat_id}
            )
            if current_run is None:
                raise LLMEvaluationError("The chart run no longer exists.")
            if current_run.get("status") == "completed":
                break
            if current_run.get("status") == "failed":
                raise LLMEvaluationError(current_run.get("error") or "The chart could not be prepared.")
            await asyncio.sleep(0.5)
        else:
            raise LLMEvaluationError("The chart table is taking longer than expected. Please try again.")
        state = await refinement_agent.run(
            chat_id=chat_id,
            run_id=run_id,
            message_id=message_id,
            prompt=prompt,
            file_ids=file_ids,
        )
        patches = state.get("patches", [])
        action_id: ObjectId | None = None
        assistant_message_id = ObjectId()
        action_payload: dict | None = None
        if patches:
            action_id = ObjectId()
            current_run = await store.find_one("analysis_runs", {"_id": run_id, "chatId": chat_id})
            source_revision_id = current_run.get("revisionId") if current_run else None
            action_payload = {
                "actionId": str(action_id),
                "type": "table_patch",
                "status": "pending",
                "patches": patches,
                "sources": state.get("sources", []),
                "sourceRevisionId": str(source_revision_id) if isinstance(source_revision_id, ObjectId) else None,
            }
            await store.insert(
                "agent_actions",
                {
                    "_id": action_id,
                    "chatId": chat_id,
                    "runId": run_id,
                    "messageId": assistant_message_id,
                    "type": "table_patch",
                    "status": "pending",
                    "patches": patches,
                    "sources": state.get("sources", []),
                    "sourceRevisionId": source_revision_id,
                    "createdAt": now_utc(),
                },
            )
        kind = "proposal" if patches else "human_request" if state.get("needs_human") else "answer"
        assistant_message = {
            "_id": assistant_message_id,
            "chatId": chat_id,
            "runId": run_id,
            "role": "assistant",
            "content": format_refinement_response(state),
            "status": "completed",
            "kind": kind,
            "actionId": action_id,
            "action": action_payload,
            "sources": state.get("sources", []),
            "createdAt": now_utc(),
        }
        await store.insert("messages", assistant_message)
        await store.update(
            "messages",
            {"_id": message_id, "chatId": chat_id, "runId": run_id},
            {"$set": {"status": "completed", "updatedAt": now_utc()}},
        )
        await store.update("chats", {"_id": chat_id}, {"$set": {"updatedAt": now_utc()}})
    except Exception as error:
        await store.update(
            "messages",
            {"_id": message_id, "chatId": chat_id, "runId": run_id},
            {"$set": {"status": "error", "updatedAt": now_utc()}},
        )
        await store.insert(
            "messages",
            {
                "_id": ObjectId(),
                "chatId": chat_id,
                "runId": run_id,
                "role": "assistant",
                "content": f"I could not complete that table request yet: {error}",
                "status": "error",
                "kind": "error",
                "createdAt": now_utc(),
            },
        )


def validate_file_names(files: list[UploadFile], allowed_extensions: set[str], category: str) -> None:
    for upload in files:
        filename = (upload.filename or "").strip()
        if not filename:
            raise HTTPException(status_code=400, detail=f"A {category} file is missing its filename.")
        if len(Path(filename).name) > 255:
            raise HTTPException(status_code=400, detail=f"{filename} has a filename that is too long.")
        if Path(filename).suffix.lower() not in allowed_extensions:
            allowed = ", ".join(sorted(allowed_extensions))
            raise HTTPException(status_code=400, detail=f"{filename} is not a supported {category} format. Use: {allowed}.")


def validate_duplicate_names(claim_charts: list[UploadFile], supporting_documents: list[UploadFile]) -> None:
    seen: set[str] = set()
    for upload in [*claim_charts, *supporting_documents]:
        normalized_name = Path((upload.filename or "").strip()).name.lower()
        if normalized_name in seen:
            raise HTTPException(status_code=409, detail=f"Duplicate filename: {upload.filename}")
        seen.add(normalized_name)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await store.startup()
    yield
    await store.close()


app = FastAPI(title="iLumos API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.app_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str | bool]:
    if not settings.database_configured:
        return {"status": "ok", "databaseConfigured": False}
    if not store.configured:
        return {"status": "degraded", "databaseConfigured": True}
    return {"status": "ok", "databaseConfigured": True}


@app.get("/api/chats", response_model=list[ChatSummary])
async def list_chats() -> list[ChatSummary]:
    require_database()
    chats = await store.find_recent_chats()
    return [ChatSummary(id=str(chat["_id"]), title=chat["title"]) for chat in chats]


@app.get("/api/chats/{chat_id}/workspace", response_model=ChatWorkspaceResponse)
async def get_chat_workspace(chat_id: str) -> ChatWorkspaceResponse:
    """Return the latest persisted run needed to fully reopen a chart workspace."""
    return await get_workspace_response(chat_id)


@app.get("/api/assets/{asset_id}")
async def get_chart_asset(asset_id: str) -> Response:
    require_database()
    try:
        object_id = ObjectId(asset_id)
    except Exception as error:
        raise HTTPException(status_code=404, detail="Asset not found.") from error
    record = await store.find_one("files", {"_id": object_id, "category": "chart_asset"})
    if record is None:
        raise HTTPException(status_code=404, detail="Asset not found.")
    data = await store.read_file(record["gridfsId"])
    safe_filename = Path(record.get("filename", "asset.bin")).name.replace('"', "")
    return Response(
        content=data,
        media_type=record.get("contentType") or "application/octet-stream",
        headers={"Content-Disposition": f"inline; filename=\"{safe_filename}\""},
    )


@app.post("/api/chats", response_model=ChatCreateResponse, status_code=201)
async def create_chat(
    background_tasks: BackgroundTasks,
    claim_charts: Annotated[list[UploadFile], File(...)] ,
    supporting_documents: Annotated[list[UploadFile] | None, File()] = None,
    prompt: Annotated[str | None, Form()] = None,
) -> ChatCreateResponse:
    require_database()
    supporting_documents = supporting_documents or []
    validate_file_names(claim_charts, CLAIM_EXTENSIONS, "claim chart")
    validate_file_names(supporting_documents, SUPPORTING_EXTENSIONS, "supporting document")
    validate_duplicate_names(claim_charts, supporting_documents)

    created_at = now_utc()
    chat_id = ObjectId()
    run_id = ObjectId()
    first_filename = Path(claim_charts[0].filename or "New chart").stem
    title = first_filename if len(claim_charts) == 1 else f"{first_filename} + {len(claim_charts) - 1} charts"

    file_records: list[dict] = []
    gridfs_ids: list[ObjectId] = []
    try:
        await store.insert("chats", {
            "_id": chat_id,
            "title": title,
            "createdAt": created_at,
            "updatedAt": created_at,
        })
        await store.insert("analysis_runs", {
            "_id": run_id,
            "chatId": chat_id,
            "prompt": prompt or "",
            "createdAt": created_at,
            "updatedAt": created_at,
            "workflow": "chart_preparation",
            "status": "queued",
        })

        for upload, category, max_bytes in [
            *[(file, "claim_chart", MAX_CLAIM_BYTES) for file in claim_charts],
            *[(file, "supporting_document", MAX_SUPPORTING_BYTES) for file in supporting_documents],
        ]:
            filename = Path((upload.filename or "").strip()).name
            gridfs_id, file_size = await store.store_file(
                filename=filename,
                content_type=upload.content_type,
                metadata={"chatId": chat_id, "runId": run_id, "category": category},
                upload_file=upload,
                max_bytes=max_bytes,
            )
            gridfs_ids.append(gridfs_id)
            file_records.append({
                "_id": ObjectId(),
                "chatId": chat_id,
                "runId": run_id,
                "gridfsId": gridfs_id,
                "filename": filename,
                "normalizedName": filename.lower(),
                "category": category,
                "contentType": upload.content_type,
                "sizeBytes": file_size,
                "createdAt": created_at,
            })

        for record in file_records:
            await store.insert("files", record)
        supporting_records = [
            {
                "_id": ObjectId(),
                "chatId": chat_id,
                "runId": run_id,
                "fileId": record["_id"],
                "filename": record["filename"],
                "gridfsId": record["gridfsId"],
                "status": "stored",
                "createdAt": created_at,
            }
            for record in file_records
            if record["category"] == "supporting_document"
        ]
        await store.insert_many("supporting_documents", supporting_records)
        if prompt and prompt.strip():
            await store.insert("messages", {
                "_id": ObjectId(),
                "chatId": chat_id,
                "runId": run_id,
                "role": "user",
                "content": prompt.strip(),
                "fileIds": [record["_id"] for record in file_records],
                "createdAt": created_at,
            })
    except ValueError as error:
        if store.bucket is not None:
            for gridfs_id in gridfs_ids:
                await store.bucket.delete(gridfs_id)
        if store.database is not None:
            await store.database.files.delete_many({"runId": run_id})
            await store.database.supporting_documents.delete_many({"runId": run_id})
            await store.database.messages.delete_many({"runId": run_id})
            await store.database.analysis_runs.delete_one({"_id": run_id})
            await store.database.chats.delete_one({"_id": chat_id})
        raise HTTPException(status_code=413, detail=str(error)) from error
    except Exception:
        if store.bucket is not None:
            for gridfs_id in gridfs_ids:
                await store.bucket.delete(gridfs_id)
        if store.database is not None:
            await store.database.files.delete_many({"runId": run_id})
            await store.database.supporting_documents.delete_many({"runId": run_id})
            await store.database.messages.delete_many({"runId": run_id})
            await store.database.analysis_runs.delete_one({"_id": run_id})
            await store.database.chats.delete_one({"_id": chat_id})
        raise

    await store.update(
        "analysis_runs",
        {"_id": run_id},
        {"$set": {"status": "processing", "updatedAt": now_utc()}},
    )
    background_tasks.add_task(process_chart_import, chat_id, run_id, file_records)

    claim_file_ids = [str(record["_id"]) for record in file_records if record["category"] == "claim_chart"]
    supporting_file_ids = [str(record["_id"]) for record in file_records if record["category"] == "supporting_document"]
    return ChatCreateResponse(
        chat_id=str(chat_id),
        run_id=str(run_id),
        title=title,
        claim_chart_file_ids=claim_file_ids,
        supporting_file_ids=supporting_file_ids,
        chart_revision_id=None,
        processing_engine="langgraph",
        verification_mode="ollama-chart-evaluation",
        processing_status="processing",
        chart_rows=[],
    )


@app.get("/api/chats/{chat_id}/table", response_model=ChartTableResponse)
async def get_chart_table(chat_id: str, run_id: str) -> ChartTableResponse:
    require_database()
    try:
        chat_object_id = ObjectId(chat_id)
        run_object_id = ObjectId(run_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid chat or run identifier.") from error

    run = await store.find_one(
        "analysis_runs", {"_id": run_object_id, "chatId": chat_object_id}
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Chart run not found.")
    revision_id = run.get("revisionId")
    rows = []
    if run.get("status") == "completed" and isinstance(revision_id, ObjectId):
        rows = await store.find_many("chart_rows", {"revisionId": revision_id})
        rows = sorted(rows, key=lambda item: item["rowIndex"])
    return ChartTableResponse(
        chat_id=chat_id,
        run_id=run_id,
        processing_status=run.get("status", "processing"),
        chart_revision_id=str(revision_id) if isinstance(revision_id, ObjectId) else None,
        error=run.get("error"),
        chart_rows=[serialize_chart_row(row) for row in rows],
    )


def _pdf_text(value: object, fallback: str = "-") -> str:
    """Escape table text while keeping intentional line breaks readable in the PDF."""
    text = str(value or fallback).replace("\r\n", "\n").replace("\r", "\n")
    # ReportLab's built-in Helvetica does not include several typographic
    # characters commonly copied from patent PDFs. Normalize them to their
    # readable ASCII equivalents so the export never contains missing-glyph
    # boxes (for example, U+2011 in “built‑in”).
    text = text.translate(str.maketrans({
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "--",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }))
    return escape(text).replace("\n", "<br/>")


def _build_chart_pdf(*, title: str, rows: list[dict]) -> bytes:
    """Build a paginated, landscape claim-chart PDF from the current revision."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = BytesIO()
    page_width, page_height = landscape(A4)
    document = SimpleDocTemplate(
        output,
        pagesize=(page_width, page_height),
        leftMargin=14 * mm,
        rightMargin=14 * mm,
        topMargin=13 * mm,
        bottomMargin=13 * mm,
        title=title,
        author="iLumos",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ExportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=colors.HexColor("#202426"),
        alignment=TA_LEFT,
        spaceAfter=3 * mm,
    )
    metadata_style = ParagraphStyle(
        "ExportMetadata",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#747979"),
        spaceAfter=5 * mm,
    )
    header_style = ParagraphStyle(
        "ExportHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=10,
        textColor=colors.HexColor("#3d4445"),
    )
    cell_style = ParagraphStyle(
        "ExportCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#33383a"),
        alignment=TA_LEFT,
    )
    confidence_style = ParagraphStyle(
        "ExportConfidence",
        parent=cell_style,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#2f6d4a"),
    )

    table_data = [[
        Paragraph("S. No.", header_style),
        Paragraph("Patent Claim", header_style),
        Paragraph("Evidence of Infringement", header_style),
        Paragraph("AI Evidence", header_style),
    ]]
    for row in sorted(rows, key=lambda item: item.get("rowIndex", 0)):
        confidence_score = row.get("confidenceScore")
        confidence_remark = row.get("confidenceRemark") or ""
        if isinstance(confidence_score, (int, float)):
            ai_content = f"<b>{confidence_score:g}% confidence</b>"
            if confidence_remark:
                ai_content += f"<br/>{_pdf_text(confidence_remark, '')}"
            ai_cell = Paragraph(ai_content, confidence_style)
        else:
            ai_cell = Paragraph(_pdf_text(row.get("aiEvidence") or "-"), cell_style)
        table_data.append([
            Paragraph(str(row.get("rowIndex", "-")), cell_style),
            Paragraph(_pdf_text(row.get("patentClaim")), cell_style),
            Paragraph(_pdf_text(row.get("evidenceOfInfringement")), cell_style),
            ai_cell,
        ])

    table = Table(
        table_data,
        colWidths=[14 * mm, 55 * mm, 98 * mm, 74 * mm],
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2eee7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#3d4445")),
        ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#dedbd5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fcfbf8")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    document.build([
        Paragraph(_pdf_text(title, "iLumos Claim Chart"), title_style),
        Paragraph(
            f"{len(rows)} row{'s' if len(rows) != 1 else ''} - Exported from the latest approved table revision",
            metadata_style,
        ),
        table,
        Spacer(1, 3 * mm),
    ])
    return output.getvalue()


@app.get("/api/chats/{chat_id}/export.pdf")
async def export_chart_pdf(chat_id: str, run_id: str) -> Response:
    """Export the latest persisted chart revision, including current confidence values."""
    require_database()
    try:
        chat_object_id = ObjectId(chat_id)
        run_object_id = ObjectId(run_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid chat or run identifier.") from error

    run = await store.find_one(
        "analysis_runs", {"_id": run_object_id, "chatId": chat_object_id}
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Chart run not found.")
    revision_id = run.get("revisionId")
    if run.get("status") != "completed" or not isinstance(revision_id, ObjectId):
        raise HTTPException(status_code=409, detail="The chart table is not ready to export yet.")

    rows = await store.find_many("chart_rows", {"revisionId": revision_id})
    rows = sorted(rows, key=lambda item: item.get("rowIndex", 0))
    chat = await store.find_one("chats", {"_id": chat_object_id})
    title = str((chat or {}).get("title") or "iLumos Claim Chart")
    pdf_bytes = _build_chart_pdf(title=title, rows=rows)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="ilumos-claim-chart.pdf"'},
    )


@app.post("/api/chats/{chat_id}/attachments", response_model=AttachmentResponse)
async def upload_chat_attachments(
    chat_id: str,
    run_id: Annotated[str, Form(...)],
    supporting_documents: Annotated[list[UploadFile], File(...)],
) -> AttachmentResponse:
    require_database()
    try:
        chat_object_id = ObjectId(chat_id)
        run_object_id = ObjectId(run_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid chat or run identifier.") from error

    run = await store.find_one(
        "analysis_runs", {"_id": run_object_id, "chatId": chat_object_id}
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Chart run not found.")
    validate_file_names(supporting_documents, SUPPORTING_EXTENSIONS, "supporting document")
    validate_duplicate_names([], supporting_documents)
    existing_files = await store.find_many("files", {"runId": run_object_id})
    existing_names = {record.get("normalizedName", "") for record in existing_files}
    for upload in supporting_documents:
        normalized_name = Path((upload.filename or "").strip()).name.lower()
        if normalized_name in existing_names:
            raise HTTPException(status_code=409, detail=f"Duplicate filename: {upload.filename}")

    created_at = now_utc()
    records: list[dict] = []
    gridfs_ids: list[ObjectId] = []
    try:
        for upload in supporting_documents:
            filename = Path((upload.filename or "").strip()).name
            gridfs_id, file_size = await store.store_file(
                filename=filename,
                content_type=upload.content_type,
                metadata={"chatId": chat_object_id, "runId": run_object_id, "category": "supporting_document"},
                upload_file=upload,
                max_bytes=MAX_SUPPORTING_BYTES,
            )
            gridfs_ids.append(gridfs_id)
            records.append(
                {
                    "_id": ObjectId(),
                    "chatId": chat_object_id,
                    "runId": run_object_id,
                    "gridfsId": gridfs_id,
                    "filename": filename,
                    "normalizedName": filename.lower(),
                    "category": "supporting_document",
                    "contentType": upload.content_type,
                    "sizeBytes": file_size,
                    "createdAt": created_at,
                }
            )
        await store.insert_many("files", records)
        await store.insert_many(
            "supporting_documents",
            [
                {
                    "_id": ObjectId(),
                    "chatId": chat_object_id,
                    "runId": run_object_id,
                    "fileId": record["_id"],
                    "filename": record["filename"],
                    "gridfsId": record["gridfsId"],
                    "status": "stored",
                    "createdAt": created_at,
                }
                for record in records
            ],
        )
    except ValueError as error:
        if store.bucket is not None:
            for gridfs_id in gridfs_ids:
                await store.bucket.delete(gridfs_id)
        raise HTTPException(status_code=413, detail=str(error)) from error
    except Exception:
        if store.bucket is not None:
            for gridfs_id in gridfs_ids:
                await store.bucket.delete(gridfs_id)
        if store.database is not None:
            await store.database.files.delete_many({"_id": {"$in": [record["_id"] for record in records]}})
            await store.database.supporting_documents.delete_many({"fileId": {"$in": [record["_id"] for record in records]}})
        raise

    return AttachmentResponse(
        file_ids=[str(record["_id"]) for record in records],
        filenames=[record["filename"] for record in records],
    )


@app.post("/api/chats/{chat_id}/messages", response_model=MessageResponse)
async def create_chat_message(
    chat_id: str,
    background_tasks: BackgroundTasks,
    request: PromptMessageRequest,
) -> MessageResponse:
    require_database()
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")
    try:
        chat_object_id = ObjectId(chat_id)
        run_object_id = ObjectId(request.run_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid chat or run identifier.") from error
    run = await store.find_one(
        "analysis_runs", {"_id": run_object_id, "chatId": chat_object_id}
    )
    if run is None or run.get("status") not in {"processing", "completed"}:
        raise HTTPException(status_code=409, detail="The chart is not ready for messages yet.")
    if len(request.file_ids) != len(set(request.file_ids)):
        raise HTTPException(status_code=400, detail="A message cannot reference the same file twice.")
    try:
        file_object_ids = [ObjectId(file_id) for file_id in request.file_ids]
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid attachment identifier.") from error
    if file_object_ids:
        referenced_files = await store.find_many(
            "files",
            {
                "_id": {"$in": file_object_ids},
                "chatId": chat_object_id,
                "runId": run_object_id,
                "category": "supporting_document",
            },
        )
        if len(referenced_files) != len(file_object_ids):
            raise HTTPException(status_code=400, detail="One or more attachments do not belong to this chart run.")
    message_content = request.content.strip()
    message_id = ObjectId()
    message = {
        "_id": message_id,
        "chatId": chat_object_id,
        "runId": run_object_id,
        "role": "user",
        "content": message_content,
        "fileIds": file_object_ids,
        "status": "queued",
        "createdAt": now_utc(),
    }
    message_id = await store.insert(
        "messages",
        message,
    )
    background_tasks.add_task(
        process_refinement_message,
        chat_object_id,
        run_object_id,
        message_id,
        message_content,
        request.file_ids,
    )
    return MessageResponse(
        message_id=str(message_id),
        status="queued",
        content=message_content,
        file_ids=[str(file_id) for file_id in file_object_ids],
    )


async def _persist_evidence_revision(
    *,
    chat_id: ObjectId,
    run_id: ObjectId,
    run: dict,
    current_revision_id: ObjectId,
    current_rows: list[dict],
    evidence_updates: dict[str, str],
    revision_type: str,
    audit_event: str,
    action_id: ObjectId | None = None,
) -> tuple[ObjectId, list[dict]]:
    revision_id = ObjectId()
    row_documents: list[dict] = []
    for current_row in current_rows:
        row_id = str(current_row["_id"])
        row_document = {
            **current_row,
            "_id": ObjectId(),
            "revisionId": revision_id,
            "createdAt": now_utc(),
        }
        if row_id in evidence_updates:
            row_document["evidenceOfInfringement"] = evidence_updates[row_id]
            row_document["aiEvidence"] = ""
            row_document.pop("confidenceScore", None)
            row_document.pop("confidenceRemark", None)
        row_documents.append(row_document)

    await store.insert_many("chart_rows", row_documents)
    await store.insert(
        "chart_revisions",
        {
            "_id": revision_id,
            "chatId": chat_id,
            "runId": run_id,
            "rowCount": len(row_documents),
            "sourceClaimChartCount": run.get("sourceClaimChartCount", 0),
            "supersedesRevisionId": current_revision_id,
            "revisionType": revision_type,
            "verificationMode": "human-approved-table-refinement",
            "evaluationApproved": True,
            "createdAt": now_utc(),
        },
    )
    await store.update(
        "analysis_runs",
        {"_id": run_id},
        {
            "$set": {
                "revisionId": revision_id,
                "confidenceStatus": "pending",
                "updatedAt": now_utc(),
            },
            "$unset": {"confidenceError": ""},
        },
    )
    audit: dict[str, Any] = {
        "_id": ObjectId(),
        "chatId": chat_id,
        "runId": run_id,
        "event": audit_event,
        "revisionId": revision_id,
        "patchCount": len(evidence_updates),
        "createdAt": now_utc(),
    }
    if action_id is not None:
        audit["actionId"] = action_id
    await store.insert("audit_events", audit)
    return revision_id, row_documents


async def apply_table_action(
    *,
    chat_id: ObjectId,
    run_id: ObjectId,
    action: dict,
) -> tuple[ObjectId, list[dict]]:
    run = await store.find_one("analysis_runs", {"_id": run_id, "chatId": chat_id})
    if run is None or run.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The chart table is not ready for changes.")
    if run.get("confidenceStatus") == "generating":
        raise HTTPException(status_code=409, detail="Wait for confidence scoring to finish before editing evidence.")
    current_revision_id = run.get("revisionId")
    action_revision_id = action.get("sourceRevisionId")
    if not isinstance(current_revision_id, ObjectId) or str(current_revision_id) != str(action_revision_id):
        raise HTTPException(status_code=409, detail="The table changed while this proposal was waiting. Ask iLumos to review it again.")

    current_rows = sorted(
        await store.find_many("chart_rows", {"revisionId": current_revision_id}),
        key=lambda row: row["rowIndex"],
    )
    rows_by_id = {str(row["_id"]): row for row in current_rows}
    evidence_updates: dict[str, str] = {}
    for patch in action.get("patches", []):
        row_id = str(patch.get("rowId", ""))
        column = str(patch.get("column", ""))
        row = rows_by_id.get(row_id)
        if row is None or column != "evidenceOfInfringement":
            raise HTTPException(status_code=409, detail="Only the Evidence of Infringement column can be changed.")
        new_value = patch.get("newValue")
        if not isinstance(new_value, str) or not new_value.strip() or len(new_value) > 16000:
            raise HTTPException(status_code=409, detail="The proposed evidence value is invalid.")
        if str(row.get(column, "")) != str(patch.get("oldValue", "")):
            raise HTTPException(status_code=409, detail="The table changed while this proposal was waiting. Ask iLumos to review it again.")
        evidence_updates[row_id] = new_value

    revision_id, row_documents = await _persist_evidence_revision(
        chat_id=chat_id,
        run_id=run_id,
        run=run,
        current_revision_id=current_revision_id,
        current_rows=current_rows,
        evidence_updates=evidence_updates,
        revision_type="assistant_evidence_refinement",
        audit_event="assistant_evidence_patch_applied",
        action_id=action["_id"],
    )
    await store.update(
        "agent_actions",
        {"_id": action["_id"], "status": "applying"},
        {"$set": {"status": "applied", "appliedRevisionId": revision_id, "decidedAt": now_utc()}},
    )
    return revision_id, row_documents


@app.patch(
    "/api/chats/{chat_id}/rows/{row_id}/evidence",
    response_model=EvidenceEditResponse,
)
async def edit_evidence_cell(
    chat_id: str,
    row_id: str,
    request: EvidenceEditRequest,
) -> EvidenceEditResponse:
    require_database()
    try:
        chat_object_id = ObjectId(chat_id)
        run_object_id = ObjectId(request.run_id)
        row_object_id = ObjectId(row_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid chat, run, or row identifier.") from error

    value = request.value.strip()
    if not value or len(value) > 16000:
        raise HTTPException(status_code=400, detail="Evidence must contain between 1 and 16,000 characters.")
    run = await store.find_one(
        "analysis_runs", {"_id": run_object_id, "chatId": chat_object_id}
    )
    if run is None or run.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The chart table is not ready for changes.")
    if run.get("confidenceStatus") == "generating":
        raise HTTPException(status_code=409, detail="Wait for confidence scoring to finish before editing evidence.")
    current_revision_id = run.get("revisionId")
    if not isinstance(current_revision_id, ObjectId):
        raise HTTPException(status_code=409, detail="The chart has no editable table revision.")
    current_rows = sorted(
        await store.find_many("chart_rows", {"revisionId": current_revision_id}),
        key=lambda row: row["rowIndex"],
    )
    current_row = next((row for row in current_rows if row.get("_id") == row_object_id), None)
    if current_row is None:
        raise HTTPException(status_code=404, detail="The requested table row was not found.")
    if request.old_value != str(current_row.get("evidenceOfInfringement", "")):
        raise HTTPException(status_code=409, detail="The evidence changed while you were editing. Reload the row and try again.")
    if value == request.old_value:
        return EvidenceEditResponse(
            chat_id=chat_id,
            run_id=request.run_id,
            chart_revision_id=str(current_revision_id),
            chart_rows=[serialize_chart_row(row) for row in current_rows],
        )

    revision_id, row_documents = await _persist_evidence_revision(
        chat_id=chat_object_id,
        run_id=run_object_id,
        run=run,
        current_revision_id=current_revision_id,
        current_rows=current_rows,
        evidence_updates={row_id: value},
        revision_type="manual_evidence_edit",
        audit_event="manual_evidence_edit_applied",
    )
    return EvidenceEditResponse(
        chat_id=chat_id,
        run_id=request.run_id,
        chart_revision_id=str(revision_id),
        chart_rows=[serialize_chart_row(row) for row in row_documents],
    )


async def mark_assistant_action_message(
    *,
    action: dict,
    chat_id: ObjectId,
    run_id: ObjectId,
    status: str,
) -> None:
    """Keep the proposal card in sync with its persisted approval record."""
    message_id = action.get("messageId")
    filters: list[dict] = []
    if isinstance(message_id, ObjectId):
        filters.append({"_id": message_id})
    filters.append({"actionId": action.get("_id")})
    proposal_message = await store.find_one(
        "messages",
        {"chatId": chat_id, "runId": run_id, "$or": filters},
    )
    if proposal_message is None:
        proposal_message = await store.find_one(
            "messages",
            {"chatId": chat_id, "runId": run_id, "actionId": action.get("_id")},
        )
    if proposal_message is None:
        proposal_message = await store.find_one(
            "messages",
            {"chatId": chat_id, "runId": run_id, "actionId": str(action.get("_id"))},
        )
    if proposal_message is None:
        # Be tolerant of older action records that did not persist messageId
        # in the same shape; the proposal still carries the actionId.
        messages = await store.find_many("messages", {"chatId": chat_id, "runId": run_id})
        proposal_message = next(
            (
                message for message in messages
                if str(message.get("actionId")) == str(action.get("_id"))
            ),
            None,
        )
    if proposal_message is None:
        return
    updated_action = dict(proposal_message.get("action") or {})
    updated_action["status"] = status
    await store.update(
        "messages",
        {"_id": proposal_message["_id"]},
        {"$set": {"action": updated_action}},
    )


@app.post(
    "/api/chats/{chat_id}/assistant/actions/{action_id}",
    response_model=AssistantActionResponse,
)
async def decide_assistant_action(
    chat_id: str,
    action_id: str,
    request: AssistantActionDecisionRequest,
) -> AssistantActionResponse:
    require_database()
    try:
        chat_object_id = ObjectId(chat_id)
        action_object_id = ObjectId(action_id)
        run_object_id = ObjectId(request.run_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid chat, action, or run identifier.") from error
    action = await store.find_one(
        "agent_actions",
        {"_id": action_object_id, "chatId": chat_object_id, "runId": run_object_id},
    )
    if action is None:
        raise HTTPException(status_code=404, detail="Assistant action not found.")
    if action.get("status") != "pending":
        raise HTTPException(status_code=409, detail="This assistant action has already been decided.")

    if request.decision == "reject":
        await store.update(
            "agent_actions",
            {"_id": action_object_id, "status": "pending"},
            {"$set": {"status": "rejected", "decidedAt": now_utc()}},
        )
        await mark_assistant_action_message(
            action=action,
            chat_id=chat_object_id,
            run_id=run_object_id,
            status="rejected",
        )
        await store.insert(
            "messages",
            {
                "_id": ObjectId(),
                "chatId": chat_object_id,
                "runId": run_object_id,
                "role": "assistant",
                "content": "No table changes were applied.",
                "status": "completed",
                "kind": "action_result",
                "actionId": action_object_id,
                "createdAt": now_utc(),
            },
        )
        await store.insert(
            "audit_events",
            {
                "_id": ObjectId(),
                "chatId": chat_object_id,
                "runId": run_object_id,
                "event": "assistant_table_patch_rejected",
                "actionId": action_object_id,
                "createdAt": now_utc(),
            },
        )
        return AssistantActionResponse(
            action_id=action_id,
            status="rejected",
            message="No table changes were applied.",
        )

    claimed = await store.compare_and_set(
        "agent_actions",
        {"_id": action_object_id, "status": "pending"},
        {"$set": {"status": "applying", "decidedAt": now_utc()}},
    )
    if not claimed:
        raise HTTPException(status_code=409, detail="This assistant action has already been decided.")
    action["status"] = "applying"
    try:
        revision_id, rows = await apply_table_action(
            chat_id=chat_object_id,
            run_id=run_object_id,
            action=action,
        )
    except Exception:
        await store.update(
            "agent_actions",
            {"_id": action_object_id, "status": "applying"},
            {"$set": {"status": "pending", "decidedAt": None}},
        )
        raise
    await mark_assistant_action_message(
        action=action,
        chat_id=chat_object_id,
        run_id=run_object_id,
        status="applied",
    )
    await store.insert(
        "messages",
        {
            "_id": ObjectId(),
            "chatId": chat_object_id,
            "runId": run_object_id,
            "role": "assistant",
            "content": f"Applied {len(action.get('patches', []))} approved table change(s).",
            "status": "completed",
            "kind": "action_result",
            "actionId": action_object_id,
            "createdAt": now_utc(),
        },
    )
    return AssistantActionResponse(
        action_id=action_id,
        status="applied",
        message="Approved changes were applied to a new table revision.",
        chart_revision_id=str(revision_id),
        chart_rows=[serialize_chart_row(row) for row in rows],
    )


@app.get("/api/chats/{chat_id}/messages", response_model=list[ChatMessage])
async def list_chat_messages(chat_id: str, run_id: str) -> list[ChatMessage]:
    require_database()
    try:
        chat_object_id = ObjectId(chat_id)
        run_object_id = ObjectId(run_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid chat or run identifier.") from error
    run = await store.find_one("analysis_runs", {"_id": run_object_id, "chatId": chat_object_id})
    if run is None:
        raise HTTPException(status_code=404, detail="Chart run not found.")
    messages = await store.find_many("messages", {"chatId": chat_object_id, "runId": run_object_id})
    messages.sort(key=lambda item: item.get("createdAt", datetime.min.replace(tzinfo=timezone.utc)))
    return [serialize_message(message) for message in messages if message.get("content")]


@app.post(
    "/api/chats/{chat_id}/confidence",
    response_model=ConfidenceGenerationResponse,
)
async def generate_chat_confidence(
    chat_id: str,
    request: ConfidenceGenerationRequest,
) -> ConfidenceGenerationResponse:
    require_database()
    try:
        chat_object_id = ObjectId(chat_id)
        run_object_id = ObjectId(request.run_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid chat or run identifier.") from error

    run = await store.find_one(
        "analysis_runs", {"_id": run_object_id, "chatId": chat_object_id}
    )
    if run is None or run.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The chart table is not ready yet.")

    if run.get("confidenceStatus") == "completed":
        revision_id = run.get("revisionId")
        existing_rows = await store.find_many("chart_rows", {"revisionId": revision_id})
        return ConfidenceGenerationResponse(
            chat_id=chat_id,
            run_id=request.run_id,
            chart_revision_id=str(revision_id),
            chart_rows=[
                serialize_chart_row(row)
                for row in sorted(existing_rows, key=lambda item: item["rowIndex"])
            ],
        )

    if run.get("confidenceStatus") == "generating":
        raise HTTPException(status_code=409, detail="Confidence generation is already running.")
    if not await claim_confidence_generation(run_object_id):
        refreshed_run = await store.find_one("analysis_runs", {"_id": run_object_id, "chatId": chat_object_id})
        if refreshed_run and refreshed_run.get("confidenceStatus") == "completed":
            revision_id = refreshed_run.get("revisionId")
            existing_rows = await store.find_many("chart_rows", {"revisionId": revision_id})
            return ConfidenceGenerationResponse(
                chat_id=chat_id,
                run_id=request.run_id,
                chart_revision_id=str(revision_id),
                chart_rows=[
                    serialize_chart_row(row)
                    for row in sorted(existing_rows, key=lambda item: item["rowIndex"])
                ],
            )
        raise HTTPException(status_code=409, detail="Confidence generation is already running.")
    try:
        workflow_state = await workflow.run_confidence(
            chat_id=chat_object_id,
            run_id=run_object_id,
        )
    except Exception as error:
        await store.update(
            "analysis_runs",
            {"_id": run_object_id},
            {
                "$set": {
                    "confidenceStatus": "failed",
                    "confidenceError": str(error),
                    "updatedAt": now_utc(),
                }
            },
        )
        raise HTTPException(status_code=422, detail=f"Confidence generation failed: {error}") from error

    return ConfidenceGenerationResponse(
        chat_id=chat_id,
        run_id=request.run_id,
        chart_revision_id=str(workflow_state["revision_id"]),
        chart_rows=[serialize_chart_row(row) for row in workflow_state.get("rows", [])],
    )


@app.get("/api/chats/{chat_id}/confidence/stream")
async def stream_chat_confidence(chat_id: str, run_id: str) -> StreamingResponse:
    require_database()
    try:
        chat_object_id = ObjectId(chat_id)
        run_object_id = ObjectId(run_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="Invalid chat or run identifier.") from error

    run = await store.find_one(
        "analysis_runs", {"_id": run_object_id, "chatId": chat_object_id}
    )
    if run is None or run.get("status") != "completed":
        raise HTTPException(status_code=409, detail="The chart table is not ready yet.")
    if run.get("confidenceStatus") == "generating":
        raise HTTPException(status_code=409, detail="Confidence generation is already running.")
    if run.get("confidenceStatus") != "completed" and not await claim_confidence_generation(run_object_id):
        refreshed_run = await store.find_one("analysis_runs", {"_id": run_object_id, "chatId": chat_object_id})
        if refreshed_run and refreshed_run.get("confidenceStatus") == "completed":
            run = refreshed_run
        else:
            raise HTTPException(status_code=409, detail="Confidence generation is already running.")

    async def events():
        try:
            if run.get("confidenceStatus") == "completed":
                existing_rows = await store.find_many(
                    "chart_rows", {"revisionId": run.get("revisionId")}
                )
                for row in sorted(existing_rows, key=lambda item: item["rowIndex"]):
                    yield f"event: row_completed\ndata: {json.dumps(serialize_chart_row(row))}\n\n"
                yield f"event: completed\ndata: {json.dumps({'chartRevisionId': str(run.get('revisionId')), 'rows': [serialize_chart_row(row) for row in sorted(existing_rows, key=lambda item: item['rowIndex'])]})}\n\n"
                return

            async for item in workflow.iter_confidence(
                chat_id=chat_object_id,
                run_id=run_object_id,
            ):
                if item["type"] == "row_started":
                    payload = {"rowIndex": item["rowIndex"]}
                    yield f"event: row_started\ndata: {json.dumps(payload)}\n\n"
                elif item["type"] == "row_completed":
                    yield f"event: row_completed\ndata: {json.dumps(serialize_chart_row(item['row']))}\n\n"
                elif item["type"] == "completed":
                    state = item["state"]
                    payload = {
                        "chartRevisionId": str(state["revision_id"]),
                        "rows": [serialize_chart_row(row) for row in state.get("rows", [])],
                    }
                    yield f"event: completed\ndata: {json.dumps(payload)}\n\n"
        except asyncio.CancelledError:
            await store.update(
                "analysis_runs",
                {"_id": run_object_id, "confidenceStatus": "generating"},
                {
                    "$set": {
                        "confidenceStatus": "failed",
                        "confidenceError": "Confidence stream disconnected; retry is available.",
                        "updatedAt": now_utc(),
                    }
                },
            )
            raise
        except Exception as error:
            await store.update(
                "analysis_runs",
                {"_id": run_object_id},
                {
                    "$set": {
                        "confidenceStatus": "failed",
                        "confidenceError": str(error),
                        "updatedAt": now_utc(),
                    }
                },
            )
            yield f"event: error\ndata: {json.dumps({'message': str(error)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
