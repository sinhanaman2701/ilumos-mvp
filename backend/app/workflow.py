from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, TypedDict

from bson import ObjectId

from .confidence_evaluator import generate_confidence_with_ollama
from .db import MongoStore
from .extraction import ExtractionError, extract_claim_chart
from .llm_evaluator import LLMEvaluationError, evaluate_claim_charts_with_ollama
from .settings import Settings

try:
    from langgraph.graph import END, StateGraph
except ImportError:
    END = None
    StateGraph = None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowState(TypedDict, total=False):
    chat_id: ObjectId
    run_id: ObjectId
    claim_files: list[dict[str, Any]]
    supporting_files: list[dict[str, Any]]
    draft_documents: list[dict[str, Any]]
    rows: list[dict[str, Any]]
    evaluation: dict[str, Any]
    confidence_generation: dict[str, Any]
    verification_results: list[dict[str, Any]]
    revision_id: ObjectId
    source_revision_id: ObjectId
    verification_mode: str
    engine: str


class ChartWorkflow:
    def __init__(self, store: MongoStore, settings: Settings) -> None:
        self.store = store
        self.settings = settings

    async def checkpoint(self, state: WorkflowState, node: str) -> None:
        await self.store.insert(
            "langgraph_checkpoints",
            {
                "_id": ObjectId(),
                "chatId": state["chat_id"],
                "runId": state["run_id"],
                "node": node,
                "rowCount": len(state.get("rows", [])),
                "draftDocumentCount": len(state.get("draft_documents", [])),
                "createdAt": now_utc(),
            },
        )

    async def extract_node(self, state: WorkflowState) -> WorkflowState:
        draft_documents: list[dict[str, Any]] = []
        for file_record in state["claim_files"]:
            data = await self.store.read_file(file_record["gridfsId"])
            try:
                extracted = extract_claim_chart(
                    filename=file_record["filename"],
                    data=data,
                    source_file_id=str(file_record["_id"]),
                )
            except ExtractionError:
                raise

            asset_storage_limit = asyncio.Semaphore(6)

            async def store_asset(asset: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
                async with asset_storage_limit:
                    asset_metadata_id = ObjectId()
                    gridfs_id, size_bytes = await self.store.store_bytes(
                        filename=asset["filename"],
                        content_type=asset["contentType"],
                        metadata={
                            "chatId": state["chat_id"],
                            "runId": state["run_id"],
                            "category": "chart_asset",
                            "parentFileId": file_record["_id"],
                            "kind": asset["kind"],
                            "source": asset["source"],
                        },
                        data=asset["bytes"],
                    )
                metadata = {
                    "_id": asset_metadata_id,
                    "chatId": state["chat_id"],
                    "runId": state["run_id"],
                    "gridfsId": gridfs_id,
                    "filename": asset["filename"],
                    "normalizedName": f"__asset__{asset['assetKey']}",
                    "category": "chart_asset",
                    "assetKind": asset["kind"],
                    "parentFileId": file_record["_id"],
                    "contentType": asset["contentType"],
                    "sizeBytes": size_bytes,
                    "source": asset["source"],
                    "createdAt": now_utc(),
                }
                reference = {
                    "assetId": str(asset_metadata_id),
                    "gridfsId": str(gridfs_id),
                    "filename": asset["filename"],
                    "kind": asset["kind"],
                    "source": asset["source"],
                }
                return asset["assetKey"], metadata, reference

            stored_assets = await asyncio.gather(*(store_asset(asset) for asset in extracted.assets))
            asset_refs_by_key: dict[str, dict[str, Any]] = {}
            asset_metadata: list[dict[str, Any]] = []
            for asset_key, metadata, reference in stored_assets:
                asset_metadata.append(metadata)
                asset_refs_by_key[asset_key] = reference
            await self.store.insert_many("files", asset_metadata)

            rows: list[dict[str, Any]] = []
            for row in extracted.rows:
                rows.append(
                    {
                        **row,
                        "imageRefs": [
                            asset_refs_by_key[key]
                            for key in row.get("assetKeys", [])
                            if key in asset_refs_by_key
                        ],
                    }
                )
                rows[-1].pop("assetKeys", None)

            draft_documents.append(
                {
                    "documentId": str(file_record["_id"]),
                    "filename": file_record["filename"],
                    "sourceText": extracted.source_text,
                    "rows": rows,
                }
            )

        if not draft_documents or not any(document["rows"] for document in draft_documents):
            raise ExtractionError("No claim-chart rows were extracted.")
        next_state = {**state, "draft_documents": draft_documents}
        await self.checkpoint(next_state, "deterministic_extract_chart")
        return next_state

    async def evaluate_node(self, state: WorkflowState) -> WorkflowState:
        try:
            evaluation = await evaluate_claim_charts_with_ollama(
                settings=self.settings,
                documents=state["draft_documents"],
            )
        except LLMEvaluationError as error:
            # The source-preserving draft is still usable when the evaluator
            # is unavailable or cannot satisfy its response contract. Keep
            # the LLM failure visible in run metadata, but do not discard the
            # analyst's uploaded chart.
            fallback_rows = [
                {
                    **row,
                    "aiEvidence": "",
                    "changes": [],
                }
                for document in state["draft_documents"]
                for row in document["rows"]
            ]
            evaluation = {
                "approved": True,
                "llmApproved": False,
                "issues": [f"AI evaluation unavailable; source-preserving extraction was used: {error}"],
                "rows": fallback_rows,
                "provider": "deterministic-source-fallback",
                "model": self.settings.ollama_model,
            }
        next_state = {
            **state,
            "evaluation": evaluation,
            "rows": evaluation["rows"],
        }
        await self.checkpoint(next_state, "llm_evaluate_chart")
        return next_state

    async def merge_node(self, state: WorkflowState) -> WorkflowState:
        merged_rows = [
            {**row, "rowIndex": index}
            for index, row in enumerate(state.get("rows", []), start=1)
        ]
        next_state = {**state, "rows": merged_rows}
        await self.checkpoint(next_state, "merge_approved_charts")
        return next_state

    async def confidence_node(self, state: WorkflowState) -> WorkflowState:
        rows = state.get("rows", [])
        if len(rows) != 1:
            raise RuntimeError("Confidence generation expects one chart row at a time.")
        confidence = await generate_confidence_with_ollama(
            settings=self.settings,
            rows=rows,
        )
        scores_by_id = {item["draftRowId"]: item for item in confidence["rows"]}
        scored_rows: list[dict[str, Any]] = []
        for row in state.get("rows", []):
            score = scores_by_id[row["draftRowId"]]
            scored_rows.append(
                {
                    **row,
                    "confidenceScore": score["confidenceScore"],
                    "confidenceRemark": score["confidenceRemark"],
                    "aiEvidence": (
                        f"Confidence: {score['confidenceScore']}%\n"
                        f"{score['confidenceRemark']}"
                    ),
                }
            )
        next_state = {
            **state,
            "confidence_generation": confidence,
            "rows": scored_rows,
        }
        await self.checkpoint(next_state, "llm_confidence_score")
        return next_state

    async def verify_node(self, state: WorkflowState) -> WorkflowState:
        evaluation = state.get("evaluation", {})
        if evaluation.get("approved") is not True:
            raise RuntimeError("The LLM evaluation did not approve the chart.")

        source_ids = {document["documentId"] for document in state.get("draft_documents", [])}
        verification_results: list[dict[str, Any]] = []
        for row in state.get("rows", []):
            source_id = str(row.get("source", {}).get("fileId", ""))
            checks = {
                "patentClaimPresent": bool(row.get("patentClaim")),
                "evidencePresent": bool(row.get("evidenceOfInfringement")),
                "sourceDocumentExists": source_id in source_ids,
                "imageReferencesPreserved": isinstance(row.get("imageRefs", []), list),
            }
            verification_results.append(
                {
                    "_id": ObjectId(),
                    "rowId": None,
                    "runId": state["run_id"],
                    "verified": all(checks.values()),
                    "checks": checks,
                    "provider": evaluation.get("provider", "ollama-chart-evaluator"),
                    "model": evaluation.get("model", self.settings.ollama_model),
                    "createdAt": now_utc(),
                }
            )
        if not verification_results or not all(result["verified"] for result in verification_results):
            raise RuntimeError("The evaluated chart failed final source validation.")
        next_state = {
            **state,
            "verification_results": verification_results,
            "verification_mode": (
                "ollama-chart-evaluation"
                if evaluation.get("llmApproved", True)
                else "deterministic-source-validation-with-llm-warning"
            ),
        }
        await self.checkpoint(next_state, "final_source_validation")
        return next_state

    async def persist_node(self, state: WorkflowState) -> WorkflowState:
        revision_id = ObjectId()
        row_documents: list[dict[str, Any]] = []
        verification_results = state.get("verification_results", [])

        if len(state.get("rows", [])) != len(verification_results):
            raise RuntimeError("The chart rows and verification results do not have the same length.")

        for row, verification in zip(state.get("rows", []), verification_results, strict=True):
            row_id = ObjectId()
            row_documents.append(
                {
                    "_id": row_id,
                    "chatId": state["chat_id"],
                    "runId": state["run_id"],
                    "revisionId": revision_id,
                    "rowIndex": row["rowIndex"],
                    "patentClaim": row["patentClaim"],
                    "evidenceOfInfringement": row["evidenceOfInfringement"],
                    "aiEvidence": row.get("aiEvidence", ""),
                    "source": row["source"],
                    "imageRefs": row.get("imageRefs", []),
                    "llmChanges": row.get("changes", []),
                    "createdAt": now_utc(),
                }
            )
            verification["rowId"] = row_id

        await self.store.insert_many("chart_rows", row_documents)
        await self.store.insert_many("verification_results", verification_results)
        await self.store.insert_many(
            "supporting_documents",
            [
                {
                    "_id": ObjectId(),
                    "chatId": state["chat_id"],
                    "runId": state["run_id"],
                    "fileId": file_record["_id"],
                    "filename": file_record["filename"],
                    "gridfsId": file_record["gridfsId"],
                    "status": "stored",
                    "createdAt": now_utc(),
                }
                for file_record in state.get("supporting_files", [])
            ],
        )
        await self.store.insert(
            "chart_revisions",
            {
                "_id": revision_id,
                "chatId": state["chat_id"],
                "runId": state["run_id"],
                "rowCount": len(row_documents),
                "sourceClaimChartCount": len(state["claim_files"]),
                "verificationMode": state.get(
                    "verification_mode", "ollama-chart-evaluation"
                ),
                "evaluationApproved": True,
                "llmEvaluationApproved": state.get("evaluation", {}).get("llmApproved", True),
                "evaluationIssues": state.get("evaluation", {}).get("issues", []),
                "evaluationProvider": state.get("evaluation", {}).get("provider"),
                "evaluationModel": state.get("evaluation", {}).get("model"),
                "createdAt": now_utc(),
            },
        )
        await self.store.update(
            "analysis_runs",
            {"_id": state["run_id"]},
            {
                "$set": {
                    "status": "completed",
                    "revisionId": revision_id,
                    "rowCount": len(row_documents),
                    "verificationMode": state.get(
                        "verification_mode", "ollama-chart-evaluation"
                    ),
                    "evaluationApproved": True,
                    "llmEvaluationApproved": state.get("evaluation", {}).get("llmApproved", True),
                    "evaluationIssues": state.get("evaluation", {}).get("issues", []),
                    "confidenceStatus": "pending",
                    "updatedAt": now_utc(),
                }
            },
        )
        await self.store.update(
            "chats",
            {"_id": state["chat_id"]},
            {"$set": {"updatedAt": now_utc()}},
        )
        await self.store.insert(
            "audit_events",
            {
                "_id": ObjectId(),
                "chatId": state["chat_id"],
                "runId": state["run_id"],
                "event": "evaluated_chart_revision_created",
                "revisionId": revision_id,
                "createdAt": now_utc(),
            },
        )
        next_state = {**state, "revision_id": revision_id}
        await self.checkpoint(next_state, "persist_evaluated_chart")
        return next_state

    def confidence_graph(self) -> Any:
        if StateGraph is None or END is None:
            return None
        builder = StateGraph(WorkflowState)
        builder.add_node("confidence", self.confidence_node)
        builder.set_entry_point("confidence")
        builder.add_edge("confidence", END)
        return builder.compile()

    async def persist_confidence_node(self, state: WorkflowState) -> WorkflowState:
        source_revision_id = state["source_revision_id"]
        source_revision = await self.store.find_one(
            "chart_revisions", {"_id": source_revision_id}
        )
        source_rows = await self.store.find_many(
            "chart_rows", {"revisionId": source_revision_id}
        )
        source_rows_by_id = {str(row["_id"]): row for row in source_rows}
        revision_id = ObjectId()
        row_documents: list[dict[str, Any]] = []
        for row in sorted(state.get("rows", []), key=lambda item: item["rowIndex"]):
            source_row = source_rows_by_id.get(row["draftRowId"])
            if source_row is None:
                raise RuntimeError("A confidence result did not map to a persisted chart row.")
            row_documents.append(
                {
                    "_id": ObjectId(),
                    "chatId": state["chat_id"],
                    "runId": state["run_id"],
                    "revisionId": revision_id,
                    "rowIndex": source_row["rowIndex"],
                    "patentClaim": source_row["patentClaim"],
                    "evidenceOfInfringement": source_row["evidenceOfInfringement"],
                    "aiEvidence": row["aiEvidence"],
                    "confidenceScore": row["confidenceScore"],
                    "confidenceRemark": row["confidenceRemark"],
                    "source": source_row["source"],
                    "imageRefs": source_row.get("imageRefs", []),
                    "llmChanges": source_row.get("llmChanges", []),
                    "createdAt": now_utc(),
                }
            )

        await self.store.insert_many("chart_rows", row_documents)
        await self.store.insert(
            "chart_revisions",
            {
                "_id": revision_id,
                "chatId": state["chat_id"],
                "runId": state["run_id"],
                "rowCount": len(row_documents),
                "sourceClaimChartCount": (source_revision or {}).get("sourceClaimChartCount", 0),
                "supersedesRevisionId": source_revision_id,
                "verificationMode": "ollama-chart-evaluation-and-confidence-generation",
                "evaluationApproved": True,
                "confidenceProvider": state.get("confidence_generation", {}).get("provider"),
                "confidenceModel": state.get("confidence_generation", {}).get("model"),
                "createdAt": now_utc(),
            },
        )
        await self.store.update(
            "analysis_runs",
            {"_id": state["run_id"]},
            {
                "$set": {
                    "revisionId": revision_id,
                    "confidenceStatus": "completed",
                    "confidenceProvider": state.get("confidence_generation", {}).get("provider"),
                    "confidenceModel": state.get("confidence_generation", {}).get("model"),
                    "updatedAt": now_utc(),
                }
            },
        )
        await self.store.insert(
            "audit_events",
            {
                "_id": ObjectId(),
                "chatId": state["chat_id"],
                "runId": state["run_id"],
                "event": "confidence_generation_completed",
                "revisionId": revision_id,
                "createdAt": now_utc(),
            },
        )
        return {**state, "revision_id": revision_id, "rows": row_documents}

    async def run_confidence(self, *, chat_id: ObjectId, run_id: ObjectId) -> WorkflowState:
        final_state: WorkflowState | None = None
        async for event in self.iter_confidence(chat_id=chat_id, run_id=run_id):
            if event["type"] == "completed":
                final_state = event["state"]
        if final_state is None:
            raise RuntimeError("Confidence generation did not complete.")
        return final_state

    async def iter_confidence(
        self,
        *,
        chat_id: ObjectId,
        run_id: ObjectId,
    ) -> AsyncIterator[dict[str, Any]]:
        run = await self.store.find_one(
            "analysis_runs", {"_id": run_id, "chatId": chat_id}
        )
        if run is None or run.get("status") != "completed":
            raise RuntimeError("The chart table is not ready for confidence generation.")
        source_revision_id = run.get("revisionId")
        if not isinstance(source_revision_id, ObjectId):
            raise RuntimeError("The chart has no persisted table revision.")
        source_rows = await self.store.find_many(
            "chart_rows", {"revisionId": source_revision_id}
        )
        if not source_rows:
            raise RuntimeError("The chart table has no rows to score.")
        source_rows = sorted(source_rows, key=lambda item: item["rowIndex"])
        scored_rows: list[dict[str, Any]] = []
        confidence_generation: dict[str, Any] = {}
        graph = self.confidence_graph()
        for source_row in source_rows:
            yield {"type": "row_started", "rowIndex": source_row["rowIndex"]}
            state: WorkflowState = {
                "chat_id": chat_id,
                "run_id": run_id,
                "source_revision_id": source_revision_id,
                "rows": [
                    {
                        **source_row,
                        "draftRowId": str(source_row["_id"]),
                    }
                ],
            }
            if graph is not None:
                row_state = await graph.ainvoke(state)
            else:
                row_state = await self.confidence_node(state)
            confidence_generation = row_state.get("confidence_generation", confidence_generation)
            scored_row = row_state["rows"][0]
            scored_rows.append(scored_row)
            yield {"type": "row_completed", "row": scored_row}

        final_state: WorkflowState = {
            "chat_id": chat_id,
            "run_id": run_id,
            "source_revision_id": source_revision_id,
            "rows": scored_rows,
            "confidence_generation": confidence_generation,
        }
        yield {"type": "completed", "state": await self.persist_confidence_node(final_state)}

    def graph(self) -> Any:
        if StateGraph is None or END is None:
            return None
        builder = StateGraph(WorkflowState)
        builder.add_node("extract", self.extract_node)
        builder.add_node("evaluate", self.evaluate_node)
        builder.add_node("merge", self.merge_node)
        builder.add_node("verify", self.verify_node)
        builder.add_node("persist", self.persist_node)
        builder.set_entry_point("extract")
        builder.add_edge("extract", "evaluate")
        builder.add_edge("evaluate", "merge")
        builder.add_edge("merge", "verify")
        builder.add_edge("verify", "persist")
        builder.add_edge("persist", END)
        return builder.compile()

    async def run(
        self,
        *,
        chat_id: ObjectId,
        run_id: ObjectId,
        claim_files: list[dict[str, Any]],
        supporting_files: list[dict[str, Any]],
    ) -> WorkflowState:
        state: WorkflowState = {
            "chat_id": chat_id,
            "run_id": run_id,
            "claim_files": claim_files,
            "supporting_files": supporting_files,
            "draft_documents": [],
            "rows": [],
            "verification_results": [],
            "engine": "langgraph" if self.graph() is not None else "fallback",
        }
        graph = self.graph()
        if graph is not None:
            return await graph.ainvoke(state)

        for node in (
            self.extract_node,
            self.evaluate_node,
            self.merge_node,
            self.verify_node,
            self.persist_node,
        ):
            state = await node(state)
        return state
