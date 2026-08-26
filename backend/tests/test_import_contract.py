import io
import asyncio
import unittest
from unittest.mock import patch

from openpyxl import Workbook
from bson import ObjectId

from app.extraction import _pdf_table_rows, _remove_duplicate_pdf_lead, extract_claim_chart
from app.llm_evaluator import (
    LLMEvaluationError,
    _assert_source_grounded,
    evaluate_claim_charts_with_ollama,
    request_ollama_json,
)
from app.refinement_agent import REFINEMENT_SYSTEM_PROMPT, RefinementAgent, _ollama_payload
from app.settings import Settings


class ImportContractTests(unittest.TestCase):
    def _refinement_state(self, *, intent: str = "edit", response: dict | None = None) -> dict:
        row_id = ObjectId()
        return {
            "route": {"intent": intent},
            "rows": [{
                "_id": row_id,
                "rowIndex": 1,
                "patentClaim": "A wireless communication module",
                "evidenceOfInfringement": "WiFi connects to the network.",
                "aiEvidence": "",
            }],
            "source_catalog": {
                "doc:source-1": {
                    "sourceId": "doc:source-1",
                    "type": "document",
                    "label": "specification.pdf",
                },
            },
            "response": response or {},
        }

    def test_refinement_graph_uses_same_model_with_isolated_prompt(self) -> None:
        settings = Settings(ollama_model="nemotron-3-nano:30b-cloud")
        payload = _ollama_payload(settings, [], 700)

        self.assertEqual(payload["model"], "nemotron-3-nano:30b-cloud")
        self.assertIn("iLumos Table Assistant", REFINEMENT_SYSTEM_PROMPT)
        self.assertNotIn("claim-chart quality evaluator", REFINEMENT_SYSTEM_PROMPT)
        self.assertIsNotNone(RefinementAgent(object(), settings).graph())

    def test_refinement_read_request_can_never_create_a_patch(self) -> None:
        state = self._refinement_state(
            intent="read",
            response={
                "answer": "The table has one row.",
                "sources": [{"sourceId": "doc:source-1"}],
                "patches": [{
                    "rowId": "not-allowed-for-read",
                    "column": "evidenceOfInfringement",
                    "oldValue": "",
                    "newValue": "Changed",
                    "sourceRefs": ["doc:source-1"],
                }],
            },
        )

        result = RefinementAgent(object(), Settings()).validate_node(state)

        self.assertEqual(result["patches"], [])
        self.assertTrue(any("cannot change" in issue for issue in result["validation_issues"]))

    def test_refinement_edit_requires_existing_row_and_source(self) -> None:
        valid = self._refinement_state(response={
            "answer": "The supporting specification gives a more precise description.",
            "sources": [{"sourceId": "doc:source-1"}],
            "patches": [{
                "rowId": "placeholder",
                "column": "evidenceOfInfringement",
                "oldValue": "WiFi connects to the network.",
                "newValue": "WiFi connects the device to the home network.",
                "reason": "The specification states the fuller behavior.",
                "sourceRefs": ["doc:source-1"],
            }],
        })
        row_id = str(valid["rows"][0]["_id"])
        valid["response"]["patches"][0]["rowId"] = row_id
        result = RefinementAgent(object(), Settings()).validate_node(valid)
        self.assertEqual(len(result["patches"]), 1)
        self.assertEqual(result["patches"][0]["rowId"], row_id)

        invalid = self._refinement_state(response={
            "patches": [{
                "rowId": "placeholder",
                "column": "evidenceOfInfringement",
                "oldValue": "WiFi connects to the network.",
                "newValue": "Unsupported replacement.",
                "sourceRefs": ["doc:missing"],
            }],
        })
        invalid["response"]["patches"][0]["rowId"] = str(invalid["rows"][0]["_id"])
        invalid_result = RefinementAgent(object(), Settings()).validate_node(invalid)
        self.assertEqual(invalid_result["patches"], [])
        self.assertTrue(any("valid source" in issue for issue in invalid_result["validation_issues"]))

    def test_refinement_rejects_read_only_columns(self) -> None:
        state = self._refinement_state(response={
            "answer": "I found a source-backed evidence update.",
            "patches": [
                {
                    "rowId": "placeholder",
                    "column": "patentClaim",
                    "oldValue": "A wireless communication module",
                    "newValue": "A changed claim",
                    "sourceRefs": ["doc:source-1"],
                },
                {
                    "rowId": "placeholder",
                    "column": "aiEvidence",
                    "oldValue": "",
                    "newValue": "A changed confidence remark",
                    "sourceRefs": ["doc:source-1"],
                },
                {
                    "rowId": "placeholder",
                    "column": "evidenceOfInfringement",
                    "oldValue": "WiFi connects to the network.",
                    "newValue": "The device connects to the home network over WiFi.",
                    "sourceRefs": ["doc:source-1"],
                },
            ],
        })
        row_id = str(state["rows"][0]["_id"])
        for patch in state["response"]["patches"]:
            patch["rowId"] = row_id

        result = RefinementAgent(object(), Settings()).validate_node(state)

        self.assertEqual(len(result["patches"]), 1)
        self.assertEqual(result["patches"][0]["column"], "evidenceOfInfringement")
        self.assertEqual(sum("read-only" in issue for issue in result["validation_issues"]), 2)

    def test_row_reference_and_follow_up_are_routed_to_evidence_edit(self) -> None:
        class FakeStore:
            async def insert(self, _collection: str, document: dict) -> ObjectId:
                return document["_id"]

        row_id = ObjectId()
        base_state = {
            "chat_id": ObjectId(),
            "run_id": ObjectId(),
            "message_id": ObjectId(),
            "rows": [{"_id": row_id, "rowIndex": 7}],
            "table_context": '{"rows":[{"rowId":"%s","rowIndex":7}]}' % row_id,
        }
        fake_route = {
            "intent": "clarify",
            "targetRowIds": [],
            "targetColumns": [],
            "useWebSearch": False,
            "searchQuery": "",
        }
        agent = RefinementAgent(FakeStore(), Settings())

        first = {
            **base_state,
            "prompt": "Use this document to refine the evidence of infringement of row 7.",
            "history": [],
        }
        with patch("app.refinement_agent.request_ollama_json", return_value=fake_route):
            first_result = asyncio.run(agent.route_node(first))
        self.assertEqual(first_result["route"]["intent"], "edit")
        self.assertEqual(first_result["route"]["targetRowIds"], [str(row_id)])
        self.assertEqual(first_result["route"]["targetColumns"], ["evidenceOfInfringement"])

        follow_up = {
            **base_state,
            "prompt": "A calibration profile stored in nonvolatile memory. For this row, I want to refine and update the evidence of infringement.",
            "history": [{"role": "user", "content": first["prompt"]}],
        }
        with patch("app.refinement_agent.request_ollama_json", return_value=fake_route):
            follow_up_result = asyncio.run(agent.route_node(follow_up))
        self.assertEqual(follow_up_result["route"]["intent"], "edit")
        self.assertEqual(follow_up_result["route"]["targetRowIds"], [str(row_id)])
        self.assertEqual(follow_up_result["route"]["targetColumns"], ["evidenceOfInfringement"])

        read_turn = {
            **base_state,
            "prompt": "What does row 7 say now?",
            "history": [
                {"role": "user", "content": first["prompt"]},
                {"role": "user", "content": follow_up["prompt"]},
            ],
        }
        with patch("app.refinement_agent.request_ollama_json", return_value={**fake_route, "intent": "read"}):
            read_result = asyncio.run(agent.route_node(read_turn))
        self.assertEqual(read_result["route"]["intent"], "read")
        self.assertEqual(read_result["route"]["targetRowIds"], [str(row_id)])

    def test_manual_evidence_edit_creates_revision_and_resets_row_confidence(self) -> None:
        from app import main as main_module

        chat_id = ObjectId()
        run_id = ObjectId()
        revision_id = ObjectId()
        row_id = ObjectId()

        class FakeStore:
            configured = True

            def __init__(self) -> None:
                self.run = {
                    "_id": run_id,
                    "chatId": chat_id,
                    "status": "completed",
                    "revisionId": revision_id,
                    "confidenceStatus": "completed",
                    "sourceClaimChartCount": 1,
                }
                self.rows = [{
                    "_id": row_id,
                    "revisionId": revision_id,
                    "rowIndex": 7,
                    "patentClaim": "A calibration profile stored in nonvolatile memory.",
                    "evidenceOfInfringement": "The product stores a calibration profile.",
                    "aiEvidence": "84% confidence",
                    "confidenceScore": 84,
                    "confidenceRemark": "Existing confidence",
                    "source": {},
                    "imageRefs": [],
                }]
                self.inserted_rows: list[dict] = []
                self.inserted_revision: dict | None = None

            async def find_one(self, collection: str, _query: dict) -> dict | None:
                return self.run if collection == "analysis_runs" else None

            async def find_many(self, collection: str, _query: dict) -> list[dict]:
                return self.rows if collection == "chart_rows" else []

            async def insert_many(self, collection: str, documents: list[dict]) -> list[ObjectId]:
                if collection == "chart_rows":
                    self.inserted_rows = documents
                return [document["_id"] for document in documents]

            async def insert(self, collection: str, document: dict) -> ObjectId:
                if collection == "chart_revisions":
                    self.inserted_revision = document
                return document["_id"]

            async def update(self, collection: str, _query: dict, update: dict) -> None:
                if collection == "analysis_runs":
                    self.run.update(update.get("$set", {}))

        fake_store = FakeStore()
        request = main_module.EvidenceEditRequest(
            run_id=str(run_id),
            old_value="The product stores a calibration profile.",
            value="The product stores a calibration profile in nonvolatile memory.",
        )
        with patch.object(main_module, "store", fake_store):
            result = asyncio.run(main_module.edit_evidence_cell(
                chat_id=str(chat_id),
                row_id=str(row_id),
                request=request,
            ))

        self.assertNotEqual(result.chart_revision_id, str(revision_id))
        self.assertEqual(result.chart_rows[0]["evidenceOfInfringement"], request.value)
        self.assertEqual(result.chart_rows[0]["aiEvidence"], "")
        self.assertIsNone(result.chart_rows[0]["confidenceScore"])
        self.assertIsNone(result.chart_rows[0]["confidenceRemark"])
        self.assertEqual(fake_store.run["confidenceStatus"], "pending")
        self.assertEqual(fake_store.rows[0]["evidenceOfInfringement"], request.old_value)

    def test_approved_refinement_creates_new_revision_and_resets_confidence(self) -> None:
        from app import main as main_module

        chat_id = ObjectId()
        run_id = ObjectId()
        revision_id = ObjectId()
        row_id = ObjectId()

        class FakeStore:
            def __init__(self) -> None:
                self.run = {
                    "_id": run_id,
                    "chatId": chat_id,
                    "status": "completed",
                    "revisionId": revision_id,
                    "sourceClaimChartCount": 1,
                }
                self.rows = [{
                    "_id": row_id,
                    "revisionId": revision_id,
                    "rowIndex": 1,
                    "patentClaim": "A wireless communication module",
                    "evidenceOfInfringement": "WiFi connects to the network.",
                    "aiEvidence": "85% confidence",
                    "confidenceScore": 85,
                    "confidenceRemark": "Existing confidence",
                }]
                self.inserted_rows: list[dict] = []
                self.inserted_revision: dict | None = None

            async def find_one(self, collection: str, _query: dict) -> dict | None:
                if collection == "analysis_runs":
                    return self.run
                return None

            async def find_many(self, collection: str, _query: dict) -> list[dict]:
                if collection == "chart_rows":
                    return self.rows
                return []

            async def insert_many(self, collection: str, documents: list[dict]) -> list[ObjectId]:
                if collection == "chart_rows":
                    self.inserted_rows = documents
                return [document["_id"] for document in documents]

            async def insert(self, collection: str, document: dict) -> ObjectId:
                if collection == "chart_revisions":
                    self.inserted_revision = document
                return document["_id"]

            async def update(self, collection: str, _query: dict, update: dict) -> None:
                if collection == "analysis_runs":
                    self.run.update(update.get("$set", {}))

        fake_store = FakeStore()
        action = {
            "_id": ObjectId(),
            "sourceRevisionId": revision_id,
            "patches": [{
                "rowId": str(row_id),
                "column": "evidenceOfInfringement",
                "oldValue": "WiFi connects to the network.",
                "newValue": "WiFi connects the device to the home network.",
            }],
        }

        with patch.object(main_module, "store", fake_store):
            new_revision_id, new_rows = asyncio.run(main_module.apply_table_action(
                chat_id=chat_id,
                run_id=run_id,
                action=action,
            ))

        self.assertNotEqual(new_revision_id, revision_id)
        self.assertEqual(new_rows[0]["evidenceOfInfringement"], "WiFi connects the device to the home network.")
        self.assertEqual(new_rows[0]["aiEvidence"], "")
        self.assertNotIn("confidenceScore", new_rows[0])
        self.assertEqual(fake_store.rows[0]["evidenceOfInfringement"], "WiFi connects to the network.")

    def test_user_approval_updates_evidence_then_confidence_stream_regenerates(self) -> None:
        from app import main as main_module
        from app.workflow import ChartWorkflow

        chat_id = ObjectId()
        run_id = ObjectId()
        revision_id = ObjectId()
        row_id = ObjectId()
        action_id = ObjectId()

        class ApprovalStore:
            configured = True

            def __init__(self) -> None:
                self.run = {
                    "_id": run_id,
                    "chatId": chat_id,
                    "status": "completed",
                    "revisionId": revision_id,
                    "confidenceStatus": "completed",
                    "sourceClaimChartCount": 1,
                }
                self.rows = [{
                    "_id": row_id,
                    "revisionId": revision_id,
                    "rowIndex": 7,
                    "patentClaim": "A calibration profile stored in nonvolatile memory.",
                    "evidenceOfInfringement": "The product stores a calibration profile.",
                    "aiEvidence": "84% confidence",
                    "confidenceScore": 84,
                    "confidenceRemark": "Existing confidence",
                    "source": {},
                    "imageRefs": [],
                }]
                self.action = {
                    "_id": action_id,
                    "chatId": chat_id,
                    "runId": run_id,
                    "status": "pending",
                    "sourceRevisionId": revision_id,
                    "patches": [{
                        "rowId": str(row_id),
                        "column": "evidenceOfInfringement",
                        "oldValue": "The product stores a calibration profile.",
                        "newValue": "The product stores a calibration profile in nonvolatile memory.",
                    }],
                }
                self.revisions = {revision_id: {"_id": revision_id, "sourceClaimChartCount": 1}}
                self.inserted_rows: list[dict] = []

            async def find_one(self, collection: str, query: dict) -> dict | None:
                if collection == "analysis_runs":
                    return self.run
                if collection == "agent_actions":
                    return self.action
                if collection == "chart_revisions":
                    return self.revisions.get(query.get("_id"))
                return None

            async def find_many(self, collection: str, _query: dict) -> list[dict]:
                if collection == "chart_rows":
                    return self.rows
                return []

            async def compare_and_set(self, collection: str, query: dict, update: dict) -> bool:
                target = self.action if collection == "agent_actions" else self.run
                expected_status = query.get("status")
                if expected_status is not None:
                    if isinstance(expected_status, dict):
                        allowed = expected_status.get("$in", [])
                        if target.get("confidenceStatus") not in allowed:
                            return False
                    elif target.get("status") != expected_status:
                        return False
                target.update(update.get("$set", {}))
                return True

            async def insert_many(self, collection: str, documents: list[dict]) -> list[ObjectId]:
                if collection == "chart_rows":
                    self.inserted_rows = documents
                return [document["_id"] for document in documents]

            async def insert(self, collection: str, document: dict) -> ObjectId:
                if collection == "chart_revisions":
                    self.revisions[document["_id"]] = document
                return document["_id"]

            async def update(self, collection: str, _query: dict, update: dict) -> None:
                if collection == "analysis_runs":
                    self.run.update(update.get("$set", {}))
                    for field in update.get("$unset", {}):
                        self.run.pop(field, None)
                elif collection == "agent_actions":
                    self.action.update(update.get("$set", {}))

        fake_store = ApprovalStore()
        with patch.object(main_module, "store", fake_store):
            approval = asyncio.run(main_module.decide_assistant_action(
                chat_id=str(chat_id),
                action_id=str(action_id),
                request=main_module.AssistantActionDecisionRequest(
                    run_id=str(run_id),
                    decision="approve",
                ),
            ))

        self.assertEqual(approval.status, "applied")
        self.assertEqual(approval.chart_rows[0]["evidenceOfInfringement"], fake_store.action["patches"][0]["newValue"])
        self.assertEqual(approval.chart_rows[0]["aiEvidence"], "")
        self.assertEqual(fake_store.action["status"], "applied")
        self.assertEqual(fake_store.run["confidenceStatus"], "pending")

        fake_store.rows = fake_store.inserted_rows
        fake_store.run["revisionId"] = ObjectId(approval.chart_revision_id)
        workflow = ChartWorkflow(fake_store, Settings())
        workflow.confidence_graph = lambda: None
        confidence_result = {
            "rows": [{
                "draftRowId": str(fake_store.rows[0]["_id"]),
                "confidenceScore": 92,
                "confidenceRemark": "The evidence directly identifies the claimed storage behavior.",
            }],
            "provider": "test-confidence-provider",
            "model": "test-model",
        }
        with patch.object(main_module, "store", fake_store), patch.object(main_module, "workflow", workflow):
            with patch("app.workflow.generate_confidence_with_ollama", return_value=confidence_result):
                response = asyncio.run(main_module.stream_chat_confidence(
                    chat_id=str(chat_id),
                    run_id=str(run_id),
                ))

                async def collect_events() -> str:
                    chunks = []
                    async for chunk in response.body_iterator:
                        chunks.append(chunk.decode() if isinstance(chunk, bytes) else str(chunk))
                    return "".join(chunks)

                events = asyncio.run(collect_events())

        self.assertIn("event: row_completed", events)
        self.assertIn('"confidenceScore": 92', events)
        self.assertEqual(fake_store.run["confidenceStatus"], "completed")
        self.assertEqual(fake_store.rows[0]["evidenceOfInfringement"], fake_store.action["patches"][0]["newValue"])

        rejected_store = ApprovalStore()
        original_evidence = rejected_store.rows[0]["evidenceOfInfringement"]
        with patch.object(main_module, "store", rejected_store):
            rejection = asyncio.run(main_module.decide_assistant_action(
                chat_id=str(chat_id),
                action_id=str(action_id),
                request=main_module.AssistantActionDecisionRequest(
                    run_id=str(run_id),
                    decision="reject",
                ),
            ))

        self.assertEqual(rejection.status, "rejected")
        self.assertEqual(rejected_store.action["status"], "rejected")
        self.assertEqual(rejected_store.rows[0]["evidenceOfInfringement"], original_evidence)
        self.assertEqual(rejected_store.run["revisionId"], revision_id)
        self.assertEqual(rejected_store.run["confidenceStatus"], "completed")

    def test_xlsx_keeps_review_columns_out_of_evidence(self) -> None:
        workbook = Workbook()
        worksheet = workbook.active
        worksheet.append(["Patent Claim", "Evidence of Infringement", "AI Evidence", "Reviewer Notes"])
        worksheet.append(["A wireless module", "WiFi connects to the network", "", "Do not import"])
        output = io.BytesIO()
        workbook.save(output)
        workbook.close()

        result = extract_claim_chart(
            filename="chart.xlsx",
            data=output.getvalue(),
            source_file_id="file-1",
        )

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]["patentClaim"], "A wireless module")
        self.assertEqual(result.rows[0]["evidenceOfInfringement"], "WiFi connects to the network")

    def test_pdf_merged_header_does_not_hide_claim_body_column(self) -> None:
        class Rect:
            width = 100
            height = 100

            def __iter__(self):
                return iter((0, 0, 100, 100))

        class Row:
            bbox = (0, 0, 100, 20)

        class Table:
            bbox = (0, 0, 100, 100)
            rows = [Row() for _ in range(5)]

            def extract(self):
                return [
                    ["", None, None, ""],
                    ["", "Challenged Claims of the Patent", "", "Evidence of Infringement"],
                    [None, "Patent", None, None],
                    ["", None, None, ""],
                    ["26. A method for searching", None, None, "The accused product performs the method."],
                ]

        class Page:
            rect = Rect()

            def find_tables(self):
                return type("Tables", (), {"tables": [Table()]})()

        rows, _ = _pdf_table_rows(
            Page(),
            page_number=1,
            table_index=0,
            source_file_id="file-1",
            filename="chart.pdf",
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["patentClaim"], "26. A method for searching")
        self.assertEqual(rows[0]["evidenceOfInfringement"], "The accused product performs the method.")

    def test_pdf_duplicate_cover_claim_is_removed(self) -> None:
        rows = [
            {
                "patentClaim": "A non-semantical method for numerically representing objects in a database",
                "source": {"pageNumber": 1},
            },
            {
                "patentClaim": "26. A non-semantical method for numerically representing objects in a computer database",
                "source": {"pageNumber": 2},
            },
        ]

        result = _remove_duplicate_pdf_lead(rows)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"]["pageNumber"], 2)

    def test_source_guard_rejects_recombined_text(self) -> None:
        source = "A wireless module connects to the network."
        _assert_source_grounded("A wireless module", source, "patentClaim")

        with self.assertRaises(LLMEvaluationError):
            _assert_source_grounded("A network wireless module", source, "patentClaim")

    def test_ollama_json_request_retries_one_malformed_response(self) -> None:
        class FakeResponse:
            status_code = 200
            is_error = False
            text = ""

            def __init__(self, content: str) -> None:
                self.content = content

            def json(self) -> dict:
                return {"choices": [{"message": {"content": self.content}}]}

        class FakeClient:
            def __init__(self, response: FakeResponse) -> None:
                self.response = response

            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *_args: object) -> None:
                return None

            async def post(self, *_args: object, **_kwargs: object) -> FakeResponse:
                return self.response

        clients = iter([
            FakeClient(FakeResponse("not json")),
            FakeClient(FakeResponse('{"rows": []}')),
        ])
        settings = Settings(ollama_base_url="https://ollama.com/v1", ollama_api_key="test")

        with patch("app.llm_evaluator.httpx.AsyncClient", side_effect=lambda **_kwargs: next(clients)):
            result = asyncio.run(request_ollama_json(
                settings=settings,
                payload={"messages": []},
                purpose="test",
            ))

        self.assertEqual(result, {"rows": []})

    def test_ollama_json_request_extracts_fenced_json_after_model_prose(self) -> None:
        class FakeResponse:
            status_code = 200
            is_error = False
            text = ""

            def json(self) -> dict:
                return {
                    "choices": [{
                        "message": {
                            "content": "I reviewed the row.\n```json\n"
                            '{"rows":[{"draftRowId":"row-1","confidenceScore":72,"remark":"Direct feature overlap."}]}\n'
                            "```",
                        }
                    }]
                }

        class FakeClient:
            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *_args: object) -> None:
                return None

            async def post(self, *_args: object, **_kwargs: object) -> FakeResponse:
                return FakeResponse()

        settings = Settings(ollama_base_url="https://ollama.com/v1", ollama_api_key="test")
        with patch("app.llm_evaluator.httpx.AsyncClient", return_value=FakeClient()):
            result = asyncio.run(request_ollama_json(
                settings=settings,
                payload={"messages": []},
                purpose="confidence evaluation",
            ))

        self.assertEqual(result["rows"][0]["confidenceScore"], 72)

    def test_invalid_evaluator_correction_keeps_deterministic_source_text(self) -> None:
        settings = Settings(ollama_base_url="https://ollama.com/v1", ollama_api_key="test")
        documents = [{
            "documentId": "doc-1",
            "filename": "chart.pdf",
            "sourceText": "A wireless module connects to the network. WiFi connects to the network.",
            "rows": [{
                "draftRowId": "row-1",
                "patentClaim": "A wireless module",
                "evidenceOfInfringement": "WiFi connects to the network.",
                "source": {"fileId": "doc-1"},
                "imageRefs": [],
            }],
        }]

        with patch(
            "app.llm_evaluator.request_ollama_json",
            return_value={
                "approved": True,
                "issues": [],
                "corrections": [{
                    "draftRowId": "row-1",
                    "patentClaim": "A network wireless module",
                }],
            },
        ):
            result = asyncio.run(evaluate_claim_charts_with_ollama(settings=settings, documents=documents))

        self.assertEqual(result["rows"][0]["patentClaim"], "A wireless module")
        self.assertTrue(any("deterministic source text" in issue for issue in result["issues"]))

    def test_evaluator_warning_does_not_block_source_preserving_rows(self) -> None:
        settings = Settings(ollama_base_url="https://ollama.com/v1", ollama_api_key="test")
        documents = [{
            "documentId": "doc-1",
            "filename": "chart.pdf",
            "sourceText": "A wireless module connects to the network.",
            "rows": [{
                "draftRowId": "row-1",
                "patentClaim": "A wireless module",
                "evidenceOfInfringement": "connects to the network",
                "source": {"fileId": "doc-1"},
                "imageRefs": [],
            }],
        }]

        with patch(
            "app.llm_evaluator.request_ollama_json",
            return_value={
                "approved": False,
                "issues": ["A placeholder was detected in the model analysis."],
                "corrections": [],
            },
        ):
            result = asyncio.run(evaluate_claim_charts_with_ollama(settings=settings, documents=documents))

        self.assertTrue(result["approved"])
        self.assertFalse(result["llmApproved"])
        self.assertEqual(len(result["rows"]), 1)


if __name__ == "__main__":
    unittest.main()
