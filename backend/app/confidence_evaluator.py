from __future__ import annotations

import json
import re
from typing import Any

from .llm_evaluator import LLMEvaluationError, request_ollama_json
from .settings import Settings


CONFIDENCE_SYSTEM_PROMPT = """You are a patent claim-chart triage analyst.

For every supplied row, compare only the Patent Claim with the Evidence of Infringement.
Estimate how strongly the evidence supports a potential infringement mapping. This is a
triage confidence score, not a legal conclusion, legal advice, or a substitute for human
patent counsel.

Your entire response must be one valid JSON object in this exact shape:
{"rows":[{"draftRowId":"...","confidenceScore":0,"remark":"..."}]}

Rules:
- Do not include analysis, reasoning, markdown, code fences, or any text before or after the JSON object.
- Return exactly one result for every supplied draftRowId, with no extra rows.
- Copy each supplied draftRowId exactly; do not invent, reorder, or alter row IDs.
- confidenceScore must be an integer from 0 to 100.
- 0 means the evidence does not meaningfully support the claim element; 100 means the
  supplied evidence appears to directly and specifically support it.
- Keep the remark short, specific, and tied only to the two supplied columns.
- Do not change, rewrite, or return either source column.
- Do not use images, outside knowledge, or supporting documents.
"""


def _confidence_payload(rows: list[dict[str, Any]]) -> str:
    lines = [
        "[CONFIDENCE_EVALUATION]",
        "Review every row independently. The first field is the patent claim; the second is the evidence.",
        "[ROWS]",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            json.dumps(
                {
                    # Compact keys reduce malformed IDs when the source ID is a long
                    # PDF-derived location string. The evaluator maps these back below.
                    "draftRowId": f"row-{index}",
                    "patentClaim": row.get("patentClaim", ""),
                    "evidenceOfInfringement": row.get("evidenceOfInfringement", ""),
                },
                ensure_ascii=False,
            )
        )
    lines.extend(
        [
            "[REQUIRED_DRAFT_ROW_IDS]",
            json.dumps(
                [f"row-{index}" for index in range(1, len(rows) + 1)],
                ensure_ascii=False,
            ),
            "[END_CONFIDENCE_EVALUATION]",
        ]
    )
    return "\n".join(lines)


def _score(value: Any) -> int:
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if not match:
            raise LLMEvaluationError("The confidence evaluator returned an invalid score.")
        parsed = float(match.group(0))
        if "%" not in value and 0 <= parsed <= 1:
            parsed *= 100
    elif isinstance(value, (int, float)):
        parsed = float(value * 100 if 0 <= value <= 1 else value)
    else:
        raise LLMEvaluationError("The confidence evaluator returned an invalid score.")
    if parsed < 0 or parsed > 100:
        raise LLMEvaluationError("The confidence evaluator returned a score outside 0-100.")
    return round(parsed)


async def generate_confidence_with_ollama(
    *,
    settings: Settings,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not settings.ollama_api_key and settings.ollama_base_url.startswith("https://ollama.com"):
        raise LLMEvaluationError("OLLAMA_API_KEY is required for direct Ollama Cloud API access.")
    if not rows:
        raise LLMEvaluationError("There are no chart rows to score.")

    actual_ids = [str(row["draftRowId"]) for row in rows]
    if len(set(actual_ids)) != len(actual_ids):
        raise LLMEvaluationError("The chart contains duplicate row IDs for confidence scoring.")
    model_ids = [f"row-{index}" for index in range(1, len(rows) + 1)]
    actual_by_model_id = dict(zip(model_ids, actual_ids, strict=True))

    user_content = _confidence_payload(rows)
    openai_compatible = settings.ollama_base_url.rstrip("/").endswith("/v1")
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": CONFIDENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "temperature": 0,
        "seed": 42,
        # One row is scored per graph step; keep the completion small so the
        # cloud model has less room to drift into prose or truncate JSON.
        "max_tokens": 900,
    }
    if openai_compatible:
        payload["response_format"] = {"type": "json_object"}
        payload["reasoning_effort"] = "none"
    else:
        payload["options"] = {"temperature": 0, "seed": 42}
    result = await request_ollama_json(
        settings=settings,
        payload=payload,
        purpose="confidence evaluation",
    )
    evaluated_rows = result.get("rows")
    if not isinstance(evaluated_rows, list):
        raise LLMEvaluationError("The confidence evaluator did not return rows.")

    expected_ids = model_ids
    expected_set = set(expected_ids)
    seen: set[str] = set()
    scores: list[dict[str, Any]] = []
    for evaluated in evaluated_rows:
        if not isinstance(evaluated, dict):
            raise LLMEvaluationError("The confidence evaluator returned a malformed row.")
        draft_id = str(evaluated.get("draftRowId", ""))
        if draft_id not in expected_set or draft_id in seen:
            raise LLMEvaluationError("The confidence evaluator returned an unknown or duplicate row.")
        remark = re.sub(r"\s+", " ", str(evaluated.get("remark", "")).strip())
        if not remark:
            raise LLMEvaluationError(f"The confidence evaluator returned no remark for {draft_id}.")
        seen.add(draft_id)
        scores.append(
            {
                "draftRowId": draft_id,
                "confidenceScore": _score(evaluated.get("confidenceScore", evaluated.get("score"))),
                "confidenceRemark": remark[:320],
            }
        )

    if seen != expected_set:
        missing = sorted(expected_set - seen)
        raise LLMEvaluationError(
            "The confidence evaluator did not score every row: " + ", ".join(missing[:5])
        )

    scores_by_id = {item["draftRowId"]: item for item in scores}
    return {
        "rows": [
            {
                **scores_by_id[model_id],
                "draftRowId": actual_by_model_id[model_id],
            }
            for model_id in expected_ids
        ],
        "provider": "ollama-confidence-evaluator",
        "model": settings.ollama_model,
    }
