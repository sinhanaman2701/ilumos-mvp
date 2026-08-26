from __future__ import annotations

import json
import re
from typing import Any

import httpx

from .settings import Settings


EVALUATOR_SYSTEM_PROMPT = """You are the claim-chart quality evaluator for a patent-analysis application.

The deterministic importer has already read the uploaded claim-chart documents and created draft rows.
Compare every draft row with the labeled source text before it is shown to the analyst.

Return only JSON with approved, issues, and corrections. The corrections array may be empty. Do not
return a new full row set. Each correction must contain one existing draftRowId and only the fields
that need a source-grounded correction. The draftRowId list is authoritative; never create rows for
sentences, sub-elements, or paragraphs that are not represented by one of those IDs.
You may correct a row only by copying or recombining text from that same document's source text.
If the draft contains a clipped fragment, repair it only when the intact wording appears elsewhere
in that same document's source text. Do not paraphrase, summarize, or rewrite legal wording.
Keep the draft source document and locator.
Do not interpret images or relocate image references. aiEvidence must always be an empty string.
Set approved=true only when every draft row has been reviewed and the draft is complete or any
correction is source-grounded. Report uncertainty in issues, but do not reject a row merely because
you are uncertain or because a placeholder appears in your own analysis. The backend decides whether
the deterministic source text is safe to persist. Do not interpret images or relocate image
references; aiEvidence remains empty in the application.
"""


class LLMEvaluationError(RuntimeError):
    pass


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _contains_source_span(value: str, source_text: str) -> bool:
    """Require the candidate text to appear in source order, not just share vocabulary."""
    candidate_tokens = _tokens(value)
    source_tokens = _tokens(source_text)
    if not candidate_tokens:
        return True
    if len(candidate_tokens) == 1:
        return candidate_tokens[0] in source_tokens
    width = len(candidate_tokens)
    return any(
        source_tokens[index : index + width] == candidate_tokens
        for index in range(len(source_tokens) - width + 1)
    )


def _assert_source_grounded(value: str, source_text: str, field_name: str) -> None:
    # Preserve source wording and order. A vocabulary-only check would allow
    # the model to invent a new sentence by recombining unrelated source words.
    if not _contains_source_span(value, source_text):
        raise LLMEvaluationError(
            f"The evaluator introduced unsupported or reordered text in {field_name}."
        )


def _parse_json(
    content: str,
    *,
    shape_keys: tuple[str, ...] = ("corrections", "rows", "issues", "warnings"),
) -> dict[str, Any]:
    """Extract the first evaluator-shaped JSON value from a model response.

    Cloud models can still add a short preamble, a markdown fence, or more than
    one JSON-looking fragment even when JSON mode is requested. Do not repair or
    reinterpret malformed JSON; only accept a complete value that the standard
    decoder can parse, then validate its expected top-level shape.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []

    # raw_decode lets us recover a valid object after prose or a markdown fence,
    # without using the unsafe first-brace/last-brace substring heuristic.
    for index, character in enumerate(cleaned):
        if character not in "[{":
            continue
        try:
            parsed, _end = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
        elif isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            # A few models return the requested rows as a top-level array.
            candidates.append({"rows": parsed})

    if not candidates:
        raise LLMEvaluationError("Ollama returned non-JSON evaluation output.")

    for parsed in candidates:
        if any(
            isinstance(parsed.get(key), list)
            for key in shape_keys
        ):
            return parsed

    raise LLMEvaluationError("Ollama evaluation did not return corrections.")


def _message_content(message: dict[str, Any]) -> str | None:
    """Normalize text content from OpenAI-compatible message shapes."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ]
        if text_parts:
            return "\n".join(text_parts)
    # Some thinking-capable adapters put the final text in a secondary field.
    for key in ("reasoning_content", "reasoning"):
        fallback = message.get(key)
        if isinstance(fallback, str) and fallback.strip():
            return fallback
    return None


async def request_ollama_json(
    *,
    settings: Settings,
    payload: dict[str, Any],
    purpose: str,
    shape_keys: tuple[str, ...] = ("corrections", "rows", "issues", "warnings"),
) -> dict[str, Any]:
    """Call Ollama and retry once when the model violates the JSON contract."""
    headers = {"Content-Type": "application/json"}
    if settings.ollama_api_key and settings.ollama_base_url.startswith("https://ollama.com"):
        headers["Authorization"] = f"Bearer {settings.ollama_api_key.strip()}"
    openai_compatible = settings.ollama_base_url.rstrip("/").endswith("/v1")
    endpoint = (
        settings.ollama_base_url.rstrip("/") + "/chat/completions"
        if openai_compatible
        else settings.ollama_base_url.rstrip("/") + "/chat"
    )
    current_payload = payload
    last_parse_error: LLMEvaluationError | None = None

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
                response = await client.post(endpoint, headers=headers, json=current_payload)
        except httpx.HTTPError as error:
            if settings.ollama_base_url.startswith("http://localhost:11434"):
                raise LLMEvaluationError(
                    "Could not reach the local Ollama daemon. Start Ollama, sign in with `ollama signin`, "
                    f"and make the {settings.ollama_model} cloud model available."
                ) from error
            raise LLMEvaluationError(f"Could not reach Ollama: {error}") from error
        if response.status_code in {401, 403}:
            raise LLMEvaluationError(
                "Ollama authentication failed. Create or rotate an Ollama API key, "
                "set OLLAMA_API_KEY in backend/.env, and restart FastAPI."
            )
        if response.is_error:
            raise LLMEvaluationError(f"Ollama returned HTTP {response.status_code}: {response.text[:500]}")

        try:
            body = response.json()
            if openai_compatible:
                choices = body.get("choices") or []
                message = choices[0].get("message", {}) if choices else {}
            else:
                message = body.get("message", {})
            content = _message_content(message) if isinstance(message, dict) else None
            if not isinstance(content, str):
                raise LLMEvaluationError("Ollama returned no JSON content.")
            return _parse_json(content, shape_keys=shape_keys)
        except (ValueError, LLMEvaluationError) as error:
            last_parse_error = error if isinstance(error, LLMEvaluationError) else LLMEvaluationError(
                "Ollama returned an unreadable response."
            )
            if attempt == 0:
                current_payload = {
                    **payload,
                    "messages": [
                        *payload.get("messages", []),
                        {
                            "role": "user",
                            "content": (
                                f"Your previous {purpose} response was not valid JSON. "
                                "Return only the requested JSON object, with no prose, markdown, or code fences."
                            ),
                        },
                    ],
                }
                continue

    raise LLMEvaluationError(
        f"Ollama returned invalid JSON for {purpose} after one retry: {last_parse_error}"
    ) from last_parse_error


def _source_context(document: dict[str, Any]) -> str:
    """Build bounded, row-focused context while retaining full source for validation."""
    source_text = document["sourceText"]
    ranges: list[tuple[int, int]] = []

    def add_match(value: str) -> None:
        words = re.findall(r"\S+", value.replace("\n", " "))[:18]
        if not words:
            return
        pattern = r"\s+".join(re.escape(word) for word in words)
        match = re.search(pattern, source_text, flags=re.IGNORECASE)
        if match:
            ranges.append((max(0, match.start() - 1100), min(len(source_text), match.end() + 1500)))

    for row in document["rows"]:
        add_match(str(row.get("patentClaim", "")))
        add_match(str(row.get("evidenceOfInfringement", "")))
        source = row.get("source", {})
        pages = [source.get("pageNumber"), *(source.get("continuedPages") or [])]
        for page in {int(page) for page in pages if page is not None}:
            page_match = re.search(rf"\[PAGE number={page}\]", source_text)
            if page_match:
                ranges.append((page_match.start(), min(len(source_text), page_match.start() + 900)))

    ranges.sort()
    merged: list[list[int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 200:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    snippets = [source_text[start:end] for start, end in merged]
    if not snippets:
        snippets = [source_text[:6000]]
    return "[ROW_FOCUSED_SOURCE_CONTEXT]\n" + "\n\n---\n\n".join(snippets)


def _document_payload(document: dict[str, Any]) -> str:
    lines = [
        f"[CLAIM_CHART documentId={document['documentId']} filename={document['filename']}]",
        "[SOURCE_CONTEXT]",
        _source_context(document),
        "[DRAFT_ROWS]",
    ]
    for row in document["rows"]:
        lines.append(
            json.dumps(
                {
                    "draftRowId": row["draftRowId"],
                    "patentClaim": row["patentClaim"],
                    "evidenceOfInfringement": row["evidenceOfInfringement"],
                    "aiEvidence": "",
                    "source": row["source"],
                    "imageRefs": row.get("imageRefs", []),
                },
                ensure_ascii=False,
            )
        )
    lines.append("[REQUIRED_OUTPUT] Review every draftRowId. Return only corrections for rows that need them; return [] if none need correction:")
    lines.append(json.dumps([row["draftRowId"] for row in document["rows"]], ensure_ascii=False))
    lines.append("[END_CLAIM_CHART]")
    return "\n".join(lines)


async def evaluate_claim_charts_with_ollama(
    *,
    settings: Settings,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    if not settings.ollama_api_key and settings.ollama_base_url.startswith("https://ollama.com"):
        raise LLMEvaluationError("OLLAMA_API_KEY is required for direct Ollama Cloud API access.")
    if not documents:
        raise LLMEvaluationError("There are no claim-chart drafts to evaluate.")

    user_content = "\n\n".join(_document_payload(document) for document in documents)
    if len(user_content) > settings.ollama_max_input_chars:
        raise LLMEvaluationError(
            f"The evaluation payload is larger than the MVP limit of {settings.ollama_max_input_chars:,} characters."
        )

    openai_compatible = settings.ollama_base_url.rstrip("/").endswith("/v1")
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 20000,
    }
    if openai_compatible:
        payload["response_format"] = {"type": "json_object"}
        payload["reasoning_effort"] = "none"
    else:
        payload["options"] = {"temperature": 0, "seed": 42}
    result = await request_ollama_json(
        settings=settings,
        payload=payload,
        purpose="claim-chart evaluation",
    )

    draft_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for document in documents:
        for row in document["rows"]:
            draft_by_id[row["draftRowId"]] = (row, document)

    returned_corrections = result.get("corrections")
    legacy_full_rows = returned_corrections is None
    if legacy_full_rows:
        returned_corrections = result.get("rows", [])

    corrections_by_id: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    ignored_extra_rows = 0
    rejected_corrections = 0
    evaluation_issues: list[str] = list(result.get("issues") or [])
    for evaluated in returned_corrections:
        if not isinstance(evaluated, dict):
            ignored_extra_rows += 1
            continue
        draft_id = str(evaluated.get("draftRowId", ""))
        if draft_id not in draft_by_id or draft_id in seen_ids:
            ignored_extra_rows += 1
            continue
        seen_ids.add(draft_id)
        draft, document = draft_by_id[draft_id]
        claim = str(evaluated.get("patentClaim", draft["patentClaim"]))
        evidence = str(evaluated.get("evidenceOfInfringement", draft["evidenceOfInfringement"]))
        validation_source = document.get("validationSourceText", document["sourceText"])
        # The deterministic extraction is the source of truth. An evaluator
        # correction is advisory and must never make an otherwise usable
        # import fail or replace source text with a reordered reconstruction.
        try:
            _assert_source_grounded(claim, validation_source, "patentClaim")
        except LLMEvaluationError as error:
            claim = draft["patentClaim"]
            rejected_corrections += 1
            evaluation_issues.append(f"Ignored evaluator correction for {draft_id}: {error}")
        try:
            _assert_source_grounded(evidence, validation_source, "evidenceOfInfringement")
        except LLMEvaluationError as error:
            evidence = draft["evidenceOfInfringement"]
            rejected_corrections += 1
            evaluation_issues.append(f"Ignored evaluator correction for {draft_id}: {error}")
        corrections_by_id[draft_id] = {
            "patentClaim": claim,
            "evidenceOfInfringement": evidence,
            "changes": evaluated.get("changes", []),
        }

    final_rows: list[dict[str, Any]] = []
    for document in documents:
        for draft in document["rows"]:
            correction = corrections_by_id.get(draft["draftRowId"], {})
            final_rows.append({
                "draftRowId": draft["draftRowId"],
                "patentClaim": correction.get("patentClaim", draft["patentClaim"]),
                "evidenceOfInfringement": correction.get(
                    "evidenceOfInfringement", draft["evidenceOfInfringement"]
                ),
                "aiEvidence": "",
                "source": draft["source"],
                "imageRefs": draft.get("imageRefs", []),
                "changes": correction.get("changes", []),
            })

    issues = evaluation_issues
    llm_approved = result.get("approved") is True
    if not llm_approved:
        issues.append(
            "The LLM evaluator returned a warning; deterministic source validation remained authoritative."
        )
    if ignored_extra_rows:
        issues.append(
            f"Ignored {ignored_extra_rows} duplicate or unrecognized evaluator correction(s)."
        )
    if rejected_corrections:
        issues.append(
            f"Retained deterministic source text for {rejected_corrections} unsupported evaluator correction(s)."
        )
    if legacy_full_rows:
        issues.append("Converted legacy full-row evaluator output into source-preserving corrections.")

    return {
        "approved": True,
        "llmApproved": llm_approved,
        "issues": issues,
        "rows": final_rows,
        "provider": "ollama-chart-evaluator",
        "model": settings.ollama_model,
    }
