"""Andromeda-based agentic workflow for document processing"""
from __future__ import annotations
from collections import Counter
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    from andromeda.utils import get_chat_model
    from andromeda.utils.langtils import BaseChatModel
    from andromeda.config import ModelConfig
except ImportError:
    get_chat_model = None
    BaseChatModel = object
    ModelConfig = object
try:
    from app.domain import (
        ClassificationResult as DomainClassificationResult,
        SubClassificationResult as DomainSubClassificationResult,
        SummaryResult as DomainSummaryResult,
    )
    from app.helpers.prompt_builder import PromptBuilder
    from app.helpers.config import PRIMARY_LLM, DEFAULT_PROVIDER, model_other_args
    from app.utils.parser import parse_llm_json
except Exception:
    from helpers.document_config import DocumentTypeConfig
    from app.helpers.prompt_builder import PromptBuilder
    from app.helpers.config import PRIMARY_LLM, DEFAULT_PROVIDER, model_other_args
    from app.utils.parser import parse_llm_json
    try:
        from domain import (
            ClassificationResult as DomainClassificationResult,
            SubClassificationResult as DomainSubClassificationResult,
            SummaryResult as DomainSummaryResult,
        )
    except Exception:
        DomainClassificationResult = None  # type: ignore
        DomainSubClassificationResult = None  # type: ignore
        DomainSummaryResult = None  # type: ignore

if DomainSummaryResult is None:
    try:
        from app.domain.base_types import SummaryResult as DomainSummaryResult
    except Exception:
        try:
            from domain.base_types import SummaryResult as DomainSummaryResult
        except Exception:
            DomainSummaryResult = None  # type: ignore

# -----------------------------
# Model configuration and setup
# -----------------------------



# Example provider configuration can be set here if needed.

class ClassificationType(TypedDict, total=False):
    """Output schema for classification tasks"""
    start_page: int
    end_page: int
    document_type: str
    reason: str  # Not used, just for the model to think about the reason
    confidence: float

class ClassificationResult(TypedDict):
    """Output schema for classification tasks"""
    types: List[ClassificationType]


class ClassificationOutput(TypedDict, total=False):
    """Output schema for single-label classification tasks"""
    document_type: str
    confidence: float
    reason: str  # Not used, just for the model to think about the reason


_SEGMENT_TOKEN_LIMIT = 25000
_SEGMENT_MAX_PAGE_CHARS = 12000
_SUMMARY_CHUNK_TOKEN_LIMIT = int(os.getenv("SUMMARY_CHUNK_TOKEN_LIMIT", "30000"))
_SUMMARY_COMBINED_MAX_CHARS = int(os.getenv("SUMMARY_COMBINED_MAX_CHARS", "6000"))
_LLM_SUMMARY_TIMEOUT_SECONDS = int(os.getenv("LLM_SUMMARY_TIMEOUT_SECONDS", "90"))
_SUMMARY_MODEL_ARGS = {
    "num_ctx": 40960,
    "reasoning": "medium",
}
_SUMMARY_FAILURE_THRESHOLD = 2
_summary_failure_count = 0
_summary_circuit_logged = False
_SUMMARY_STOPWORDS = {
    "about", "above", "after", "again", "against", "also", "among", "because",
    "been", "before", "being", "below", "between", "both", "could", "does",
    "doing", "during", "each", "from", "further", "have", "having", "here",
    "into", "itself", "more", "most", "only", "other", "over", "same",
    "should", "some", "such", "than", "that", "their", "then", "there",
    "these", "they", "this", "those", "through", "under", "until", "very",
    "were", "what", "when", "where", "which", "while", "with", "would",
    "your", "summary", "document", "claim", "page", "pages",
}


def _load_segment_tokenizer():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return None


_SEGMENT_TOKENIZER = _load_segment_tokenizer()


def _summary_tokens(value: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'-]{2,}", (value or "").lower())
        if token not in _SUMMARY_STOPWORDS
    }


def _split_summary_claims(summary_text: str) -> List[str]:
    claims = [
        part.strip(" -:\t\r\n")
        for part in re.split(r"(?:\n+|(?<=[.!?])\s+|;\s+)", summary_text or "")
    ]
    return [claim for claim in claims if len(claim) >= 12]


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def summary_grounding_confidence(summary_text: str, source_text: str) -> float:
    """Estimate summary confidence from how much of it is supported by source text."""
    summary = (summary_text or "").strip()
    source = (source_text or "").strip()
    if not summary or not source:
        return 0.0

    source_tokens = _summary_tokens(source)
    summary_tokens = _summary_tokens(summary)
    if not source_tokens or not summary_tokens:
        return 0.0

    token_coverage = len(summary_tokens & source_tokens) / len(summary_tokens)
    claims = _split_summary_claims(summary)
    if not claims:
        return round(_clamp_confidence(token_coverage), 3)

    claim_scores = []
    for claim in claims:
        claim_tokens = _summary_tokens(claim)
        if not claim_tokens:
            continue
        claim_scores.append(len(claim_tokens & source_tokens) / len(claim_tokens))

    if not claim_scores:
        return round(_clamp_confidence(token_coverage), 3)

    claim_support = sum(claim_scores) / len(claim_scores)
    confidence = (token_coverage * 0.65) + (claim_support * 0.35)
    return round(_clamp_confidence(confidence), 3)


def _summary_confidence_from_result(summary: Dict[str, Any], source_text: str) -> float:
    try:
        confidence = float(summary.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    if confidence > 0.0:
        return round(_clamp_confidence(confidence), 3)
    return summary_grounding_confidence(str(summary.get("text") or ""), source_text)


def _fallback_summary_text(text: str, max_sentences: int = 8, max_chars: int = 1800) -> str:
    """Create a local extractive summary when the model cannot summarize reliably."""
    source = re.sub(r"\s+", " ", str(text or "")).strip()
    if not source:
        return ""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", source)
        if sentence.strip()
    ]
    summary = " ".join(sentences[:max_sentences]) if sentences else source
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].strip()
    return summary


def _summary_error_message(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if not text:
        return type(exc).__name__
    first_sentence = re.split(r"(?<=\.)\s+", text, maxsplit=1)[0]
    return f"{type(exc).__name__}: {first_sentence[:260]}"


def _summary_circuit_open() -> bool:
    return _summary_failure_count >= _SUMMARY_FAILURE_THRESHOLD


def _reset_summary_circuit() -> None:
    global _summary_failure_count, _summary_circuit_logged
    _summary_failure_count = 0
    _summary_circuit_logged = False


def _record_summary_failure(scope: str, exc: BaseException) -> None:
    global _summary_failure_count, _summary_circuit_logged

    _summary_failure_count += 1
    print(
        f"[LOG] {scope} summarization failed; using local fallback summary "
        f"({_summary_error_message(exc)})"
    )
    if _summary_circuit_open() and not _summary_circuit_logged:
        _summary_circuit_logged = True
        print(
            "[LOG] LLM summarization fallback mode enabled for this run after "
            f"{_summary_failure_count} failures."
        )


def _build_chat_model(*, temperature: float = 0.0, provider: Optional[str] = None) -> BaseChatModel:
    if get_chat_model is None or ModelConfig is None:
        raise ImportError("Andromeda chat model APIs are not available in the current environment")
    return get_chat_model(
        ModelConfig(
            name=PRIMARY_LLM,
            provider=provider or DEFAULT_PROVIDER,
            temperature=temperature,
            other_args=model_other_args(PRIMARY_LLM, **_SUMMARY_MODEL_ARGS),
        )
    )


def _is_connection_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in [
            "connecterror",
            "connection refused",
            "connection failed",
            "could not connect",
            "all connection attempts failed",
            "httpcore",
            "socket.gaierror",
            "connection reset",
        ]
    )


def _build_local_chat_model() -> BaseChatModel:
    return get_chat_model(
        ModelConfig(
            name=PRIMARY_LLM,
            provider="litellm",
            other_args=model_other_args(PRIMARY_LLM, **_SUMMARY_MODEL_ARGS),
        )
    )


async def _safe_ainvoke(chat_model: BaseChatModel, messages: Any) -> Any:
    try:
        return await chat_model.ainvoke(messages)
    except Exception as e:
        if _is_connection_failure(e):
            fallback = _build_local_chat_model()
            return await fallback.ainvoke(messages)
        raise


async def _safe_ainvoke_with_timeout(
    chat_model: BaseChatModel,
    messages: Any,
    timeout_seconds: int = _LLM_SUMMARY_TIMEOUT_SECONDS,
) -> Any:
    return await asyncio.wait_for(_safe_ainvoke(chat_model, messages), timeout=timeout_seconds)


def _safe_invoke(chat_model: BaseChatModel, messages: Any) -> Any:
    try:
        return chat_model.invoke(messages)
    except Exception as e:
        if _is_connection_failure(e):
            fallback = _build_local_chat_model()
            return fallback.invoke(messages)
        raise


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _token_count(text: str) -> int:
    if not text:
        return 0
    if _SEGMENT_TOKENIZER is None:
        return len(text.split())
    return len(_SEGMENT_TOKENIZER.encode(text))


def _chunk_text_for_summary(text: str) -> List[str]:
    """Split large grouped text into bounded chunks before LLM summarization."""
    source = str(text or "").strip()
    if not source:
        return []
    if _token_count(source) <= _SUMMARY_CHUNK_TOKEN_LIMIT:
        return [source]

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", source) if part.strip()]
    if not paragraphs:
        paragraphs = [source]

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush_current() -> None:
        nonlocal current, current_tokens
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = _token_count(paragraph)
        if paragraph_tokens > _SUMMARY_CHUNK_TOKEN_LIMIT:
            flush_current()
            words = paragraph.split()
            batch: List[str] = []
            batch_tokens = 0
            for word in words:
                word_tokens = _token_count(word)
                if batch and batch_tokens + word_tokens > _SUMMARY_CHUNK_TOKEN_LIMIT:
                    chunks.append(" ".join(batch).strip())
                    batch = []
                    batch_tokens = 0
                batch.append(word)
                batch_tokens += word_tokens
            if batch:
                chunks.append(" ".join(batch).strip())
            continue

        if current and current_tokens + paragraph_tokens > _SUMMARY_CHUNK_TOKEN_LIMIT:
            flush_current()
        current.append(paragraph)
        current_tokens += paragraph_tokens

    flush_current()
    return chunks


def _is_zero_based_page_numbers(grouped_page_numbers: List[List[Any]]) -> bool:
    pages: List[int] = []
    for group in grouped_page_numbers or []:
        for p in (group or []):
            ip = _safe_int(p)
            if ip is not None:
                pages.append(ip)
    return any(p == 0 for p in pages)


def _collect_sub_types() -> Set[str]:
    out: Set[str] = {"OTHER"}
    for defs in DocumentTypeConfig.SUB_TYPE_DEFINITIONS.values():
        out.update(defs.keys())
    return out


def _infer_hl_type(full_type: str) -> str:
    ft = (full_type or "").upper().strip()
    hl_candidates = DocumentTypeConfig.get_high_level_candidates()
    if ft in hl_candidates:
        return ft
    for hl, defs in DocumentTypeConfig.SUB_TYPE_DEFINITIONS.items():
        if ft in defs:
            return hl
    return "OTHER"


def _normalize_doc_type(value: str, allowed_types: Set[str], fallback: str = "OTHER") -> str:
    t = (value or "").upper().strip()
    if t in allowed_types:
        return t
    return fallback if fallback in allowed_types else "OTHER"


def _resolve_sub_type_prediction(hl_type: str, sub_type: str) -> str:
    """Return subtype when valid; otherwise collapse OTHER/invalid subtype to high-level."""
    hlt = _normalize_doc_type(
        hl_type,
        set(DocumentTypeConfig.get_high_level_candidates()) | {"OTHER"},
        fallback="OTHER",
    )
    sub = (sub_type or "").upper().strip()
    if not sub or sub == "OTHER":
        return hlt
    if sub in DocumentTypeConfig.get_sub_type_candidates(hlt):
        return sub
    return hlt


def _format_response_text(response: Any) -> str:
    def _collect_text(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    parsed = parse_llm_json(stripped)
                    if isinstance(parsed, dict):
                        return _collect_text(parsed)
                except Exception:
                    pass
            return [stripped] if stripped else []
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                parts.extend(_collect_text(item))
            return parts
        if isinstance(value, dict):
            item_type = str(value.get("type") or "").lower()
            if item_type in {"reasoning", "thinking"}:
                return []
            for key in ("text", "content", "output_text", "answer", "result", "value"):
                if key in value:
                    parts = _collect_text(value[key])
                    if parts:
                        return parts

            parts: List[str] = []
            for key, nested in value.items():
                if key in {"type", "index", "id", "name", "role"}:
                    continue
                parts.extend(_collect_text(nested))
            return parts
        return [str(value)]

    if response is None:
        return ""
    if hasattr(response, "content"):
        response = response.content
    return "\n".join(_collect_text(response)).strip()


def _resolve_document_definition(sub_type: str, hl_type: str):
    definition = DocumentTypeConfig.get_definition((sub_type or "").strip().upper())
    if definition is not None:
        return definition
    return DocumentTypeConfig.get_definition((hl_type or "").strip().upper())


def _parse_structured_response(response: Any, schema: Any) -> Optional[Any]:
    if response is None or schema is None:
        return None

    if isinstance(response, dict):
        try:
            return schema.model_validate(response)
        except Exception:
            pass

    raw = response
    if hasattr(response, "model_dump"):
        try:
            return schema.model_validate(response.model_dump())
        except Exception:
            pass

    if hasattr(response, "content"):
        raw = response.content

    if isinstance(raw, str):
        raw_text = raw.strip()
        try:
            parsed_text = parse_llm_json(raw_text)
            return schema.model_validate(parsed_text)
        except Exception:
            try:
                return schema.model_validate_json(raw_text)
            except Exception:
                return None

    try:
        return schema.model_validate(raw)
    except Exception:
        return None


def _page_text_lookup(extracted_text: List[str], zero_based: bool) -> Dict[int, str]:
    lookup: Dict[int, str] = {}
    for i, txt in enumerate(extracted_text or []):
        p = i if zero_based else (i + 1)
        lookup[p] = txt or ""
    return lookup


def _serialize_page(page_number: int, text: str) -> str:
    body = (text or "").strip()
    if len(body) > _SEGMENT_MAX_PAGE_CHARS:
        body = body[:_SEGMENT_MAX_PAGE_CHARS]
    return f"\n{body}\n"


def _chunk_pages_by_tokens(page_numbers: List[int], page_texts: Dict[int, str]) -> List[List[int]]:
    chunks: List[List[int]] = []
    current: List[int] = []
    current_tokens = 0

    for p in page_numbers:
        block = _serialize_page(p, page_texts.get(p, ""))
        block_tokens = _token_count(block)
        if current and (current_tokens + block_tokens > _SEGMENT_TOKEN_LIMIT):
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(p)
        current_tokens += block_tokens

    if current:
        chunks.append(current)
    return chunks


def _normalize_ranges(
    raw_ranges: List[Dict[str, Any]],
    chunk_pages: List[int],
    allowed_types: Set[str],
) -> List[ClassificationType]:
    if not chunk_pages:
        return []

    ordered_pages = sorted(set(chunk_pages))
    min_page = ordered_pages[0]
    max_page = ordered_pages[-1]
    page_to_type: Dict[int, str] = {}

    for item in raw_ranges or []:
        start = _safe_int((item or {}).get("start_page"))
        end = _safe_int((item or {}).get("end_page"))
        if start is None or end is None:
            continue
        if start > end:
            start, end = end, start
        start = max(min_page, start)
        end = min(max_page, end)
        if start > end:
            continue
        doc_type = _normalize_doc_type((item or {}).get("document_type", ""), allowed_types)
        for p in ordered_pages:
            if start <= p <= end:
                page_to_type.setdefault(p, doc_type)

    # Fill coverage gaps; bridge filler/title/end pages to surrounding dominant type when possible.
    for idx, p in enumerate(ordered_pages):
        if p in page_to_type:
            continue
        prev_type = None
        next_type = None
        for j in range(idx - 1, -1, -1):
            pp = ordered_pages[j]
            if pp in page_to_type:
                prev_type = page_to_type[pp]
                break
        for j in range(idx + 1, len(ordered_pages)):
            np = ordered_pages[j]
            if np in page_to_type:
                next_type = page_to_type[np]
                break
        if prev_type and next_type and prev_type == next_type:
            page_to_type[p] = prev_type
        elif prev_type:
            page_to_type[p] = prev_type
        elif next_type:
            page_to_type[p] = next_type
        else:
            page_to_type[p] = "OTHER"

    normalized: List[ClassificationType] = []
    run_start = ordered_pages[0]
    run_type = page_to_type[run_start]

    for i in range(1, len(ordered_pages)):
        p = ordered_pages[i]
        t = page_to_type[p]
        if t == run_type:
            continue
        normalized.append(
            {
                "start_page": run_start,
                "end_page": ordered_pages[i - 1],
                "document_type": run_type,
                "reason": "normalized",
            }
        )
        run_start = p
        run_type = t

    normalized.append(
        {
            "start_page": run_start,
            "end_page": ordered_pages[-1],
            "document_type": run_type,
            "reason": "normalized",
        }
    )
    return normalized


def _merge_adjacent_same_type(ranges: List[ClassificationType]) -> List[ClassificationType]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda r: (r["start_page"], r["end_page"]))
    merged: List[ClassificationType] = [ordered[0]]
    for curr in ordered[1:]:
        prev = merged[-1]
        if curr["document_type"] == prev["document_type"] and curr["start_page"] <= (prev["end_page"] + 1):
            prev["end_page"] = max(prev["end_page"], curr["end_page"])
        else:
            merged.append(curr)
    return merged


def _segment_group_by_pages(
    page_numbers: List[int],
    page_texts: Dict[int, str],
    allowed_type_descriptions: List[str],
    allowed_type_labels: Set[str],
) -> List[ClassificationType]:
    if not page_numbers:
        return []

    chat_model = _build_chat_model()
    chunks = _chunk_pages_by_tokens(page_numbers, page_texts)
    all_ranges: List[ClassificationType] = []

    for chunk_pages in chunks:
        page_block = "\n".join(_serialize_page(p, page_texts.get(p, "")) for p in chunk_pages)
        sys_p, usr_p = PromptBuilder.build_page_range_classification_prompt(
            page_block=page_block,
            allowed_types=allowed_type_descriptions,
            page_numbers=chunk_pages,
        )

        response = None
        attempts = 0
        while attempts < 3:
            try:
                response = _safe_invoke(
                    chat_model.with_structured_output(ClassificationResult),
                    PromptBuilder.build_messages(sys_p, usr_p),
                )
                break
            except Exception:
                attempts += 1

        raw_types = (response or {}).get("types", []) if isinstance(response, dict) else []
        all_ranges.extend(_normalize_ranges(raw_types, chunk_pages, allowed_type_labels))

    all_ranges = _normalize_ranges(all_ranges, page_numbers, allowed_type_labels)
    return _merge_adjacent_same_type(all_ranges)


# -----------------------------
# Classification Functions
# -----------------------------

def classify_document(text: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Classify document and return (full_type, high_level_type, classification_metadata)
    """
    chat_model = _build_chat_model()
    sys_p, usr_p = PromptBuilder.build_classification_prompt(text)
    response = None
    attempts = 0
    last_error: Optional[Exception] = None
    while attempts < 3:
        try:
            response = _safe_invoke(
                chat_model.with_structured_output(DomainClassificationResult),
                PromptBuilder.build_messages(sys_p, usr_p),
            )
            break
        except Exception as exc:
            last_error = exc
            attempts += 1
    if response is None and last_error is not None:
        raise RuntimeError("High-level classification failed after 3 attempts") from last_error

    classification_response = _parse_structured_response(response, DomainClassificationResult)
    hl_type = ""
    classification_metadata: Dict[str, Any] = {"confidence": None}

    if classification_response is not None:
        classification_data = classification_response.model_dump()
        hl_type = (classification_data.get("document_type") or "").upper().strip()
        classification_metadata["confidence"] = classification_data.get("confidence")

    hl_candidates = DocumentTypeConfig.get_high_level_candidates()
    if hl_type not in hl_candidates:
        raise ValueError(f"Classifier returned unsupported high-level document type: {hl_type!r}")

    return hl_type, hl_type, classification_metadata


def classify_sub_type(text: str, hl_type: str) -> Tuple[str, Dict[str, Any]]:
    """
    Classify only sub-type for an already known high-level type.
    Falls back to hl_type when sub-classification is unavailable.
    """
    hlt = (hl_type or "").upper().strip()
    classification_metadata: Dict[str, Any] = {"confidence": None}

    candidates = DocumentTypeConfig.get_sub_type_candidates(hlt)
    if not candidates:
        return hlt or "OTHER", classification_metadata

    chat_model = _build_chat_model()
    sys_p2, usr_p2 = PromptBuilder.build_subclassification_prompt(text, hlt)
    response2 = None
    attempts = 0
    last_error: Optional[Exception] = None
    while attempts < 3:
        try:
            response2 = _safe_invoke(
                chat_model.with_structured_output(DomainSubClassificationResult),
                PromptBuilder.build_messages(sys_p2, usr_p2),
            )
            break
        except Exception as exc:
            last_error = exc
            attempts += 1
    if response2 is None and last_error is not None:
        raise RuntimeError(f"Subtype classification failed for {hlt} after 3 attempts") from last_error

    classification_response = _parse_structured_response(response2, DomainSubClassificationResult)
    sub_type = ""
    if classification_response is not None:
        classification_data = classification_response.model_dump()
        sub_type = (classification_data.get("sub_type") or "").upper().strip()
        classification_metadata["confidence"] = classification_data.get("confidence")

    return _resolve_sub_type_prediction(hlt, sub_type), classification_metadata


def _filter_predictions(results: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Filter predictions to focus on major types"""
    if len(results) <= 6:
        filtered = results
    else:
        # Take first 3 and last 3
        filtered = results[:3] + results[-3:]
    
    minor_types = ("MEDICAL", "OTHER", "POLICE_REPORT", "EMAIL_CORRESPONDENCE")
    # Check if we have major (non-medical/other) types
    has_major_type = any(hl not in minor_types for _, hl in filtered)
    
    if has_major_type:
        # Keep only major types
        return [(ft, hl) for ft, hl in filtered if hl not in minor_types]
    
    return filtered


def _get_most_common_type(results: List[Tuple[str, str]]) -> str:
    """Get most common high-level type"""
    if not results:
        return "OTHER"
    
    hl_types = [hl for _, hl in results]
    return Counter(hl_types).most_common(1)[0][0]


def _get_final_document_type(results: List[Tuple[str, str]], common_hl_type: str) -> str:
    """Determine final document type from filtered results"""
    predictions = [full_type for full_type, _ in results if full_type != 'OTHER']

    high_level_types = DocumentTypeConfig.get_high_level_candidates()
    sub_type_defs = DocumentTypeConfig.SUB_TYPE_DEFINITIONS.get(common_hl_type, {})
    
    if sub_type_defs:
        # If there are sub-types defined, filter predictions to valid sub-types
        valid_predictions = [
            p for p in predictions 
            if p in sub_type_defs.keys() or p == common_hl_type
        ]
    else:
        valid_predictions = [p for p in predictions if p in high_level_types]
    
    if valid_predictions:
        return Counter(valid_predictions).most_common(1)[0][0]
    
    return Counter(predictions).most_common(1)[0][0] if predictions else 'OTHER'


def predict_document_type(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node: Predict document types for all grouped texts
    Returns updated state with predictions
    """
    texts = state.get("grouped_texts", []) or []
    grouped_page_numbers = state.get("grouped_page_numbers", []) or []
    extracted_text = state.get("extracted_text", []) or []
    if not texts and not grouped_page_numbers:
        return state

    segment_hl_types: List[str] = []

    # Refine rough clusters into contiguous ranges using structured page-range prediction.
    if grouped_page_numbers and extracted_text:
        zero_based = _is_zero_based_page_numbers(grouped_page_numbers)
        page_texts = _page_text_lookup(extracted_text, zero_based=zero_based)
        hl_allowed = [f"{k}: {v}" for k, v in DocumentTypeConfig.TYPE_DEFINITIONS.items()]
        hl_allowed_raw = DocumentTypeConfig.get_high_level_candidates()
        hl_allowed.append("OTHER: Uncategorized documents")

        refined_groups: List[List[int]] = []
        refined_texts: List[str] = []

        for group in grouped_page_numbers:
            pages = sorted(
                {
                    p for p in (_safe_int(x) for x in (group or []))
                    if p is not None and p in page_texts
                }
            )
            if not pages:
                continue

            predicted_ranges = _segment_group_by_pages(pages, page_texts, hl_allowed, hl_allowed_raw)
            if not predicted_ranges:
                predicted_ranges = [
                    {
                        "start_page": pages[0],
                        "end_page": pages[-1],
                        "document_type": "OTHER",
                        "reason": "fallback",
                    }
                ]

            for rr in predicted_ranges:
                start = _safe_int(rr.get("start_page"))
                end = _safe_int(rr.get("end_page"))
                if start is None or end is None:
                    continue
                if start > end:
                    start, end = end, start
                segment_pages = [p for p in pages if start <= p <= end]
                if not segment_pages:
                    continue
                segment_text = "\n".join(page_texts.get(p, "") for p in segment_pages).strip()
                hl_from_range = _normalize_doc_type(
                    (rr.get("document_type") or ""),
                    set(DocumentTypeConfig.get_high_level_candidates()) | {"OTHER"},
                    fallback="OTHER",
                )
                refined_groups.append(segment_pages)
                refined_texts.append(segment_text)
                segment_hl_types.append(hl_from_range)

        if refined_groups and len(refined_groups) == len(refined_texts):
            state["grouped_page_numbers"] = refined_groups
            state["grouped_texts"] = refined_texts
            texts = refined_texts
            if len(segment_hl_types) != len(refined_texts):
                segment_hl_types = ["OTHER"] * len(refined_texts)

    # Fallback: if we do not have segment high-level types, infer from existing classifier.
    if texts and (not segment_hl_types or len(segment_hl_types) != len(texts)):
        inferred: List[str] = []
        for text in texts:
            _, hl, _ = classify_document(text)
            inferred.append(
                _normalize_doc_type(
                    hl,
                    set(DocumentTypeConfig.get_high_level_candidates()) | {"OTHER"},
                    fallback="OTHER",
                )
            )
        segment_hl_types = inferred
    
    # Second-stage refinement for sub-types: use the same range-segmentation approach.
    final_groups: List[List[int]] = []
    final_texts: List[str] = []
    final_hl_types: List[str] = []
    final_predictions: List[str] = []

    page_texts: Dict[int, str] = {}
    if extracted_text and state.get("grouped_page_numbers"):
        zero_based = _is_zero_based_page_numbers(state.get("grouped_page_numbers") or [])
        page_texts = _page_text_lookup(extracted_text, zero_based=zero_based)

    current_grouped_page_numbers = state.get("grouped_page_numbers") or []
    for idx, text in enumerate(texts):
        hl_type = segment_hl_types[idx] if idx < len(segment_hl_types) else "OTHER"
        hl_type = _normalize_doc_type(
            hl_type,
            set(DocumentTypeConfig.get_high_level_candidates()) | {"OTHER"},
            fallback="OTHER",
        )
        pages = []
        if idx < len(current_grouped_page_numbers):
            pages = sorted(
                {
                    p for p in (_safe_int(x) for x in (current_grouped_page_numbers[idx] or []))
                    if p is not None
                }
            )

        sub_type_defs = DocumentTypeConfig.SUB_TYPE_DEFINITIONS.get(hl_type, {})

        can_segment_sub_type = bool(sub_type_defs) and bool(pages) and bool(page_texts)

        if can_segment_sub_type:
            sub_allowed = [f"{k}: {v}" for k, v in sub_type_defs.items()]
            sub_allowed.append("OTHER: Uncategorized documents")
            sub_allowed_raw = DocumentTypeConfig.get_sub_type_candidates(hl_type) | {"OTHER"}
            sub_ranges = _segment_group_by_pages(pages, page_texts, sub_allowed, sub_allowed_raw)
            if not sub_ranges:
                sub_ranges = [
                    {
                        "start_page": pages[0],
                        "end_page": pages[-1],
                        "document_type": "OTHER",
                        "reason": "fallback",
                    }
                ]
            for sr in sub_ranges:
                start = _safe_int(sr.get("start_page"))
                end = _safe_int(sr.get("end_page"))
                if start is None or end is None:
                    continue
                if start > end:
                    start, end = end, start
                sub_pages = [p for p in pages if start <= p <= end]
                if not sub_pages:
                    continue
                sub_text = "\n".join(page_texts.get(p, "") for p in sub_pages).strip()
                predicted_sub_type = _normalize_doc_type(
                    sr.get("document_type", ""),
                    set(DocumentTypeConfig.get_sub_type_candidates(hl_type)) | {"OTHER"},
                    fallback=hl_type,
                )
                predicted_sub_type = _resolve_sub_type_prediction(hl_type, predicted_sub_type)
                final_groups.append(sub_pages)
                final_texts.append(sub_text)
                final_hl_types.append(hl_type)
                final_predictions.append(predicted_sub_type)
        else:
            full_type = hl_type
            if DocumentTypeConfig.get_sub_type_candidates(hl_type):
                full_type, _ = classify_sub_type(text, hl_type)
            full_type = _normalize_doc_type(
                full_type,
                set(DocumentTypeConfig.get_high_level_candidates()) | _collect_sub_types(),
                fallback=hl_type,
            )
            if full_type == "OTHER":
                full_type = hl_type
            final_groups.append(pages)
            final_texts.append(text)
            final_hl_types.append(_infer_hl_type(full_type))
            final_predictions.append(full_type)

    if final_groups and len(final_groups) == len(final_predictions) == len(final_hl_types):
        state["grouped_page_numbers"] = final_groups
        state["grouped_texts"] = final_texts

    # Final predictions list maps 1:1 with grouped pages/texts.
    results: List[Tuple[str, str]] = list(zip(final_predictions, final_hl_types))
    
    # Store all predictions
    predictions = [full_type for full_type, _ in results]
    hl_types = [hl_type for _, hl_type in results]
    state["predictions"] = predictions
    state["hl_types"] = hl_types

    sub_documents: List[Dict[str, Any]] = []
    for idx, pages in enumerate(state.get("grouped_page_numbers", []) or []):
        full_type = predictions[idx] if idx < len(predictions) else "OTHER"
        hl_type = hl_types[idx] if idx < len(hl_types) else _infer_hl_type(full_type)
        sub_documents.append(
            {
                "document_type": hl_type,
                "sub_type": None if full_type == hl_type else full_type,
                "pages": pages,
                "text": state.get("grouped_texts", [])[idx] if idx < len(state.get("grouped_texts", [])) else "",
            }
        )
    state["sub_documents"] = sub_documents
    
    # Filter and determine common type
    filtered_results = _filter_predictions(results)
    common_hl_type = _get_most_common_type(filtered_results)
    document_type = _get_final_document_type(filtered_results, common_hl_type)
    
    state["common_hl_type"] = common_hl_type
    state["document_type"] = document_type
    
    return state


# -----------------------------
# Summarization Functions
# -----------------------------

async def summarize_page(
    text: str,
    sub_type: str = "OTHER",
    hl_type: str = "OTHER",
    include_overview: bool = True,
) -> str:
    """
    Summarize a single page.
    Returns the overview string.
    """
    if not include_overview:
        return ""
    if _summary_circuit_open():
        return _fallback_summary_text(text)

    chat_model: BaseChatModel = get_chat_model(
        ModelConfig(
            name=PRIMARY_LLM,
            provider=DEFAULT_PROVIDER,
            temperature=0.3,
            other_args=model_other_args(PRIMARY_LLM, **_SUMMARY_MODEL_ARGS),
        )
    )

    schema_hint = None
    doc_def = _resolve_document_definition(sub_type, hl_type)
    summary_format = getattr(doc_def, "summary_format", None) if doc_def is not None else None

    sys_p, usr_p = PromptBuilder.build_summarize_prompt(
        text, sub_type, hl_type, schema_hint, include_actions=True, summary_format=summary_format
    )
    try:
        response = await _safe_ainvoke_with_timeout(
            chat_model.with_structured_output(DomainSummaryResult),
            PromptBuilder.build_messages(sys_p, usr_p),
        )
    except Exception as e:
        _record_summary_failure("Page", e)
        return _fallback_summary_text(text)
    parsed_summary = _parse_structured_response(response, DomainSummaryResult)
    if parsed_summary is not None:
        overview = parsed_summary.text.strip()
    else:
        overview = _format_response_text(response).strip()
    return overview


async def summarize_document(
    text: str,
    sub_type: str = "OTHER",
    hl_type: str = "OTHER",
) -> Dict[str, Any]:
    """Produce a final summary record with text and confidence."""
    if _summary_circuit_open():
        summary_text = _fallback_summary_text(text)
        return {
            "text": summary_text,
            "confidence": summary_grounding_confidence(summary_text, text),
        }

    chat_model: BaseChatModel = get_chat_model(
        ModelConfig(
            name=PRIMARY_LLM,
            provider=DEFAULT_PROVIDER,
            temperature=0.3,
            other_args=model_other_args(PRIMARY_LLM, **_SUMMARY_MODEL_ARGS),
        )
    )

    op_format = DocumentTypeConfig.get_schema(sub_type)
    schema_hint = op_format.model_json_schema() if op_format else None
    doc_def = _resolve_document_definition(sub_type, hl_type)
    summary_format = getattr(doc_def, "summary_format", None) if doc_def is not None else None
    sys_p, usr_p = PromptBuilder.build_summarize_prompt(
        text, sub_type, hl_type, schema_hint, include_actions=False, summary_format=summary_format
    )
    try:
        response = await _safe_ainvoke_with_timeout(
            chat_model.with_structured_output(DomainSummaryResult),
            PromptBuilder.build_messages(sys_p, usr_p),
        )
    except Exception as e:
        _record_summary_failure("Document", e)
        summary_text = _fallback_summary_text(text)
        return {
            "text": summary_text,
            "confidence": summary_grounding_confidence(summary_text, text),
        }
    parsed_summary = _parse_structured_response(response, DomainSummaryResult)
    if parsed_summary is not None:
        summary = parsed_summary.model_dump()
        summary["confidence"] = _summary_confidence_from_result(summary, text)
        return summary

    summary_text = _format_response_text(response).strip()
    return {
        "text": summary_text,
        "confidence": summary_grounding_confidence(summary_text, text),
    }


async def _summarize_group_text(
    text: str,
    document_type: str,
    common_hl_type: str,
) -> str:
    """Summarize one grouped document, chunking large PDFs before model calls."""
    chunks = _chunk_text_for_summary(text)
    if not chunks:
        return ""
    if len(chunks) == 1:
        return await summarize_page(chunks[0], document_type, common_hl_type) or ""

    print(
        f"[LOG] Large summary input split into {len(chunks)} chunks "
        f"(tokens~{_token_count(text)})."
    )
    chunk_overviews: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        overview = await summarize_page(chunk, document_type, common_hl_type)
        if not overview:
            overview = _fallback_summary_text(chunk)
        chunk_overviews.append(f"Section {idx}: {overview}")

    combined = "\n".join(chunk_overviews)
    if len(combined) > _SUMMARY_COMBINED_MAX_CHARS:
        combined = _fallback_summary_text(
            combined,
            max_sentences=16,
            max_chars=_SUMMARY_COMBINED_MAX_CHARS,
        )
    return combined


async def summarize_groups(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node: Summarize all grouped texts
    - Requires: grouped_texts, document_type, common_hl_type
    - Produces: overviews, summaries
    """
    grouped_texts = state.get("grouped_texts", [])
    document_type = state.get("document_type", "OTHER")
    common_hl_type = state.get("common_hl_type", "OTHER")
    _reset_summary_circuit()
    
    if not grouped_texts:
        state["overviews"] = []
        state["summaries"] = []
        return state
    
    overviews = []
    for text in grouped_texts:
        overviews.append(await _summarize_group_text(text, document_type, common_hl_type) or "")
    
    state["overviews"] = overviews
    state["summaries"] = []
    
    return state


async def summarize_combined_overviews(state: Dict[str, Any], extract: bool = False) -> Dict[str, Any]:
    """
    Node: Create final combined overview from all page overviews
    - Requires: overviews, document_type, common_hl_type
    - Produces: final_overview, summary (combined single-page summary; not merged)
    """
    overviews = state.get("overviews", [])
    document_type = state.get("document_type", "OTHER")
    common_hl_type = state.get("common_hl_type", "OTHER")
    
    if not overviews:
        state["final_overview"] = None
        state["summary"] = None
        return state
    
    if len(overviews) > 1:
        combined_text = '\n\n'.join(overviews)
        final_overview = await summarize_page(
            combined_text, document_type, common_hl_type or 'OTHER'
        )
        summary_result = await summarize_document(
            combined_text, document_type, common_hl_type or 'OTHER'
        )
        state["final_overview"] = final_overview
        state["summary"] = summary_result
    elif extract:
        combined_text = '\n\n'.join(overviews)
        final_overview = await summarize_page(
            combined_text, document_type, common_hl_type or 'OTHER', include_overview=False
        )
        summary_result = await summarize_document(
            combined_text, document_type, common_hl_type or 'OTHER'
        )
        state["final_overview"] = final_overview
        state["summary"] = summary_result
    else:
        state["final_overview"] = overviews[0]
        state["summary"] = await summarize_document(
            overviews[0], document_type, common_hl_type or 'OTHER'
        )
    
    return state


# -----------------------------
# Claim Number Functions
# -----------------------------

async def select_claim_number_from_candidates(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node: Select most likely claim number from candidates using LLM
    - Requires: final_overview (str), potential_claim_numbers (str)
    - Produces: claim_number (str)
    """
    overview = state.get("final_overview", "") or ""
    candidates = state.get("potential_claim_numbers", "") or ""
    
    if not candidates:
        state["claim_number"] = "not found"
        return state

    candidate_tokens = {
        match.group(0)
        for match in re.finditer(r"\b[A-Za-z]?\d[A-Za-z0-9-]{5,20}[A-Za-z]?\b", candidates)
    }
    
    chat_model: BaseChatModel = get_chat_model(
        ModelConfig(
            name=PRIMARY_LLM,
            provider=DEFAULT_PROVIDER,
            temperature=0.3,
            other_args=model_other_args(PRIMARY_LLM, **_SUMMARY_MODEL_ARGS),
        )
    )
    sys_p, usr_p = PromptBuilder.build_claim_selection_prompt(overview, candidates)
    response = await _safe_ainvoke(chat_model, PromptBuilder.build_messages(sys_p, usr_p))
    
    content = _format_response_text(response)
    token = "not found"
    for raw_token in re.findall(r"\b[A-Za-z]?\d[A-Za-z0-9-]{5,20}[A-Za-z]?\b", content):
        if not candidate_tokens or raw_token in candidate_tokens:
            token = raw_token
            break

    if token == "not found":
        fallback = claim_extractor.extract_simple(candidates)
        token = fallback if fallback != "not found" else token
    
    state["claim_number"] = token
    return state


# -----------------------------
# Legacy compatibility functions
# -----------------------------

async def find_claim_number(overview: str, potential_claim_numbers: str) -> str:
    """
    Legacy compatibility function for finding claim number
    """
    state = {
        "final_overview": overview,
        "potential_claim_numbers": potential_claim_numbers,
        "claim_number": "not found"
    }
    state = await select_claim_number_from_candidates(state)
    return state["claim_number"]
