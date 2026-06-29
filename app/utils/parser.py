import json
import re
import logging
from typing import Any, Optional, List, Dict, Tuple, Callable

logger = logging.getLogger(__name__)

def normalize_llm_content(content: Any) -> str:
    """Convert AIMessage.content (str or list of blocks) to clean text, stripping reasoning."""
    if isinstance(content, str):
        return content.strip()
    
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                block_type = str(block.get("type", "")).lower()
                
                # Skip reasoning/thinking blocks
                if block_type in ("reasoning", "thinking"):
                    continue
                
                # Unwrap non_standard wrappers
                if block_type == "non_standard":
                    inner = block.get("value")
                    if isinstance(inner, dict):
                        inner_type = str(inner.get("type", "")).lower()
                        if inner_type in ("reasoning", "thinking"):
                            continue  # skip reasoning inside non_standard
                        # If inner has text, we can check it
                        text = inner.get("text") or inner.get("content") or inner.get("value")
                        if isinstance(text, str) and text.strip():
                            text_parts.append(text)
                    continue
                
                # Extract text from text blocks
                if block_type in ("text", "text-plain"):
                    text = block.get("text") or block.get("content")
                    if isinstance(text, str):
                        text_parts.append(text)
                    continue
                
                # For unknown block types, try common keys, as long as it's not reasoning
                if any(k in block for k in ("reasoning", "thinking")):
                    continue
                for key in ("text", "content", "value"):
                    val = block.get(key)
                    if isinstance(val, str) and val.strip():
                        text_parts.append(val)
                        break
        
        return "\n".join(text_parts).strip()
    
    return str(content).strip() if content else ""


def parse_llm_json(text: Any) -> dict:
    if not isinstance(text, str):
        if isinstance(text, list):
            text = normalize_llm_content(text)
        else:
            raise TypeError("Expected LLM output as string")
 
    text = text.strip()
 
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()
 
    json_candidates = []
    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                potential_json = text[start:i+1]
                json_candidates.append(potential_json)
                start = -1
    for json_str in reversed(json_candidates):
        if '"thinking"' not in json_str.lower():
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
 
    for json_str in reversed(json_candidates):
        try:
            result = json.loads(json_str)
            if "thinking" in result and isinstance(result.get("value"), dict):
                actual_data = result.get("value")
                if "thinking" not in actual_data:
                    return actual_data
            return result
        except json.JSONDecodeError:
            continue
 
    return json.loads(text)


def _looks_like_field_payload(
    payload: Dict[str, Any],
    fields: List[Dict[str, Any]],
) -> bool:
    if not isinstance(payload, dict):
        return False
    if not fields:
        return True
    field_names = {
        field["name"]
        for field in fields
        if isinstance(field, dict) and "name" in field
    }
    if not field_names:
        return True
    return bool(field_names.intersection(payload.keys()))


def _payload_value_score(
    payload: Dict[str, Any],
    fields: List[Dict[str, Any]],
) -> int:
    score = 0
    for field in fields:
        if not isinstance(field, dict):
            continue
        name = field.get("name")
        if not name:
            continue
        raw = payload.get(name)
        if raw is not None:
            # Check if it is a dict from extraction_agent (value / confidence / snippet)
            if isinstance(raw, dict) and ("value" in raw or "confidence" in raw):
                val = raw.get("value")
                confidence = raw.get("confidence")
                if val not in (None, "", []):
                    score += 2
                try:
                    if float(confidence or 0.0) > 0.0:
                        score += 1
                except (TypeError, ValueError):
                    pass
            else:
                val = raw
                if val not in (None, "", []):
                    score += 1
    return score


def _tool_call_name_and_args(call: Any) -> Tuple[str, Any]:
    if hasattr(call, "model_dump"):
        call = call.model_dump()
    if not isinstance(call, dict):
        return "", None

    name = str(call.get("name") or "")
    args = call.get("args")

    if isinstance(args, str):
        try:
            args = parse_llm_json(args)
        except Exception:
            args = None
    return name, args


def _extract_response_format_tool_call(
    msg: Any,
    response_tool_name: str,
    fields: List[Dict[str, Any]],
    coerce_list_fn: Optional[Callable[[Any, List[Dict[str, Any]]], Any]] = None,
) -> Optional[Dict[str, Any]]:
    tool_calls = list(getattr(msg, "tool_calls", None) or [])

    for call in reversed(tool_calls):
        name, args = _tool_call_name_and_args(call)
        if coerce_list_fn is not None:
            args = coerce_list_fn(args, fields)
        if not isinstance(args, dict):
            continue
        if name == response_tool_name or _looks_like_field_payload(
            args, fields
        ):
            return args
    return None


def _extract_from_messages(
    messages: List[Any],
    fields: List[Dict[str, Any]],
    schema_name: str,
    coerce_list_fn: Optional[Callable[[Any, List[Dict[str, Any]]], Any]] = None,
) -> Optional[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for msg in reversed(messages):
        parsed = _extract_response_format_tool_call(
            msg, schema_name, fields, coerce_list_fn
        )
        if parsed is not None:
            candidates.append(parsed)

    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if not content:
            continue
        try:
            parsed = parse_llm_json(content)
        except Exception:
            continue
        if coerce_list_fn is not None:
            parsed = coerce_list_fn(parsed, fields)
        if isinstance(parsed, dict) and _looks_like_field_payload(
            parsed, fields
        ):
            candidates.append(parsed)

    if not candidates:
        return None

    best = max(
        candidates, key=lambda payload: _payload_value_score(payload, fields)
    )
    return best


def extract_agent_payload(
    response: Any,
    fields: List[Dict[str, Any]],
    schema_name: str = "",
    coerce_list_fn: Optional[Callable[[Any, List[Dict[str, Any]]], Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Consolidated helper to parse structured response payload from LangChain message response,
    dictionary, list, or model object.
    """
    if coerce_list_fn is not None:
        response = coerce_list_fn(response, fields)

    if hasattr(response, "content") or hasattr(response, "tool_calls"):
        return _extract_from_messages([response], fields, schema_name, coerce_list_fn)

    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict) and _looks_like_field_payload(dumped, fields):
            return dumped
        return None

    if isinstance(response, dict):
        for key in ("structured_response", "response", "output", "final"):
            nested = response.get(key)
            parsed = extract_agent_payload(
                nested, fields, schema_name, coerce_list_fn
            )
            if parsed is not None:
                return parsed
        messages = response.get("messages")
        if isinstance(messages, list):
            parsed = _extract_from_messages(
                messages, fields, schema_name, coerce_list_fn
            )
            if parsed is not None:
                return parsed
        if _looks_like_field_payload(response, fields):
            return response
        return None

    if isinstance(response, list):
        candidates: List[Dict[str, Any]] = []
        parsed_messages = _extract_from_messages(
            response, fields, schema_name, coerce_list_fn
        )
        if parsed_messages is not None:
            candidates.append(parsed_messages)
        for item in reversed(response):
            parsed = extract_agent_payload(
                item, fields, schema_name, coerce_list_fn
            )
            if parsed is not None:
                candidates.append(parsed)
        if candidates:
            def get_dict_score(cand):
                if hasattr(cand, "model_dump"):
                    return _payload_value_score(cand.model_dump(), fields)
                if isinstance(cand, dict):
                    return _payload_value_score(cand, fields)
                return 0
            return max(candidates, key=get_dict_score)
        return None

    return None

