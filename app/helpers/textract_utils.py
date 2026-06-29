"""
AWS Textract helper utilities.

This module provides S3-backed async text extraction suitable for PDFs/TIFFs via:
  - textract.start_document_text_detection
  - textract.get_document_text_detection

Why S3-backed?
Textract async APIs require the document to be in S3. This is the most reliable path
for multi-page PDFs and TIFFs and works well in Fargate.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Optional, Set, Tuple

import boto3

_FORM_ID_OR_FOOTER_RE = re.compile(
    r"(?i)\b("
    r"acord\s*\d+"
    r"|page\s+\d+\s+of\s+\d+"
    r"|all rights reserved"
    r"|registered marks of acord"
    r")\b|©\s*\d{4}"
)

_TEMPLATE_PROMPT_HINTS = (
    "address",
    "annual revenues",
    "any area leased to others",
    "business phone",
    "city",
    "city limits",
    "contact name",
    "contact type",
    "county",
    "description of operations",
    "e-mail",
    "email",
    "fax",
    "full time empl",
    "interest",
    "loc #",
    "name and mailing address",
    "open to public area",
    "part time empl",
    "phone",
    "sq ft",
    "state",
    "street",
    "website",
    "zip",
)


@dataclass(frozen=True)
class TextractSettings:
    s3_bucket: str
    s3_prefix: str = "textract-input/"
    poll_interval_seconds: float = 2.0
    poll_timeout_seconds: int = 900
    delete_s3_object: bool = True


def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_textract_settings_from_env() -> TextractSettings:
    bucket = os.getenv("TEXTRACT_S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError(
            "Missing required env var TEXTRACT_S3_BUCKET (Textract async requires S3)."
        )

    prefix = os.getenv("TEXTRACT_S3_PREFIX", "textract-input/").strip() or "textract-input/"
    if not prefix.endswith("/"):
        prefix += "/"

    poll_interval = float(os.getenv("TEXTRACT_POLL_INTERVAL_SECONDS", "2.0"))
    poll_timeout = int(os.getenv("TEXTRACT_POLL_TIMEOUT_SECONDS", "900"))
    delete_obj = _bool_env("TEXTRACT_DELETE_S3_OBJECT", True)

    return TextractSettings(
        s3_bucket=bucket,
        s3_prefix=prefix,
        poll_interval_seconds=poll_interval,
        poll_timeout_seconds=poll_timeout,
        delete_s3_object=delete_obj,
    )


def _get_region_name() -> Optional[str]:
    # boto3 can resolve region implicitly in ECS, but being explicit helps local runs.
    return os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")


def _clients():
    region = _get_region_name()
    if region:
        return boto3.client("s3", region_name=region), boto3.client("textract", region_name=region)
    return boto3.client("s3"), boto3.client("textract")


def _put_s3_object(s3_client, bucket: str, key: str, file_path: str) -> None:
    # Upload as binary; content-type is optional for Textract.
    s3_client.upload_file(file_path, bucket, key)


def _delete_s3_object_quietly(s3_client, bucket: str, key: str) -> None:
    try:
        s3_client.delete_object(Bucket=bucket, Key=key)
    except Exception:
        # Non-fatal cleanup.
        pass


def _bbox(block: Dict) -> Tuple[float, float, float, float]:
    """
    Return (left, top, width, height) in normalized page coordinates (0..1).
    Missing geometry defaults to zeros.
    """
    bb = ((block or {}).get("Geometry") or {}).get("BoundingBox") or {}
    left = float(bb.get("Left") or 0.0)
    top = float(bb.get("Top") or 0.0)
    width = float(bb.get("Width") or 0.0)
    height = float(bb.get("Height") or 0.0)
    return left, top, width, height


def _normalize_line_text(text: str) -> str:
    """Normalize a LINE block into stable plain text."""
    t = (text or "").strip()
    if not t:
        return ""

    for src, dst in {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }.items():
        t = t.replace(src, dst)

    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"\$\s+(\d)", r"$\1", t)
    t = re.sub(r"(?<=\d)\s+%", "%", t)
    return t.strip()


def _is_checkbox_mark(token: str) -> bool:
    return token.strip().lower() in {"x", "✓", "✔", "☒", "☑"}


def _is_form_id_or_footer(text: str) -> bool:
    return bool(_FORM_ID_OR_FOOTER_RE.search((text or "").strip()))


def _is_value_like_cell(cell: str) -> bool:
    c = (cell or "").strip()
    if not c:
        return False
    if _is_checkbox_mark(c):
        return False
    if c in {"$", "%", "-", "/"}:
        return False
    if re.search(r"\d", c):
        return True
    lower = c.lower()
    if "@" in c or "www." in lower:
        return True
    if lower in {"yes", "no", "y", "n", "true", "false", "am", "pm"}:
        return True
    letters = [ch for ch in c if ch.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return upper_ratio < 0.85 and not c.endswith(":")


def _template_prompt_score(cell: str) -> int:
    c = (cell or "").strip()
    if not c:
        return 0
    lower = c.lower()
    score = 0
    if c.endswith(":"):
        score += 2
    if "#" in c:
        score += 1
    if c in {"$", "%", "SQ FT", "sq ft"}:
        score += 1
    if "mm/dd/yyyy" in lower:
        score += 2
    if any(hint in lower for hint in _TEMPLATE_PROMPT_HINTS):
        score += 2
    return score


def _is_template_row_cells(cells: List[str]) -> bool:
    non_empty = [c.strip() for c in cells if c.strip()]
    if len(non_empty) < 2:
        return False
    value_count = sum(1 for c in non_empty if _is_value_like_cell(c))
    prompt_score = sum(_template_prompt_score(c) for c in non_empty)
    if len(non_empty) >= 3 and value_count == 0 and prompt_score >= 3:
        return True
    if len(non_empty) >= 4 and value_count <= 1 and prompt_score >= 7:
        return True
    return False


def _is_template_prompt_line(text: str) -> bool:
    line = (text or "").strip()
    if not line:
        return False
    lower = line.lower()
    if line.endswith(":") and not _is_value_like_cell(line):
        return True
    if any(hint in lower for hint in _TEMPLATE_PROMPT_HINTS):
        return not _is_value_like_cell(line)
    return False


def _find_checkbox_label(cells: List[str], idx: int) -> Optional[str]:
    for j in range(idx + 1, min(idx + 4, len(cells))):
        cand = cells[j].strip()
        if not cand or _is_checkbox_mark(cand):
            continue
        if cand in {"$", "%", "-", "/"} or cand.endswith(":"):
            continue
        if re.search(r"[A-Za-z0-9]", cand):
            return cand
    for j in range(max(0, idx - 2), idx):
        cand = cells[j].strip()
        if not cand or _is_checkbox_mark(cand):
            continue
        if cand in {"$", "%", "-", "/"} or cand.endswith(":"):
            continue
        if re.search(r"[A-Za-z0-9]", cand):
            return cand
    return None


def _normalize_checkbox_cells(cells: List[str]) -> List[str]:
    selected: List[str] = []
    out: List[str] = []
    for idx, raw in enumerate(cells):
        cell = raw.strip()
        if not cell:
            continue
        if _is_checkbox_mark(cell):
            label = _find_checkbox_label(cells, idx) or "true"
            selected.append(label)
            continue
        out.append(cell)

    if selected:
        uniq: List[str] = []
        seen: Set[str] = set()
        for val in selected:
            key = val.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(val)
        out.append(f"selected: {', '.join(uniq)}")
    return out


def _is_label_like_row(cells: List[str]) -> bool:
    non_empty = [c.strip() for c in cells if c.strip()]
    if len(non_empty) < 4:
        return False
    value_count = sum(1 for c in non_empty if _is_value_like_cell(c))
    return value_count <= max(1, len(non_empty) // 4)


def _is_value_like_row(cells: List[str]) -> bool:
    non_empty = [c.strip() for c in cells if c.strip()]
    if not non_empty:
        return False
    value_count = sum(1 for c in non_empty if _is_value_like_cell(c))
    return value_count >= max(1, len(non_empty) // 3)


def _median(values: List[float], default: float) -> float:
    if not values:
        return default
    vals = sorted(values)
    return vals[len(vals) // 2]


def _align_columns_with_empties(
    template_cols: List[Dict],
    value_cols: List[Dict],
    *,
    col_gap: float,
    empty_token: str = "[EMPTY]",
) -> Optional[List[str]]:
    """
    Align a sparse value row to a denser template row and synthesize missing cells.
    Returns None when alignment is too uncertain.
    """
    if not template_cols or not value_cols:
        return None
    if len(value_cols) > len(template_cols):
        return None

    template_centers = [((c["left"] + c["right"]) / 2.0) for c in template_cols]
    template_widths = [max(0.0, float(c["right"] - c["left"])) for c in template_cols]
    center_gaps = [b - a for a, b in zip(template_centers, template_centers[1:]) if (b - a) > 0]
    typical_gap = _median(center_gaps, default=max(col_gap, 0.03))
    typical_width = _median(template_widths, default=0.03)
    distance_limit = max(typical_gap * 2.0, typical_width * 2.2, 0.06)

    assigned: List[Optional[str]] = [None] * len(template_cols)
    nearest_distances: List[float] = []
    sorted_values = sorted(value_cols, key=lambda x: x["left"])
    prev_idx = -1
    m = len(sorted_values)
    n = len(template_cols)
    for k, col in enumerate(sorted_values):
        center = (col["left"] + col["right"]) / 2.0
        min_idx = prev_idx + 1
        max_idx = n - (m - k)
        if min_idx > max_idx:
            return None
        ranked = sorted(
            ((abs(center - template_centers[idx]), idx) for idx in range(min_idx, max_idx + 1)),
            key=lambda x: x[0],
        )
        chosen_dist, chosen_idx = ranked[0]
        if chosen_dist > distance_limit:
            return None
        assigned[chosen_idx] = col["text"].strip()
        nearest_distances.append(chosen_dist)
        prev_idx = chosen_idx

    if not nearest_distances:
        return None

    mean_distance = sum(nearest_distances) / len(nearest_distances)
    if mean_distance > (distance_limit * 0.75):
        return None

    return [val if val is not None else empty_token for val in assigned]


def _canonical_repeat_signature(text: str) -> str:
    """
    Build a canonical signature for deduping repeated header/footer lines.
    """
    sig = (text or "").strip().lower()
    if not sig:
        return ""
    sig = re.sub(r"\bpage\s+\d+\s+of\s+\d+\b", "page # of #", sig)
    sig = re.sub(r"\s+", " ", sig)
    sig = re.sub(r"[^a-z0-9# ]+", "", sig)
    return sig.strip()


def _is_margin_item(item: Dict, *, top_cutoff: float = 0.13, bottom_cutoff: float = 0.87) -> bool:
    return float(item.get("top") or 0.0) <= top_cutoff or float(item.get("bottom") or 0.0) >= bottom_cutoff


def _is_probable_heading(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return uppercase_ratio >= 0.75 and len(text.split()) <= 8


def _should_join_wrapped_lines(prev: str, curr: str) -> bool:
    """
    Conservative wrap join for paragraph continuations.
    """
    if not prev or not curr:
        return False
    if "\t" in prev or "\t" in curr:
        return False
    if prev.endswith(":") or prev.endswith((".", "!", "?")):
        return False
    if _is_probable_heading(prev) or _is_probable_heading(curr):
        return False
    if prev.endswith("-"):
        return True
    if len(prev) >= 25 and curr[:1].islower():
        return True
    tail = prev.split()[-1].lower() if prev.split() else ""
    return tail in {"a", "an", "and", "for", "in", "of", "on", "or", "the", "to", "with"}


def _compact_lines(lines: List[str]) -> List[str]:
    """
    Remove noisy duplicates and merge obvious line-wrap continuations.
    """
    out: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue

        if out and out[-1] == line:
            continue

        if out and out[-1] != "" and _should_join_wrapped_lines(out[-1], line):
            if out[-1].endswith("-"):
                out[-1] = f"{out[-1][:-1]}{line.lstrip()}"
            else:
                out[-1] = f"{out[-1]} {line}"
            continue

        out.append(line)

    while out and out[-1] == "":
        out.pop()
    return out


def _group_into_bands(items: List[Dict], *, y_tolerance: float) -> List[List[Dict]]:
    """
    Group items that share a similar vertical band (row).

    Items must include: center_y (float).
    """
    if not items:
        return []
    items_sorted = sorted(items, key=lambda x: (x["center_y"], x["left"]))
    bands: List[List[Dict]] = []
    current: List[Dict] = [items_sorted[0]]
    current_y = items_sorted[0]["center_y"]
    for item in items_sorted[1:]:
        y = item["center_y"]
        if abs(y - current_y) <= y_tolerance:
            current.append(item)
        else:
            bands.append(current)
            current = [item]
            current_y = y
    bands.append(current)
    return bands


def _iter_line_items(blocks: Iterable[Dict]) -> Iterable[Dict]:
    for b in blocks or []:
        if not isinstance(b, dict) or b.get("BlockType") != "LINE":
            continue
        left, top, width, height = _bbox(b)
        page_num = int(b.get("Page") or 1)
        text = _normalize_line_text((b.get("Text") or ""))
        if not text.strip():
            continue
        yield {
            "page": page_num,
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "right": left + width,
            "bottom": top + height,
            "center_y": top + (height / 2.0),
            "text": text,
        }


def _detect_repeated_margin_signatures(pages: Dict[int, List[Dict]]) -> Set[str]:
    """
    Find lines repeated on most pages near top/bottom margins.
    """
    if len(pages) < 2:
        return set()

    per_sig_count: Dict[str, int] = {}
    for line_items in pages.values():
        page_sigs: Set[str] = set()
        for item in line_items:
            if not _is_margin_item(item):
                continue
            sig = _canonical_repeat_signature(item.get("text", ""))
            if len(sig) < 8:
                continue
            page_sigs.add(sig)
        for sig in page_sigs:
            per_sig_count[sig] = per_sig_count.get(sig, 0) + 1

    min_pages = max(2, ceil(len(pages) * 0.60))
    return {sig for sig, count in per_sig_count.items() if count >= min_pages}


def _band_has_columns(band: List[Dict], *, gap_threshold: float) -> bool:
    if len(band) < 2:
        return False
    gaps = [nxt["left"] - cur["right"] for cur, nxt in zip(band, band[1:])]
    return any(g >= gap_threshold for g in gaps)


def _render_page_normalized(
    line_items: List[Dict],
    *,
    page_num: int,
    repeated_margin_signatures: Set[str],
) -> str:
    """
    Produce normalized plain text suitable for downstream LLM extraction.

    - No synthetic markdown.
    - Preserve row structure for obvious multi-column bands using tab separators.
    - Strip repeated header/footer lines and known form/footer boilerplate.
    - Drop low-information template rows and normalize checkbox markers.
    """
    header = f"--- PAGE {page_num} ---"
    if not line_items:
        return header

    heights = sorted([li["height"] for li in line_items if li.get("height", 0) > 0])
    median_h = heights[len(heights) // 2] if heights else 0.012
    y_tol = max(median_h * 0.7, 0.012)
    v_gap = max(median_h * 1.8, 0.03)
    col_gap = max(median_h * 1.3, 0.025)

    bands = _group_into_bands(line_items, y_tolerance=y_tol)

    out_lines: List[str] = []
    prev_bottom: Optional[float] = None
    active_template_cols: Optional[List[Dict]] = None
    active_template_bottom: Optional[float] = None
    for band in bands:
        band.sort(key=lambda x: x["left"])
        band_top = min(b["top"] for b in band)
        band_bottom = max(b["bottom"] for b in band)

        if prev_bottom is not None and band_top - prev_bottom >= v_gap and out_lines and out_lines[-1] != "":
            out_lines.append("")

        pruned_band: List[Dict] = []
        for item in band:
            text = item.get("text", "")
            if _is_form_id_or_footer(text):
                continue
            sig = _canonical_repeat_signature(item.get("text", ""))
            if (
                sig
                and sig in repeated_margin_signatures
                and _is_margin_item(item)
            ):
                continue
            pruned_band.append(item)

        if pruned_band:
            if len(pruned_band) == 1:
                line = pruned_band[0]["text"]
                if _is_template_prompt_line(line):
                    continue
                out_lines.append(line)
                active_template_cols = None
                active_template_bottom = None
            elif _band_has_columns(pruned_band, gap_threshold=col_gap):
                geom_cells = [
                    {
                        "text": col["text"].strip(),
                        "left": col["left"],
                        "right": col["right"],
                    }
                    for col in pruned_band
                    if col["text"].strip()
                ]
                cells = [c["text"] for c in geom_cells]

                if (
                    active_template_cols
                    and active_template_bottom is not None
                    and (band_top - active_template_bottom) <= (v_gap * 2.2)
                    and len(cells) <= len(active_template_cols)
                    and _is_value_like_row(cells)
                ):
                    aligned_cells = _align_columns_with_empties(
                        active_template_cols,
                        geom_cells,
                        col_gap=col_gap,
                    )
                    if aligned_cells:
                        cells = aligned_cells

                cells = _normalize_checkbox_cells(cells)
                if not cells:
                    continue
                if _is_template_row_cells(cells):
                    continue
                row_text = "\t".join(cells)
                out_lines.append(row_text)

                if _is_label_like_row([c["text"] for c in geom_cells]):
                    active_template_cols = geom_cells
                    active_template_bottom = band_bottom
                elif active_template_bottom is not None and (band_top - active_template_bottom) > (v_gap * 2.2):
                    active_template_cols = None
                    active_template_bottom = None
            else:
                line = " ".join(col["text"] for col in pruned_band)
                if _is_template_prompt_line(line):
                    continue
                out_lines.append(line)
                active_template_cols = None
                active_template_bottom = None

        prev_bottom = max(prev_bottom or 0.0, band_bottom)

    compacted = _compact_lines(out_lines)
    if not compacted:
        return header
    return f"{header}\n" + "\n".join(compacted).strip()


def _extract_pages_from_blocks(blocks: List[Dict]) -> List[str]:
    # Textract returns LINE blocks with a "Page" number for multi-page documents.
    # We re-order by geometry and emit normalized plain text suitable for LLM extraction.
    if not blocks:
        return []

    pages: Dict[int, List[Dict]] = {}
    for li in _iter_line_items(blocks):
        pages.setdefault(li["page"], []).append(li)

    if not pages:
        return []

    repeated_margin_signatures = _detect_repeated_margin_signatures(pages)
    return [
        _render_page_normalized(
            pages[p],
            page_num=p,
            repeated_margin_signatures=repeated_margin_signatures,
        )
        for p in sorted(pages.keys())
    ]


def extract_text_pages_via_textract_s3(
    *,
    file_path: str,
    original_filename: str,
    settings: TextractSettings,
) -> List[str]:
    """
    Extract text per page using Textract async job (S3-backed).
    Returns a list of page texts (1-indexed order).

    Output is normalized plain text (page headers + row-ordered lines, with tabs for
    obvious multi-column rows) to improve downstream LLM extraction without using
    Textract AnalyzeDocument.
    """
    s3_client, textract_client = _clients()

    safe_name = os.path.basename(original_filename or os.path.basename(file_path))
    key = f"{settings.s3_prefix}{uuid.uuid4().hex}_{safe_name}"

    _put_s3_object(s3_client, settings.s3_bucket, key, file_path)

    job_id = None
    try:
        start = textract_client.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": settings.s3_bucket, "Name": key}}
        )
        job_id = start["JobId"]

        # Poll until SUCCEEDED/FAILED, then page through results.
        deadline = time.time() + settings.poll_timeout_seconds
        next_token = None
        blocks: List[Dict] = []

        # First, wait for terminal status.
        status = "IN_PROGRESS"
        while status in {"IN_PROGRESS"}:
            if time.time() > deadline:
                raise TimeoutError(
                    f"Textract job timed out after {settings.poll_timeout_seconds}s (job_id={job_id})."
                )
            resp = textract_client.get_document_text_detection(JobId=job_id, MaxResults=1000)
            status = resp.get("JobStatus", "IN_PROGRESS")
            if status == "SUCCEEDED":
                blocks.extend(resp.get("Blocks", []) or [])
                next_token = resp.get("NextToken")
                break
            if status == "FAILED":
                msg = resp.get("StatusMessage") or "Textract job failed"
                raise RuntimeError(f"{msg} (job_id={job_id})")
            time.sleep(settings.poll_interval_seconds)

        # Then fetch remaining pages if present.
        while next_token:
            resp = textract_client.get_document_text_detection(
                JobId=job_id, NextToken=next_token, MaxResults=1000
            )
            blocks.extend(resp.get("Blocks", []) or [])
            next_token = resp.get("NextToken")

        return _extract_pages_from_blocks(blocks)
    finally:
        if settings.delete_s3_object:
            _delete_s3_object_quietly(s3_client, settings.s3_bucket, key)
