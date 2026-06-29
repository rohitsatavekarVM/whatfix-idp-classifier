"""Andromeda workflow-based document processor"""

import os

# Ensure headless-friendly matplotlib backend unless explicitly overridden
if not os.environ.get("MPLBACKEND"):
    os.environ["MPLBACKEND"] = "Agg"

import asyncio
import inspect
import logging
import time
from datetime import date, datetime
import json
from pathlib import Path
import shutil
import multiprocessing
import tempfile
import re
import base64
import io

from typing import Any, Dict, List

from andromeda.config import ModelConfig
from andromeda.utils import get_chat_model
from andromeda import HumanMessage

from .common import (
    textract_extract_pages,
    combine_dicts,
    run_tesseract_on_pages,
)
from .image_classifier import classify_image

logger = logging.getLogger(__name__)

# Optional WorkflowBuilder
try:
    from andromeda.core.workflow import WorkflowBuilder
except ImportError:
    WorkflowBuilder = None

# Andromeda helper functions
from app.helpers.document_classifier import predict_document_type


async def andromeda_process_files(
    file_name: str,
    file_path: str,
    file_id: str,
    summarize: bool = True,
    session_id=None,
    context: Dict[str, Any] | None = None,
):
    """
    Andromeda workflow version of file processing.

    Uses WorkflowBuilder to orchestrate document processing and
    andromeda_flow for LLM-powered document classification.

    Yields:
        progress,
        message,
        is_done,
        result,
        page_count,
        file_id
    """

    print("===== Andromeda workflow started =====")

    if WorkflowBuilder is None:
        raise ImportError(
            "Andromeda WorkflowBuilder not available."
        )

    # Initial progress
    yield (
        10,
        "Processing file",
        False,
        None,
        None,
        file_id,
    )

    await asyncio.sleep(0)

    state = {
        "session_id": session_id,
        "submission_context": context or {},
        "file_name": file_name,
        "file_path": file_path,
        "file_id": file_id,
        "page_count": 0,
        "raw_pages": {},
        "extracted_text": [],
        "grouped_texts": [],
        "grouped_page_numbers": [],
        "predictions": [],
        "hl_types": [],
        "document_type": "OTHER",
        "common_hl_type": "OTHER",
        "overviews": [],
        "summaries": [],
        "final_overview": None,
        "final_summary": None,
        "claim_number": "not found",
        "integration_data": {},
        "integration_results": {},
        "group_pdf_paths": [],
        "clustering": {
            "pass1": {},
            "pass2": {},
        },
        "deconflicted_fields": {},
        "deconfliction_log": {},
        "hitl_exceptions": [],
        "hitl_required": False,
        "specialized_agent_results": {},
        "next_best_actions": {
            "enabled": False,
            "actions": [],
        },
        "input_processing_route": "unknown",
        "page_processing_modes": {},
        "digital_pages": [],
        "ocr_pages": [],
        "vision_pages": [],
        "classified_images": [],
        "debug_visualize": False,
        "result": None,
    }

    _ctx: Dict[str, Any] = {
        "images": [],
    }

    # Progress signalling helper
    yield_progress = [(None, None)]

    def _mark(progress: int, message: str):
        yield_progress[0] = (progress, message)

    def _state_log_summary(s: Any) -> Dict[str, Any]:
        """Compact workflow state summary for logs without dumping payloads."""
        if not isinstance(s, dict):
            return {"state_type": type(s).__name__}

        def _safe_len(value: Any) -> int:
            try:
                return len(value or [])
            except Exception:
                return 0

        return {
            "file_id": s.get("file_id"),
            "file_type": s.get("file_type"),
            "pages": s.get("page_count"),
            "route": s.get("input_processing_route"),
            "digital_pages": _safe_len(s.get("digital_pages")),
            "ocr_pages": _safe_len(s.get("ocr_pages")),
            "vision_pages": _safe_len(s.get("vision_pages")),
            "extracted_pages": _safe_len(s.get("extracted_text")),
            "groups": _safe_len(s.get("grouped_page_numbers")),
            "predictions": _safe_len(s.get("predictions")),
            "document_type": s.get("document_type"),
            "common_hl_type": s.get("common_hl_type"),
            "sub_documents": _safe_len(s.get("sub_documents")),
            "extraction_results": _safe_len(s.get("extraction_results")),
            "assembled_fields": _safe_len(s.get("assembled_fields")),
            "deconflicted_fields": _safe_len(s.get("deconflicted_fields")),
            "claim_number": s.get("claim_number"),
            "hitl_required": s.get("hitl_required"),
            "integration_results": sorted((s.get("integration_results") or {}).keys()),
            "result_ready": bool(s.get("result")),
        }

    def _logged_step(step_name: str, step_func):
        async def _runner(s: dict) -> dict:
            start_time = time.perf_counter()
            logger.info("[WORKFLOW] START %s | state=%s", step_name, _state_log_summary(s))
            try:
                result = step_func(s)
                if inspect.isawaitable(result):
                    result = await result
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                logger.info(
                    "[WORKFLOW] END %s | duration_ms=%s state=%s",
                    step_name,
                    duration_ms,
                    _state_log_summary(result),
                )
                return result
            except Exception:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                logger.exception(
                    "[WORKFLOW] ERROR %s | duration_ms=%s state=%s",
                    step_name,
                    duration_ms,
                    _state_log_summary(s),
                )
                raise

        return _runner

    def step_detect_file_type(s: dict) -> dict:
        """
        Detect the file type based on file extension and update workflow state.
        """

        file_name = s.get("file_name", "").lower()

        if not file_name:
            raise Exception("File name missing in state")

        ext = file_name.split(".")[-1]

        if ext == "pdf":
            file_type = "pdf"

        elif ext in ("tif", "tiff", "jpg", "jpeg", "png"):
            file_type = "image"

        elif ext == "docx":
            file_type = "docx"

        elif ext == "xlsx":
            file_type = "xlsx"

        else:
            raise Exception(f"Unsupported file type: {ext}")

        s["file_type"] = file_type
        s["extension"] = ext
        print("Current State after file type detection:", s["file_type"])
        return s
    
    def step_parse_document(s: dict) -> dict:

        file_type = s.get("file_type")
        file_path = s.get("file_path")

        if not file_type or not file_path:
            raise Exception("Missing file_type or file_path in state")

        raw_pages = {}
        candidates = []

        # -------------------------
        # PDF (PyMuPDF)
        # -------------------------
        if file_type == "pdf":
            import fitz
            from PIL import Image
            import io

            doc = fitz.open(file_path)

            for page_index in range(len(doc)):
                page = doc[page_index]

                # TEXT
                text = page.get_text("text")
                raw_pages[page_index + 1] = text or ""

                page_area = page.rect.width * page.rect.height

                # IMAGES
                images = page.get_images(full=True)

                for img in images:
                    xref = img[0]
                    rects = page.get_image_rects(xref)

                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    for rect in rects:
                        image_area = rect.width * rect.height
                        ratio = image_area / page_area if page_area > 0 else 0

                        candidates.append({
                            "page": page_index + 1,
                            "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                            "image": pil_image,
                            "area_ratio": ratio
                        })

            doc.close()

        # -------------------------
        # IMAGE
        # -------------------------
        elif file_type == "image":
            from PIL import Image

            img = Image.open(file_path).convert("RGB")

            raw_pages = {1: ""}

            candidates = [{
                "page": 1,
                "bbox": None,
                "image": img,
                "area_ratio": 1.0  # entire image
            }]

        # -------------------------
        # DOCX
        # -------------------------
        elif file_type == "docx":
            from docx2python import docx2python
            import zipfile
            from PIL import Image
            import io

            result = docx2python(file_path)

            # safer text extraction
            text = str(result.text)
            raw_pages = {1: text}

            with zipfile.ZipFile(file_path) as docx:
                for name in docx.namelist():
                    if name.startswith("word/media/"):
                        try:
                            image_bytes = docx.read(name)
                            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                            candidates.append({
                                "page": 1,
                                "bbox": None,
                                "image": img,
                                "area_ratio": None
                            })
                        except Exception as e:
                            print(f"[WARN] DOCX image read failed: {e}")

        # -------------------------
        # XLSX
        # -------------------------
        elif file_type == "xlsx":
            import openpyxl
            from PIL import Image
            import io

            wb = openpyxl.load_workbook(file_path)

            for i, sheet in enumerate(wb.worksheets, start=1):

                text_lines = []

                        
                for row in sheet.iter_rows(values_only=True):
                    row_text = " ".join([str(c) for c in row if c])
                    if row_text:
                        text_lines.append(row_text)

                raw_pages[i] = "\n".join(text_lines)

                # images
                if hasattr(sheet, "_images"):
                    for img in sheet._images:
                        try:
                            image_bytes = img._data()
                            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                            candidates.append({
                                "page": i,
                                "bbox": None,
                                "image": pil_img,
                                "area_ratio": None
                            })
                        except Exception as e:
                            print(f"[WARN] Excel image extraction failed: {e}")

        else:
            raise Exception(f"Unsupported file_type: {file_type}")

        # FINAL STATE UPDATE
        s["page_count"] = len(raw_pages)
        s["raw_pages"] = raw_pages
        _ctx["classification_candidates"] = candidates   # with images
        
        return s

    def step_filter_classification_candidates(s: dict) -> dict:

        threshold = 0.10

        filtered = []
        ignored = []

        for c in _ctx.get("classification_candidates", []):

            ratio = c.get("area_ratio")

            # If ratio is None (docx/xlsx), keep it
            if ratio is None:
                filtered.append(c)
                continue

            if ratio >= threshold:
                filtered.append(c)
            else:
                ignored.append(c)

        _ctx["classification_candidates"] = filtered
        _ctx["ignored_candidates"] = ignored

        return s
    
    def step_classify_images(s: dict) -> dict:

        candidates = _ctx.get("classification_candidates", [])
        page_count = s.get("page_count", 0)

        if not candidates:
            s["classified_images"] = []
            s["page_types"] = {
                p: "document" for p in range(1, page_count + 1)
            }
            return s

        from .image_classifier import classify_image

        page_results = {}
        classified = []

        for c in candidates:

            img = c["image"]
            page = c["page"]

            try:
                result = classify_image(img)
                label = result["label"]
                confidence = result["confidence"]

            except Exception as e:
                print(f"[WARN] Classification failed: {e}")
                continue

            classified.append({
                "page": page,
                "label": label,
                "confidence": confidence,
                "bbox": c.get("bbox"),
                "area_ratio": c.get("area_ratio")
            })

            if page not in page_results:
                page_results[page] = []

            page_results[page].append((label, confidence))

        # PAGE LEVEL DECISION
        page_types = {
            p: "document" for p in range(1, page_count + 1)
        }

        for page, results in page_results.items():
            labels = [r[0] for r in results]
            if any("photo" in l.lower() or "photograph" in l.lower() for l in labels):
                page_types[page] = "photo"
                _ctx["classification_candidates"] = candidates
            else:
                page_types[page] = "document"

        s["classified_images"] = classified
        s["page_types"] = page_types

        return s

    def step_identify_input_processing(s: dict) -> dict:
        """Route each page through digital text, OCR, or vision processing."""

        page_types = s.get("page_types", {})
        raw_pages = s.get("raw_pages", {})
        page_count = s.get("page_count", 0)

        digital_pages = []
        ocr_pages = []
        vision_pages = []
        page_processing_modes = {}

        for page in range(1, page_count + 1):
            ptype = page_types.get(page, "document")
            raw_text = str(raw_pages.get(page, "") or "").strip()
            has_digital_text = len(raw_text) >= 40

            if ptype == "photo":
                vision_pages.append(page)
                page_processing_modes[page] = "vision"
            elif has_digital_text:
                digital_pages.append(page)
                page_processing_modes[page] = "digital_text"
            else:
                ocr_pages.append(page)
                page_processing_modes[page] = "ocr"

        if ocr_pages and vision_pages:
            route = "ocr_and_vision"
        elif ocr_pages:
            route = "ocr"
        elif vision_pages:
            route = "vision"
        else:
            route = "digital"

        s["digital_pages"] = digital_pages
        s["ocr_pages"] = ocr_pages
        s["vision_pages"] = vision_pages
        s["page_processing_modes"] = page_processing_modes
        s["input_processing_route"] = route
        s["requires_ocr"] = bool(ocr_pages)
        s["requires_vision"] = bool(vision_pages)

        print("INPUT PROCESSING ROUTE:", route)
        print("DIGITAL PAGES:", digital_pages)
        print("OCR PAGES:", ocr_pages)
        print("VISION PAGES:", vision_pages)

        return s

    def step_process_vision_pages(s: dict) -> dict:
        vision_pages = s.get("vision_pages", [])

        if not vision_pages:
            s["vision_results"] = {}
            return s

        page_image_map = {}
        for candidate in _ctx.get("classification_candidates", []):
            page = candidate.get("page")
            if page in vision_pages and page not in page_image_map:
                page_image_map[page] = candidate["image"]

        vision_results = {}

        for page_num in vision_pages:
            pil_image = page_image_map.get(page_num)

            if pil_image is None:
                print(f"[WARN] No image found for vision page {page_num}, skipping.")
                vision_results[page_num] = ""
                continue

            try:
                buffer = io.BytesIO()
                pil_image.save(buffer, format="JPEG")
                b64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")



                


                message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": (
                                "Describe exactly what you see in this image. "
                                "Be specific and objective - include visible objects, their physical condition, "
                                "materials, colors, spatial layout, and the extent of any damage. "
                                "Do not speculate about context, purpose, or what the image might be used for. "
                                "Only describe what is directly visible. "
                                "If the image is purely decorative or does not contain useful content for extraction, "
                                "respond with REMOVE_MARKER only."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            }
                        }
                    ]
                )
                response = _vision_chat_model().invoke([message])
                description = _coerce_multimodal_response(response).strip()
                if description == "REMOVE_MARKER":
                    description = ""
                vision_results[page_num] = description
                print(f"\nVision Description:\n{description}\n")

            except Exception as e:
                print(f"[ERROR] Vision processing failed for page {page_num}: {e}")
                vision_results[page_num] = ""

        raw_pages = s.get("raw_pages", {})
        for page_num, description in vision_results.items():
            if description:
                existing = raw_pages.get(page_num, "")
                raw_pages[page_num] = (existing + "\n" + description).strip()

        s["raw_pages"] = raw_pages
        s["vision_results"] = vision_results
        return s
    
    async def step_ocr_extract(s: dict) -> dict:

        _mark(30, "Running OCR")

        file_path = s["file_path"]
        ocr_pages = s.get("ocr_pages", [])
        vision_pages = s.get("vision_pages", [])     
        vision_results = s.get("vision_results", {})  
        page_count = s.get("page_count", 0)
        ocr_engine = s.get("ocr_engine", "tesseract")

        try:
            pages_dict = {}
            if ocr_pages:
                if ocr_engine == "textract":
                    pages_dict, _ = await textract_extract_pages(
                        file_path=file_path,
                        pages=ocr_pages
                    )
                elif ocr_engine == "tesseract":
                    pages_dict = await asyncio.to_thread(
                        run_tesseract_on_pages,
                        {"file_path": file_path},
                        ocr_pages,
                    )
                else:
                    raise Exception(f"Unsupported OCR engine: {ocr_engine}")

            # Normalize to the full ordered page list.
            extracted_list = []
            page_map = []

            page_count = s.get("page_count", 0)
            raw_pages = s.get("raw_pages", {})  # from PyMuPDF

            extracted_text = []
            page_map = []

            for p in range(1, page_count + 1):
                ocr_text = pages_dict.get(p, "").strip()
                pymu_text = raw_pages.get(p, "").strip()
                vision_text = vision_results.get(p, "").strip()

                # Merge logic, giving OCR priority where available.
                #final_text = ocr_text if ocr_text else pymu_text

                if p in vision_pages:
                    final_text = vision_text or pymu_text
                elif p in ocr_pages:
                    final_text = ocr_text if ocr_text else pymu_text
                else:
                    final_text = pymu_text

                extracted_text.append(final_text)
                page_map.append(p)

            s["extracted_text"] = extracted_text
            s["page_map"] = page_map
        except Exception as e:
            print("OCR ERROR:", str(e))
            s["extracted_text"] = []
            s["page_map"] = []

        return s

    def step_predict_types(s: dict) -> dict:
        """
        Predict document type using the LLM classifier.
        """

        _mark(60, "Predicting document type")

        state = predict_document_type(s)

        return state
    
    def step_save_results(s: dict) -> dict:
        """
        Final workflow step.
        Builds the result consumed by IDPClassifier.
        """

        s["result"] = {
            "file_name": s.get("file_name"),
            "document_type": s.get("document_type", "UNKNOWN"),
            "confidence": float(s.get("confidence", 0.0)),
            "predictions": s.get("predictions", []),
            "page_count": s.get("page_count", 0),
            "input_processing_route": s.get("input_processing_route"),
            "page_processing_modes": s.get("page_processing_modes", {}),
            "digital_pages": s.get("digital_pages", []),
            "ocr_pages": s.get("ocr_pages", []),
            "vision_pages": s.get("vision_pages", []),
            "extracted_text": s.get("extracted_text", []),
        }

        return s

    wf = WorkflowBuilder(name="WhatfixClassification")
    (
        wf.start("step_detect_file_type") \
            .run(step_detect_file_type)
        .then("step_parse_document") \
            .run(step_parse_document)
        .then("step_filter_classification_candidates") \
            .run(step_filter_classification_candidates)
        .then("step_classify_images") \
            .run(step_classify_images)
        .then("step_identify_input_processing") \
            .run(step_identify_input_processing)
        .then("step_process_vision_pages") \
            .run(step_process_vision_pages)
        .then("step_ocr_extract") \
            .run(step_ocr_extract)
        .then("step_predict_types") \
            .run(step_predict_types)
        .finish("step_save_results") \
            .run(step_save_results)
    )

    try:
        async for partial_state in wf.astream(state=state):

            progress = yield_progress[0]

            if progress != (None, None):
                progress_value, message = progress

                yield (
                    progress_value or 50,
                    message or "Processing...",
                    False,
                    None,
                    partial_state.get("page_count"),
                    file_id,
                )

                yield_progress[0] = (None, None)

            await asyncio.sleep(0)

        result = {
            "file_name": partial_state.get("file_name"),
            "document_type": partial_state.get("document_type", "UNKNOWN"),
            "confidence": partial_state.get("confidence", 0.0),
            "predictions": partial_state.get("predictions"),
            "page_count": partial_state.get("page_count"),
            "input_processing_route": partial_state.get("input_processing_route"),
            "page_processing_modes": partial_state.get("page_processing_modes"),
            "digital_pages": partial_state.get("digital_pages"),
            "ocr_pages": partial_state.get("ocr_pages"),
            "vision_pages": partial_state.get("vision_pages"),
            "extracted_text": partial_state.get("extracted_text"),
        }

        yield (
            100,
            "Classification Complete",
            True,
            result,
            result["page_count"],
            file_id,
        )

    except Exception as e:
        logger.exception("Classification workflow failed")

        yield (
            100,
            "Error",
            True,
            {"error": str(e)},
            state.get("page_count"),
            file_id,
        )