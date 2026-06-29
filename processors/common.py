"""Common utilities for document processing"""
import asyncio
import multiprocessing
from typing import Any, Dict, List, Tuple
from PIL import Image
from pdf2image import convert_from_path
from pypdf import PdfReader, PdfWriter
from pdf2image import convert_from_path
import uuid
import os
import logging


logger = logging.getLogger(__name__)

try:
    from app.helpers import ocr
    from app.helpers.textract_utils import (
        extract_text_pages_via_textract_s3,
        load_textract_settings_from_env,
    )
except Exception:
    from helpers import ocr
    from helpers.textract_utils import (
        extract_text_pages_via_textract_s3,
        load_textract_settings_from_env,
    )

# Set start method to spawn for CUDA compatibility
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set
    pass


def pdf_process(file_path: str):
    """Read pdf, convert it to images"""
    try:
        images = convert_from_path(file_path)
        return images
    except Exception as e:
        print("Error in PDF processing: ", e)
        return None


def tif_process(file_path: str):
    """Read TIFF image"""
    return Image.open(file_path)


async def combine_dicts(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine multiple dictionaries with intelligent merging"""
    if not dict_list:
        return {}
    
    EXCLUDE_VALUES = set(value.lower() for value in [
        '[Unreadable]', '[Specified]', '[Redacted]', 'Missing', 'Unknown',
        'Not Specified', '[Multiple entries, unreadable]', '[Missing]', ''
    ])

    async def value_is_excluded(val):
        if isinstance(val, str):
            return val.strip().lower() in EXCLUDE_VALUES or len(val) < 4
        return False

    def convert_to_hashable(x):
        if isinstance(x, dict):
            return frozenset((k, convert_to_hashable(v)) for k, v in sorted(x.items()))
        elif isinstance(x, list):
            return tuple(convert_to_hashable(i) for i in x)
        return x

    async def recursive_combine(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            combined = {}
            all_keys = set().union(*(d.keys() for d in dicts if isinstance(d, dict)))
            for key in all_keys:
                values = []
                for d in dicts:
                    if key in d:
                        val = d[key]
                        if val not in ['', {}, None]:
                            if isinstance(val, str):
                                if not await value_is_excluded(val):
                                    values.append(val)
                            elif isinstance(val, list):
                                filtered_list = []
                                for item in val:
                                    if isinstance(item, dict):
                                        item = await recursive_combine([item])
                                        if item:
                                            filtered_list.append(item)
                                    elif isinstance(item, str):
                                        if not await value_is_excluded(item):
                                            filtered_list.append(item)
                                    elif isinstance(item, bool):
                                        item = "Yes" if item else "No"
                                        filtered_list.append(item)
                                    else:
                                        filtered_list.append(item)
                                if filtered_list:
                                    values.append(filtered_list)
                            elif isinstance(val, dict):
                                val = await recursive_combine([val])
                                if isinstance(val, bool):
                                    val = "Yes" if val else "No"
                                if val:
                                    values.append(val)
                            else:
                                if isinstance(val, bool):
                                    val = "Yes" if val else "No"
                                if not await value_is_excluded(val):
                                    values.append(val)
                if not values:
                    continue
                if all(isinstance(v, dict) for v in values):
                    combined_value = await recursive_combine(values)
                    if combined_value:
                        combined[key] = combined_value
                elif all(isinstance(v, list) for v in values):
                    merged_list = []
                    for v in values:
                        merged_list.extend(v)
                    seen = set()
                    dedup_list = []
                    for item in merged_list:
                        hashed_item = convert_to_hashable(item)
                        if hashed_item not in seen:
                            seen.add(hashed_item)
                            dedup_list.append(item)
                    if dedup_list:
                        combined[key] = dedup_list
                else:
                    if all(isinstance(v, str) for v in values):
                        unique_values = list(set(values))
                        if unique_values:
                            combined[key] = unique_values[0] if len(unique_values) == 1 else unique_values
                    elif all(isinstance(v, (int, float)) for v in values):
                        combined[key] = sum(values)
                    else:
                        combined_values = []
                        for v in values:
                            if v not in ['', {}, None]:
                                combined_values.append(v)
                        if combined_values:
                            seen = set()
                            dedup_list = []
                            for cv in combined_values:
                                hashed_cv = convert_to_hashable(cv)
                                if hashed_cv not in seen:
                                    seen.add(hashed_cv)
                                    dedup_list.append(cv)
                            if dedup_list:
                                combined[key] = dedup_list
            return combined
        except Exception as e:
            logger.exception("Error in recursive combine: %s", e)
            return dict_list[0]
    
    combined_result = await recursive_combine(dict_list)
    return combined_result


def ocr_single_image(img):
    final_image = ocr.preprocessImage(img)
    config = '--oem 3 --psm 6'

    text = ocr.extract_and_process_hocr(final_image, config, timeout=30)

    if not text.strip():
        text = ocr.extract_and_process_hocr(final_image, config, timeout=60)

    return text

def batch_process_ocr_text_extraction(batch_data):
    """Process batch of images for OCR text extraction"""
    result = []
    idx = batch_data[0]
    for image in batch_data[1]:
        try:
            final_image = ocr.preprocessImage(image)
            config = '--oem 3 --psm 12'
            # Try with initial short timeout
            extracted_text = ocr.extract_and_process_hocr(final_image, config, timeout=30)
            if not extracted_text.strip():
                # If failed, retry with longer timeout
                extracted_text = ocr.extract_and_process_hocr(final_image, config, timeout=60)
            result.append(extracted_text)
        except Exception as e:
            print(f"OCR processing error: {str(e)}")
            result.append("")
    return idx, result



def run_tesseract_on_pages(s, ocr_pages):
    import fitz
    import pytesseract
    import os
    import shutil
    try:
        from app.helpers.config import TESSERACT_CMD
    except Exception:
        from helpers.config import TESSERACT_CMD

    # Configure pytesseract tesseract_cmd robustly (env, config, or system)
    if TESSERACT_CMD:
        if os.path.isabs(TESSERACT_CMD) and os.path.exists(TESSERACT_CMD):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        elif not os.path.isabs(TESSERACT_CMD):
            # allow non-absolute custom commands (user-managed PATH entry)
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        else:
            fallback = shutil.which("tesseract")
            if fallback:
                pytesseract.pytesseract.tesseract_cmd = fallback
    else:
        fallback = shutil.which("tesseract")
        if fallback:
            pytesseract.pytesseract.tesseract_cmd = fallback

    from PIL import Image
    import io

    file_path = s.get("file_path")

    doc = fitz.open(file_path)
    if not ocr_pages:
        ocr_pages = list(range(1, doc.page_count + 1))

    results = {}

    for p in ocr_pages:
        page = doc[p - 1]

        # render page → image
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # better OCR quality
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        results[p] = ocr_single_image(img)

    doc.close()
    return results

    doc.close()
    return results


# async def textract_extract_pages(file_path: str, file_name: str) -> Tuple[List[str], int]:
#     """
#     Extract text per page using AWS Textract (S3-backed async job).
#     Returns (pages, page_count).
#     """
#     settings = load_textract_settings_from_env()
#     pages = await asyncio.to_thread(
#         extract_text_pages_via_textract_s3,
#         file_path=file_path,
#         original_filename=file_name,
#         settings=settings,
#     )
#     return pages, len(pages)

async def textract_extract_pages(
    file_path: str,
    pages: list[int] | None = None
) -> Tuple[dict, int]:

    try:
        settings = load_textract_settings_from_env()
    except RuntimeError as e:
        print(f"[WARN] Textract not configured: {e}. Falling back to local OCR.")
        masked_pages = pages if pages else []
        pages_dict = await asyncio.to_thread(
            run_tesseract_on_pages,
            {"file_path": file_path},
            masked_pages,
        )
        return pages_dict, len(pages_dict)

    # If no filtering → keep existing behavior
    if not pages:
        all_pages = await asyncio.to_thread(
            extract_text_pages_via_textract_s3,
            file_path=file_path,
            original_filename=file_path,
            settings=settings,
        )
        return all_pages, len(all_pages)

    # -----------------------------
    # TRUE SELECTIVE OCR (using helper)
    # -----------------------------

    temp_dir = "temp_files"  # or your existing dir

    # ✅ create filtered PDF
    tmp_path = create_filtered_pdf(file_path, pages, temp_dir)

    try:
        result = await asyncio.to_thread(
            extract_text_pages_via_textract_s3,
            file_path=tmp_path,
            original_filename=tmp_path,
            settings=settings,
        )

        # ✅ map back to original page numbers
        pages = sorted(pages)

        mapped = {}
        for idx, original_page in enumerate(pages):
            mapped[original_page] = result.get(idx + 1, "")

        return mapped, len(mapped)

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    
def create_filtered_pdf(file_path, pages, temp_dir):
    reader = PdfReader(file_path)
    writer = PdfWriter()

    # ✅ FIX 1: ensure order
    pages = sorted(pages)

    for p in pages:
        writer.add_page(reader.pages[p - 1])

    # ✅ FIX 2: ensure dir exists
    os.makedirs(temp_dir, exist_ok=True)

    tmp_path = os.path.join(temp_dir, f"filtered_{uuid.uuid4().hex}.pdf")

    with open(tmp_path, "wb") as f:
        writer.write(f)

    return tmp_path
