"""Lightweight image/document page classifier without model-heavy dependencies."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytesseract


DOCUMENT_LABEL = "a document page with printed text"
EVIDENCE_LABEL = "a real world photographic evidence of property damage or vehicle damage"


def classify_image(pil_image) -> Dict[str, object]:
    gray = pil_image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return {"label": DOCUMENT_LABEL, "confidence": 0.5, "probs": [0.5, 0.5]}

    text = pytesseract.image_to_string(gray) or ""
    text_density = len(text.strip()) / max(1, arr.size / 1000)
    edge_density = _edge_density(arr)

    document_score = min(1.0, 0.35 + text_density * 0.08 + edge_density * 1.5)
    evidence_score = max(0.0, 1.0 - document_score)

    if document_score >= evidence_score:
        label = DOCUMENT_LABEL
        confidence = document_score
    else:
        label = EVIDENCE_LABEL
        confidence = evidence_score

    return {
        "label": label,
        "confidence": float(round(confidence, 4)),
        "probs": [float(round(document_score, 4)), float(round(evidence_score, 4))],
    }


def _edge_density(arr: np.ndarray) -> float:
    if arr.shape[0] < 2 or arr.shape[1] < 2:
        return 0.0
    vertical = np.abs(np.diff(arr, axis=0)) > 35
    horizontal = np.abs(np.diff(arr, axis=1)) > 35
    return float((vertical.mean() + horizontal.mean()) / 2.0)
