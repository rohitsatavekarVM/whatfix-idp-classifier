"""
LLM-based document classifier.

Replaces the old YAML/DocumentTypeConfig based classifier.
"""

import json
from typing import Dict, Any

from andromeda import HumanMessage
from andromeda.config import ModelConfig
from andromeda.utils import get_chat_model

from app.helpers.config import PRIMARY_LLM


PROMPT = """
You are an expert document classification system.

Classify the following document.

Return ONLY valid JSON.

Example:

{{
  "document_type": "INVOICE",
  "confidence": 0.94,
  "reason": "Contains invoice number, billing address and total amount."
}}

Rules:

- Confidence must be between 0 and 1.
- Use ONLY the supplied document.
- Never invent information.
- Return ONLY valid JSON.
- Do not wrap the response in markdown.
- Do not explain your reasoning.

Document:

{document}
"""


def predict_document_type(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict the document type using the extracted text.
    """

    extracted_text = state.get("extracted_text", [])

    document = "\n\n".join(
        page.strip()
        for page in extracted_text
        if page and page.strip()
    )

    if not document:
        state["document_type"] = "UNKNOWN"
        state["confidence"] = 0.0
        state["predictions"] = []
        return state

    chat_model = get_chat_model(
        ModelConfig(
            name=PRIMARY_LLM,
            provider="litellm",
            temperature=0,
        )
    )

    response = chat_model.invoke(
        [
            HumanMessage(
                content=PROMPT.format(
                    document=document[:120000]
                )
            )
        ]
    )

    # -----------------------------
    # Extract response text safely
    # -----------------------------
    if hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    # LangChain / Andromeda sometimes returns content blocks
    if isinstance(content, list):

        text = ""

        for block in content:

            if isinstance(block, dict):

                if block.get("type") == "text":
                    text += block.get("text", "")

            elif hasattr(block, "text"):
                text += block.text

            else:
                text += str(block)

        content = text

    content = str(content).strip()

    # Remove markdown code fences if present
    if content.startswith("```"):

        content = (
            content.replace("```json", "")
            .replace("```", "")
            .strip()
        )

    print("\n========== RAW LLM RESPONSE ==========")
    print(content)
    print("======================================\n")

    # -----------------------------
    # Parse JSON
    # -----------------------------
    try:
        result = json.loads(content)

    except Exception as e:

        print("JSON Parse Error:", e)

        result = {
            "document_type": "UNKNOWN",
            "confidence": 0.0,
            "reason": content,
        }

    state["document_type"] = result.get(
        "document_type",
        "UNKNOWN",
    )

    try:
        state["confidence"] = float(
            result.get("confidence", 0.0)
        )
    except Exception:
        state["confidence"] = 0.0

    state["predictions"] = [
        {
            "document_type": state["document_type"],
            "confidence": state["confidence"],
            "reason": result.get("reason", ""),
        }
    ]

    return state