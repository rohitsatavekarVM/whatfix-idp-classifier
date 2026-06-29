"""
Wrapper around the IDP classification endpoint.
"""

import os
import requests

from app.config.whatfix_config import WhatfixConfig
from app.models.classification import ClassificationResult


class IDPClassifier:

    def __init__(self):
        self.config = WhatfixConfig()

    def classify(self, file_path: str) -> ClassificationResult:
        """
        Sends a document to IDP for classification.
        """

        with open(file_path, "rb") as f:
            files = {
                "file": (
                    os.path.basename(file_path),
                    f,
                    "application/pdf"
                )
            }

            response = requests.post(
                self.config.IDP_ENDPOINT,
                files=files,
                timeout=self.config.REQUEST_TIMEOUT,
            )

        response.raise_for_status()

        result = response.json()

        return ClassificationResult(
            artifact_id=result.get("artifact_id"),
            source="WHATFIX",
            file_name=os.path.basename(file_path),
            document_type=result.get("document_type", "UNKNOWN"),
            confidence=float(result.get("confidence", 0.0)),
            metadata=result,
        )