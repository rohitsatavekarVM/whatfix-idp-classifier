"""
Builds the canonical document from the IDP classification result.
"""

from app.models.classification import ClassificationResult
from app.models.canonical import CanonicalDocument


class CanonicalBuilder:

    @staticmethod
    def build(
        classification: ClassificationResult,
        file_path: str
    ) -> CanonicalDocument:

        return CanonicalDocument(
            artifact_id=classification.artifact_id,
            source=classification.source,
            file_name=classification.file_name,
            document_type=classification.document_type,
            confidence=classification.confidence,
            file_path=file_path,
            metadata=classification.metadata,
        )