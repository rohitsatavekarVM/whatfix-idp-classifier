"""
Classification models used across the Whatfix → IDP pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ClassificationResult:
    """
    Represents the document classification returned by IDP.
    """

    file_name: str
    document_type: str
    confidence: float = 0.0

    source: str = "WHATFIX"

    artifact_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "source": self.source,
            "file_name": self.file_name,
            "document_type": self.document_type,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            artifact_id=data.get("artifact_id"),
            source=data.get("source", "WHATFIX"),
            file_name=data.get("file_name", ""),
            document_type=data.get("document_type", ""),
            confidence=float(data.get("confidence", 0.0)),
            metadata=data.get("metadata", {}),
        )