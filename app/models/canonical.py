"""
Canonical model for the Whatfix → eGain migration pipeline.

Every classified document is converted into this format before it
is passed to downstream systems.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class CanonicalDocument:
    """
    Canonical representation of a classified Whatfix document.
    """

    artifact_id: Optional[str]

    source: str

    file_name: str

    document_type: str

    confidence: float

    file_path: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "source": self.source,
            "file_name": self.file_name,
            "document_type": self.document_type,
            "confidence": self.confidence,
            "file_path": self.file_path,
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
            file_path=data.get("file_path"),
            metadata=data.get("metadata", {}),
        )