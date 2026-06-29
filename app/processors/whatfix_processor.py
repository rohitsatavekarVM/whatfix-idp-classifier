"""
Orchestrates the Whatfix -> IDP -> Canonical flow.
"""

from app.helpers.artifact_manager import ArtifactManager
from app.helpers.canonical_builder import CanonicalBuilder
from app.processors.idp_classifier import IDPClassifier


class WhatfixProcessor:

    def __init__(self):
        self.artifacts = ArtifactManager()
        self.classifier = IDPClassifier()

    async def process(self, uploaded_file_path: str) -> dict:
        """
        Process a Whatfix document through
        the local Andromeda IDP workflow.
        """

        print("===== Inside WhatfixProcessor =====")
        print(uploaded_file_path)

        artifact = self.artifacts.create_artifact(
            uploaded_file_path
        )

        classification = await self.classifier.classify(
            file_path=artifact["file_path"],
            artifact_id=artifact["artifact_id"],
        )

        self.artifacts.save_classification(
            artifact["artifact_id"],
            classification.to_dict(),
        )

        canonical = CanonicalBuilder.build(
            classification=classification,
            file_path=artifact["file_path"],
        )

        self.artifacts.save_canonical(
            artifact["artifact_id"],
            canonical.to_dict(),
        )

        return canonical.to_dict()