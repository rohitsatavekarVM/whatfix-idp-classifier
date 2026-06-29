"""
Artifact management utilities.
"""

import os
import uuid
import shutil
from datetime import datetime

from app.config.whatfix_config import WhatfixConfig


class ArtifactManager:

    def __init__(self):
        self.config = WhatfixConfig()
        self.config.create_directories()

    def create_artifact(self, source_file_path: str) -> dict:
        artifact_id = str(uuid.uuid4())

        file_name = os.path.basename(source_file_path)

        destination_path = os.path.join(
            self.config.RAW_FILES_DIR,
            f"{artifact_id}_{file_name}"
        )

        shutil.copy2(source_file_path, destination_path)

        return {
            "artifact_id": artifact_id,
            "file_name": file_name,
            "file_path": destination_path,
            "created_at": datetime.utcnow().isoformat()
        }

    def save_classification(self, artifact_id: str, data: dict) -> str:
        output_path = os.path.join(
            self.config.CLASSIFIED_DIR,
            f"{artifact_id}.json"
        )

        self._write_json(output_path, data)

        return output_path

    def save_canonical(self, artifact_id: str, data: dict) -> str:
        output_path = os.path.join(
            self.config.CANONICAL_DIR,
            f"{artifact_id}.json"
        )

        self._write_json(output_path, data)

        return output_path

    def get_artifact_path(self, artifact_id: str) -> str | None:
        for file_name in os.listdir(self.config.RAW_FILES_DIR):
            if file_name.startswith(artifact_id):
                return os.path.join(self.config.RAW_FILES_DIR, file_name)

        return None

    @staticmethod
    def _write_json(path: str, data: dict):
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)