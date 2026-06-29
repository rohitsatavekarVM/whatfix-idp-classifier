import os
import uuid

from app.models.classification import ClassificationResult
from app.processors.andromeda_processor import andromeda_process_files


class IDPClassifier:

    async def classify(self, file_path: str, artifact_id: str) -> ClassificationResult:
        """
        Runs the local Andromeda workflow instead of calling the external IDP.
        """

        workflow_result = None

        async for (
            progress,
            message,
            is_done,
            result,
            pages,
            file_id,
        ) in andromeda_process_files(
            file_name=os.path.basename(file_path),
            file_path=file_path,
            file_id=str(uuid.uuid4()),
            summarize=False,
            session_id=None,
            context={},
        ):

            if is_done:
                if result is None:
                    raise Exception("Andromeda returned no result.")

                if "error" in result:
                    raise Exception(result["error"])

                workflow_result = result

        if workflow_result is None:
            raise Exception("Document classification failed.")

        return ClassificationResult(
            artifact_id=artifact_id,
            source="WHATFIX",
            file_name=os.path.basename(file_path),
            document_type=workflow_result.get(
                "document_type",
                "UNKNOWN",
            ),
            confidence=0.0,
            metadata=workflow_result,
        )