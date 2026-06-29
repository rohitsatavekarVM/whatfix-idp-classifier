"""
High-level Whatfix extractor.

Responsible for:
- Fetching content inventory
- Fetching PDF inventory
- Downloading PDFs

Does NOT perform:
- IDP classification
- Canonical conversion
- eGain integration
"""

import os

from app.integrations.whatfix.client import WhatfixClient
from app.integrations.whatfix.downloader import WhatfixDownloader


class WhatfixExtractor:

    def __init__(self):
        self.client = WhatfixClient()
        self.downloader = WhatfixDownloader()

    def extract_content_inventory(self):
        """
        Fetch all Whatfix content.
        """
        return self.client.get_content()

    def extract_content_details(self, content_id: str):
        """
        Fetch detailed information for a content item.
        """
        return self.client.get_content_by_id(content_id)

    def extract_pdf_inventory(self):
        """
        Fetch the PDF inventory from Whatfix.
        """

        url = (
            f"{self.client.config.BASE_URL}"
            f"/v1/accounts/{self.client.config.ACCOUNT_ID}"
            "/content/upload/cloud/pdf/listS3DirContent"
        )

        response = self.client.session.get(
            url,
            headers=self.client.config.headers,
            timeout=self.client.config.REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        return response.json()

    def download_pdfs(self, output_directory: str):
        """
        Download all PDFs from the Whatfix PDF inventory.

        Returns:
            List of downloaded local file paths.
        """

        inventory = self.extract_pdf_inventory()

        downloaded_files = []

        for pdf in inventory:

            download_url = (
                pdf.get("url")
                or pdf.get("downloadUrl")
                or pdf.get("blobUrl")
                or pdf.get("signedUrl")
            )

            if not download_url:
                continue

            file_name = (
                pdf.get("fileName")
                or pdf.get("filename")
                or pdf.get("name")
                or "document.pdf"
            )

            local_path = self.downloader.download(
                download_url=download_url,
                output_directory=output_directory,
                file_name=file_name,
            )

            downloaded_files.append(local_path)

        return downloaded_files