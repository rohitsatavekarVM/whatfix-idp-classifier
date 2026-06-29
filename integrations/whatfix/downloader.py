"""
Downloads and stores Whatfix assets locally.
"""

import os

from app.integrations.whatfix.client import WhatfixClient


class WhatfixDownloader:

    def __init__(self):
        self.client = WhatfixClient()

    def download(
        self,
        download_url: str,
        output_directory: str,
        file_name: str,
    ) -> str:
        os.makedirs(output_directory, exist_ok=True)

        output_path = os.path.join(
            output_directory,
            file_name
        )

        self.client.download_file(
            download_url=download_url,
            output_path=output_path,
        )

        return output_path