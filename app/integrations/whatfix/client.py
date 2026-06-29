"""
Whatfix API client.
"""

import requests

from app.config.whatfix_config import WhatfixConfig


class WhatfixClient:

    def __init__(self):
        self.config = WhatfixConfig()

    def get_content(self):
        url = (
            f"{self.config.BASE_URL}/v1/accounts/"
            f"{self.config.ACCOUNT_ID}/content"
        )

        response = requests.get(
            url,
            headers=self.config.headers,
            timeout=self.config.REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        return response.json()

    def get_content_by_id(self, content_id: str):
        url = (
            f"{self.config.BASE_URL}/v1/accounts/"
            f"{self.config.ACCOUNT_ID}/content/{content_id}"
        )

        response = requests.get(
            url,
            headers=self.config.headers,
            timeout=self.config.REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        return response.json()

    def download_file(self, download_url: str, output_path: str):
        response = requests.get(
            download_url,
            headers=self.config.headers,
            stream=True,
            timeout=self.config.REQUEST_TIMEOUT,
        )

        response.raise_for_status()

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        return output_path