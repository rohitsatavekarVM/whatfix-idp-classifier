"""
Configuration for the Whatfix -> IDP Classification flow.

This module centralizes:
- Whatfix API configuration
- Artifact storage locations
- IDP endpoint configuration
"""

from dataclasses import dataclass
import os


@dataclass
class WhatfixConfig:
    """Configuration for the Whatfix integration."""

    # -----------------------------
    # Whatfix API
    # -----------------------------
    BASE_URL: str = os.getenv("WHATFIX_BASE_URL", "")
    API_KEY: str = os.getenv("WHATFIX_API_KEY", "")
    USER_EMAIL: str = os.getenv("WHATFIX_USER_EMAIL", "")
    ACCOUNT_ID: str = os.getenv("WHATFIX_ACCOUNT_ID", "")

    # -----------------------------
    # Local Storage
    # -----------------------------
    ARTIFACT_ROOT: str = os.getenv(
        "WHATFIX_ARTIFACT_ROOT",
        "artifacts/whatfix"
    )

    RAW_FILES_DIR: str = os.path.join(
        ARTIFACT_ROOT,
        "raw"
    )

    CLASSIFIED_DIR: str = os.path.join(
        ARTIFACT_ROOT,
        "classified"
    )

    CANONICAL_DIR: str = os.path.join(
        ARTIFACT_ROOT,
        "canonical"
    )

    # -----------------------------
    # IDP
    # -----------------------------
    IDP_ENDPOINT: str = os.getenv(
        "IDP_ENDPOINT",
        "http://localhost:11003/whatfix/upload"
    )

    REQUEST_TIMEOUT: int = 300

    # -----------------------------
    # Runtime
    # -----------------------------
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def create_directories(self):
        """
        Ensure required directories exist.
        """
        os.makedirs(self.RAW_FILES_DIR, exist_ok=True)
        os.makedirs(self.CLASSIFIED_DIR, exist_ok=True)
        os.makedirs(self.CANONICAL_DIR, exist_ok=True)

    @property
    def headers(self):
        """
        Headers for Whatfix API.
        """
        return {
            "x-whatfix-integration-key": self.API_KEY,
            "x-whatfix-user": self.USER_EMAIL,
            "Accept": "application/json"
        }