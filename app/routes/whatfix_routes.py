"""
Quart routes for Whatfix document classification.
"""

import os
import tempfile

from quart import Blueprint, jsonify, request

from app.processors.whatfix_processor import WhatfixProcessor

whatfix_bp = Blueprint("whatfix", __name__)


@whatfix_bp.route("/whatfix/upload", methods=["POST"])
async def upload_document():

    print("===== Upload endpoint called =====")

    if "file" not in (await request.files):
        return jsonify({"error": "No file uploaded"}), 400

    files = await request.files
    uploaded_file = files["file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "Invalid filename"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_path = temp_file.name

    await uploaded_file.save(temp_path)

    try:
        processor = WhatfixProcessor()

        # Await the async processor
        result = await processor.process(temp_path)

        return jsonify(result), 200

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)