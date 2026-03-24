#!/bin/bash
# setup_env.sh
# Sequential installation script to bypass catastrophic dependency backtracking.

# Stop on errors
set -e

echo "--- Tier 1: Non-Negotiable AI Engine ---"
pip install -U "transformers>=5.3.0" "tokenizers==0.22.2" "huggingface-hub>=1.7.0"

echo "--- Tier 2: Core Requirements (No Dependencies) ---"
pip install -r backend/requirements.txt --no-deps

echo "--- Tier 3: Legacy Complainers ---"
pip install chromadb==0.5.21 docling==2.11.0 --no-deps

echo "--- Tier 4: Missing Gears (Manual Sub-dependencies) ---"
pip install pydantic fastapi uvicorn python-multipart python-dotenv
pip install bcrypt build chroma-hnswlib==0.7.6 kubernetes posthog pypika
pip install dataclasses-json deprecated dirtyjson filetype tinytag typing-inspect
pip install llama-index-workflows jsonref latex2mathml deepsearch-glm docling-parse easyocr marko pypdfium2 python-docx python-pptx jsonlines banks lxml typer

echo "Done! Tiered installation complete."
