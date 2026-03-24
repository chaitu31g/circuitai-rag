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

echo "--- Tier 4: Surgical Downgrade for Parser ---"
pip install "docling-parse>=3.0.0,<4.0.0" "pypdfium2>=4.30.0,<5.0.0" "docling-ibm-models>=2.0.6,<3.0.0"

echo "--- Tier 5: Missing Gears (Manual Sub-dependencies) ---"
pip install pydantic fastapi "uvicorn>=0.34.0" python-multipart python-dotenv \
    bcrypt build chroma-hnswlib kubernetes posthog pypika \
    dataclasses-json deprecated dirtyjson filetype tinytag typing-inspect \
    llama-index-workflows jsonref latex2mathml deepsearch-glm easyocr \
    marko python-docx python-pptx jsonlines banks lxml "typer>=0.24.0" \
    onnxruntime "opentelemetry-api==1.38.0" "opentelemetry-sdk==1.38.0" \
    "opentelemetry-exporter-otlp-proto-grpc==1.38.0" "opentelemetry-instrumentation-fastapi==0.46b0"

echo "Done! 5-Tier installation complete."
