#!/bin/bash
# setup_env.sh
# Sequential installation script to bypass catastrophic dependency backtracking.

# Stop on errors
set -e

echo "--- Tier 1: Non-Negotiable AI Engine ---"
pip install -U "transformers>=5.3.0" "tokenizers==0.22.2" "huggingface-hub>=1.7.0"

echo "--- Tier 2: Core Requirements (No Dependencies) ---"
pip install -r backend/requirements.txt --no-deps

echo "--- Tier 3: ChromaDB Only (No Dependencies, to bypass tokenizers conflict) ---"
pip install chromadb==0.5.21 --no-deps

# Tier 4 removed: newer docling>=2.14 no longer needs docling-ibm-models surgical downgrade

echo "--- Tier 5: Missing Gears (Manual Sub-dependencies) ---"
pip install pydantic fastapi "uvicorn>=0.34.0" python-multipart python-dotenv \
    "requests==2.32.4" bcrypt build chroma-hnswlib kubernetes posthog pypika \
    dataclasses-json deprecated dirtyjson filetype tinytag typing-inspect \
    llama-index-workflows jsonref latex2mathml deepsearch-glm easyocr \
    marko python-docx python-pptx jsonlines banks lxml "typer>=0.24.0" pylatexenc "polyfactory>=2.22.2" \
    onnxruntime "opentelemetry-api==1.38.0" "opentelemetry-sdk==1.38.0" \
    "opentelemetry-exporter-otlp-proto-grpc==1.38.0" "opentelemetry-exporter-otlp-proto-http==1.38.0" \
    "opentelemetry-exporter-otlp-proto-common==1.38.0" "opentelemetry-proto==1.38.0" \
    "opentelemetry-instrumentation-fastapi==0.59b0" "opentelemetry-instrumentation-asgi==0.59b0" \
    "opentelemetry-instrumentation==0.59b0" "opentelemetry-semantic-conventions==0.59b0" "opentelemetry-util-http==0.59b0"

echo "--- Tier 6: Final Pin (Restore AI Engine Hub Version) ---"
pip install "huggingface-hub>=1.7.0" "setuptools>=80.9.0"

echo "Done! 5-Tier installation complete."
