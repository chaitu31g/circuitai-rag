# setup_env.py
import subprocess
import os
import sys

def run_pip(args):
    """Run a pip command using the current Python interpreter."""
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"\n>>> Executing: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with return code {e.returncode}")
        sys.exit(1)

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("=== PROJECT ENVIRONMENT SETUP (5-TIER INSTALLATION) ===")
    
    # Tier 1
    print("\n--- Tier 1: Non-Negotiable AI Engine ---")
    run_pip(["install", "-U", "transformers>=5.3.0", "tokenizers==0.22.2", "huggingface-hub>=1.7.0"])

    # Tier 2
    print("\n--- Tier 2: Core Requirements (No Dependencies) ---")
    run_pip(["install", "-r", "backend/requirements.txt", "--no-deps"])

    # Tier 3
    print("\n--- Tier 3: ChromaDB Only (No Dependencies, to bypass tokenizers conflict) ---")
    run_pip(["install", "chromadb==0.5.21", "--no-deps"])

    # Tier 4 removed: newer docling>=2.14 no longer needs docling-ibm-models surgical downgrade

    # Tier 5
    print("\n--- Tier 5: Missing Gears (Manual Sub-dependencies) ---")
    run_pip([
        "install", "pydantic", "fastapi", "uvicorn>=0.34.0", "python-multipart", "python-dotenv",
        "requests==2.32.4", "bcrypt", "build", "chroma-hnswlib", "kubernetes", "posthog", "pypika",
        "dataclasses-json", "deprecated", "dirtyjson", "filetype", "tinytag", "typing-inspect",
        "llama-index-workflows", "jsonref", "latex2mathml", "deepsearch-glm", "easyocr",
        "marko", "python-docx", "python-pptx", "jsonlines", "banks", "lxml", "typer>=0.24.0",
        "pylatexenc", "polyfactory>=2.22.2",
        "onnxruntime", "opentelemetry-api==1.38.0", "opentelemetry-sdk==1.38.0",
        "opentelemetry-exporter-otlp-proto-grpc==1.38.0", "opentelemetry-exporter-otlp-proto-http==1.38.0",
        "opentelemetry-exporter-otlp-proto-common==1.38.0", "opentelemetry-proto==1.38.0",
        "opentelemetry-instrumentation-fastapi==0.59b0", "opentelemetry-instrumentation-asgi==0.59b0",
        "opentelemetry-instrumentation==0.59b0", "opentelemetry-semantic-conventions==0.59b0", "opentelemetry-util-http==0.59b0"
    ])

    print("\n--- Tier 6: Final Pin (Restore AI Engine Hub Version) ---")
    run_pip([
        "install",
        "huggingface-hub>=1.7.0",   # Tier 4 docling-ibm-models downgrades this — restore it
        "setuptools>=80.9.0",        # llama-index-core requires this
    ])

    print("\n=== All Tiers Installed! Environment is fixed. ===")

if __name__ == "__main__":
    main()
