# setup_env.py
import subprocess
import os
import sys

def run_pip(args):
    """Run a pip command using the current Python interpreter."""
    cmd = [sys.executable, "-m", "pip"] + args
    print(f"\n>>> Executing: {' '.join(cmd)}")
    # Use check=True to stop on errors (same as set -e in bash)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with return code {e.returncode}")
        sys.exit(1)

def main():
    # Ensure current directory is the project root (where backend/ is)
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    print("=== PROJECT ENVIRONMENT SETUP (TIERED INSTALLATION) ===")
    print("Strategy: Forced installation to bypass dependency backtracking.")

    # Tier 1
    print("\n--- Tier 1: Non-Negotiable AI Engine ---")
    run_pip(["install", "-U", "transformers>=5.3.0", "tokenizers==0.22.2", "huggingface-hub>=1.7.0"])

    # Tier 2
    print("\n--- Tier 2: Core Requirements (No Dependencies) ---")
    run_pip(["install", "-r", "backend/requirements.txt", "--no-deps"])

    # Tier 3
    print("\n--- Tier 3: Legacy Complainers ---")
    run_pip(["install", "chromadb==0.5.21", "docling==2.11.0", "--no-deps"])

    # Tier 4
    print("\n--- Tier 4: Missing Gears (Manual Sub-dependencies) ---")
    run_pip(["install", "pydantic", "fastapi", "uvicorn", "python-multipart", "python-dotenv"])
    run_pip(["install", "bcrypt", "build", "chroma-hnswlib==0.7.6", "kubernetes", "posthog", "pypika"])
    run_pip(["install", "dataclasses-json", "deprecated", "dirtyjson", "filetype", "tinytag", "typing-inspect"])
    run_pip(["install", "llama-index-workflows", "jsonref", "latex2mathml", "deepsearch-glm", "docling-parse", "easyocr", "marko", "pypdfium2", "python-docx", "python-pptx", "jsonlines", "banks", "lxml", "typer"])

    print("\n=== All Tiers Installed! Tiered installation complete. ===")

if __name__ == "__main__":
    main()
