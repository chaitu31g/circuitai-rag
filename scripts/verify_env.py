# verify_env.py
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CircuitAI-Env-Check")

def check_package(name, expected_version=None):
    try:
        import importlib.metadata
        version = importlib.metadata.version(name)
        logger.info(f"✓ {name.ljust(25)}: {version}")
        return True
    except Exception:
        logger.error(f"✗ {name.ljust(25)}: MISSING")
        return False

def main():
    print("\n" + "="*50)
    print(" CIRCUITAI ENVIRONMENT VALIDATION ")
    print("="*50 + "\n")

    # Tier 1: Core AI & PDF
    print("--- Core Engine ---")
    check_package("transformers")
    check_package("torch")
    check_package("docling")
    check_package("pdfplumber")
    check_package("fitz")  # PyMuPDF

    # Tier 2: Vector DB
    print("\n--- Vector Database ---")
    check_package("chromadb")

    # Tier 3: Critical Pins (The Conflict Zone)
    print("\n--- Critical Pins (Conflict Zone) ---")
    check_package("opentelemetry.api")
    check_package("typer")
    check_package("pydantic")

    print("\n--- Critical Imports ---")
    try:
        import chromadb
        import docling
        from transformers import AutoModel
        import fastapi
        print("✓ All critical imports succeeded!")
    except Exception as e:
        print(f"✗ Import failure: {e}")

    print("\n" + "="*50)

if __name__ == "__main__":
    main()
