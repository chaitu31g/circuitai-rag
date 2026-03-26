#!/bin/bash
# setup_env.sh – CircuitAI Colab Installation Script
# =====================================================
# Designed to resolve all known dependency conflicts between:
#   - chromadb 0.5.21 (needs tokenizers <=0.20.3)
#   - docling 2.82 (needs huggingface-hub <1.0, typer <0.22)
#   - docling-ibm-models (needs transformers <5.0)
#   - Microsoft Table Transformer (needs transformers >=4.42)
#   - Colab base packages (numpy, requests, fsspec, etc.)
#
# Strategy:
#   Step 1 – Install pinned conflict-sensitive packages FIRST
#   Step 2 – Install the rest with --no-deps to prevent pip from auto-upgrading
#   Step 3 – Install docling-ibm-models with --no-deps (protects transformers pin)

set -e

echo ""
echo "━━━ Step 1: Pin conflict-sensitive packages ━━━"
pip install -q \
    "transformers>=4.42.0,<5.0.0" \
    "tokenizers>=0.13.2,<=0.20.3" \
    "huggingface-hub>=0.23,<1.0.0" \
    "typer>=0.12.5,<0.22.0" \
    "numpy>=1.24.0,<2.0.0" \
    "requests==2.32.4" \
    "PyMuPDF==1.25.1"

echo ""
echo "━━━ Step 2: Install main requirements (no-deps to protect pins) ━━━"
pip install -q -r backend/requirements.txt --no-deps

echo ""
echo "━━━ Step 3: Install docling-ibm-models with --no-deps ━━━"
pip install -q "docling-ibm-models>=3.12.0,<4.0.0" "rapidocr>=3.3,<4.0.0" --no-deps

echo ""
echo "━━━ Step 4: Install remaining sub-deps that need their own deps ━━━"
pip install -q \
    sentence-transformers \
    accelerate \
    bitsandbytes \
    safetensors \
    sentencepiece \
    timm \
    pdfplumber \
    easyocr \
    onnxruntime \
    chromadb==0.5.21 --no-deps

echo ""
echo "━━━ Step 5: Verify no regressions ━━━"
python - <<'EOF'
import importlib, sys

checks = {
    "transformers":     ("4.42", "5.0"),
    "tokenizers":       ("0.13", "0.21"),
    "huggingface_hub":  ("0.23", "1.0"),
    "fitz":             None,      # PyMuPDF
    "pdfplumber":       None,
    "chromadb":         None,
    "docling":          None,
}

ok = True
for pkg, bounds in checks.items():
    try:
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "?")
        if bounds:
            from packaging.version import Version
            lo, hi = bounds
            if not (Version(lo) <= Version(ver) < Version(hi)):
                print(f"  ✗ {pkg}=={ver}  [expected >={lo},<{hi}]")
                ok = False
                continue
        print(f"  ✓ {pkg}=={ver}")
    except ImportError:
        print(f"  ✗ {pkg} not installed")
        ok = False

sys.exit(0 if ok else 1)
EOF

echo ""
echo "✅ Installation complete! All constraints satisfied."
