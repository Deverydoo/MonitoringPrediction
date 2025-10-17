#!/bin/bash
# Install security dependencies for TFT Inference Daemon
# Run this before deploying to production

set -e  # Exit on error

echo "========================================"
echo "Installing Security Dependencies"
echo "========================================"
echo ""

echo "[1/3] Installing slowapi (rate limiting)..."
pip install slowapi
echo "OK: slowapi installed"
echo ""

echo "[2/3] Installing pyarrow (secure Parquet persistence)..."
pip install pyarrow
echo "OK: pyarrow installed"
echo ""

echo "[3/3] Verifying installation..."
python3 -c "import slowapi, pyarrow; print('Security dependencies verified!')"
echo ""

echo "========================================"
echo "SUCCESS: All security dependencies installed"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Set TFT_API_KEY environment variable"
echo "   Example: export TFT_API_KEY=your-secret-key-here"
echo ""
echo "2. Set CORS_ORIGINS environment variable"
echo "   Example: export CORS_ORIGINS=http://localhost:8501"
echo ""
echo "3. Start the daemon:"
echo "   python3 tft_inference_daemon.py"
echo ""
