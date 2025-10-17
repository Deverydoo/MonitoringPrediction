#!/bin/bash
# Setup API key for TFT Monitoring System
# This script copies the example .env file and sets up the API key

echo ""
echo "================================================"
echo "TFT Monitoring System - API Key Setup"
echo "================================================"
echo ""

if [ -f .env ]; then
    echo "[WARNING] .env file already exists!"
    echo ""
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping .env file creation (already exists)"
        exit 0
    fi
fi

echo "Creating .env file from .env.example..."
cp .env.example .env

echo ""
echo "================================================"
echo "API Key Setup Complete!"
echo "================================================"
echo ""
echo "The API key has been configured:"
echo "  - Dashboard: .streamlit/secrets.toml"
echo "  - Daemon:    .env (TFT_API_KEY)"
echo ""
echo "Next steps:"
echo "  1. Load environment: source .env"
echo "  2. Start the daemon: python tft_inference_daemon.py"
echo "  3. Start the dashboard: streamlit run tft_dashboard_web.py"
echo ""
echo "The dashboard and daemon will now authenticate with the API key."
echo ""
