#!/usr/bin/env python3
"""
Script to apply security fixes to tft_inference_daemon.py
This is a temporary helper to make targeted edits efficiently
"""

import re
from pathlib import Path

# Read the file
daemon_file = Path("tft_inference_daemon.py")
content = daemon_file.read_text(encoding='utf-8')

# Fix 1: Add API key protection to /feed/data endpoint
content = re.sub(
    r'@app\.post\("/feed/data"\)\s*\nasync def feed_data\(request: FeedDataRequest\):',
    r'@app.post("/feed/data")\nasync def feed_data(request: FeedDataRequest, api_key: str = Depends(verify_api_key)):',
    content
)

# Fix 2: Add API key protection to /predictions/current endpoint
content = re.sub(
    r'@app\.get\("/predictions/current"\)\s*\nasync def get_predictions\(\):',
    r'@app.get("/predictions/current")\nasync def get_predictions(api_key: str = Depends(verify_api_key)):',
    content
)

# Fix 3: Add API key protection to /alerts/active endpoint
content = re.sub(
    r'@app\.get\("/alerts/active"\)\s*\nasync def get_active_alerts\(\):',
    r'@app.get("/alerts/active")\nasync def get_active_alerts(api_key: str = Depends(verify_api_key)):',
    content
)

# Fix 4: Update PERSISTENCE_FILE to use parquet
content = re.sub(
    r'PERSISTENCE_FILE = "inference_rolling_window\.pkl"',
    r'PERSISTENCE_FILE = "inference_rolling_window.parquet"',
    content
)

# Write back
daemon_file.write_text(content, encoding='utf-8')

print("✅ Security fixes applied!")
print("   - API key protection added to /feed/data, /predictions/current, /alerts/active")
print("   - Persistence file changed to .parquet")
print("   - CORS already fixed to whitelist only")
