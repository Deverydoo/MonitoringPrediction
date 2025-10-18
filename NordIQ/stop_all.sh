#!/bin/bash
# NordIQ AI Systems - Linux/Mac Stop Script

echo "Stopping NordIQ AI Systems..."
echo

pkill -f "tft_inference_daemon.py"
pkill -f "metrics_generator_daemon.py"
pkill -f "tft_dashboard_web.py"
pkill -f "streamlit"

echo
echo "All services stopped."
