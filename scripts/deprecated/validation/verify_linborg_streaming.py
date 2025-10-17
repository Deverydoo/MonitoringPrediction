#!/usr/bin/env python3
"""
Quick verification script to check LINBORG metrics are flowing through the system.
Run this AFTER restarting both daemons.
"""

import requests
import json

print("=" * 80)
print("LINBORG METRICS VERIFICATION")
print("=" * 80)

LINBORG_METRICS = [
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
    'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
    'net_in_mb_s', 'net_out_mb_s',
    'back_close_wait', 'front_close_wait',
    'load_average', 'uptime_days'
]

# Step 1: Check metrics daemon
print("\n1️⃣  Checking Metrics Generator Daemon (port 8001)...")
try:
    response = requests.get("http://localhost:8001/", timeout=2)
    if response.ok:
        data = response.json()
        print(f"   ✅ Daemon running: {data.get('status', 'unknown')}")
        print(f"   ✅ Scenario: {data.get('scenario', 'unknown')}")
        print(f"   ✅ Fleet size: {data.get('fleet_size', 0)} servers")
    else:
        print(f"   ❌ Daemon returned error: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"   ❌ Cannot connect to metrics daemon: {e}")
    print(f"   Start it with: python metrics_generator_daemon.py --stream --servers 20")
    exit(1)

# Step 2: Check inference daemon
print("\n2️⃣  Checking Inference Daemon (port 8000)...")
try:
    response = requests.get("http://localhost:8000/status", timeout=2)
    if response.ok:
        data = response.json()
        print(f"   ✅ Daemon running: {data.get('running', False)}")
        warmup = data.get('warmup', {})
        print(f"   ✅ Warmed up: {warmup.get('is_warmed_up', False)}")
        print(f"   ✅ Window size: {warmup.get('current_size', 0)} records")
    else:
        print(f"   ❌ Daemon returned error: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"   ❌ Cannot connect to inference daemon: {e}")
    print(f"   Start it with: python tft_inference_daemon.py --port 8000")
    exit(1)

# Step 3: Check predictions output
print("\n3️⃣  Checking Predictions Output...")
try:
    response = requests.get("http://localhost:8000/predictions/current", timeout=5)
    if response.ok:
        data = response.json()
        predictions = data.get('predictions', {})

        if not predictions:
            print(f"   ⚠️  No predictions yet - daemon may still be warming up")
            print(f"   Wait 30 seconds and try again")
            exit(0)

        # Check first server
        first_server = list(predictions.keys())[0]
        server_data = predictions[first_server]

        print(f"   ✅ Predictions available for {len(predictions)} servers")
        print(f"   📊 Checking server: {first_server}")
        print(f"\n   LINBORG Metrics Check:")

        missing = []
        present = []
        zero_values = []

        for metric in LINBORG_METRICS:
            if metric in server_data:
                current_val = server_data[metric].get('current', 0)
                if current_val == 0:
                    zero_values.append(metric)
                    print(f"      ⚠️  {metric:20s} - Present but ZERO (current: {current_val})")
                else:
                    present.append(metric)
                    print(f"      ✅ {metric:20s} - OK (current: {current_val:.2f})")
            else:
                missing.append(metric)
                print(f"      ❌ {metric:20s} - MISSING")

        print(f"\n   📈 Summary:")
        print(f"      Present & Non-zero: {len(present)}/14")
        print(f"      Present but Zero: {len(zero_values)}/14")
        print(f"      Missing: {len(missing)}/14")

        if missing:
            print(f"\n   ❌ PROBLEM: Missing metrics: {missing}")
            print(f"      → Check inference daemon logs for errors")
            exit(1)
        elif len(present) == 14:
            print(f"\n   ✅ SUCCESS! All 14 LINBORG metrics flowing correctly!")
            exit(0)
        else:
            print(f"\n   ⚠️  WARNING: All metrics present but {len(zero_values)} have zero values")
            print(f"      Zero metrics: {zero_values}")
            print(f"      → This might be normal if system just started")
            print(f"      → Wait 10 seconds and check dashboard")
            exit(0)
    else:
        print(f"   ❌ Predictions endpoint error: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"   ❌ Error fetching predictions: {e}")
    exit(1)
