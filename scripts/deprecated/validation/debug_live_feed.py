#!/usr/bin/env python3
"""
Debug script to intercept and show what data the metrics daemon is actually sending.
"""

import requests
import json
import time

print("=" * 80)
print("DEBUGGING LIVE DATA FEED")
print("=" * 80)

# Intercept a batch from the metrics daemon
print("\nWaiting for metrics daemon to send a batch...")
print("(This monitors the inference daemon's feed endpoint)")

# Just check the rolling window in the inference daemon
for attempt in range(5):
    try:
        response = requests.get("http://localhost:8000/status", timeout=2)
        if response.ok:
            status = response.json()
            window_size = status.get('window_size', 0)
            tick = status.get('tick_count', 0)

            print(f"\nAttempt {attempt + 1}/5:")
            print(f"  Window size: {window_size} records")
            print(f"  Tick count: {tick}")

            if window_size > 0:
                print("\n✅ Inference daemon HAS data in rolling window!")
                print("   Now let's check what columns are in that data...")

                # Get predictions to trigger a DataFrame creation
                pred_response = requests.get("http://localhost:8000/predictions/current", timeout=10)
                if pred_response.ok:
                    preds = pred_response.json()

                    if 'predictions' in preds and preds['predictions']:
                        first_server = list(preds['predictions'].keys())[0]
                        metrics_present = list(preds['predictions'][first_server].keys())

                        print(f"\n   Metrics in predictions for {first_server}:")
                        for m in metrics_present:
                            current = preds['predictions'][first_server][m].get('current', 'N/A')
                            print(f"      {m:20s}: current={current}")

                        print(f"\n   Total metrics returned: {len(metrics_present)}")

                        # Check for missing LINBORG metrics
                        LINBORG = [
                            'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
                            'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
                            'net_in_mb_s', 'net_out_mb_s',
                            'back_close_wait', 'front_close_wait',
                            'load_average', 'uptime_days'
                        ]

                        missing = [m for m in LINBORG if m not in metrics_present]
                        if missing:
                            print(f"\n   ❌ MISSING METRICS: {missing}")
                        else:
                            print(f"\n   ✅ All 14 LINBORG metrics present!")
                    else:
                        print("\n   ⚠️  Predictions dict is empty")
                else:
                    print(f"\n   ❌ Predictions request failed: {pred_response.status_code}")

                break
            else:
                print("   ⚠️  No data yet, waiting...")
                time.sleep(2)
        else:
            print(f"   ❌ Status request failed: {response.status_code}")
            time.sleep(2)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        time.sleep(2)
else:
    print("\n❌ No data received after 5 attempts")
    print("   Check if metrics daemon is running: python metrics_generator_daemon.py --stream --servers 20")
