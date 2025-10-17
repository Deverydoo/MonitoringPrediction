#!/usr/bin/env python3
"""Debug script to trace LINBORG metrics through the entire pipeline."""

import pandas as pd
import json

print("=" * 80)
print("STEP 1: VALIDATE TRAINING DATA (metrics_generator.py output)")
print("=" * 80)

# Check training data
df = pd.read_parquet('training/server_metrics.parquet')

print(f"\n✅ Training data loaded: {len(df):,} rows")
print(f"✅ Unique servers: {df['server_name'].nunique()}")
print(f"✅ Total columns: {len(df.columns)}")

# Expected LINBORG metrics
LINBORG_METRICS = [
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
    'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
    'net_in_mb_s', 'net_out_mb_s',
    'back_close_wait', 'front_close_wait',
    'load_average', 'uptime_days'
]

print(f"\n📊 LINBORG Metrics Check (14 expected):")
print("-" * 80)

missing = []
present = []
for metric in LINBORG_METRICS:
    if metric in df.columns:
        present.append(metric)
        # Get sample values
        sample_val = df[metric].iloc[0]
        non_zero = (df[metric] != 0).sum()
        print(f"  ✅ {metric:20s} - Sample: {sample_val:.2f}, Non-zero rows: {non_zero:,}")
    else:
        missing.append(metric)
        print(f"  ❌ {metric:20s} - MISSING")

print(f"\n📈 Summary:")
print(f"  Present: {len(present)}/14")
print(f"  Missing: {len(missing)}/14")

if missing:
    print(f"\n⚠️  MISSING METRICS: {missing}")
else:
    print(f"\n✅ All 14 LINBORG metrics present in training data!")

# Show all columns
print(f"\n📋 All columns in training data:")
for i, col in enumerate(sorted(df.columns), 1):
    print(f"  {i:2d}. {col}")

# Sample row
print(f"\n📝 Sample row (first server, first timestamp):")
sample = df.iloc[0]
for metric in LINBORG_METRICS:
    if metric in df.columns:
        print(f"  {metric:20s}: {sample[metric]:.2f}")
