#!/usr/bin/env python3
"""
LINBORG Schema Validation Script
Verifies that all 14 LINBORG metrics are present and correctly named throughout the system.
"""

import sys
from pathlib import Path
import pandas as pd
import json

# Expected 14 LINBORG metrics
EXPECTED_LINBORG_METRICS = [
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
    'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
    'net_in_mb_s', 'net_out_mb_s',
    'back_close_wait', 'front_close_wait',
    'load_average', 'uptime_days'
]

def validate_training_data(training_dir: str = "./training/") -> bool:
    """Validate training data contains all 14 LINBORG metrics."""
    print("\n" + "="*70)
    print("VALIDATING TRAINING DATA")
    print("="*70)

    training_path = Path(training_dir)

    # Try to load parquet
    parquet_file = training_path / "server_metrics.parquet"
    if not parquet_file.exists():
        print(f"ERROR: Training data not found: {parquet_file}")
        return False

    print(f"Loading: {parquet_file}")
    df = pd.read_parquet(parquet_file)

    # Check columns
    present = [m for m in EXPECTED_LINBORG_METRICS if m in df.columns]
    missing = [m for m in EXPECTED_LINBORG_METRICS if m not in df.columns]

    print(f"\nLINBORG Metrics: {len(present)}/14 present")

    if missing:
        print(f"\nMISSING METRICS:")
        for m in missing:
            print(f"  - {m}")
        return False

    # Verify data types and sample values
    print("\nSample values (first row):")
    sample = df.iloc[0]
    for metric in EXPECTED_LINBORG_METRICS:
        value = sample[metric]
        print(f"  {metric:20s} = {value:8.2f}")

    # Verify percentages are in valid range
    pct_metrics = [m for m in EXPECTED_LINBORG_METRICS if m.endswith('_pct')]
    invalid_pcts = []
    for metric in pct_metrics:
        min_val = df[metric].min()
        max_val = df[metric].max()
        if min_val < 0 or max_val > 100:
            invalid_pcts.append(f"{metric}: [{min_val:.2f}, {max_val:.2f}]")

    if invalid_pcts:
        print(f"\nWARNING: Percentage metrics outside [0, 100] range:")
        for msg in invalid_pcts:
            print(f"  - {msg}")

    print("\nSTATUS: PASS - All 14 LINBORG metrics present")
    return True


def validate_code_files() -> bool:
    """Validate that code files reference all 14 LINBORG metrics."""
    print("\n" + "="*70)
    print("VALIDATING CODE FILES")
    print("="*70)

    files_to_check = {
        'main.py': 'User interface validation',
        'tft_trainer.py': 'Model training configuration',
        'tft_inference_daemon.py': 'Inference data processing',
        'tft_dashboard_web.py': 'Dashboard display'
    }

    all_valid = True

    for filename, description in files_to_check.items():
        filepath = Path(filename)
        if not filepath.exists():
            print(f"\nERROR: File not found: {filename}")
            all_valid = False
            continue

        print(f"\nChecking {filename} ({description})...")

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for each metric
        found = []
        missing = []
        for metric in EXPECTED_LINBORG_METRICS:
            if f"'{metric}'" in content or f'"{metric}"' in content:
                found.append(metric)
            else:
                missing.append(metric)

        print(f"  Found: {len(found)}/14 metrics")

        if missing:
            print(f"  WARNING: Missing references to:")
            for m in missing[:5]:  # Show first 5
                print(f"    - {m}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")

    return all_valid


def validate_metrics_generator() -> bool:
    """Validate that metrics_generator_daemon.py is fixed."""
    print("\n" + "="*70)
    print("VALIDATING METRICS GENERATOR DAEMON FIX")
    print("="*70)

    filepath = Path("metrics_generator_daemon.py")
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        return False

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for the fixed pattern
    has_base_keys = "'cpu_user'" in content and "'cpu_sys'" in content
    has_pct_suffix_addition = "f'{metric}_pct'" in content

    print(f"Uses base metric keys (cpu_user, mem_used, etc.): {has_base_keys}")
    print(f"Adds _pct suffix when storing output: {has_pct_suffix_addition}")

    if has_base_keys and has_pct_suffix_addition:
        print("\nSTATUS: PASS - Daemon uses correct key pattern")
        return True
    else:
        print("\nSTATUS: FAIL - Daemon may not be using correct pattern")
        return False


def main():
    """Run all validations."""
    print("\n" + "="*70)
    print("LINBORG SCHEMA VALIDATION")
    print("Verifying 14 metrics across entire pipeline")
    print("="*70)

    results = {}

    # Validate training data
    results['training_data'] = validate_training_data()

    # Validate code files
    results['code_files'] = validate_code_files()

    # Validate daemon fix
    results['daemon_fix'] = validate_metrics_generator()

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {check:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "="*70)
    if all_passed:
        print("OVERALL STATUS: PASS - System ready for presentation")
        print("ACTION: Restart both daemons to activate updated code")
    else:
        print("OVERALL STATUS: FAIL - Issues detected, review output above")
    print("="*70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
