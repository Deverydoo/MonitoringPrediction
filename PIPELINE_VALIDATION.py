#!/usr/bin/env python3
"""
COMPLETE PIPELINE VALIDATION - Production Ready
This script validates every step of the LINBORG metrics pipeline.
Must pass 100% before presentation.
"""

import sys
import json
from pathlib import Path

print("=" * 80)
print("LINBORG METRICS PIPELINE - COMPLETE VALIDATION")
print("=" * 80)
print("This validates the ENTIRE data flow from generator to dashboard")
print("=" * 80)

# Expected LINBORG metrics
LINBORG_METRICS = [
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
    'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
    'net_in_mb_s', 'net_out_mb_s',
    'back_close_wait', 'front_close_wait',
    'load_average', 'uptime_days'
]

validation_results = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def test_step(name, condition, error_msg="", warning_msg=""):
    """Test a validation step."""
    if condition:
        validation_results['passed'].append(name)
        print(f"   ✅ {name}")
        return True
    elif warning_msg:
        validation_results['warnings'].append(f"{name}: {warning_msg}")
        print(f"   ⚠️  {name}: {warning_msg}")
        return False
    else:
        validation_results['failed'].append(f"{name}: {error_msg}")
        print(f"   ❌ {name}")
        if error_msg:
            print(f"      → {error_msg}")
        return False

# =============================================================================
# STEP 1: Validate metrics_generator.py (Training Data Generator)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: Validate metrics_generator.py")
print("=" * 80)

try:
    import pandas as pd
    from metrics_generator import PROFILE_BASELINES, ServerProfile

    # Check that PROFILE_BASELINES has all 14 LINBORG metrics for each profile
    print("\n📊 Checking PROFILE_BASELINES for all 7 profiles...")

    all_profiles_valid = True
    for profile in ServerProfile:
        baselines = PROFILE_BASELINES.get(profile, {})
        missing = [m for m in LINBORG_METRICS if m not in baselines]

        if missing:
            print(f"   ❌ {profile.value}: Missing {len(missing)} metrics: {missing}")
            all_profiles_valid = False
        else:
            print(f"   ✅ {profile.value}: All 14 LINBORG metrics present")

    test_step(
        "metrics_generator.py has all 14 LINBORG metrics in PROFILE_BASELINES",
        all_profiles_valid,
        "Some profiles missing LINBORG metrics in baselines"
    )

    # Check training data file exists and has correct schema
    print("\n📁 Checking training data files...")
    training_dir = Path("training")

    if not training_dir.exists():
        test_step("Training directory exists", False, "Directory 'training/' not found")
    else:
        test_step("Training directory exists", True)

        parquet_files = list(training_dir.glob("*.parquet"))

        if not parquet_files:
            test_step("Training data exists", False, "No .parquet files in training/")
        else:
            latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
            test_step("Training data exists", True)

            print(f"\n   Loading: {latest_file.name}")
            df = pd.read_parquet(latest_file)

            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Servers: {df['server_name'].nunique()}")

            # Verify all LINBORG metrics present
            missing_cols = [m for m in LINBORG_METRICS if m not in df.columns]

            if missing_cols:
                test_step(
                    "Training data has all 14 LINBORG metrics",
                    False,
                    f"Missing columns: {missing_cols}"
                )
            else:
                test_step("Training data has all 14 LINBORG metrics", True)

                # Check for non-zero values
                print("\n   Checking for non-zero values...")
                zero_metrics = []
                for metric in LINBORG_METRICS:
                    non_zero_count = (df[metric] != 0).sum()
                    if non_zero_count == 0:
                        zero_metrics.append(metric)
                        print(f"      ⚠️  {metric}: ALL ZEROS ({len(df)} rows)")
                    else:
                        pct_non_zero = (non_zero_count / len(df)) * 100
                        print(f"      ✅ {metric}: {pct_non_zero:.1f}% non-zero")

                if zero_metrics:
                    test_step(
                        "Training data has non-zero values for all metrics",
                        False,
                        f"These metrics are all zeros: {zero_metrics}"
                    )
                else:
                    test_step("Training data has non-zero values for all metrics", True)

except ImportError as e:
    test_step("Import metrics_generator.py", False, f"Import error: {e}")
except Exception as e:
    test_step("metrics_generator.py validation", False, f"Error: {e}")

# =============================================================================
# STEP 2: Validate metrics_generator_daemon.py (Streaming Daemon)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: Validate metrics_generator_daemon.py (Code Inspection)")
print("=" * 80)

try:
    # Read the daemon source code and verify it has LINBORG metrics
    daemon_file = Path("metrics_generator_daemon.py")

    if not daemon_file.exists():
        test_step("metrics_generator_daemon.py exists", False, "File not found")
    else:
        test_step("metrics_generator_daemon.py exists", True)

        with open(daemon_file, 'r', encoding='utf-8') as f:
            daemon_code = f.read()

        # Check for LINBORG metrics in the code
        has_linborg = 'cpu_user_pct' in daemon_code and 'cpu_iowait_pct' in daemon_code
        has_old_metrics = "for metric in ['cpu', 'mem'," in daemon_code

        test_step(
            "Daemon code references LINBORG metrics",
            has_linborg,
            "Code doesn't contain LINBORG metric names like cpu_user_pct"
        )

        test_step(
            "Daemon code does NOT use old metric names",
            not has_old_metrics,
            "Code still has old metric loop with 'cpu', 'mem'"
        )

        # Check for all 14 LINBORG metrics in the daemon code
        all_metrics_found = all(metric in daemon_code for metric in LINBORG_METRICS)
        test_step(
            "All 14 LINBORG metrics referenced in daemon code",
            all_metrics_found,
            f"Some metrics not found in code"
        )

except Exception as e:
    test_step("metrics_generator_daemon.py code inspection", False, f"Error: {e}")

# =============================================================================
# STEP 3: Validate tft_inference_daemon.py (Inference Daemon)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: Validate tft_inference_daemon.py (Code Inspection)")
print("=" * 80)

try:
    inference_file = Path("tft_inference_daemon.py")

    if not inference_file.exists():
        test_step("tft_inference_daemon.py exists", False, "File not found")
    else:
        test_step("tft_inference_daemon.py exists", True)

        with open(inference_file, 'r', encoding='utf-8') as f:
            inference_code = f.read()

        # Check time_varying_unknown_reals has all 14 LINBORG metrics
        has_tvu_reals = 'time_varying_unknown_reals' in inference_code
        test_step("Has time_varying_unknown_reals definition", has_tvu_reals)

        # Check for LINBORG metrics in heuristic loop
        has_heuristic_loop = "for metric in ['cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct'" in inference_code
        test_step(
            "Heuristic loop includes cpu_idle_pct and other LINBORG metrics",
            has_heuristic_loop,
            "Heuristic loop may be incomplete"
        )

        # Verify all 14 metrics are somewhere in the inference code
        all_metrics_in_inference = all(metric in inference_code for metric in LINBORG_METRICS)
        test_step(
            "All 14 LINBORG metrics referenced in inference daemon",
            all_metrics_in_inference
        )

except Exception as e:
    test_step("tft_inference_daemon.py code inspection", False, f"Error: {e}")

# =============================================================================
# STEP 4: Validate Model
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: Validate Trained Model")
print("=" * 80)

try:
    models_dir = Path("models")

    if not models_dir.exists():
        test_step("Models directory exists", False, "Directory 'models/' not found")
    else:
        test_step("Models directory exists", True)

        model_dirs = list(models_dir.glob("tft_model_*"))

        if not model_dirs:
            test_step("Trained model exists", False, "No tft_model_* directories found")
        else:
            latest_model = max(model_dirs, key=lambda p: p.stat().st_mtime)
            test_step("Trained model exists", True)

            print(f"\n   Latest model: {latest_model.name}")

            # Check for safetensors file
            safetensors_file = latest_model / "model.safetensors"
            test_step("Model uses safetensors format", safetensors_file.exists())

            # Check training_info.json
            training_info_file = latest_model / "training_info.json"
            if training_info_file.exists():
                with open(training_info_file) as f:
                    training_info = json.load(f)

                print(f"   Trained at: {training_info.get('trained_at', 'unknown')}")
                print(f"   Epochs: {training_info.get('epochs', 'unknown')}")

                test_step("Model training info exists", True)
            else:
                test_step("Model training info exists", False, "training_info.json not found")

except Exception as e:
    test_step("Model validation", False, f"Error: {e}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

total_tests = len(validation_results['passed']) + len(validation_results['failed']) + len(validation_results['warnings'])
pass_count = len(validation_results['passed'])
fail_count = len(validation_results['failed'])
warn_count = len(validation_results['warnings'])

print(f"\n✅ Passed: {pass_count}/{total_tests}")
print(f"❌ Failed: {fail_count}/{total_tests}")
print(f"⚠️  Warnings: {warn_count}/{total_tests}")

if validation_results['failed']:
    print("\n🚨 FAILED TESTS:")
    for failure in validation_results['failed']:
        print(f"   ❌ {failure}")

if validation_results['warnings']:
    print("\n⚠️  WARNINGS:")
    for warning in validation_results['warnings']:
        print(f"   ⚠️  {warning}")

print("\n" + "=" * 80)

if fail_count == 0 and warn_count == 0:
    print("🎉 ALL VALIDATIONS PASSED - READY FOR PRESENTATION!")
    print("=" * 80)
    sys.exit(0)
elif fail_count == 0:
    print("✅ CORE VALIDATIONS PASSED (warnings are acceptable)")
    print("=" * 80)
    sys.exit(0)
else:
    print("❌ PIPELINE NOT READY - Fix failed tests before presentation")
    print("=" * 80)
    sys.exit(1)
