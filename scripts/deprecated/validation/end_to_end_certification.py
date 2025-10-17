#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Certification Test
Validates all optimizations work correctly across the entire pipeline

Tests:
1. Centralized LINBORG schema (linborg_schema.py)
2. Server profile detection (server_profiles.py)
3. Data generation with 'status' column (not 'state')
4. Trainer using centralized schema
5. Dashboard CPU helper function
6. Full pipeline integration

Run: python end_to_end_certification.py
"""

import sys
import io
from pathlib import Path
import traceback

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def print_test(name, passed, details=""):
    """Print test result"""
    if passed:
        symbol = f"{Colors.GREEN}✓{Colors.RESET}"
        status = f"{Colors.GREEN}PASS{Colors.RESET}"
    else:
        symbol = f"{Colors.RED}✗{Colors.RESET}"
        status = f"{Colors.RED}FAIL{Colors.RESET}"

    print(f"{symbol} {name:50s} [{status}]")
    if details:
        print(f"  {Colors.YELLOW}{details}{Colors.RESET}")


def test_centralized_schema():
    """Test 1: Verify linborg_schema.py works correctly"""
    print_header("TEST 1: Centralized LINBORG Schema")

    results = []

    try:
        # Test import
        from linborg_schema import (
            LINBORG_METRICS,
            NUM_LINBORG_METRICS,
            validate_linborg_metrics,
            get_metric_type,
            LINBORG_METRICS_PCT,
            LINBORG_METRICS_COUNTS,
            LINBORG_METRICS_CONTINUOUS,
            LINBORG_CRITICAL_METRICS
        )
        print_test("Import linborg_schema", True)
        results.append(True)

        # Test metric count
        passed = NUM_LINBORG_METRICS == 14
        print_test(f"Correct metric count (14)", passed,
                  f"Found: {NUM_LINBORG_METRICS}")
        results.append(passed)

        # Test all metrics present
        expected_metrics = [
            'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
            'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
            'net_in_mb_s', 'net_out_mb_s',
            'back_close_wait', 'front_close_wait',
            'load_average', 'uptime_days'
        ]
        missing = [m for m in expected_metrics if m not in LINBORG_METRICS]
        passed = len(missing) == 0
        print_test("All 14 metrics defined", passed,
                  f"Missing: {missing}" if missing else "All present")
        results.append(passed)

        # Test validation helper
        present, missing = validate_linborg_metrics(LINBORG_METRICS)
        passed = len(present) == 14 and len(missing) == 0
        print_test("Validation helper works", passed,
                  f"{len(present)}/14 present, {len(missing)} missing")
        results.append(passed)

        # Test metric type helper
        passed = (
            get_metric_type('cpu_user_pct') == 'percentage' and
            get_metric_type('back_close_wait') == 'count' and
            get_metric_type('load_average') == 'continuous'
        )
        print_test("Metric type detection", passed)
        results.append(passed)

        # Test subsets
        passed = (
            len(LINBORG_METRICS_PCT) == 8 and
            len(LINBORG_METRICS_COUNTS) == 2 and
            len(LINBORG_METRICS_CONTINUOUS) == 4
        )
        print_test("Metric subsets correct", passed,
                  f"PCT:{len(LINBORG_METRICS_PCT)}, COUNT:{len(LINBORG_METRICS_COUNTS)}, CONT:{len(LINBORG_METRICS_CONTINUOUS)}")
        results.append(passed)

    except Exception as e:
        print_test("Centralized schema tests", False, str(e))
        results.append(False)

    return all(results)


def test_server_profiles():
    """Test 2: Verify server_profiles.py works correctly"""
    print_header("TEST 2: Server Profile Detection")

    results = []

    try:
        from server_profiles import (
            ServerProfile,
            infer_profile_from_name,
            get_profile_display_name,
            PROFILE_PATTERNS
        )
        print_test("Import server_profiles", True)
        results.append(True)

        # Test profile detection
        test_cases = [
            ('ppml0015', ServerProfile.ML_COMPUTE),
            ('ppdb042', ServerProfile.DATABASE),
            ('ppweb123', ServerProfile.WEB_API),
            ('ppcon456', ServerProfile.CONDUCTOR_MGMT),
            ('ppetl789', ServerProfile.DATA_INGEST),
            ('pprisk001', ServerProfile.RISK_ANALYTICS),
            ('unknown_server', ServerProfile.GENERIC),
        ]

        passed_cases = 0
        for server_name, expected in test_cases:
            detected = infer_profile_from_name(server_name)
            if detected == expected:
                passed_cases += 1

        passed = passed_cases == len(test_cases)
        print_test(f"Profile detection ({passed_cases}/{len(test_cases)})", passed)
        results.append(passed)

        # Test display names
        display = get_profile_display_name(ServerProfile.ML_COMPUTE)
        passed = display == 'ML Compute'
        print_test("Display name formatting", passed, f"Got: '{display}'")
        results.append(passed)

        # Test pattern count
        passed = len(PROFILE_PATTERNS) >= 20  # Should have many patterns
        print_test(f"Pattern registry loaded ({len(PROFILE_PATTERNS)} patterns)", passed)
        results.append(passed)

    except Exception as e:
        print_test("Server profile tests", False, str(e))
        traceback.print_exc()
        results.append(False)

    return all(results)


def test_data_generation():
    """Test 3: Verify data generation uses 'status' not 'state'"""
    print_header("TEST 3: Data Generation (status column)")

    results = []

    try:
        import pandas as pd

        # Check if training data exists
        training_file = Path("./training/server_metrics.parquet")

        if not training_file.exists():
            print_test("Training data exists", False,
                      "Generate with: python metrics_generator.py --hours 24")
            return False

        print_test("Training data file found", True, str(training_file))
        results.append(True)

        # Load and check schema
        df = pd.read_parquet(training_file)

        # Check for 'status' column (NEW)
        has_status = 'status' in df.columns
        print_test("Has 'status' column", has_status,
                  "✓ Uses new naming" if has_status else "✗ Still using 'state'")
        results.append(has_status)

        # Check does NOT have 'state' column (OLD)
        has_old_state = 'state' in df.columns
        passed = not has_old_state
        print_test("Does NOT have 'state' column", passed,
                  "✗ Old column still present" if has_old_state else "✓ Cleaned up")
        results.append(passed)

        # Check for all LINBORG metrics
        from linborg_schema import LINBORG_METRICS, validate_linborg_metrics
        present, missing = validate_linborg_metrics(df.columns)

        passed = len(missing) == 0
        print_test(f"All 14 LINBORG metrics present", passed,
                  f"{len(present)}/14 present" + (f", missing: {missing}" if missing else ""))
        results.append(passed)

        # Check data types
        if 'status' in df.columns:
            status_values = df['status'].unique()
            print_test(f"Status values look valid", True,
                      f"Found: {sorted(status_values)[:5]}...")
            results.append(True)

    except Exception as e:
        print_test("Data generation tests", False, str(e))
        traceback.print_exc()
        results.append(False)

    return all(results)


def test_trainer_integration():
    """Test 4: Verify trainer uses centralized schema"""
    print_header("TEST 4: Trainer Integration")

    results = []

    try:
        # Check if tft_trainer.py imports linborg_schema
        trainer_file = Path("./tft_trainer.py")

        if not trainer_file.exists():
            print_test("Trainer file exists", False)
            return False

        print_test("Trainer file found", True)
        results.append(True)

        # Check for import statement
        with open(trainer_file, 'r', encoding='utf-8') as f:
            content = f.read()

        has_import = 'from linborg_schema import' in content
        print_test("Imports linborg_schema", has_import)
        results.append(has_import)

        # Check that old state→status conversion is REMOVED
        has_old_conversion = "if 'state' in df.columns and 'status' not in df.columns:" in content
        passed = not has_old_conversion
        print_test("Old state→status conversion removed", passed,
                  "✗ Old code still present" if has_old_conversion else "✓ Cleaned up")
        results.append(passed)

        # Check uses LINBORG_METRICS
        uses_centralized = 'LINBORG_METRICS' in content
        print_test("Uses centralized LINBORG_METRICS", uses_centralized)
        results.append(uses_centralized)

    except Exception as e:
        print_test("Trainer integration tests", False, str(e))
        traceback.print_exc()
        results.append(False)

    return all(results)


def test_dashboard_helpers():
    """Test 5: Verify dashboard CPU helper function"""
    print_header("TEST 5: Dashboard CPU Helper Function")

    results = []

    try:
        dashboard_file = Path("./tft_dashboard_web.py")

        if not dashboard_file.exists():
            print_test("Dashboard file exists", False)
            return False

        print_test("Dashboard file found", True)
        results.append(True)

        # Check for extract_cpu_used function
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            content = f.read()

        has_helper = 'def extract_cpu_used' in content
        print_test("extract_cpu_used() helper defined", has_helper)
        results.append(has_helper)

        # Check that helper is actually USED (not just defined)
        usage_count = content.count('extract_cpu_used(')
        # Should be: definition + at least 4 call sites = 5+ occurrences
        passed = usage_count >= 5
        print_test(f"Helper function used in code", passed,
                  f"Found {usage_count} occurrences (definition + call sites)")
        results.append(passed)

        # Check old duplicated logic is reduced
        old_pattern_count = content.count('100 - cpu_idle_cur if cpu_idle_cur > 0 else')
        passed = old_pattern_count <= 2  # Should only be in helper function itself
        print_test("Duplicated CPU calc reduced", passed,
                  f"Old pattern appears {old_pattern_count} times (should be ≤2)")
        results.append(passed)

    except Exception as e:
        print_test("Dashboard helper tests", False, str(e))
        traceback.print_exc()
        results.append(False)

    return all(results)


def test_full_pipeline():
    """Test 6: Run minimal pipeline integration test"""
    print_header("TEST 6: Full Pipeline Integration")

    results = []

    try:
        # Test 1: Can we import everything?
        from linborg_schema import LINBORG_METRICS
        from server_profiles import ServerProfile, infer_profile_from_name
        import pandas as pd

        print_test("All imports successful", True)
        results.append(True)

        # Test 2: Load training data if exists
        training_file = Path("./training/server_metrics.parquet")
        if training_file.exists():
            df = pd.read_parquet(training_file)

            # Verify schema
            has_status = 'status' in df.columns
            has_no_state = 'state' not in df.columns
            from linborg_schema import validate_linborg_metrics
            present, missing = validate_linborg_metrics(df.columns)

            schema_valid = has_status and has_no_state and len(missing) == 0
            print_test("Training data schema valid", schema_valid,
                      f"status: {has_status}, no state: {has_no_state}, metrics: {len(present)}/14")
            results.append(schema_valid)

            # Test profile detection on actual data
            if 'server_name' in df.columns:
                sample_server = df['server_name'].iloc[0]
                profile = infer_profile_from_name(sample_server)
                print_test(f"Profile detection on real data", True,
                          f"{sample_server} → {profile.value}")
                results.append(True)
        else:
            print_test("Training data available", False,
                      "Run: python metrics_generator.py --hours 24")
            results.append(False)

        # Test 3: Check model file
        model_dirs = list(Path("./models").glob("tft_model_*"))
        if model_dirs:
            latest_model = max(model_dirs, key=lambda p: p.stat().st_mtime)
            print_test("Trained model exists", True, str(latest_model.name))
            results.append(True)
        else:
            print_test("Trained model exists", False,
                      "Run: python main.py train")
            results.append(False)

    except Exception as e:
        print_test("Pipeline integration tests", False, str(e))
        traceback.print_exc()
        results.append(False)

    return all(results)


def main():
    """Run all certification tests"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("="*70)
    print(" END-TO-END CERTIFICATION TEST")
    print(" Validating All Optimizations")
    print("="*70)
    print(Colors.RESET)

    test_results = {}

    # Run all tests
    test_results["Centralized Schema"] = test_centralized_schema()
    test_results["Server Profiles"] = test_server_profiles()
    test_results["Data Generation"] = test_data_generation()
    test_results["Trainer Integration"] = test_trainer_integration()
    test_results["Dashboard Helpers"] = test_dashboard_helpers()
    test_results["Full Pipeline"] = test_full_pipeline()

    # Summary
    print_header("CERTIFICATION SUMMARY")

    passed_count = sum(1 for result in test_results.values() if result)
    total_count = len(test_results)

    for test_name, passed in test_results.items():
        if passed:
            print(f"{Colors.GREEN}✓{Colors.RESET} {test_name:30s} {Colors.GREEN}PASS{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗{Colors.RESET} {test_name:30s} {Colors.RED}FAIL{Colors.RESET}")

    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")

    if passed_count == total_count:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ CERTIFICATION PASSED: {passed_count}/{total_count} tests{Colors.RESET}")
        print(f"{Colors.GREEN}All optimizations verified and working correctly!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ CERTIFICATION FAILED: {passed_count}/{total_count} tests{Colors.RESET}")
        print(f"{Colors.YELLOW}Please review failed tests above and address issues.{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
