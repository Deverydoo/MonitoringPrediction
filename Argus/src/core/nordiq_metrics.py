#!/usr/bin/env python3
"""
NordIQ Metrics Framework - Single Source of Truth
Production-grade infrastructure monitoring metrics

Version: 1.2.1
Date: 2025-10-18
"""

# =============================================================================
# ALL 15 NORDIQ METRICS
# =============================================================================

NORDIQ_METRICS = [
    # CPU metrics (5 percentages)
    'cpu_user_pct',      # User-space CPU usage (0-100%)
    'cpu_sys_pct',       # System/kernel CPU usage (0-100%)
    'cpu_iowait_pct',    # I/O wait - CRITICAL troubleshooting metric (0-100%)
    'cpu_idle_pct',      # CPU idle time (0-100%)
    'java_cpu_pct',      # JVM-specific CPU usage (0-100%)

    # Memory metrics (3 percentages)
    'mem_used_pct',      # Memory utilization (0-100%)
    'swap_used_pct',     # Swap usage - thrashing indicator (0-100%)
    'disk_usage_pct',    # Disk space usage (0-100%)

    # Network metrics (2 continuous)
    'net_in_mb_s',       # Network ingress (MB/s, ≥0)
    'net_out_mb_s',      # Network egress (MB/s, ≥0)

    # Connection metrics (2 counts)
    'back_close_wait',   # Backend connections in CLOSE_WAIT (count, ≥0)
    'front_close_wait',  # Frontend connections in CLOSE_WAIT (count, ≥0)

    # System metrics (2 continuous)
    'load_average',      # System load average (≥0)
    'uptime_days',       # Days since last boot (0-30)

    # Cascade/dependency metrics (1 continuous)
    'cascade_impact',    # Inter-server dependency impact score (0-1)
]

# =============================================================================
# METRIC SUBSETS BY TYPE
# =============================================================================

# Percentage metrics (8 total) - stored as 0-100
NORDIQ_METRICS_PCT = [
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
    'mem_used_pct', 'swap_used_pct', 'disk_usage_pct'
]

# Connection count metrics (2 total) - integer counts
NORDIQ_METRICS_COUNTS = [
    'back_close_wait',
    'front_close_wait'
]

# Continuous float metrics (5 total)
NORDIQ_METRICS_CONTINUOUS = [
    'net_in_mb_s',
    'net_out_mb_s',
    'load_average',
    'uptime_days',       # Actually integer, but stored as float in some places
    'cascade_impact'     # Inter-server dependency impact (0-1)
]

# =============================================================================
# CRITICAL METRICS FOR ALERTING
# =============================================================================

# Metrics that indicate immediate trouble
NORDIQ_CRITICAL_METRICS = [
    'cpu_iowait_pct',   # High I/O wait = disk/storage bottleneck
    'swap_used_pct',    # Swap usage = memory thrashing
    'mem_used_pct',     # Memory pressure
]

# Metrics prominently displayed in dashboard
NORDIQ_DISPLAY_METRICS = [
    'cpu_idle_pct',     # Displayed as "CPU Used = 100 - idle"
    'cpu_iowait_pct',   # I/O Wait - CRITICAL
    'mem_used_pct',     # Memory
    'swap_used_pct',    # Swap
    'load_average',     # Load
]

# =============================================================================
# CORE COLUMNS (NON-METRIC)
# =============================================================================

CORE_COLUMNS = [
    'timestamp',        # datetime64[ns] - Time of observation
    'server_name',      # string - Server hostname
    'server_id',        # string - Encoded server ID (for model)
    'profile',          # string - Server workload type (ml_compute, database, etc.)
    'status',           # string - Operational status (healthy, critical_issue, etc.)
    'problem_child',    # bool - Persistent troublemaker flag
    'notes',            # string - Human-readable annotations
]

# Time-based features (generated from timestamp)
TIME_FEATURES = [
    'hour',             # int - Hour of day (0-23)
    'day_of_week',      # int - Day of week (0=Monday, 6=Sunday)
    'month',            # int - Month (1-12)
    'is_weekend',       # int - Weekend flag (0/1)
    'time_idx',         # int - Sequential time index for TFT
]

# =============================================================================
# COMPLETE SCHEMA
# =============================================================================

# Expected columns in training data
TRAINING_SCHEMA = CORE_COLUMNS + TIME_FEATURES + NORDIQ_METRICS

# Schema version for compatibility tracking
SCHEMA_VERSION = "1.0.0_nordiq"

# Number of metrics (for validation)
NUM_NORDIQ_METRICS = len(NORDIQ_METRICS)
assert NUM_NORDIQ_METRICS == 15, f"Expected 15 NordIQ Metrics Framework metrics, got {NUM_NORDIQ_METRICS}"

# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_nordiq_metrics(df_columns: list) -> tuple:
    """
    Validate that DataFrame contains all NordIQ Metrics Framework metrics.

    Args:
        df_columns: List of column names from DataFrame

    Returns:
        (present_metrics, missing_metrics) tuple
    """
    present = [m for m in NORDIQ_METRICS if m in df_columns]
    missing = [m for m in NORDIQ_METRICS if m not in df_columns]
    return present, missing


def get_metric_type(metric_name: str) -> str:
    """
    Get the type of a NordIQ Metrics Framework metric.

    Args:
        metric_name: Name of the metric

    Returns:
        'percentage', 'count', 'continuous', or 'unknown'
    """
    if metric_name in NORDIQ_METRICS_PCT:
        return 'percentage'
    elif metric_name in NORDIQ_METRICS_COUNTS:
        return 'count'
    elif metric_name in NORDIQ_METRICS_CONTINUOUS:
        return 'continuous'
    else:
        return 'unknown'


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("NordIQ Metrics Framework Schema v" + SCHEMA_VERSION)
    print("=" * 70)
    print(f"\nTotal Metrics: {NUM_NORDIQ_METRICS}")
    print(f"  - Percentages: {len(NORDIQ_METRICS_PCT)}")
    print(f"  - Counts: {len(NORDIQ_METRICS_COUNTS)}")
    print(f"  - Continuous: {len(NORDIQ_METRICS_CONTINUOUS)}")

    print("\n" + "=" * 70)
    print("ALL METRICS:")
    print("=" * 70)
    for i, metric in enumerate(NORDIQ_METRICS, 1):
        metric_type = get_metric_type(metric)
        print(f"{i:2d}. {metric:20s} ({metric_type})")

    print("\n" + "=" * 70)
    print("CRITICAL METRICS:")
    print("=" * 70)
    for metric in NORDIQ_CRITICAL_METRICS:
        print(f"  - {metric}")

    print("\n" + "=" * 70)
    print("SCHEMA VALIDATION EXAMPLE:")
    print("=" * 70)

    # Example: Validate a sample schema
    sample_columns = ['timestamp', 'server_name'] + NORDIQ_METRICS
    present, missing = validate_nordiq_metrics(sample_columns)

    print(f"Present: {len(present)}/{NUM_NORDIQ_METRICS}")
    if missing:
        print(f"Missing: {missing}")
    else:
        print("✅ All NordIQ Metrics Framework metrics present!")
