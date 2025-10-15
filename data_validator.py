#!/usr/bin/env python3
"""
Data Contract Validator

Validates data against DATA_CONTRACT.md specification to prevent
pipeline breakage from schema drift.

Conforms to: DATA_CONTRACT.md v1.0.0
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd


# Data Contract Constants (v2.0.0 - LINBORG Metrics)
CONTRACT_VERSION = "2.0.0"

VALID_STATES = [
    'critical_issue',
    'healthy',
    'heavy_load',
    'idle',
    'maintenance',
    'morning_spike',
    'offline',
    'recovery'
]

# LINBORG Metrics (14 total)
REQUIRED_COLUMNS = [
    'timestamp',
    'server_name',
    'state',
    # CPU metrics (5)
    'cpu_user_pct',
    'cpu_sys_pct',
    'cpu_iowait_pct',
    'cpu_idle_pct',
    'java_cpu_pct',
    # Memory metrics (2)
    'mem_used_pct',
    'swap_used_pct',
    # Disk metrics (1)
    'disk_usage_pct',
    # Network metrics (2)
    'net_in_mb_s',
    'net_out_mb_s',
    # Connection metrics (2)
    'back_close_wait',
    'front_close_wait',
    # System metrics (2)
    'load_average',
    'uptime_days'
]

NUMERIC_RANGES = {
    # CPU percentages
    'cpu_user_pct': (0.0, 100.0),
    'cpu_sys_pct': (0.0, 100.0),
    'cpu_iowait_pct': (0.0, 100.0),
    'cpu_idle_pct': (0.0, 100.0),
    'java_cpu_pct': (0.0, 100.0),
    # Memory percentages
    'mem_used_pct': (0.0, 100.0),
    'swap_used_pct': (0.0, 100.0),
    # Disk percentage
    'disk_usage_pct': (0.0, 100.0),
    # Network throughput (MB/s)
    'net_in_mb_s': (0.0, float('inf')),
    'net_out_mb_s': (0.0, float('inf')),
    # Connection counts (integers)
    'back_close_wait': (0, float('inf')),
    'front_close_wait': (0, float('inf')),
    # System metrics
    'load_average': (0.0, float('inf')),
    'uptime_days': (0, 365)
}


class DataValidator:
    """
    Validator for data contract compliance.

    Checks:
    - Required columns presence
    - State values validity
    - Numeric ranges
    - Timestamp format
    - Data types
    """

    def __init__(self, strict: bool = True):
        """
        Initialize validator.

        Args:
            strict: If True, raise errors. If False, return warnings only.
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """
        Validate DataFrame against data contract.

        Args:
            df: DataFrame to validate

        Returns:
            (is_valid, list_of_errors, list_of_warnings)
        """
        self.errors = []
        self.warnings = []

        # Check required columns
        self._validate_columns(df)

        # Check state values
        if 'state' in df.columns:
            self._validate_states(df)

        # Check numeric ranges
        self._validate_numeric_ranges(df)

        # Check timestamp format
        if 'timestamp' in df.columns:
            self._validate_timestamps(df)

        # Check for missing data
        self._validate_missing_data(df)

        # Check for duplicate timestamps per server
        if 'timestamp' in df.columns and 'server_name' in df.columns:
            self._validate_temporal_uniqueness(df)

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """Check for required columns."""
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            self.errors.append(f"Missing required columns: {sorted(missing)}")

        # Allow profile and training-related columns
        extra = set(df.columns) - set(REQUIRED_COLUMNS) - {
            'server_id', 'profile', 'hour', 'day_of_week', 'month', 'is_weekend',
            'is_business_hours', 'time_idx', 'notes', 'problem_child', 'status'
        }
        if extra:
            self.warnings.append(f"Extra columns present: {sorted(extra)}")

    def _validate_states(self, df: pd.DataFrame) -> None:
        """Check state values validity."""
        invalid_states = set(df['state'].unique()) - set(VALID_STATES)
        if invalid_states:
            self.errors.append(
                f"Invalid state values found: {sorted(invalid_states)}\n"
                f"   Valid states: {VALID_STATES}"
            )

        # Check state distribution
        state_counts = df['state'].value_counts()
        total = len(df)

        for state in VALID_STATES:
            if state not in state_counts:
                self.warnings.append(f"State '{state}' not present in data")
            elif state_counts[state] / total < 0.001:  # Less than 0.1%
                self.warnings.append(
                    f"State '{state}' underrepresented: "
                    f"{state_counts[state]} samples ({state_counts[state]/total*100:.2f}%)"
                )

    def _validate_numeric_ranges(self, df: pd.DataFrame) -> None:
        """Check numeric columns are within valid ranges."""
        for col, (min_val, max_val) in NUMERIC_RANGES.items():
            if col not in df.columns:
                continue

            # Check data type
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.errors.append(f"Column '{col}' must be numeric, got {df[col].dtype}")
                continue

            # Check range
            out_of_range = (df[col] < min_val) | (df[col] > max_val)
            if out_of_range.any():
                count = out_of_range.sum()
                examples = df[out_of_range][col].head(3).tolist()
                self.errors.append(
                    f"Column '{col}' has {count} values out of range [{min_val}, {max_val}]\n"
                    f"   Examples: {examples}"
                )

            # Check for unrealistic values (warnings only)
            if col == 'cpu_user_pct' and (df[col] > 99.9).sum() > len(df) * 0.05:
                self.warnings.append(
                    f"More than 5% of samples have cpu_user_pct > 99.9% "
                    f"(may indicate data quality issues)"
                )

            # I/O Wait validation - CRITICAL metric
            if col == 'cpu_iowait_pct' and (df[col] > 50).sum() > len(df) * 0.01:
                self.warnings.append(
                    f"More than 1% of samples have cpu_iowait_pct > 50% "
                    f"(CRITICAL: severe I/O bottleneck - 'system troubleshooting 101')"
                )

    def _validate_timestamps(self, df: pd.DataFrame) -> None:
        """Check timestamp format and consistency."""
        # Check data type
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                # Try to convert
                pd.to_datetime(df['timestamp'])
                self.warnings.append(
                    "Column 'timestamp' is not datetime type but can be converted"
                )
            except Exception as e:
                self.errors.append(f"Column 'timestamp' has invalid format: {e}")
                return

        # Check for future timestamps
        now = pd.Timestamp.now(tz='UTC')
        # Make sure we compare with compatible timezone
        timestamps = df['timestamp']
        if timestamps.dt.tz is None:
            now = now.tz_localize(None)
        future_timestamps = timestamps > now
        if future_timestamps.any():
            count = future_timestamps.sum()
            self.warnings.append(
                f"{count} timestamps are in the future (synthetic data?)"
            )

        # Check for chronological order within servers
        if 'server_name' in df.columns:
            for server in df['server_name'].unique()[:5]:  # Check first 5 servers
                server_df = df[df['server_name'] == server].sort_index()
                if not server_df['timestamp'].is_monotonic_increasing:
                    self.warnings.append(
                        f"Timestamps for '{server}' are not chronologically ordered"
                    )
                    break

    def _validate_missing_data(self, df: pd.DataFrame) -> None:
        """Check for missing values."""
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                continue

            missing = df[col].isna().sum()
            if missing > 0:
                pct = (missing / len(df)) * 100
                if pct > 5.0:
                    self.errors.append(
                        f"Column '{col}' has {missing} missing values ({pct:.1f}%)"
                    )
                else:
                    self.warnings.append(
                        f"Column '{col}' has {missing} missing values ({pct:.2f}%)"
                    )

    def _validate_temporal_uniqueness(self, df: pd.DataFrame) -> None:
        """Check for duplicate timestamps per server."""
        duplicates = df.duplicated(subset=['server_name', 'timestamp'], keep=False)
        if duplicates.any():
            count = duplicates.sum()
            self.errors.append(
                f"Found {count} duplicate (server_name, timestamp) pairs"
            )

    def validate_model_compatibility(
        self,
        model_dir: Path,
        data_df: pd.DataFrame
    ) -> Tuple[bool, List[str]]:
        """
        Verify model can load data without dimension mismatches.

        Args:
            model_dir: Path to model directory
            data_df: DataFrame to validate against model

        Returns:
            (is_compatible, list_of_errors)
        """
        errors = []
        model_dir = Path(model_dir)

        # Check training_info.json exists
        training_info_file = model_dir / 'training_info.json'
        if not training_info_file.exists():
            errors.append(f"training_info.json not found in {model_dir}")
            return False, errors

        # Load training info
        with open(training_info_file) as f:
            training_info = json.load(f)

        # Validate state count
        trained_states = training_info.get('unique_states', [])
        current_states = sorted(data_df['state'].unique().tolist())

        if set(trained_states) != set(VALID_STATES):
            errors.append(
                f"Model trained with {len(trained_states)} states: {trained_states}\n"
                f"   Contract requires {len(VALID_STATES)} states: {VALID_STATES}"
            )

        if set(current_states) - set(trained_states):
            extra_states = set(current_states) - set(trained_states)
            errors.append(
                f"Data contains states not seen during training: {extra_states}"
            )

        # Validate server mapping exists
        server_mapping_file = model_dir / 'server_mapping.json'
        if not server_mapping_file.exists():
            errors.append(f"server_mapping.json not found in {model_dir}")

        # Check data contract version
        data_contract_version = training_info.get('data_contract_version')
        if data_contract_version != CONTRACT_VERSION:
            errors.append(
                f"Model trained with contract v{data_contract_version}, "
                f"current contract is v{CONTRACT_VERSION}"
            )

        is_compatible = len(errors) == 0
        return is_compatible, errors

    def print_report(self) -> None:
        """Print validation report."""
        print("\n" + "=" * 60)
        print("DATA CONTRACT VALIDATION REPORT")
        print("=" * 60)

        if not self.errors and not self.warnings:
            print("[OK] All validations passed!")
        else:
            if self.errors:
                print(f"\n[ERROR] {len(self.errors)} error(s) found:")
                for i, error in enumerate(self.errors, 1):
                    print(f"   {i}. {error}")

            if self.warnings:
                print(f"\n[WARNING] {len(self.warnings)} warning(s):")
                for i, warning in enumerate(self.warnings, 1):
                    print(f"   {i}. {warning}")

        print("=" * 60 + "\n")


def validate_file(file_path: Path, strict: bool = True) -> bool:
    """
    Validate a data file against the contract.

    Args:
        file_path: Path to CSV or Parquet file
        strict: If True, errors cause failure

    Returns:
        True if valid (or warnings only in non-strict mode)
    """
    file_path = Path(file_path)

    print(f"\n[VALIDATE] {file_path.name}")

    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return False

    # Load data
    try:
        if file_path.suffix == '.parquet':
            df = pd.read_parquet(file_path)
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            print(f"[ERROR] Unsupported file format: {file_path.suffix}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}")
        return False

    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Validate
    validator = DataValidator(strict=strict)
    is_valid, errors, warnings = validator.validate_schema(df)

    validator.print_report()

    return is_valid if strict else True


# CLI usage
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("DATA CONTRACT VALIDATOR")
    print(f"Contract Version: {CONTRACT_VERSION}")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage: python data_validator.py <file_path> [--strict]")
        print("\nExample:")
        print("  python data_validator.py training/server_metrics.parquet")
        print("  python data_validator.py training/server_metrics.parquet --strict")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    strict = "--strict" in sys.argv

    result = validate_file(file_path, strict=strict)

    if result:
        print("[OK] Validation passed!")
        sys.exit(0)
    else:
        print("[ERROR] Validation failed!")
        sys.exit(1)
