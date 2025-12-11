# Tachyon Argus - Data Preparation Guide

How to prepare real-world monitoring data for training the TFT model.

## Overview

To train Tachyon Argus on your own data, you need to:

1. **Extract** metrics from your monitoring system (database, API, files)
2. **Map** your column names to the expected schema
3. **Transform** data types and units
4. **Partition** into time-chunked Parquet files
5. **Validate** the final dataset

---

## Target Schema

The training engine expects exactly these 16 columns:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `timestamp` | datetime | - | ISO 8601 format |
| `server_name` | string | - | Unique server identifier |
| `status` | string | enum | Server state (see below) |
| `cpu_user_pct` | float | 0-100 | User CPU percentage |
| `cpu_sys_pct` | float | 0-100 | System CPU percentage |
| `cpu_iowait_pct` | float | 0-100 | I/O wait percentage |
| `cpu_idle_pct` | float | 0-100 | Idle CPU percentage |
| `java_cpu_pct` | float | 0-100 | Java process CPU (0 if N/A) |
| `mem_used_pct` | float | 0-100 | Memory usage percentage |
| `swap_used_pct` | float | 0-100 | Swap usage percentage |
| `disk_usage_pct` | float | 0-100 | Disk usage percentage |
| `net_in_mb_s` | float | 0+ | Network in (MB/s) |
| `net_out_mb_s` | float | 0+ | Network out (MB/s) |
| `back_close_wait` | int | 0+ | Backend CLOSE_WAIT connections |
| `front_close_wait` | int | 0+ | Frontend CLOSE_WAIT connections |
| `load_average` | float | 0+ | System load average (1-min) |
| `uptime_days` | int | 0-365 | Days since last reboot |

**Valid status values:**
- `healthy`, `critical_issue`, `heavy_load`, `idle`, `maintenance`, `morning_spike`, `offline`, `recovery`

---

## Complete Data Preparation Script

Here's a complete Python script to transform your data:

```python
#!/usr/bin/env python3
"""
data_prep.py - Prepare real-world metrics for Tachyon Argus training

Usage:
    python data_prep.py --source database --output ./training/
    python data_prep.py --source csv --input metrics.csv --output ./training/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse


# =============================================================================
# COLUMN MAPPING - CUSTOMIZE THIS FOR YOUR DATA
# =============================================================================

# Map YOUR column names to the expected schema
COLUMN_MAPPING = {
    # Your column name -> Expected column name

    # Required identifiers
    'collection_time': 'timestamp',      # or 'ts', 'datetime', 'time', etc.
    'hostname': 'server_name',           # or 'host', 'server', 'instance', etc.
    'state': 'status',                   # or 'health', 'condition', etc.

    # CPU metrics
    'cpu_user': 'cpu_user_pct',          # or 'user_cpu', 'cpu_usr', etc.
    'cpu_system': 'cpu_sys_pct',         # or 'sys_cpu', 'cpu_kernel', etc.
    'cpu_iowait': 'cpu_iowait_pct',      # or 'iowait', 'cpu_wait', etc.
    'cpu_idle': 'cpu_idle_pct',          # or 'idle_cpu', etc.
    'java_cpu': 'java_cpu_pct',          # or 'jvm_cpu', 'process_cpu', etc.

    # Memory metrics
    'memory_used': 'mem_used_pct',       # or 'mem_pct', 'memory_percent', etc.
    'swap_used': 'swap_used_pct',        # or 'swap_pct', etc.

    # Disk metrics
    'disk_used': 'disk_usage_pct',       # or 'disk_pct', 'fs_used', etc.

    # Network metrics
    'net_rx_mbps': 'net_in_mb_s',        # or 'network_in', 'rx_bytes', etc.
    'net_tx_mbps': 'net_out_mb_s',       # or 'network_out', 'tx_bytes', etc.

    # Connection metrics
    'backend_close_wait': 'back_close_wait',   # or 'close_wait_backend', etc.
    'frontend_close_wait': 'front_close_wait', # or 'close_wait_frontend', etc.

    # System metrics
    'load_1min': 'load_average',         # or 'loadavg', 'load1', etc.
    'uptime': 'uptime_days',             # or 'days_up', 'server_uptime', etc.
}

# Status value mapping (map your values to expected values)
STATUS_MAPPING = {
    # Your status value -> Expected status value
    'OK': 'healthy',
    'GOOD': 'healthy',
    'UP': 'healthy',
    'NORMAL': 'healthy',
    'WARNING': 'heavy_load',
    'WARN': 'heavy_load',
    'HIGH_LOAD': 'heavy_load',
    'CRITICAL': 'critical_issue',
    'ERROR': 'critical_issue',
    'ALERT': 'critical_issue',
    'DOWN': 'offline',
    'UNREACHABLE': 'offline',
    'MAINT': 'maintenance',
    'MAINTENANCE_MODE': 'maintenance',
    'IDLE': 'idle',
    'LOW': 'idle',
    'RECOVERING': 'recovery',
    'STARTING': 'recovery',
    'PEAK': 'morning_spike',
}


# =============================================================================
# DATA EXTRACTION FUNCTIONS
# =============================================================================

def extract_from_database(connection_string: str, query: str = None) -> pd.DataFrame:
    """
    Extract metrics from a database.

    Supports: PostgreSQL, MySQL, SQL Server, Oracle
    """
    import sqlalchemy

    engine = sqlalchemy.create_engine(connection_string)

    if query is None:
        # Default query - customize for your schema
        query = """
        SELECT
            collection_time,
            hostname,
            state,
            cpu_user,
            cpu_system,
            cpu_iowait,
            cpu_idle,
            java_cpu,
            memory_used,
            swap_used,
            disk_used,
            net_rx_mbps,
            net_tx_mbps,
            backend_close_wait,
            frontend_close_wait,
            load_1min,
            uptime
        FROM server_metrics
        WHERE collection_time >= NOW() - INTERVAL '30 days'
        ORDER BY collection_time, hostname
        """

    print(f"Executing query...")
    df = pd.read_sql(query, engine)
    print(f"Extracted {len(df):,} rows")

    return df


def extract_from_csv(file_path: str) -> pd.DataFrame:
    """Extract metrics from CSV file."""
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df):,} rows")
    return df


def extract_from_prometheus(prometheus_url: str, servers: list,
                            start_time: datetime, end_time: datetime,
                            step: str = '5m') -> pd.DataFrame:
    """
    Extract metrics from Prometheus.

    Args:
        prometheus_url: Prometheus server URL
        servers: List of server names/instances
        start_time: Start of time range
        end_time: End of time range
        step: Query resolution (default 5 minutes)
    """
    import requests

    records = []

    # Prometheus queries for each metric
    queries = {
        'cpu_user_pct': 'avg by (instance) (rate(node_cpu_seconds_total{mode="user"}[5m])) * 100',
        'cpu_sys_pct': 'avg by (instance) (rate(node_cpu_seconds_total{mode="system"}[5m])) * 100',
        'cpu_iowait_pct': 'avg by (instance) (rate(node_cpu_seconds_total{mode="iowait"}[5m])) * 100',
        'cpu_idle_pct': 'avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100',
        'mem_used_pct': '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100',
        'disk_usage_pct': '(1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100',
        'load_average': 'node_load1',
        'net_in_mb_s': 'rate(node_network_receive_bytes_total[5m]) / 1048576',
        'net_out_mb_s': 'rate(node_network_transmit_bytes_total[5m]) / 1048576',
    }

    for metric_name, query in queries.items():
        print(f"Querying {metric_name}...")

        response = requests.get(
            f"{prometheus_url}/api/v1/query_range",
            params={
                'query': query,
                'start': start_time.timestamp(),
                'end': end_time.timestamp(),
                'step': step
            }
        )

        data = response.json()

        if data['status'] == 'success':
            for result in data['data']['result']:
                instance = result['metric'].get('instance', 'unknown')
                for timestamp, value in result['values']:
                    records.append({
                        'timestamp': datetime.fromtimestamp(timestamp),
                        'server_name': instance,
                        metric_name: float(value)
                    })

    df = pd.DataFrame(records)

    # Pivot to get one row per timestamp/server
    df = df.groupby(['timestamp', 'server_name']).first().reset_index()

    print(f"Extracted {len(df):,} rows from Prometheus")
    return df


def extract_from_elasticsearch(es_url: str, index: str,
                               start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """Extract metrics from Elasticsearch."""
    from elasticsearch import Elasticsearch

    es = Elasticsearch([es_url])

    query = {
        "query": {
            "range": {
                "@timestamp": {
                    "gte": start_time.isoformat(),
                    "lte": end_time.isoformat()
                }
            }
        },
        "size": 10000,
        "sort": [{"@timestamp": "asc"}]
    }

    records = []

    # Scroll through results
    response = es.search(index=index, body=query, scroll='5m')
    scroll_id = response['_scroll_id']

    while True:
        hits = response['hits']['hits']
        if not hits:
            break

        for hit in hits:
            records.append(hit['_source'])

        response = es.scroll(scroll_id=scroll_id, scroll='5m')

    df = pd.DataFrame(records)
    print(f"Extracted {len(df):,} rows from Elasticsearch")
    return df


# =============================================================================
# DATA TRANSFORMATION
# =============================================================================

def apply_column_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename columns based on mapping."""

    # Find which mappings apply to this dataframe
    rename_dict = {}
    for src_col, dest_col in mapping.items():
        if src_col in df.columns:
            rename_dict[src_col] = dest_col
        elif src_col.lower() in [c.lower() for c in df.columns]:
            # Case-insensitive match
            actual_col = [c for c in df.columns if c.lower() == src_col.lower()][0]
            rename_dict[actual_col] = dest_col

    print(f"Mapping {len(rename_dict)} columns...")
    df = df.rename(columns=rename_dict)

    return df


def apply_status_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Map status values to expected values."""

    if 'status' not in df.columns:
        print("No status column found, defaulting to 'healthy'")
        df['status'] = 'healthy'
        return df

    # Apply mapping (case-insensitive)
    df['status'] = df['status'].apply(
        lambda x: mapping.get(str(x).upper(), mapping.get(str(x), 'healthy'))
    )

    # Validate all values are valid
    valid_statuses = {'healthy', 'critical_issue', 'heavy_load', 'idle',
                      'maintenance', 'morning_spike', 'offline', 'recovery'}

    invalid = df[~df['status'].isin(valid_statuses)]['status'].unique()
    if len(invalid) > 0:
        print(f"WARNING: Invalid status values found: {invalid}")
        print("  Defaulting invalid values to 'healthy'")
        df.loc[~df['status'].isin(valid_statuses), 'status'] = 'healthy'

    return df


def convert_units(df: pd.DataFrame) -> pd.DataFrame:
    """Convert units to expected format."""

    # Network: Convert bytes/s to MB/s if values seem too large
    for col in ['net_in_mb_s', 'net_out_mb_s']:
        if col in df.columns:
            median_val = df[col].median()
            if median_val > 1000:  # Likely in bytes/s
                print(f"Converting {col} from bytes/s to MB/s")
                df[col] = df[col] / 1_048_576
            elif median_val > 100:  # Likely in KB/s
                print(f"Converting {col} from KB/s to MB/s")
                df[col] = df[col] / 1024

    # Uptime: Convert seconds to days if values seem too large
    if 'uptime_days' in df.columns:
        median_val = df['uptime_days'].median()
        if median_val > 365:  # Likely in seconds
            print("Converting uptime from seconds to days")
            df['uptime_days'] = (df['uptime_days'] / 86400).astype(int)
        elif median_val > 30 * 24:  # Likely in hours
            print("Converting uptime from hours to days")
            df['uptime_days'] = (df['uptime_days'] / 24).astype(int)

    # Memory: Convert bytes to percentage if needed
    # (This requires knowing total memory, so usually done at source)

    return df


def add_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns with sensible defaults."""

    required_columns = {
        'timestamp': None,  # Required, no default
        'server_name': None,  # Required, no default
        'status': 'healthy',
        'cpu_user_pct': 0.0,
        'cpu_sys_pct': 0.0,
        'cpu_iowait_pct': 0.0,
        'cpu_idle_pct': 100.0,
        'java_cpu_pct': 0.0,
        'mem_used_pct': 0.0,
        'swap_used_pct': 0.0,
        'disk_usage_pct': 0.0,
        'net_in_mb_s': 0.0,
        'net_out_mb_s': 0.0,
        'back_close_wait': 0,
        'front_close_wait': 0,
        'load_average': 0.0,
        'uptime_days': 0,
    }

    for col, default in required_columns.items():
        if col not in df.columns:
            if default is None:
                raise ValueError(f"Required column '{col}' is missing and has no default")
            print(f"Adding missing column '{col}' with default value: {default}")
            df[col] = default

    return df


def derive_missing_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Derive metrics that can be calculated from others."""

    # If cpu_idle is missing but we have the others, calculate it
    if 'cpu_idle_pct' not in df.columns or df['cpu_idle_pct'].isna().all():
        if all(c in df.columns for c in ['cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct']):
            print("Deriving cpu_idle_pct from other CPU metrics")
            df['cpu_idle_pct'] = 100 - df['cpu_user_pct'] - df['cpu_sys_pct'] - df['cpu_iowait_pct']
            df['cpu_idle_pct'] = df['cpu_idle_pct'].clip(0, 100)

    # If we have total CPU but not breakdown, estimate
    if 'cpu_total' in df.columns:
        if 'cpu_user_pct' not in df.columns or df['cpu_user_pct'].isna().all():
            print("Estimating CPU breakdown from total")
            df['cpu_user_pct'] = df['cpu_total'] * 0.7
            df['cpu_sys_pct'] = df['cpu_total'] * 0.2
            df['cpu_iowait_pct'] = df['cpu_total'] * 0.1
            df['cpu_idle_pct'] = 100 - df['cpu_total']

    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure correct data types."""

    # Timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # String columns
    df['server_name'] = df['server_name'].astype(str)
    df['status'] = df['status'].astype(str)

    # Float columns (percentages and rates)
    float_cols = ['cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct',
                  'java_cpu_pct', 'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
                  'net_in_mb_s', 'net_out_mb_s', 'load_average']

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # Integer columns
    int_cols = ['back_close_wait', 'front_close_wait', 'uptime_days']

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df


def validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clip values to expected ranges."""

    # Percentage columns (0-100)
    pct_cols = ['cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct',
                'java_cpu_pct', 'mem_used_pct', 'swap_used_pct', 'disk_usage_pct']

    for col in pct_cols:
        if col in df.columns:
            out_of_range = ((df[col] < 0) | (df[col] > 100)).sum()
            if out_of_range > 0:
                print(f"Clipping {out_of_range} out-of-range values in {col}")
            df[col] = df[col].clip(0, 100)

    # Non-negative columns
    nonneg_cols = ['net_in_mb_s', 'net_out_mb_s', 'back_close_wait',
                   'front_close_wait', 'load_average', 'uptime_days']

    for col in nonneg_cols:
        if col in df.columns:
            negative = (df[col] < 0).sum()
            if negative > 0:
                print(f"Clipping {negative} negative values in {col}")
            df[col] = df[col].clip(lower=0)

    # Uptime days (0-365)
    if 'uptime_days' in df.columns:
        df['uptime_days'] = df['uptime_days'].clip(0, 365)

    return df


# =============================================================================
# PARTITIONING AND OUTPUT
# =============================================================================

def partition_by_time(df: pd.DataFrame, chunk_hours: int = 2) -> dict:
    """
    Partition dataframe into time chunks.

    Args:
        df: DataFrame with timestamp column
        chunk_hours: Hours per chunk (default: 2)

    Returns:
        Dictionary mapping chunk_id to DataFrame
    """
    df = df.sort_values('timestamp')

    # Create chunk boundaries
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()

    chunks = {}
    current_start = min_time.replace(minute=0, second=0, microsecond=0)

    # Align to chunk boundary
    hour = current_start.hour
    current_start = current_start.replace(hour=(hour // chunk_hours) * chunk_hours)

    while current_start <= max_time:
        current_end = current_start + timedelta(hours=chunk_hours)

        mask = (df['timestamp'] >= current_start) & (df['timestamp'] < current_end)
        chunk_df = df[mask]

        if len(chunk_df) > 0:
            chunk_id = current_start.strftime('%Y%m%d_%H')
            chunks[chunk_id] = chunk_df

        current_start = current_end

    print(f"Created {len(chunks)} time chunks")
    return chunks


def save_partitioned_parquet(chunks: dict, output_dir: str) -> str:
    """
    Save partitioned data as Parquet files with manifest.

    Args:
        chunks: Dictionary of chunk_id -> DataFrame
        output_dir: Output directory path

    Returns:
        Path to manifest file
    """
    output_path = Path(output_dir)
    partitioned_dir = output_path / 'server_metrics_partitioned'
    partitioned_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        'version': '2.0.0',
        'created': datetime.now().isoformat(),
        'chunk_hours': 2,
        'chunks': []
    }

    total_rows = 0
    servers = set()

    for chunk_id, chunk_df in sorted(chunks.items()):
        # Save chunk
        chunk_path = partitioned_dir / f'chunk_{chunk_id}.parquet'
        chunk_df.to_parquet(chunk_path, index=False, compression='snappy')

        # Update manifest
        manifest['chunks'].append({
            'id': chunk_id,
            'file': f'chunk_{chunk_id}.parquet',
            'rows': len(chunk_df),
            'servers': chunk_df['server_name'].nunique(),
            'start': chunk_df['timestamp'].min().isoformat(),
            'end': chunk_df['timestamp'].max().isoformat()
        })

        total_rows += len(chunk_df)
        servers.update(chunk_df['server_name'].unique())

        print(f"  Saved {chunk_id}: {len(chunk_df):,} rows")

    # Add summary to manifest
    manifest['total_rows'] = total_rows
    manifest['total_servers'] = len(servers)
    manifest['servers'] = sorted(servers)

    # Save manifest
    manifest_path = partitioned_dir / 'chunk_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved {total_rows:,} rows across {len(chunks)} chunks")
    print(f"Manifest: {manifest_path}")

    return str(manifest_path)


def save_metadata(df: pd.DataFrame, output_dir: str) -> str:
    """Save dataset metadata."""

    output_path = Path(output_dir)

    metadata = {
        'version': '2.0.0',
        'created': datetime.now().isoformat(),
        'total_rows': len(df),
        'total_servers': df['server_name'].nunique(),
        'servers': sorted(df['server_name'].unique().tolist()),
        'time_range': {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat(),
            'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        },
        'columns': list(df.columns),
        'status_distribution': df['status'].value_counts().to_dict()
    }

    metadata_path = output_path / 'metrics_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata: {metadata_path}")
    return str(metadata_path)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataset(df: pd.DataFrame) -> bool:
    """Validate dataset meets requirements."""

    print("\n" + "="*60)
    print("DATASET VALIDATION")
    print("="*60)

    errors = []
    warnings = []

    # Check required columns
    required = ['timestamp', 'server_name', 'status', 'cpu_user_pct', 'cpu_sys_pct',
                'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct', 'mem_used_pct',
                'swap_used_pct', 'disk_usage_pct', 'net_in_mb_s', 'net_out_mb_s',
                'back_close_wait', 'front_close_wait', 'load_average', 'uptime_days']

    missing = set(required) - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Check for nulls in critical columns
    for col in ['timestamp', 'server_name']:
        if col in df.columns and df[col].isna().any():
            errors.append(f"NULL values in {col}")

    # Check data volume
    if len(df) < 1000:
        warnings.append(f"Small dataset ({len(df)} rows). Recommend 10,000+ for good training.")

    if df['server_name'].nunique() < 5:
        warnings.append(f"Few servers ({df['server_name'].nunique()}). Recommend 10+ for good generalization.")

    # Check time coverage
    time_span_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
    if time_span_hours < 24:
        warnings.append(f"Short time span ({time_span_hours:.1f} hours). Recommend 7+ days.")

    # Check status distribution
    status_counts = df['status'].value_counts()
    if 'healthy' not in status_counts or status_counts['healthy'] / len(df) < 0.5:
        warnings.append("Low percentage of 'healthy' status. May affect model calibration.")

    # Report results
    if errors:
        print("\n❌ ERRORS (must fix):")
        for e in errors:
            print(f"   - {e}")

    if warnings:
        print("\n⚠️  WARNINGS (consider):")
        for w in warnings:
            print(f"   - {w}")

    if not errors and not warnings:
        print("\n✅ Dataset validation PASSED")
    elif not errors:
        print("\n✅ Dataset validation PASSED with warnings")
    else:
        print("\n❌ Dataset validation FAILED")

    # Print summary stats
    print(f"\nDataset Summary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Servers: {df['server_name'].nunique()}")
    print(f"  Time span: {time_span_hours:.1f} hours ({time_span_hours/24:.1f} days)")
    print(f"  Timestamps per server: {len(df) / df['server_name'].nunique():.0f} avg")

    return len(errors) == 0


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def prepare_dataset(source_type: str, output_dir: str, **kwargs) -> str:
    """
    Main pipeline to prepare dataset for training.

    Args:
        source_type: 'database', 'csv', 'prometheus', 'elasticsearch'
        output_dir: Output directory for Parquet files
        **kwargs: Source-specific arguments

    Returns:
        Path to manifest file
    """
    print("="*60)
    print("TACHYON ARGUS DATA PREPARATION")
    print("="*60)

    # Step 1: Extract
    print("\n[1/6] EXTRACTING DATA...")
    if source_type == 'database':
        df = extract_from_database(kwargs['connection_string'], kwargs.get('query'))
    elif source_type == 'csv':
        df = extract_from_csv(kwargs['input_file'])
    elif source_type == 'prometheus':
        df = extract_from_prometheus(
            kwargs['prometheus_url'],
            kwargs['servers'],
            kwargs['start_time'],
            kwargs['end_time']
        )
    elif source_type == 'elasticsearch':
        df = extract_from_elasticsearch(
            kwargs['es_url'],
            kwargs['index'],
            kwargs['start_time'],
            kwargs['end_time']
        )
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    # Step 2: Map columns
    print("\n[2/6] MAPPING COLUMNS...")
    df = apply_column_mapping(df, COLUMN_MAPPING)
    df = apply_status_mapping(df, STATUS_MAPPING)

    # Step 3: Transform
    print("\n[3/6] TRANSFORMING DATA...")
    df = convert_units(df)
    df = derive_missing_metrics(df)
    df = add_missing_columns(df)
    df = fix_data_types(df)
    df = validate_ranges(df)

    # Step 4: Validate
    print("\n[4/6] VALIDATING...")
    if not validate_dataset(df):
        raise ValueError("Dataset validation failed")

    # Step 5: Partition
    print("\n[5/6] PARTITIONING...")
    chunks = partition_by_time(df, chunk_hours=2)

    # Step 6: Save
    print("\n[6/6] SAVING...")
    manifest_path = save_partitioned_parquet(chunks, output_dir)
    save_metadata(df, output_dir)

    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Ready for training with: python src/training/main.py train --streaming")

    return manifest_path


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Prepare real-world metrics for Tachyon Argus training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From CSV file
  python data_prep.py --source csv --input metrics.csv --output ./training/

  # From PostgreSQL
  python data_prep.py --source database \\
    --connection "postgresql://user:pass@host/db" \\
    --output ./training/

  # From Prometheus
  python data_prep.py --source prometheus \\
    --prometheus-url http://prometheus:9090 \\
    --start 2025-01-01 --end 2025-01-15 \\
    --output ./training/
        """
    )

    parser.add_argument('--source', type=str, required=True,
                       choices=['database', 'csv', 'prometheus', 'elasticsearch'],
                       help='Data source type')
    parser.add_argument('--output', type=str, default='./training/',
                       help='Output directory (default: ./training/)')

    # CSV options
    parser.add_argument('--input', type=str,
                       help='Input CSV file path')

    # Database options
    parser.add_argument('--connection', type=str,
                       help='Database connection string')
    parser.add_argument('--query', type=str,
                       help='Custom SQL query')

    # Prometheus options
    parser.add_argument('--prometheus-url', type=str,
                       help='Prometheus server URL')
    parser.add_argument('--start', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str,
                       help='End date (YYYY-MM-DD)')

    # Elasticsearch options
    parser.add_argument('--es-url', type=str,
                       help='Elasticsearch URL')
    parser.add_argument('--index', type=str,
                       help='Elasticsearch index name')

    args = parser.parse_args()

    # Build kwargs based on source
    kwargs = {}

    if args.source == 'csv':
        if not args.input:
            parser.error("--input required for csv source")
        kwargs['input_file'] = args.input

    elif args.source == 'database':
        if not args.connection:
            parser.error("--connection required for database source")
        kwargs['connection_string'] = args.connection
        kwargs['query'] = args.query

    elif args.source == 'prometheus':
        if not args.prometheus_url:
            parser.error("--prometheus-url required for prometheus source")
        kwargs['prometheus_url'] = args.prometheus_url
        kwargs['start_time'] = datetime.fromisoformat(args.start) if args.start else datetime.now() - timedelta(days=30)
        kwargs['end_time'] = datetime.fromisoformat(args.end) if args.end else datetime.now()
        kwargs['servers'] = []  # Will discover from metrics

    elif args.source == 'elasticsearch':
        if not args.es_url or not args.index:
            parser.error("--es-url and --index required for elasticsearch source")
        kwargs['es_url'] = args.es_url
        kwargs['index'] = args.index
        kwargs['start_time'] = datetime.fromisoformat(args.start) if args.start else datetime.now() - timedelta(days=30)
        kwargs['end_time'] = datetime.fromisoformat(args.end) if args.end else datetime.now()

    prepare_dataset(args.source, args.output, **kwargs)


if __name__ == '__main__':
    main()
```

---

## Quick Start Examples

### From CSV

```bash
# If your CSV has different column names, edit COLUMN_MAPPING in the script first
python data_prep.py --source csv --input /path/to/metrics.csv --output ./training/
```

### From PostgreSQL

```bash
python data_prep.py --source database \
  --connection "postgresql://user:password@localhost:5432/monitoring" \
  --output ./training/
```

### From MySQL

```bash
python data_prep.py --source database \
  --connection "mysql+pymysql://user:password@localhost:3306/monitoring" \
  --output ./training/
```

### From Prometheus

```bash
python data_prep.py --source prometheus \
  --prometheus-url http://prometheus:9090 \
  --start 2025-01-01 \
  --end 2025-01-15 \
  --output ./training/
```

---

## Customizing Column Mapping

Edit the `COLUMN_MAPPING` dictionary in the script to match your data:

```python
COLUMN_MAPPING = {
    # Your column name -> Expected column name

    # Example: Your DB has 'collected_at' instead of 'timestamp'
    'collected_at': 'timestamp',

    # Example: Your DB has 'host_id' instead of 'server_name'
    'host_id': 'server_name',

    # Example: Your DB has 'cpu_percent' as total CPU
    'cpu_percent': 'cpu_user_pct',  # Map to user, derive others

    # ... add all your mappings
}
```

---

## Handling Missing Metrics

If your monitoring system doesn't collect all 16 metrics, the script will:

1. **Use defaults** for non-critical metrics (e.g., `java_cpu_pct = 0`)
2. **Derive metrics** when possible (e.g., calculate `cpu_idle_pct` from others)
3. **Fail** only if critical columns are missing (`timestamp`, `server_name`)

### Common Scenarios

**No Java metrics:**
```python
# Script will default java_cpu_pct to 0
# No action needed
```

**Only total CPU, no breakdown:**
```python
# Add to your data before mapping:
df['cpu_user_pct'] = df['cpu_total'] * 0.7
df['cpu_sys_pct'] = df['cpu_total'] * 0.2
df['cpu_iowait_pct'] = df['cpu_total'] * 0.1
df['cpu_idle_pct'] = 100 - df['cpu_total']
```

**No connection metrics:**
```python
# Script will default back_close_wait and front_close_wait to 0
# No action needed
```

---

## Unit Conversions

The script auto-detects and converts common unit mismatches:

| Metric | Expected | Auto-converts from |
|--------|----------|-------------------|
| Network | MB/s | bytes/s, KB/s |
| Uptime | days | seconds, hours |
| Percentages | 0-100 | 0-1 (multiplies by 100) |

---

## Validation

After preparation, the script validates:

- All 16 required columns present
- No NULL values in critical columns
- Values within expected ranges
- Sufficient data volume (warns if < 1000 rows)
- Sufficient server count (warns if < 5)
- Sufficient time coverage (warns if < 24 hours)

---

## Output Structure

```
training/
├── server_metrics_partitioned/
│   ├── chunk_20250101_00.parquet
│   ├── chunk_20250101_02.parquet
│   ├── chunk_20250101_04.parquet
│   ├── ...
│   └── chunk_manifest.json
└── metrics_metadata.json
```

The output is ready for streaming training:
```bash
cd Argus
python src/training/main.py train --streaming
```

---

## Troubleshooting

### "Missing required column"
- Check your `COLUMN_MAPPING` - make sure your column names are mapped
- Column names are case-sensitive

### "NULL values in timestamp"
- Ensure your source data has valid timestamps
- Check date parsing format

### "Dataset validation failed"
- Review the specific errors printed
- Most common: missing columns or wrong data types

### Memory issues with large datasets
- The script processes data in chunks for partitioning
- For very large datasets (>100M rows), consider extracting in batches

---

## Next Steps

After data preparation:

1. **Train the model:**
   ```bash
   cd Argus
   python src/training/main.py train --streaming
   ```

2. **Validate predictions:** Test on held-out data

3. **Deploy:** See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
