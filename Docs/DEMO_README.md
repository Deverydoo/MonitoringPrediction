# TFT Monitoring Dashboard - Demo Guide

## Overview

The refactored dashboard system provides reproducible, explainable monitoring demonstrations. No more random data generation - all results are based on pre-generated datasets with predictable incident patterns.

## Key Changes

### What Changed
- **Removed**: Random data generation from `tft_dashboard.py`
- **Added**: File-based data source reader (`DataSource` class)
- **Added**: Demo data generator with predictable incident patterns
- **Added**: Simple runner script for quick demos

### Why These Changes
1. **Reproducibility**: Same data = same results every time
2. **Explainability**: Know exactly what the data represents
3. **Production Ready**: Can now read from real production data sources
4. **Demo Friendly**: Predictable incident patterns make great presentations

## Quick Start

### Option 1: One-Command Demo
```bash
python run_demo.py
```

This will:
1. Generate demo data (if not already present)
2. Launch the dashboard
3. Play through the 5-minute incident scenario

### Option 2: Step-by-Step

#### Step 1: Generate Demo Data
```bash
python demo_data_generator.py --output-dir ./demo_data --duration 5 --fleet-size 10
```

Output:
- `./demo_data/demo_dataset.csv` - CSV format
- `./demo_data/demo_dataset.parquet` - Parquet format (faster)
- `./demo_data/demo_dataset_metadata.json` - Metadata and incident info

#### Step 2: Run Dashboard
```bash
python tft_dashboard_refactored.py ./demo_data/demo_dataset.parquet --refresh 5
```

## Demo Data Structure

### Incident Timeline (5 minutes)
```
0:00 - 1:30  Stable Baseline
             └─ All metrics healthy
             └─ Green health status

1:30 - 2:30  Escalation Phase
             └─ Gradual increase in CPU, latency
             └─ Warning thresholds approached
             └─ 30-min probability increases

2:30 - 3:30  Peak Incident
             └─ CPU spikes to 80-90%
             └─ Latency increases significantly
             └─ Critical alerts triggered
             └─ Probability reaches 70-90%

3:30 - 5:00  Recovery
             └─ Gradual return to normal
             └─ Metrics stabilize
             └─ Probability decreases back to baseline
```

### Affected Servers
By default, the first server of each profile type is affected:
- `web-001` (web profile)
- `api-001` (api profile)
- `db-001` (database profile)
- `cache-001` (cache profile)

## Dashboard Features

### Figure 1: KPI Header
- **Environment Health**: GOOD / WARNING / CRITICAL
- **Current Phase**: Shows which phase of the incident you're in
- **Incident Probability**: 30-minute and 8-hour predictions
- **Fleet Status**: Active vs total servers
- **Progress**: How far through the demo data

### Figure 2: Top 5 Problem Servers
- Bar chart of highest risk servers
- Color coded: Red (>0.7), Orange (>0.4), Yellow (>0.2)
- Shows server profile and current state

### Figure 3: Probability Trend
- Time series of incident probability
- Shows how predictions change over time
- Vertical lines mark phase boundaries

### Figure 4: Fleet Risk Heat Map
- Color-coded risk strip for entire fleet
- Green = healthy, Yellow = warning, Red = high risk
- Shows at-a-glance fleet health

### Figure 5: Rolling Metrics
- CPU usage (median across fleet)
- Latency P95
- Error rate (mean)
- Shows the actual metrics driving predictions

## Configuration

### Demo Data Generator

```python
from demo_data_generator import generate_demo_dataset

generate_demo_dataset(
    output_dir="./my_demo/",
    duration_minutes=10,      # Longer demo
    fleet_size=20,            # More servers
    seed=123                  # Different random pattern
)
```

### Dashboard

```python
from tft_dashboard_refactored import run_dashboard

run_dashboard(
    data_path="./demo_data/demo_dataset.parquet",
    data_format="parquet",
    refresh_sec=3,           # Faster refresh
    save_plots=True          # Save figures to disk
)
```

## Production Usage

### Reading from Production Data

The dashboard can read from any CSV or Parquet file with the expected schema:

```python
from tft_dashboard_refactored import DataSource, LiveDashboard

# Point to your production data
data_source = DataSource(
    data_path="/path/to/production/metrics.parquet",
    data_format="parquet"
)

dashboard = LiveDashboard(data_source)
dashboard.run()
```

### Required Data Schema

Your data file must include these columns:
- `timestamp` - ISO format timestamp
- `server_name` - Server identifier
- `profile` - Server profile/type
- `state` - Server state (online, warning, critical_issue, offline)
- `cpu_pct` - CPU percentage (0-100)
- `mem_pct` - Memory percentage (0-100)
- `latency_ms` - Latency in milliseconds
- `error_rate` - Error rate (0-1)
- `disk_io_mb_s` - Disk I/O in MB/s
- `net_in_mb_s` - Network in MB/s
- `net_out_mb_s` - Network out MB/s
- `gc_pause_ms` - GC pause time in milliseconds

Optional columns:
- `incident_phase` - Current incident phase (for demo data)
- `problem_child` - Boolean flag for affected servers
- `container_oom` - OOM event flag

## Command Line Options

### demo_data_generator.py
```bash
python demo_data_generator.py [OPTIONS]

Options:
  --output-dir DIR      Output directory (default: ./demo_data/)
  --duration MINUTES    Duration in minutes (default: 5)
  --fleet-size N        Number of servers (default: 10)
  --seed N              Random seed (default: 42)
  --filename NAME       Base filename (default: demo_dataset)
```

### tft_dashboard_refactored.py
```bash
python tft_dashboard_refactored.py DATA_PATH [OPTIONS]

Arguments:
  DATA_PATH             Path to data file or directory

Options:
  --format FORMAT       Data format: csv, parquet, or auto (default: auto)
  --refresh SECONDS     Refresh interval (default: 5)
  --save-plots          Save plots to disk
```

### run_demo.py
```bash
python run_demo.py [OPTIONS]

Options:
  --duration MINUTES    Demo duration (default: 5)
  --fleet-size N        Number of servers (default: 10)
  --seed N              Random seed (default: 42)
  --refresh SECONDS     Dashboard refresh interval (default: 5)
  --output-dir DIR      Directory for demo data (default: ./demo_data/)
  --regenerate          Force regeneration of demo data
```

## Jupyter Notebook Usage

```python
# In Jupyter notebook
from demo_data_generator import generate_demo_dataset
from tft_dashboard_refactored import run_dashboard

# Generate demo data
generate_demo_dataset(
    output_dir="./demo_data/",
    duration_minutes=5,
    fleet_size=10
)

# Run dashboard (will display inline)
run_dashboard(
    data_path="./demo_data/demo_dataset.parquet",
    refresh_sec=5
)
```

## Troubleshooting

### Issue: "Data source not found"
**Solution**: Make sure you've generated the demo data first:
```bash
python demo_data_generator.py
```

### Issue: "No module named 'pyarrow'"
**Solution**: Install pyarrow for Parquet support:
```bash
pip install pyarrow
```
Alternatively, use CSV format:
```bash
python tft_dashboard_refactored.py ./demo_data/demo_dataset.csv --format csv
```

### Issue: Dashboard runs too fast/slow
**Solution**: Adjust the playback speed by modifying the sleep time in `LiveDashboard.run()` or use `--refresh` to control display refresh rate.

### Issue: Want different incident patterns
**Solution**: Modify the phase boundaries in `demo_data_generator.py`:
```python
self.phase_boundaries = {
    'stable': (0, 120),      # Longer stable period
    'escalation': (120, 180),
    'peak': (180, 240),
    'recovery': (240, 360)   # Longer recovery
}
```

## File Reference

### New Files
- `demo_data_generator.py` - Reproducible demo data generator
- `tft_dashboard_refactored.py` - File-based dashboard (replaces random generation)
- `run_demo.py` - Convenient demo runner script
- `DEMO_README.md` - This file

### Original Files (Still Present)
- `tft_dashboard.py` - Original dashboard with random generation (kept for reference)
- `metrics_generator.py` - Full-scale data generator for training
- `tft_trainer.py` - Model training
- `tft_inference.py` - Model inference

## Next Steps

1. **Run the demo** to see the incident pattern
2. **Modify the generator** to create different scenarios
3. **Connect to production data** using the same dashboard
4. **Train a real model** on your production data
5. **Replace ModelAdapter** with your trained TFT model for real predictions

## Architecture

```
┌─────────────────────┐
│  Demo Data Gen      │──> demo_dataset.parquet
│  (reproducible)     │
└─────────────────────┘

┌─────────────────────┐
│  Production System  │──> production_metrics.parquet
│  (real data)        │
└─────────────────────┘
          │
          ├─────────────────────────┐
          │                         │
          ▼                         ▼
    ┌──────────┐            ┌──────────────┐
    │ DataSource│            │ ModelAdapter │
    │ (reader) │            │ (predictions)│
    └──────────┘            └──────────────┘
          │                         │
          └────────┬────────────────┘
                   ▼
            ┌─────────────┐
            │ Dashboard   │
            │ (visualize) │
            └─────────────┘
```

## Support

For questions or issues:
1. Check the metadata file: `demo_dataset_metadata.json`
2. Review the dashboard output logs
3. Verify data schema matches requirements
4. Check Python dependencies are installed
