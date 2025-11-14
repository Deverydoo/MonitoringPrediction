# ARCHITECTURE GUIDE
## Comprehensive System Architecture and Design Documentation

**Version:** 3.0.0
**Created:** 2025-11-14
**Status:** ⚠️ AUTHORITATIVE - Production Reference Documentation

---

## Table of Contents

1. [Overview](#1-overview)
   - [System Purpose](#system-purpose)
   - [Architecture Philosophy](#architecture-philosophy)
   - [Key Components](#key-components)
2. [System Architecture](#2-system-architecture)
   - [Microservices Design](#microservices-design)
   - [Process Architecture](#process-architecture)
   - [Communication Protocols](#communication-protocols)
   - [Data Flow Architecture](#data-flow-architecture)
3. [Data Contract & Schema](#3-data-contract--schema)
   - [Schema Definition](#schema-definition)
   - [NordIQ Metrics Framework](#nordiq-metrics-framework)
   - [State Contract](#state-contract)
   - [Schema Transformations](#schema-transformations)
   - [Validation Requirements](#validation-requirements)
4. [Adapter System Design](#4-adapter-system-design)
   - [Adapter Architecture](#adapter-architecture)
   - [How Adapters Work](#how-adapters-work)
   - [Adapter Daemon Loop](#adapter-daemon-loop)
   - [Server Name Encoding](#server-name-encoding)
   - [Authentication System](#authentication-system)
5. [GPU Configuration](#5-gpu-configuration)
   - [Supported GPUs](#supported-gpus)
   - [Auto-Detection](#auto-detection)
   - [Configuration Profiles](#configuration-profiles)
   - [Performance Optimization](#performance-optimization)
6. [Component Details](#6-component-details)
   - [Inference Daemon](#inference-daemon)
   - [Data Adapters](#data-adapters)
   - [Dashboard](#dashboard)
   - [Training Pipeline](#training-pipeline)
7. [Deployment](#7-deployment)
   - [Development Mode](#development-mode)
   - [Production Mode](#production-mode)
   - [Systemd Services](#systemd-services)
   - [Docker Deployment](#docker-deployment)
   - [Windows Services](#windows-services)
8. [Troubleshooting](#8-troubleshooting)
   - [Common Issues](#common-issues)
   - [Diagnostic Tools](#diagnostic-tools)
   - [FAQ](#faq)
9. [Reference](#9-reference)
   - [API Endpoints](#api-endpoints)
   - [File Structure](#file-structure)
   - [Version History](#version-history)

---

## 1. Overview

### System Purpose

The TFT (Temporal Fusion Transformer) Monitoring Prediction System is a production-grade infrastructure monitoring and predictive analytics platform built on a microservices architecture. It provides:

- **Real-time monitoring** of server infrastructure metrics
- **Predictive analytics** using deep learning (TFT models)
- **Proactive alerting** for potential system issues
- **Scalable architecture** supporting dynamic server fleets
- **Multi-source data integration** through adapter pattern

### Architecture Philosophy

The system follows several core architectural principles:

1. **Microservices Pattern**: Independent, loosely-coupled components
2. **Push-Based Data Flow**: Active data producers, passive data consumers
3. **API-First Design**: RESTful HTTP APIs for all inter-component communication
4. **Hardware Abstraction**: Automatic GPU detection and optimization
5. **Contract-Driven Development**: Immutable data contracts ensuring pipeline integrity

### Key Components

```
┌──────────────────────────────────────────────────────────┐
│                   SYSTEM COMPONENTS                      │
└──────────────────────────────────────────────────────────┘

┌─────────────────────┐      ┌─────────────────────┐
│  Data Adapters      │      │  Inference Daemon   │
│  (MongoDB/ES)       │─────▶│  (TFT Engine)       │
│  Port: None         │ POST │  Port: 8000         │
└─────────────────────┘      └─────────────────────┘
                                      │
                                      │ GET
                                      ▼
                             ┌─────────────────────┐
                             │  Dashboard          │
                             │  (Visualization)    │
                             │  Port: 8501         │
                             └─────────────────────┘
```

**Component Roles:**

- **Data Adapters**: Active data fetchers that query databases and push metrics
- **Inference Daemon**: Central processing engine that receives data and generates predictions
- **Dashboard**: Web-based visualization interface for real-time monitoring

---

## 2. System Architecture

### Microservices Design

#### Critical Concept: Independent Daemon Processes

**Adapters are NOT called by the inference daemon.**
**Adapters actively PUSH data to the inference daemon.**

This is a **microservices architecture** where each component runs independently and communicates via HTTP APIs.

#### Three Independent Processes

```
┌──────────────────────────────────────────────────────────────┐
│                  PRODUCTION ARCHITECTURE                     │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ Process 1: Data Source Adapter (MongoDB/Elasticsearch)  │
│ Port: None (HTTP client only)                           │
│ Role: Active Data Fetcher & Pusher                      │
└──────────────────────────────────────────────────────────┘
        │
        │ Every 5 seconds:
        │ 1. Fetch from database
        │ 2. Transform to TFT format
        │ 3. HTTP POST to /feed
        │
        ↓ HTTP POST /feed

┌──────────────────────────────────────────────────────────┐
│ Process 2: TFT Inference Daemon                         │
│ Port: 8000 (HTTP server)                                │
│ Role: Data Receiver, Prediction Generator               │
└──────────────────────────────────────────────────────────┘
        │
        │ On demand:
        │ Dashboard requests predictions
        │
        ↓ HTTP GET /predict

┌──────────────────────────────────────────────────────────┐
│ Process 3: Dash Dashboard                               │
│ Port: 8501 (web server)                                 │
│ Role: Visualization & User Interface                    │
└──────────────────────────────────────────────────────────┘
```

### Process Architecture

#### Process Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    PROCESS STATES                           │
└─────────────────────────────────────────────────────────────┘

1. Inference Daemon (Must start FIRST)
   ─────────────────────────────────────
   Start: python tft_inference_daemon.py --port 8000
   State: LISTENING on port 8000
   Waits: For /feed POST requests
   Critical: Must be running before adapter starts

2. Adapter Daemon (Start SECOND)
   ─────────────────────────────────────
   Start: python adapters/mongodb_adapter.py --daemon
   State: RUNNING continuous loop
   Action: Fetching → Transforming → POSTing
   Depends: Requires inference daemon at localhost:8000

3. Dashboard (Start THIRD)
   ─────────────────────────────────────
   Start: python dash_app.py
   State: WEB SERVER on port 8501
   Action: Fetching predictions every 30s
   Depends: Requires inference daemon at localhost:8000
```

#### Process Dependencies

```
Inference Daemon (port 8000)
    ↑
    ├── MongoDB Adapter (depends on daemon)
    ├── Elasticsearch Adapter (depends on daemon)
    └── Dashboard (depends on daemon)
```

**Critical:** Inference daemon MUST be running before starting adapter or dashboard.

#### Key Characteristics

| Characteristic | Description |
|----------------|-------------|
| **Active Fetcher** | Adapter actively queries database every N seconds |
| **Push-Based** | Adapter pushes data to inference daemon (not pulled) |
| **Stateful** | Tracks last fetch time to avoid duplicate data |
| **Independent Process** | Runs separately, can restart without affecting inference |
| **HTTP Client** | Makes HTTP POST requests to daemon's `/feed` endpoint |
| **No Port** | Doesn't listen on any port (client-only) |

### Communication Protocols

#### Communication Flow

```
MongoDB/Elasticsearch
    ↓ (query)
Data Adapter (Process 1)
    ↓ (HTTP POST to /feed)
TFT Inference Daemon (Process 2)
    ↓ (HTTP GET from /predict)
Dashboard (Process 3)
    ↓ (display)
User Browser
```

#### 1. Adapter → Inference Daemon (Push)

**Endpoint:** `POST /feed`

**Request:**
```http
POST http://localhost:8000/feed HTTP/1.1
Content-Type: application/json
X-API-Key: abc123def456...

[
  {
    "timestamp": "2025-10-17T12:00:00Z",
    "server_name": "ppml0001",
    "state": "healthy",
    "cpu_user_pct": 65.4,
    "cpu_sys_pct": 12.3,
    "cpu_iowait_pct": 2.1,
    "cpu_idle_pct": 20.2,
    "java_cpu_pct": 45.0,
    "mem_used_pct": 85.2,
    "swap_used_pct": 0.0,
    "disk_usage_pct": 45.0,
    "net_in_mb_s": 25.3,
    "net_out_mb_s": 18.7,
    "back_close_wait": 0,
    "front_close_wait": 0,
    "load_average": 8.5,
    "uptime_days": 42.0
  }
]
```

**Response:**
```json
{
  "status": "ok",
  "received": 20,
  "warmup_progress": 65.2
}
```

**Frequency:** Every 5 seconds (configurable via `--interval`)

#### 2. Dashboard → Inference Daemon (Pull)

**Endpoint:** `GET /predict`

**Request:**
```http
GET http://localhost:8000/predict HTTP/1.1
X-API-Key: abc123def456...
```

**Response:**
```json
{
  "predictions": {
    "ppml0001": {
      "current": {...},
      "predictions_30m": {...},
      "predictions_8h": {...},
      "risk_score": 58
    }
  },
  "timestamp": "2025-10-17T12:00:05Z"
}
```

**Frequency:** Every 30 seconds (configurable in dashboard)

### Data Flow Architecture

#### Minute-by-Minute Timeline Example

```
Time    | Adapter Action                      | Inference Daemon              | Dashboard
--------|-------------------------------------|-------------------------------|------------------
12:00:00| ─ Fetch metrics (12:00:00-11:59:55)| ─ Waiting for data           | ─ Shows old data
12:00:01| ─ Transform 20 records             | ─ Waiting                     | ─ Shows old data
12:00:02| ─ POST /feed (20 records)          | ✅ Received 20 records        | ─ Shows old data
        |                                     | ─ Add to warmup buffer       |
        |                                     | ─ Check if warmed up         |
12:00:03| ─ Sleep 5 seconds                  | ─ Ready for predictions      | ─ Shows old data
12:00:04| ─ Sleeping...                      | ─ Idle                        | ─ Shows old data
12:00:05| ─ Sleeping...                      | ─ Idle                        | ─ Sleeping...
12:00:06| ─ Sleeping...                      | ─ Idle                        | 🔄 Refresh (30s)
12:00:07| ─ Sleeping...                      | ─ Idle                        | ← GET /predict
        |                                     | ✅ Generate predictions      |
        |                                     | → Return predictions         |
12:00:08| ─ Wake up, start next fetch        | ─ Idle                        | ✅ Display new data
12:00:09| ─ Fetch metrics (12:00:08-12:00:03)| ─ Waiting for data           | ─ Shows new data
...     | (Repeat every 5 seconds)           | (Serves predictions on req)   | (Refreshes 30s)
```

---

## 3. Data Contract & Schema

### Schema Definition

#### Purpose

This defines the **immutable data contract** for the TFT Monitoring Prediction System. ALL components (data generation, training, inference) MUST conform to this specification. Any deviation will cause model loading failures and pipeline breaks.

**DO NOT modify schemas without updating this contract first.**

#### Version

**Contract Version:** 2.0.0 (NordIQ Metrics Framework)
**Breaking Change from v1.0.0:** All old metrics (cpu_pct, mem_pct, disk_io_mb_s, latency_ms) replaced with 14 NordIQ Metrics Framework production metrics.

### NordIQ Metrics Framework

#### Core Identification (3 columns)

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `timestamp` | datetime | ISO8601 timestamp | Any valid datetime |
| `server_name` | string | Unique hostname | e.g., `ppml0001`, `ppdb001` |
| `state` | string | Operational state | See State Contract below |

#### CPU Metrics (5 columns)

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `cpu_user_pct` | float | User space CPU | 0.0 - 100.0 |
| `cpu_sys_pct` | float | System/kernel CPU | 0.0 - 100.0 |
| `cpu_iowait_pct` | float | **I/O wait (CRITICAL)** | 0.0 - 100.0 |
| `cpu_idle_pct` | float | Idle CPU (% Used = 100 - idle) | 0.0 - 100.0 |
| `java_cpu_pct` | float | Java/Spark CPU usage | 0.0 - 100.0 |

#### Memory Metrics (2 columns)

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `mem_used_pct` | float | Memory utilization | 0.0 - 100.0 |
| `swap_used_pct` | float | Swap usage (thrashing indicator) | 0.0 - 100.0 |

#### Disk Metrics (1 column)

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `disk_usage_pct` | float | Disk space usage | 0.0 - 100.0 |

#### Network Metrics (2 columns)

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `net_in_mb_s` | float | Network ingress (MB/s) | 0.0+ |
| `net_out_mb_s` | float | Network egress (MB/s) | 0.0+ |

#### Connection Metrics (2 columns)

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `back_close_wait` | int | TCP backend connections | 0+ |
| `front_close_wait` | int | TCP frontend connections | 0+ |

#### System Metrics (2 columns)

| Column | Type | Description | Range/Values |
|--------|------|-------------|--------------|
| `load_average` | float | System load average | 0.0+ |
| `uptime_days` | int | Days since reboot | 0-365 |

**Total:** 3 core + 14 NordIQ Metrics Framework metrics = **17 required columns**

#### DEPRECATED Columns (DO NOT USE)

❌ `cpu_pct` - Replaced by cpu_user_pct, cpu_sys_pct, cpu_iowait_pct, cpu_idle_pct, java_cpu_pct
❌ `mem_pct` - Replaced by mem_used_pct, swap_used_pct
❌ `disk_io_mb_s` - Replaced by net_in_mb_s, net_out_mb_s
❌ `latency_ms` - Replaced by load_average

### State Contract

#### Valid States (IMMUTABLE)

**These 8 values MUST match across all pipeline stages:**

```python
VALID_STATES = [
    'critical_issue',  # Severe problems requiring immediate attention
    'healthy',         # Normal operational state
    'heavy_load',      # High utilization but stable
    'idle',           # Low activity baseline
    'maintenance',     # Scheduled maintenance mode
    'morning_spike',   # Peak usage periods (time-based)
    'offline',        # Server unavailable/unreachable
    'recovery'        # Post-incident recovery phase
]
```

#### State Determination Logic

```python
# Based on source metrics
if anomaly_score > 0.7 or cpu_pct > 95 or mem_pct > 95:
    state = 'critical_issue'
elif cpu_pct > 80 or mem_pct > 80:
    state = 'heavy_load'
elif cpu_pct < 5 and mem_pct < 10:
    state = 'idle'
elif is_business_hours and (9 <= hour <= 11):
    state = 'morning_spike'
# ... (see full logic in metrics_generator.py)
```

### Schema Transformations

#### 1. Source → Training Data

**File:** `metrics_generator.py`

```python
# Source (Production CSV/MongoDB)
Host Name       → server_name
Timestamp       → timestamp
State           → state (derived using state determination logic)
CPU metrics     → cpu_user_pct, cpu_sys_pct, cpu_iowait_pct, cpu_idle_pct, java_cpu_pct
Memory metrics  → mem_used_pct, swap_used_pct
Disk metrics    → disk_usage_pct
Network metrics → net_in_mb_s, net_out_mb_s
Connection      → back_close_wait, front_close_wait
System metrics  → load_average, uptime_days

# Additional training features
timestamp       → hour, day_of_week, month, is_weekend (temporal)
server_name     → server_id (encoded via hash)
```

**Output Format:** Parquet (required)
**Output Location:** `training/server_metrics.parquet`

#### 2. Training Data → Model

**File:** `tft_trainer.py`

```python
# TimeSeriesDataSet configuration
time_idx: Sequential integer (0, 1, 2, ...)
target: 'cpu_user_pct' (primary target)
group_ids: ['server_id']  # Encoded server names

# Time-varying unknown (to be predicted)
- cpu_user_pct, cpu_sys_pct, cpu_iowait_pct, cpu_idle_pct, java_cpu_pct
- mem_used_pct, swap_used_pct
- disk_usage_pct
- net_in_mb_s, net_out_mb_s
- back_close_wait, front_close_wait
- load_average

# Time-varying known (known in advance)
- hour, day_of_week, month, is_weekend, is_business_hours

# Categorical
- state (8 values - see State Contract)
- server_id (hash-encoded)

# Encoders (CRITICAL)
categorical_encoders = {
    'server_id': NaNLabelEncoder(add_nan=True),  # Allow unknown servers
    'state': NaNLabelEncoder(add_nan=True)       # Allow unknown states
}
```

#### 3. Model → Predictions

**File:** `tft_inference.py`

```python
# Input: Same schema as training
# Output: Predictions with decoded server names

{
    'server_name': 'ppml0001',  # Decoded from server_id
    'prediction_time': '2025-10-11T07:30:00',
    'predictions': {
        '30min': {'cpu_user_pct': 65.2, 'mem_used_pct': 72.1, ...},
        '1hr': {...},
        '8hr': {...}
    },
    'quantiles': {
        'p10': {...},  # Lower bound
        'p50': {...},  # Median
        'p90': {...}   # Upper bound
    }
}
```

### Validation Requirements

#### Data Validation (All Stages)

**Required validations before training/inference:**

```python
def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate DataFrame against data contract.

    Returns:
        (is_valid, list_of_errors)
    """
    errors = []

    # Required columns
    required_cols = [
        'timestamp', 'server_name', 'state',
        'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
        'mem_used_pct', 'swap_used_pct',
        'disk_usage_pct',
        'net_in_mb_s', 'net_out_mb_s',
        'back_close_wait', 'front_close_wait',
        'load_average', 'uptime_days'
    ]
    missing = set(required_cols) - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    # State values
    VALID_STATES = ['critical_issue', 'healthy', 'heavy_load', 'idle',
                    'maintenance', 'morning_spike', 'offline', 'recovery']
    invalid_states = set(df['state'].unique()) - set(VALID_STATES)
    if invalid_states:
        errors.append(f"Invalid states found: {invalid_states}")

    # Numeric ranges
    for col in ['cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct',
                'cpu_idle_pct', 'java_cpu_pct', 'mem_used_pct',
                'swap_used_pct', 'disk_usage_pct']:
        if (df[col] < 0).any() or (df[col] > 100).any():
            errors.append(f"{col} out of range [0, 100]")

    # Timestamp format
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        errors.append("timestamp must be datetime type")

    return (len(errors) == 0, errors)
```

#### Model Loading Validation

**Before loading model weights:**

```python
def validate_model_compatibility(model_dir: Path, data_df: pd.DataFrame) -> bool:
    """
    Verify model can load data without dimension mismatches.

    Checks:
    - State values count matches
    - Server mapping exists
    - Feature columns match
    """
    # Load training info
    with open(model_dir / 'training_info.json') as f:
        training_info = json.load(f)

    # Validate state count
    trained_states = training_info.get('unique_states', [])

    if set(trained_states) != set(VALID_STATES):
        print(f"[ERROR] Model trained with {len(trained_states)} states, "
              f"contract requires {len(VALID_STATES)}")
        return False

    # Validate server mapping exists
    if not (model_dir / 'server_mapping.json').exists():
        print("[ERROR] server_mapping.json not found in model directory")
        return False

    return True
```

---

## 4. Adapter System Design

### Adapter Architecture

#### Overview

Adapters are independent daemon processes that bridge production data sources (MongoDB, Elasticsearch) with the TFT inference system. They follow an active push pattern, continuously fetching and forwarding data.

#### Design Patterns

- **Active Fetcher**: Queries data source on regular intervals
- **Push-Based**: Sends data to inference daemon (not called by it)
- **Stateful**: Maintains fetch state to avoid duplicates
- **Independent**: Can restart without affecting inference daemon
- **Transformational**: Converts source schemas to TFT format

### How Adapters Work

#### Adapter Daemon Loop

```python
# adapters/mongodb_adapter.py - Main Loop

def run_daemon(self, interval: int = 5):
    """Run continuous streaming daemon."""

    logger.info("🚀 Starting MongoDB adapter daemon")
    logger.info(f"   Fetch interval: {interval} seconds")

    self.last_fetch_time = datetime.utcnow() - timedelta(seconds=interval)

    while True:  # ← Infinite loop
        # Step 1: Fetch new metrics from MongoDB
        metrics = self.fetch_recent_metrics(since=self.last_fetch_time)
        # Query: db.server_metrics.find({timestamp: {$gte: last_fetch_time}})

        if metrics:
            # Step 2: Transform MongoDB docs → TFT format
            records = self.transform_to_tft_format(metrics)
            # Converts field names, handles nested structures

            # Step 3: Forward to TFT daemon
            self.forward_to_tft_daemon(records)
            # HTTP POST http://localhost:8000/feed

            # Step 4: Update state
            self.last_fetch_time = latest_metric_timestamp

        # Step 5: Sleep until next interval
        time.sleep(interval)  # Default: 5 seconds
```

#### Adapter Configuration

Adapters are configured via JSON files:

```json
{
  "database": {
    "host": "mongodb.example.com",
    "port": 27017,
    "database": "monitoring",
    "collection": "server_metrics",
    "auth": {
      "username": "monitor_user",
      "password": "secure_password"
    }
  },
  "tft_daemon": {
    "url": "http://localhost:8000",
    "api_key": "loaded_from_env_or_explicit"
  },
  "fetch_interval": 5,
  "batch_size": 100
}
```

### Server Name Encoding

#### Problem Statement

Sequential integers (1, 2, 3...) break when servers are added/removed in production. Need deterministic, stable encoding.

#### Solution: Hash-Based Encoding

```python
import hashlib

def encode_server_name(server_name: str) -> str:
    """
    Create deterministic hash-based encoding for server names.

    Args:
        server_name: Original hostname (e.g., 'ppvra00a0018')

    Returns:
        Consistent numeric string ID (e.g., '12345')
    """
    # Use first 8 chars of SHA256 hash as numeric ID
    hash_obj = hashlib.sha256(server_name.encode('utf-8'))
    hash_int = int(hash_obj.hexdigest()[:8], 16)
    return str(hash_int % 1_000_000)  # Keep it reasonable for TFT

def create_server_mapping(server_names: list) -> dict:
    """
    Create bidirectional mapping for encoding/decoding.

    Returns:
        {
            'name_to_id': {'ppvra00a0018': '123456', ...},
            'id_to_name': {'123456': 'ppvra00a0018', ...}
        }
    """
    name_to_id = {name: encode_server_name(name) for name in server_names}
    id_to_name = {v: k for k, v in name_to_id.items()}
    return {'name_to_id': name_to_id, 'id_to_name': id_to_name}
```

#### Decoding (Inference)

```python
def decode_server_name(server_id: str, mapping: dict) -> str:
    """
    Decode server ID back to original name.

    Args:
        server_id: Encoded server ID
        mapping: Server mapping dict from create_server_mapping()

    Returns:
        Original server name or 'UNKNOWN_{id}' if not found
    """
    return mapping['id_to_name'].get(server_id, f'UNKNOWN_{server_id}')
```

#### Mapping Persistence

```python
# Save mapping during training
import json

mapping = create_server_mapping(df['server_name'].unique())
with open(f'{model_dir}/server_mapping.json', 'w') as f:
    json.dump(mapping, f, indent=2)

# Load mapping during inference
with open(f'{model_dir}/server_mapping.json', 'r') as f:
    server_mapping = json.load(f)
```

**Benefits:**
- ✅ Deterministic: Same server name → same ID every time
- ✅ Stable: Adding/removing servers doesn't affect existing IDs
- ✅ Reversible: Can decode predictions back to server names
- ✅ Production-ready: Handles dynamic server fleets

### Authentication System

#### How API Keys Work

1. Run `python generate_api_key.py` (done automatically by startup scripts)
2. API key is stored in `.env` file: `TFT_API_KEY=abc123...`
3. Adapters automatically load key from `.env` when they start
4. Every HTTP request includes: `X-API-Key: abc123...`
5. Daemon validates key before processing request

#### Priority Order (Adapters)

1. `api_key` field in adapter config file (explicit override)
2. `.env` file in project root (recommended - automatic)
3. `TFT_API_KEY` environment variable (fallback)

**Security:** API keys provide authentication to prevent unauthorized data injection or prediction access.

---

## 5. GPU Configuration

### Supported GPUs

#### Consumer/Workstation

**RTX 4090** (Ada Lovelace, SM 8.9)
- Batch size (train): 32
- Batch size (inference): 128
- Workers: 8
- Tensor Cores: medium precision

**RTX 3090** (Ampere, SM 8.6)
- Batch size (train): 32
- Batch size (inference): 128
- Workers: 8
- Tensor Cores: medium precision

#### Data Center - Current Generation

**Tesla V100** (Volta, SM 7.0)
- Batch size (train): 64
- Batch size (inference): 256
- Workers: 16
- Tensor Cores: high precision (enterprise reproducibility)

**Tesla A100** (Ampere, SM 8.0)
- Batch size (train): 128
- Batch size (inference): 512
- Workers: 32
- Tensor Cores: high precision + TF32 support

#### Data Center - Next Generation

**H100** (Hopper, SM 9.0)
- Batch size (train): 256
- Batch size (inference): 1024
- Workers: 32
- Tensor Cores: high precision + FP8 support

**H200** (Hopper HBM3e, SM 9.0)
- Batch size (train): 512
- Batch size (inference): 2048
- Workers: 32
- Tensor Cores: high precision + FP8 support
- 141GB HBM3e memory

### Auto-Detection

#### Detection Phase

```python
from gpu_profiles import setup_gpu

gpu = setup_gpu()
```

The system:
1. Detects GPU model name via `torch.cuda.get_device_name()`
2. Reads compute capability via `torch.cuda.get_device_capability()`
3. Matches to predefined profiles or falls back to compute capability ranges

### Configuration Profiles

#### Automatic Configuration

Automatically applies:

**Tensor Core Precision**: `torch.set_float32_matmul_precision('medium'|'high')`
- Consumer GPUs (RTX): `'medium'` - balance speed/precision
- Data Center GPUs (Tesla/H100): `'high'` - enterprise precision

**cuDNN Settings**:
- `cudnn.benchmark`: Auto-tune convolution algorithms (True for all)
- `cudnn.deterministic`: Reproducibility (False for consumer, True for enterprise)

**Memory Allocation**:
- Consumer: 85% reservation (leave headroom)
- Data Center: 90% reservation (maximize utilization)

**Batch Sizes**: GPU-specific optimal values

**DataLoader Workers**: CPU core allocation (8-32 depending on GPU class)

#### Usage in Inference

```python
class TFTInference:
    def __init__(self, model_path=None, use_real_model=True):
        # Auto-detect GPU and apply optimal profile
        if torch.cuda.is_available():
            self.gpu = setup_gpu()
            self.device = self.gpu.device
        else:
            self.gpu = None
            self.device = torch.device('cpu')

        # Batch sizes auto-selected
        batch_size = self.gpu.get_batch_size('inference') if self.gpu else 64
        num_workers = min(self.gpu.get_num_workers(), 4) if self.gpu else 0
```

#### Usage in Training

```python
class TFTTrainer:
    def __init__(self, config=None):
        # Auto-detect GPU and apply optimal profile
        if torch.cuda.is_available():
            self.gpu = setup_gpu()
            self.device = self.gpu.device
            # Use GPU-optimal batch size if not specified
            if 'batch_size' not in self.config:
                self.config['batch_size'] = self.gpu.get_batch_size('train')

        # Workers auto-configured
        if self.gpu:
            optimal_workers = self.gpu.get_num_workers()
        else:
            optimal_workers = 2
```

### Performance Optimization

#### Training Speed

- **RTX 4090**: ~20-30 min/epoch (90 servers, 720h data)
- **A100**: ~10-15 min/epoch (estimated 2x faster)
- **H100**: ~5-8 min/epoch (estimated 4x faster with FP8)

#### Inference Throughput

- **RTX 4090**: ~128 predictions/batch
- **A100**: ~512 predictions/batch
- **H100**: ~1024 predictions/batch
- **H200**: ~2048 predictions/batch

#### Fallback Behavior

If GPU is unknown:
1. Matches by compute capability range
2. Falls back to "Generic" profile with conservative settings:
   - Batch size (train): 16
   - Batch size (inference): 64
   - Workers: 4
   - Precision: `'highest'` (safest)

#### Example Output

**RTX 4090 (Consumer):**
```
[GPU] Detected: NVIDIA GeForce RTX 4090
[GPU] Compute Capability: SM 8.9
[GPU] Profile: RTX 4090
[GPU] Consumer/Workstation GPU - Ada Lovelace architecture
[GPU] Tensor Cores: Enabled (precision=medium)
[GPU] cuDNN: benchmark=True, deterministic=False
[GPU] Memory: 85% reserved
```

**H100 (Data Center):**
```
[GPU] Detected: NVIDIA H100
[GPU] Compute Capability: SM 9.0
[GPU] Profile: H100
[GPU] Next-gen Data Center GPU - Hopper architecture with FP8
[GPU] Tensor Cores: Enabled (precision=high)
[GPU] cuDNN: benchmark=True, deterministic=True
[GPU] Memory: 90% reserved
[GPU] Auto-configured batch size: 256
```

---

## 6. Component Details

### Inference Daemon

#### Purpose

Central processing engine that:
- Receives real-time metrics via `/feed` endpoint
- Maintains warmup buffer (288 data points per server)
- Generates predictions using TFT model
- Serves predictions via `/predict` endpoint

#### Key Features

- **Warmup Management**: Tracks data collection until prediction-ready
- **GPU Optimization**: Automatic GPU detection and configuration
- **Model Loading**: Loads pre-trained TFT models with validation
- **API Server**: FastAPI-based HTTP server on port 8000
- **State Management**: Maintains server state and prediction cache

#### Configuration

```python
# Command line
python tft_inference_daemon.py --port 8000 --model models/latest

# Environment variables
TFT_API_KEY=abc123...
CUDA_VISIBLE_DEVICES=0
```

### Data Adapters

#### MongoDB Adapter

**Purpose**: Fetch metrics from MongoDB collections

**Configuration**:
```json
{
  "database": {
    "host": "mongodb.prod.example.com",
    "port": 27017,
    "database": "monitoring",
    "collection": "server_metrics"
  }
}
```

**Query Pattern**:
```javascript
db.server_metrics.find({
  timestamp: {$gte: last_fetch_time}
}).sort({timestamp: 1})
```

#### Elasticsearch Adapter

**Purpose**: Fetch metrics from Elasticsearch indices

**Configuration**:
```json
{
  "elasticsearch": {
    "hosts": ["https://es.prod.example.com:9200"],
    "index_pattern": "monitoring-*",
    "time_field": "@timestamp"
  }
}
```

**Query Pattern**:
```json
{
  "query": {
    "range": {
      "@timestamp": {
        "gte": "2025-11-14T12:00:00Z"
      }
    }
  },
  "sort": [{"@timestamp": "asc"}]
}
```

#### Data Throughput

```
Servers × Metrics × Interval = Data Rate

Example:
- 50 servers
- 17 NordIQ Metrics Framework metrics each
- 5 second interval
- ~850 metrics every 5 seconds
- ~170 metrics/second
- ~420 KB/request (JSON)
```

**Performance:** Adapters easily handle 100+ servers at 5-second intervals.

### Dashboard

#### Purpose

Web-based visualization interface providing:
- Real-time server fleet status
- Predictive analytics charts
- Risk score indicators
- Historical trends
- Alert notifications

#### Technology Stack

- **Framework**: Dash (Plotly)
- **Port**: 8501
- **Refresh Rate**: 30 seconds (configurable)
- **Authentication**: API key via secrets.toml

#### Key Visualizations

- Fleet overview dashboard
- Per-server metric charts
- Prediction confidence intervals
- Risk heatmaps
- Time-series trends

### Training Pipeline

#### Purpose

Offline model training process that:
- Loads historical training data
- Configures TimeSeriesDataSet
- Trains TFT model with GPU optimization
- Saves model artifacts and mappings

#### Key Files

- `tft_trainer.py`: Main training orchestrator
- `metrics_generator.py`: Training data generator
- `gpu_profiles.py`: GPU auto-configuration

#### Training Process

1. **Data Loading**: Load parquet training data
2. **Validation**: Validate against data contract
3. **Dataset Creation**: Create TimeSeriesDataSet
4. **GPU Setup**: Auto-detect and configure GPU
5. **Model Training**: Train TFT with optimal hyperparameters
6. **Model Saving**: Save weights, config, and mappings

---

## 7. Deployment

### Development Mode

#### Manual Start (3 Terminals)

```bash
# Terminal 1: Inference Daemon (FIRST)
conda activate py310
python tft_inference_daemon.py --port 8000
# Wait for: ✅ Inference daemon started on port 8000

# Terminal 2: Adapter (SECOND)
conda activate py310
python adapters/mongodb_adapter.py --daemon --interval 5
# Wait for: ✅ Connected to MongoDB
# Wait for: ✅ Forwarded X records to TFT daemon

# Terminal 3: Dashboard (THIRD)
conda activate py310
python dash_app.py
# Open browser: http://localhost:8501
```

### Production Mode

#### Automated Script (Windows)

Create `start_all_production.bat`:

```batch
@echo off
echo ============================================
echo TFT Production System - Starting
echo ============================================
echo.

REM Load API key
for /f "usebackq tokens=1,2 delims==" %%a in (.env) do (
    if "%%a"=="TFT_API_KEY" set TFT_API_KEY=%%b
)

REM Step 1: Start Inference Daemon (CRITICAL FIRST)
echo [1/3] Starting Inference Daemon...
start "TFT Inference Daemon" cmd /k "conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python tft_inference_daemon.py --port 8000"

REM Wait for daemon to initialize
echo [INFO] Waiting for daemon to initialize (8 seconds)...
timeout /t 8 /nobreak >nul

REM Step 2: Start Adapter (SECOND)
echo [2/3] Starting MongoDB Adapter...
start "MongoDB Adapter" cmd /k "conda activate py310 && set TFT_API_KEY=%TFT_API_KEY% && python adapters\mongodb_adapter.py --daemon --config adapters\mongodb_adapter_config.json"

REM Wait for adapter to connect
echo [INFO] Waiting for adapter to connect (3 seconds)...
timeout /t 3 /nobreak >nul

REM Step 3: Start Dashboard (THIRD)
echo [3/3] Starting Dashboard...
start "TFT Dashboard" cmd /k "conda activate py310 && python dash_app.py --server.fileWatcherType none --server.runOnSave false"

echo.
echo ============================================
echo System Started!
echo ============================================
echo Inference Daemon:   http://localhost:8000
echo MongoDB Adapter:    Streaming from production
echo Dashboard:          http://localhost:8501
echo.
pause
```

#### Automated Script (Linux)

Create `start_all_production.sh`:

```bash
#!/bin/bash
echo "============================================"
echo "TFT Production System - Starting"
echo "============================================"
echo ""

# Load API key from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Step 1: Start Inference Daemon (CRITICAL FIRST)
echo "[1/3] Starting Inference Daemon..."
gnome-terminal -- bash -c "conda activate py310 && export TFT_API_KEY='$TFT_API_KEY' && python tft_inference_daemon.py --port 8000; exec bash"

# Wait for daemon to initialize
echo "[INFO] Waiting for daemon to initialize (8 seconds)..."
sleep 8

# Step 2: Start Adapter (SECOND)
echo "[2/3] Starting MongoDB Adapter..."
gnome-terminal -- bash -c "conda activate py310 && export TFT_API_KEY='$TFT_API_KEY' && python adapters/mongodb_adapter.py --daemon --config adapters/mongodb_adapter_config.json; exec bash"

# Wait for adapter to connect
echo "[INFO] Waiting for adapter to connect (3 seconds)..."
sleep 3

# Step 3: Start Dashboard (THIRD)
echo "[3/3] Starting Dashboard..."
gnome-terminal -- bash -c "conda activate py310 && python dash_app.py --server.fileWatcherType none --server.runOnSave false; exec bash"

echo ""
echo "============================================"
echo "System Started!"
echo "============================================"
echo "Inference Daemon:   http://localhost:8000"
echo "MongoDB Adapter:    Streaming from production"
echo "Dashboard:          http://localhost:8501"
```

### Systemd Services

#### 1. Inference Daemon Service

`/etc/systemd/system/tft-inference-daemon.service`:

```ini
[Unit]
Description=TFT Inference Daemon
After=network.target
Documentation=https://github.com/yourorg/tft-monitoring

[Service]
Type=simple
User=tft
Group=tft
WorkingDirectory=/opt/tft-monitoring
EnvironmentFile=/etc/tft/.env
ExecStart=/opt/tft-monitoring/venv/bin/python tft_inference_daemon.py --port 8000
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### 2. MongoDB Adapter Service

`/etc/systemd/system/tft-mongodb-adapter.service`:

```ini
[Unit]
Description=TFT MongoDB Adapter
After=network.target tft-inference-daemon.service mongodb.service
Requires=tft-inference-daemon.service
Documentation=https://github.com/yourorg/tft-monitoring

[Service]
Type=simple
User=tft
Group=tft
WorkingDirectory=/opt/tft-monitoring/adapters
EnvironmentFile=/etc/tft/.env
ExecStart=/opt/tft-monitoring/venv/bin/python mongodb_adapter.py --daemon --config /etc/tft/mongodb_adapter_config.json
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### 3. Dashboard Service

`/etc/systemd/system/tft-dashboard.service`:

```ini
[Unit]
Description=TFT Dash Dashboard
After=network.target tft-inference-daemon.service
Requires=tft-inference-daemon.service
Documentation=https://github.com/yourorg/tft-monitoring

[Service]
Type=simple
User=tft
Group=tft
WorkingDirectory=/opt/tft-monitoring
ExecStart=/opt/tft-monitoring/venv/bin/python dash_app.py --server.fileWatcherType none --server.runOnSave false --server.port 8501
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### Service Management

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable tft-inference-daemon
sudo systemctl enable tft-mongodb-adapter
sudo systemctl enable tft-dashboard

# Start services (in order)
sudo systemctl start tft-inference-daemon
sleep 5
sudo systemctl start tft-mongodb-adapter
sleep 3
sudo systemctl start tft-dashboard

# Check status
sudo systemctl status tft-inference-daemon
sudo systemctl status tft-mongodb-adapter
sudo systemctl status tft-dashboard

# View logs
journalctl -u tft-inference-daemon -f
journalctl -u tft-mongodb-adapter -f
journalctl -u tft-dashboard -f
```

### Docker Deployment

#### Docker Compose

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  # Service 1: Inference Daemon (must start first)
  tft-inference-daemon:
    image: tft-monitoring:latest
    container_name: tft-inference-daemon
    command: python tft_inference_daemon.py --port 8000
    ports:
      - "8000:8000"
    environment:
      - TFT_API_KEY=${TFT_API_KEY}
    volumes:
      - ./models:/app/models
      - ./training:/app/training
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Service 2: MongoDB Adapter (depends on daemon)
  tft-mongodb-adapter:
    image: tft-monitoring:latest
    container_name: tft-mongodb-adapter
    command: python adapters/mongodb_adapter.py --daemon --config /config/mongodb_adapter_config.json
    depends_on:
      tft-inference-daemon:
        condition: service_healthy
    environment:
      - TFT_API_KEY=${TFT_API_KEY}
    volumes:
      - ./adapters/mongodb_adapter_config.json:/config/mongodb_adapter_config.json:ro
    restart: unless-stopped

  # Service 3: Dashboard (depends on daemon)
  tft-dashboard:
    image: tft-monitoring:latest
    container_name: tft-dashboard
    command: python dash_app.py --server.fileWatcherType none --server.runOnSave false --server.port 8501
    ports:
      - "8501:8501"
    depends_on:
      - tft-inference-daemon
    environment:
      - TFT_API_KEY=${TFT_API_KEY}
    restart: unless-stopped

networks:
  default:
    name: tft-network
```

**Start with Docker Compose:**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Stop all services
docker-compose down
```

### Windows Services

#### Using NSSM

```batch
REM Download NSSM from https://nssm.cc/

REM Service 1: Inference Daemon
nssm install TFTInferenceDaemon ^
  "C:\Python310\python.exe" ^
  "D:\tft-monitoring\tft_inference_daemon.py" ^
  --port 8000

nssm set TFTInferenceDaemon AppDirectory D:\tft-monitoring
nssm set TFTInferenceDaemon AppEnvironmentExtra TFT_API_KEY=your-api-key

REM Service 2: MongoDB Adapter (depends on daemon)
nssm install TFTMongoDBAdapter ^
  "C:\Python310\python.exe" ^
  "D:\tft-monitoring\adapters\mongodb_adapter.py" ^
  --daemon --config D:\tft-monitoring\adapters\mongodb_adapter_config.json

nssm set TFTMongoDBAdapter AppDirectory D:\tft-monitoring\adapters
nssm set TFTMongoDBAdapter DependOnService TFTInferenceDaemon

REM Service 3: Dashboard (depends on daemon)
nssm install TFTDashboard ^
  "C:\Python310\python.exe" ^
  "D:\tft-monitoring\dash_app.py"

nssm set TFTDashboard AppDirectory D:\tft-monitoring
nssm set TFTDashboard DependOnService TFTInferenceDaemon

REM Start services (in order)
nssm start TFTInferenceDaemon
timeout /t 5 /nobreak
nssm start TFTMongoDBAdapter
timeout /t 3 /nobreak
nssm start TFTDashboard
```

---

## 8. Troubleshooting

### Common Issues

#### Problem 1: "Adapter can't connect to inference daemon"

**Symptoms:**
```
❌ Error forwarding to TFT daemon: Connection refused
```

**Diagnosis:**
```bash
# Check if inference daemon is running
curl http://localhost:8000/health

# Check if port 8000 is listening
netstat -an | grep 8000        # Linux
netstat -an | findstr 8000     # Windows
```

**Solution:**
```bash
# Start inference daemon FIRST
python tft_inference_daemon.py --port 8000

# Wait 5 seconds, then start adapter
python adapters/mongodb_adapter.py --daemon
```

#### Problem 2: "Dashboard shows 0/0 servers"

**Symptoms:**
- Dashboard shows "Environment Status: Unknown"
- Fleet Status: 0/0 servers

**Diagnosis:**
```bash
# Check adapter logs
# Should show: ✅ Forwarded X records to TFT daemon

# Check inference daemon logs
# Should show: [FEED] Received X metrics

# Check dashboard connection
curl http://localhost:8000/predict -H "X-API-Key: your-key"
```

**Solution:**
1. Verify adapter is running and forwarding data
2. Check API key matches in adapter config and .env
3. Wait for warmup (288 data points per server needed)

#### Problem 3: "Adapter fetches duplicate data"

**Symptoms:**
```
📊 Fetched 1000 metrics (same timestamps repeated)
```

**Diagnosis:**
```bash
# Check adapter state tracking
# Adapter should track last_fetch_time

# Check database has new data
mongo monitoring --eval "db.server_metrics.find().sort({timestamp: -1}).limit(1)"
```

**Solution:**
- Adapter tracks state internally (last_fetch_time)
- Only fetches data newer than last fetch
- If duplicate data appears, check database clock sync

#### Problem 4: "Authentication failed" or "403 Forbidden"

**Symptoms:**
```
❌ TFT daemon error: 403 Forbidden
⚠️ No API key configured - daemon may reject request
```

**Diagnosis:**
```bash
# Step 1: Check if .env file exists
cat .env | grep TFT_API_KEY
# Should show: TFT_API_KEY=abc123...

# Step 2: Check if adapter is loading the key
python adapters/mongodb_adapter.py --once --verbose
# Should see: ✅ Loaded API key from .env

# Step 3: Test daemon authentication
curl -X POST http://localhost:8000/health \
  -H "X-API-Key: $(grep TFT_API_KEY .env | cut -d= -f2)"
# Should return: {"status": "ok"}
```

**Solution:**
```bash
# Generate API key if missing
python generate_api_key.py

# Run adapter from project root directory
cd /path/to/MonitoringPrediction
python adapters/mongodb_adapter.py --daemon

# Or set environment variable explicitly
export TFT_API_KEY=$(grep TFT_API_KEY .env | cut -d= -f2)
python adapters/mongodb_adapter.py --daemon
```

#### Problem 5: "Process crashes after system restart"

**Symptoms:**
- One or more processes not running after reboot

**Diagnosis:**
```bash
# Check systemd service status
sudo systemctl status tft-inference-daemon
sudo systemctl status tft-mongodb-adapter

# Check logs
journalctl -u tft-inference-daemon -n 50
```

**Solution:**
```bash
# Ensure services are enabled
sudo systemctl enable tft-inference-daemon
sudo systemctl enable tft-mongodb-adapter
sudo systemctl enable tft-dashboard

# Check dependencies are correct
# Adapter should have: After=tft-inference-daemon.service
```

### Diagnostic Tools

#### Checking Running Processes

```bash
# Linux
ps aux | grep -E "(tft_inference|mongodb_adapter|elasticsearch_adapter|dash)"

# Windows
tasklist | findstr /I "python"

# Expected output:
# python tft_inference_daemon.py --port 8000
# python mongodb_adapter.py --daemon --interval 5
# python dash_app.py
```

#### Health Checks

```bash
# Inference daemon health
curl http://localhost:8000/health

# Dashboard health
curl http://localhost:8501

# Check API connectivity
curl -X GET http://localhost:8000/predict \
  -H "X-API-Key: your-api-key-here"
```

### FAQ

#### Q: Can I run multiple adapters at the same time?

**A:** Yes! You can run multiple adapters simultaneously:

```bash
# Terminal 1: MongoDB adapter
python adapters/mongodb_adapter.py --daemon

# Terminal 2: Elasticsearch adapter
python adapters/elasticsearch_adapter.py --daemon
```

Both will POST to the same `/feed` endpoint. This is useful if you have metrics in multiple data sources.

#### Q: What happens if the adapter crashes?

**A:** Inference daemon continues working:

- ✅ Inference daemon keeps serving predictions (using cached data)
- ✅ Dashboard continues displaying last known predictions
- ⚠️ No new data arrives until adapter restarts
- ✅ When adapter restarts, it fetches missed data (state tracked)

**Use systemd/Docker restart policies to auto-restart crashed adapters.**

#### Q: What happens if inference daemon crashes?

**A:** Adapter and dashboard lose connectivity:

- ❌ Adapter gets "Connection refused" errors (retries automatically)
- ❌ Dashboard shows "Daemon not connected"
- ✅ When daemon restarts, both reconnect automatically
- ⚠️ Warmup period required (288 data points per server)

**Critical: Inference daemon is the core component.**

#### Q: Can I change the fetch interval while running?

**A:** No, restart required:

```bash
# Stop adapter
Ctrl+C  # or: sudo systemctl stop tft-mongodb-adapter

# Restart with new interval
python adapters/mongodb_adapter.py --daemon --interval 10
# or: sudo systemctl start tft-mongodb-adapter
```

#### Q: How do I know which data source is feeding the daemon?

**A:** Check inference daemon logs:

```bash
# With metrics generator (simulated)
[FEED] Received 20 metrics from 127.0.0.1
[FEED] Server names: ppml0001, ppml0002...

# With MongoDB adapter (production)
[FEED] Received 47 metrics from 127.0.0.1
[FEED] Server names: prod-ml-01, prod-db-03...
```

Or check adapter logs:
```bash
✅ Forwarded 47 records to TFT daemon
```

#### Q: Can adapter and inference daemon run on different machines?

**A:** Yes! Configure adapter to point to remote daemon:

```json
{
  "tft_daemon": {
    "url": "http://inference-server.example.com:8000",
    "api_key": "your-api-key"
  }
}
```

**Requirements:**
- Network connectivity between machines
- Firewall allows port 8000
- API key authentication configured

#### Q: Do I need both adapter AND metrics generator?

**A:** No, choose one:

| Scenario | Use This |
|----------|----------|
| **Development/Testing** | `metrics_generator_daemon.py` (simulated data) |
| **Production** | `mongodb_adapter.py` or `elasticsearch_adapter.py` (real data) |
| **Hybrid** | Run both (mix simulated + real data) |

**Same `/feed` endpoint - daemon doesn't care about data source.**

#### Q: How much data does the adapter send per interval?

**A:** Depends on your fleet size:

```
Servers × Metrics × Interval = Data Rate

Example:
- 50 servers
- 17 NordIQ Metrics Framework metrics each
- 5 second interval
- ~850 metrics every 5 seconds
- ~170 metrics/second
- ~420 KB/request (JSON)
```

**Performance:** Adapters easily handle 100+ servers at 5-second intervals.

---

## 9. Reference

### API Endpoints

#### Inference Daemon (Port 8000)

**POST /feed**
- Purpose: Receive metrics from adapters
- Authentication: X-API-Key header
- Payload: Array of metric objects
- Response: Status and warmup progress

**GET /predict**
- Purpose: Get predictions for all servers
- Authentication: X-API-Key header
- Response: Predictions with risk scores

**GET /health**
- Purpose: Health check endpoint
- Authentication: None
- Response: System status

### File Structure

#### Training Data Output
```
training/
├── server_metrics.parquet        # Main training data
├── metrics_metadata.json         # Generation metadata
└── server_mapping.json           # Server name mappings
```

#### Model Output
```
models/tft_model_YYYYMMDD_HHMMSS/
├── model.safetensors             # Model weights
├── config.json                   # Model architecture config
├── training_info.json            # Training metadata
├── server_mapping.json           # Server name mappings (REQUIRED)
└── data_contract_version.txt     # Contract version used
```

#### Adapter Configuration
```
adapters/
├── mongodb_adapter.py
├── mongodb_adapter_config.json
├── elasticsearch_adapter.py
└── elasticsearch_adapter_config.json
```

### Version History

| Version | Date | Changes |
|---------|------|---------|
| 3.0.0 | 2025-11-14 | Merged architecture documents into comprehensive guide |
| 2.0.0 | 2025-10-14 | NordIQ Metrics Framework (14 production metrics) |
| 1.0.0 | 2025-10-11 | Initial data contract and adapter architecture |

---

## Summary

### Key Takeaways

1. ✅ **Microservices Architecture**: Independent daemon processes communicate via HTTP
2. ✅ **Push-Based Data Flow**: Adapters actively push data to inference daemon
3. ✅ **Contract-Driven**: Immutable data contracts ensure pipeline integrity
4. ✅ **GPU Optimization**: Automatic detection and configuration for optimal performance
5. ✅ **Production-Ready**: Systemd, Docker, and Windows service support
6. ✅ **Scalable**: Handles 100+ servers with 5-second update intervals
7. ✅ **Maintainable**: Clear separation of concerns, independent restart capability

### Architecture Pattern

```
Data Source → Adapter (fetcher) → Inference Daemon (processor) → Dashboard (viewer)
  (passive)     (active)              (server)                      (client)
```

**Remember:** Adapters are the "active" component that drives the data flow!

---

**Built by Craig Giannelli and Claude Code**

---

**Document Version:** 3.0.0
**Last Updated:** 2025-11-14
**Status:** ✅ Production Critical Documentation
