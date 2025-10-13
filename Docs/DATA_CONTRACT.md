# DATA CONTRACT - Single Source of Truth

**Version:** 1.0.0
**Created:** 2025-10-11
**Status:** ⚠️ AUTHORITATIVE - All code must conform to this contract

---

## 🎯 Purpose

This document defines the **immutable data contract** for the TFT Monitoring Prediction System. ALL components (data generation, training, inference) MUST conform to this specification. Any deviation will cause model loading failures and pipeline breaks.

**DO NOT modify schemas without updating this document first.**

---

## 📊 Source Data Reference

### Production Monitoring System (AIMLP OSDS 2.0)
- **Source:** `Docs/aimlp_server_metrics.csv`
- **Description:** `Docs/metrics decription.txt`
- **Server Pattern:** `ppvra00a00XX` (production servers)

---

## 🔑 Core Schema Definition

### Required Columns (All Stages)

| Column | Type | Description | Range/Values | Source Field |
|--------|------|-------------|--------------|--------------|
| `timestamp` | datetime | ISO8601 timestamp | Any valid datetime | Timestamp |
| `server_name` | string | Unique hostname | e.g., `ppvra00a0018` | Host Name |
| `cpu_pct` | float | CPU utilization percentage | 0.0 - 100.0 | Derived from %CPU columns |
| `mem_pct` | float | Memory utilization percentage | 0.0 - 100.0 | %Mem Used |
| `disk_io_mb_s` | float | Disk I/O throughput | 0.0+ | Mb/s In + Mb/s Out |
| `latency_ms` | float | Network/system latency | 0.0+ | Derived metric |
| `state` | string | Operational state | See State Contract below | Derived from metrics |

### State Contract (IMMUTABLE)

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

**State Determination Logic:**
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

---

## 🏗️ Server Name Encoding Strategy

### Problem Statement
Current approach uses sequential integers (1, 2, 3...) which breaks when servers are added/removed in production.

### Solution: Hash-Based Encoding with Mapping Table

#### Encoding (Training/Generation)
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

---

## 📐 Schema Transformations

### 1. Source → Training Data

**File:** `metrics_generator.py`

```python
# Source (AIMLP CSV)
Host Name       → server_name
%CPU User+Sys   → cpu_pct
%Mem Used       → mem_pct
Disk Usage      → disk_io_mb_s (derived)
Load Aver       → latency_ms (proxy)
(derived)       → state (using state determination logic)

# Additional training features
timestamp       → hour, day_of_week, month, is_weekend (temporal)
server_name     → server_id (encoded via hash)
```

**Output Format:** Parquet (required)
**Output Location:** `training/server_metrics.parquet`

### 2. Training Data → Model

**File:** `tft_trainer.py`

```python
# TimeSeriesDataSet configuration
time_idx: Sequential integer (0, 1, 2, ...)
target: 'cpu_pct' (primary target)
group_ids: ['server_id']  # Encoded server names

# Time-varying unknown (to be predicted)
- cpu_pct
- mem_pct
- disk_io_mb_s
- latency_ms

# Time-varying known (known in advance)
- hour
- day_of_week
- month
- is_weekend
- is_business_hours

# Categorical
- state (8 values - see State Contract)
- server_id (hash-encoded)

# Encoders (CRITICAL)
categorical_encoders = {
    'server_id': NaNLabelEncoder(add_nan=True),  # Allow unknown servers
    'state': NaNLabelEncoder(add_nan=True)       # Allow unknown states
}
```

### 3. Model → Predictions

**File:** `tft_inference.py`

```python
# Input: Same schema as training
# Output: Predictions with decoded server names

{
    'server_name': 'ppvra00a0018',  # Decoded from server_id
    'prediction_time': '2025-10-11T07:30:00',
    'predictions': {
        '30min': {'cpu_pct': 65.2, 'mem_pct': 72.1, ...},
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

---

## 🔒 Validation Contract

### Data Validation (All Stages)

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
    required_cols = ['timestamp', 'server_name', 'cpu_pct', 'mem_pct',
                     'disk_io_mb_s', 'latency_ms', 'state']
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
    if (df['cpu_pct'] < 0).any() or (df['cpu_pct'] > 100).any():
        errors.append("cpu_pct out of range [0, 100]")

    if (df['mem_pct'] < 0).any() or (df['mem_pct'] > 100).any():
        errors.append("mem_pct out of range [0, 100]")

    # Timestamp format
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        errors.append("timestamp must be datetime type")

    return (len(errors) == 0, errors)
```

### Model Loading Validation

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
    current_states = data_df['state'].unique().tolist()

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

## 📦 File Structure Contract

### Training Data Output
```
training/
├── server_metrics.parquet        # Main training data
├── metrics_metadata.json         # Generation metadata
└── server_mapping.json           # Server name mappings
```

### Model Output
```
models/tft_model_YYYYMMDD_HHMMSS/
├── model.safetensors             # Model weights
├── config.json                   # Model architecture config
├── training_info.json            # Training metadata
├── server_mapping.json           # Server name mappings (REQUIRED)
└── data_contract_version.txt     # Contract version used
```

### Metadata Schemas

#### metrics_metadata.json
```json
{
  "generated_at": "ISO8601",
  "total_samples": 432000,
  "time_span_hours": 24,
  "servers_count": 25,
  "unique_states": ["critical_issue", "healthy", ...],
  "state_counts": {"healthy": 279598, ...},
  "data_contract_version": "1.0.0"
}
```

#### training_info.json
```json
{
  "trained_at": "ISO8601",
  "epochs": 20,
  "total_samples": 432000,
  "num_servers": 25,
  "unique_states": ["critical_issue", "healthy", ...],
  "state_encoder_size": 8,
  "server_encoder_size": 25,
  "data_contract_version": "1.0.0"
}
```

---

## 🚨 Breaking Changes Protocol

### If You Must Change The Contract:

1. **Update this document FIRST**
2. **Increment version number** (e.g., 1.0.0 → 2.0.0)
3. **Create migration guide** in `Docs/MIGRATIONS.md`
4. **Update all three stages:**
   - `metrics_generator.py`
   - `tft_trainer.py`
   - `tft_inference.py`
5. **Regenerate training data**
6. **Retrain all models**
7. **Run full validation suite**
8. **Update all documentation**

### Version Compatibility Matrix

| Contract | Generator | Trainer | Inference | Status |
|----------|-----------|---------|-----------|--------|
| 1.0.0    | ✅        | ✅      | ✅        | Current |

---

## 🛠️ Implementation Checklist

### Phase 1: Implement Server Encoding (CURRENT)
- [ ] Create `server_encoder.py` utility module
- [ ] Update `metrics_generator.py` to encode server names
- [ ] Save `server_mapping.json` during generation
- [ ] Update `tft_trainer.py` to use encoded server_id
- [ ] Save `server_mapping.json` with model
- [ ] Update `tft_inference.py` to decode server names

### Phase 2: Add Validation
- [ ] Create `data_validator.py` module
- [ ] Add validation to data generation
- [ ] Add validation before training
- [ ] Add validation before inference
- [ ] Add contract version tracking

### Phase 3: Regenerate & Retrain
- [ ] Regenerate training data with new encoder
- [ ] Validate training data against contract
- [ ] Retrain model with validated data
- [ ] Test full pipeline end-to-end
- [ ] Update all documentation

---

## 📖 Reference Materials

- **Source Data:** [Docs/aimlp_server_metrics.csv](aimlp_server_metrics.csv)
- **Field Descriptions:** [Docs/metrics decription.txt](metrics decription.txt)
- **Current Training Data:** `training/server_metrics.parquet`
- **Metadata:** `training/metrics_metadata.json`

---

## 🔍 Quick Reference

### State Values (Copy-Paste)
```python
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
```

### Required Columns (Copy-Paste)
```python
REQUIRED_COLUMNS = [
    'timestamp',
    'server_name',
    'cpu_pct',
    'mem_pct',
    'disk_io_mb_s',
    'latency_ms',
    'state'
]
```

---

**Contract Version:** 1.0.0
**Last Updated:** 2025-10-11
**Maintained By:** Project Team
**Review Frequency:** Before any schema changes

⚠️ **THIS IS THE SOURCE OF TRUTH - ALL CODE MUST CONFORM TO THIS CONTRACT**
