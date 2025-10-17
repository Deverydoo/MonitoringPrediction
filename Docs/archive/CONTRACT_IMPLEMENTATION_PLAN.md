# Data Contract Implementation Plan

**Created:** 2025-10-11
**Status:** Ready to Execute
**Priority:** HIGH - Prevents pipeline drift issues

---

## 🎯 Objective

Implement the DATA_CONTRACT.md specification across all three pipeline stages to eliminate schema drift and reduce wasted retraining effort.

---

## 📋 Completed (Phase 0)

### ✅ Foundation Work
1. **[DATA_CONTRACT.md](DATA_CONTRACT.md)** - Single source of truth created
2. **[server_encoder.py](../server_encoder.py)** - Hash-based server name encoding/decoding
3. **[data_validator.py](../data_validator.py)** - Contract validation utility
4. **Test Results:**
   - Server encoder: All tests pass ✅
   - Data validator: Working, identified 5.4% missing values in training data ⚠️

---

## 🚀 Phase 1: Update Data Generation (NEXT)

### Files to Modify
1. **metrics_generator.py**

### Changes Required

#### 1.1 Add Server Encoding
```python
from server_encoder import ServerEncoder

# In main generation function:
encoder = ServerEncoder()
encoder.create_mapping(df['server_name'].unique())

# Add encoded column
df['server_id'] = df['server_name'].apply(encoder.encode)

# Save mapping with data
encoder.save_mapping(output_dir / 'server_mapping.json')
```

#### 1.2 Add Contract Validation
```python
from data_validator import DataValidator, CONTRACT_VERSION

# Before saving data:
validator = DataValidator(strict=False)
is_valid, errors, warnings = validator.validate_schema(df)

if not is_valid:
    print("[ERROR] Data does not conform to contract!")
    validator.print_report()
    return False
```

#### 1.3 Update Metadata
```python
# In metrics_metadata.json:
{
    ...
    "unique_states": sorted(df['state'].unique().tolist()),
    "state_counts": df['state'].value_counts().to_dict(),
    "data_contract_version": CONTRACT_VERSION,
    "has_server_mapping": True
}
```

#### 1.4 Fix Missing Values
- Current issue: 5.4% missing values in cpu_pct, mem_pct, disk_io_mb_s, latency_ms
- Root cause: Some server states don't generate all metrics
- Solution: Ensure all states generate valid numeric values (use 0.0 or interpolation)

### Testing Phase 1
```bash
# Regenerate training data
python metrics_generator.py --servers 25 --hours 24 --output ./training/

# Validate output
python data_validator.py training/server_metrics.parquet --strict

# Check for server_mapping.json
ls training/server_mapping.json

# Verify metadata
cat training/metrics_metadata.json
```

**Expected Outcome:**
- ✅ training/server_metrics.parquet (no missing values)
- ✅ training/server_mapping.json (25 servers)
- ✅ training/metrics_metadata.json (with contract version)
- ✅ All validations pass

---

## 🎓 Phase 2: Update Model Training

### Files to Modify
1. **tft_trainer.py**

### Changes Required

#### 2.1 Load Server Mapping
```python
from server_encoder import ServerEncoder

# At training start:
encoder = ServerEncoder(mapping_file='./training/server_mapping.json')
print(f"[OK] Loaded server mapping: {encoder.get_stats()['total_servers']} servers")
```

#### 2.2 Validate Input Data
```python
from data_validator import DataValidator

# Before creating TimeSeriesDataSet:
validator = DataValidator(strict=True)
is_valid, errors, warnings = validator.validate_schema(df)

if not is_valid:
    print("[ERROR] Training data violates contract!")
    validator.print_report()
    sys.exit(1)
```

#### 2.3 Update Training Info
```python
# Save training_info.json:
{
    ...
    "unique_states": sorted(df['state'].unique().tolist()),
    "state_encoder_size": len(df['state'].unique()),
    "server_encoder_size": len(df['server_id'].unique()),
    "data_contract_version": CONTRACT_VERSION
}
```

#### 2.4 Copy Server Mapping to Model
```python
# After saving model:
import shutil
shutil.copy(
    './training/server_mapping.json',
    model_output_dir / 'server_mapping.json'
)
print(f"[OK] Server mapping saved with model")
```

### Testing Phase 2
```bash
# Train model
python tft_trainer.py --dataset ./training/ --epochs 20

# Verify model artifacts
ls models/tft_model_*/server_mapping.json
cat models/tft_model_*/training_info.json

# Check contract version
grep "data_contract_version" models/tft_model_*/training_info.json
```

**Expected Outcome:**
- ✅ Model trains without dimension mismatches
- ✅ training_info.json contains contract version
- ✅ server_mapping.json copied to model directory
- ✅ State encoder size = 8

---

## 🔮 Phase 3: Update Inference

### Files to Modify
1. **tft_inference.py**

### Changes Required

#### 3.1 Load Server Mapping
```python
from server_encoder import ServerEncoder

def _load_model(self):
    # Load server mapping
    mapping_file = self.model_dir / 'server_mapping.json'
    if not mapping_file.exists():
        print("[ERROR] server_mapping.json not found in model directory")
        self.use_real_model = False
        return

    self.server_encoder = ServerEncoder(mapping_file)
    print(f"[OK] Loaded server mapping: {self.server_encoder.get_stats()}")
```

#### 3.2 Validate Model Compatibility
```python
from data_validator import DataValidator

def _load_model(self):
    # ... after loading model ...

    # Validate dummy dataset against model
    validator = DataValidator()
    is_compatible, errors = validator.validate_model_compatibility(
        self.model_dir,
        dummy_df
    )

    if not is_compatible:
        print("[ERROR] Model incompatible with current contract!")
        for error in errors:
            print(f"   {error}")
        self.use_real_model = False
        return
```

#### 3.3 Decode Server Names in Predictions
```python
def _format_tft_predictions(self, raw_predictions, servers):
    """Format predictions with decoded server names."""

    formatted = {}
    for i, server_id in enumerate(servers):
        # Decode server_id back to server_name
        server_name = self.server_encoder.decode(server_id)

        formatted[server_name] = {
            'server_id': server_id,
            'server_name': server_name,
            'predictions': {
                '30min': {...},
                '8hr': {...}
            }
        }

    return formatted
```

#### 3.4 Update Simulation Generator
```python
def _generate_simulation_batch(self):
    # Use encoder for consistent server IDs
    server_id = self.server_encoder.encode(server_name)

    batch.append({
        'server_name': server_name,
        'server_id': server_id,  # Encoded ID
        ...
    })
```

### Testing Phase 3
```bash
# Start daemon
python tft_inference.py --daemon --port 8000

# Check logs for:
# [OK] Loaded server mapping: ...
# [OK] Model validated against contract
# [SUCCESS] TFT model loaded successfully

# Test prediction endpoint
curl http://localhost:8000/predictions/current

# Verify server names are decoded in output
```

**Expected Outcome:**
- ✅ Daemon starts without errors
- ✅ Server mapping loaded successfully
- ✅ Predictions contain decoded server names
- ✅ No dimension mismatch errors

---

## 🧪 Phase 4: End-to-End Testing

### Test Scenarios

#### Scenario 1: Fresh Training Pipeline
```bash
# 1. Generate data
python metrics_generator.py --servers 25 --hours 24 --output ./training/

# 2. Validate data
python data_validator.py training/server_metrics.parquet --strict

# 3. Train model
python tft_trainer.py --dataset ./training/ --epochs 20

# 4. Start inference
python tft_inference.py --daemon --port 8000

# 5. Test dashboard
python tft_dashboard_refactored.py training/server_metrics.parquet
```

#### Scenario 2: Adding New Servers (Unknown Server Handling)
```bash
# 1. Generate data with 5 new servers (30 total)
python metrics_generator.py --servers 30 --hours 24 --output ./testing/

# 2. Run inference with old model (25 servers)
python tft_inference.py --model-path models/tft_model_YYYYMMDD_HHMMSS --data testing/server_metrics.parquet

# 3. Verify unknown servers handled gracefully
#    - New servers should be encoded to UNKNOWN_<id>
#    - Predictions should use TFT with add_nan=True
```

#### Scenario 3: Model Compatibility Check
```bash
# Try loading old model (without contract) - should fail gracefully
python tft_inference.py --model-path models/tft_model_20251011_071329

# Should see:
# [ERROR] server_mapping.json not found in model directory
# [FALLBACK] Using heuristic predictions
```

### Success Criteria
- [ ] Fresh pipeline runs without errors
- [ ] No dimension mismatches
- [ ] Server names encode/decode correctly
- [ ] Unknown servers handled gracefully
- [ ] Old models fail gracefully with clear error messages
- [ ] All validation checks pass

---

## 📊 Rollback Plan

If implementation fails:

1. **Revert Code Changes**
   ```bash
   git checkout main -- metrics_generator.py tft_trainer.py tft_inference.py
   ```

2. **Use Old Model**
   ```bash
   # Point to model trained before contract
   python tft_inference.py --model-path models/tft_model_<old_date>
   ```

3. **Manual Mapping**
   - Create minimal server_mapping.json manually for old model
   - Use sequential encoding as fallback

---

## 📝 Documentation Updates

After successful implementation:

1. Update [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
   - Add contract version information
   - Update schema sections
   - Add server encoding details

2. Update [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
   - Add validation commands
   - Add contract version check

3. Create [MIGRATIONS.md](MIGRATIONS.md)
   - Document migration from pre-contract to contract v1.0.0
   - Include troubleshooting guide

4. Update README.md
   - Add contract compliance badge
   - Link to DATA_CONTRACT.md

---

## ⏱️ Time Estimates

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Update metrics_generator.py | 30 min |
| 1 | Fix missing value issues | 15 min |
| 1 | Test & validate | 15 min |
| 2 | Update tft_trainer.py | 30 min |
| 2 | Test training | 45 min (includes training time) |
| 3 | Update tft_inference.py | 45 min |
| 3 | Test inference | 15 min |
| 4 | End-to-end testing | 30 min |
| 4 | Documentation updates | 20 min |
| **Total** | | **~4 hours** |

*Note: Most time is waiting for model training*

---

## 🚦 Current Status

**Phase 0:** ✅ COMPLETE
**Phase 1:** ⏭️ READY TO START
**Phase 2:** ⏸️ PENDING
**Phase 3:** ⏸️ PENDING
**Phase 4:** ⏸️ PENDING

---

## 🎯 Next Immediate Action

```bash
# Start Phase 1:
# 1. Open metrics_generator.py
# 2. Add server_encoder import
# 3. Add data_validator import
# 4. Update main generation function
# 5. Test with small dataset first (--servers 5 --hours 1)
```

**Files to edit:** `metrics_generator.py`
**Expected time:** 45 minutes
**Risk level:** Low (can rollback easily)

---

**Plan Version:** 1.0
**Created:** 2025-10-11
**Status:** Ready for execution
