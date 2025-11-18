# TensorRT Directory Structure & Naming Convention

**Goal**: Keep TensorRT models completely separate from PyTorch models to avoid confusion

---

## Proposed Directory Structure

```
NordIQ/
├── models/                          # PyTorch models (Python inference)
│   ├── tft_model_20251118_134508/
│   │   ├── model.safetensors       # PyTorch weights
│   │   ├── config.json
│   │   ├── training_info.json
│   │   └── server_mapping.json
│   └── tft_model_20251119_080000/
│
├── models_tensorrt/                 # TensorRT optimized models ⭐ NEW
│   ├── tft_tensorrt_20251119_080000/
│   │   ├── model.onnx              # Intermediate ONNX format
│   │   ├── model_fp32.engine       # TensorRT FP32 engine
│   │   ├── model_fp16.engine       # TensorRT FP16 engine (recommended)
│   │   ├── model_int8.engine       # TensorRT INT8 engine (optional)
│   │   ├── config.json             # Model configuration
│   │   ├── training_info.json      # Copy from PyTorch model
│   │   ├── server_mapping.json     # Copy from PyTorch model
│   │   ├── conversion_log.txt      # Conversion details
│   │   └── benchmark_results.json  # Performance metrics
│   └── tft_tensorrt_20251120_120000/
│
├── src/
│   ├── training/
│   │   └── tft_trainer.py          # Existing PyTorch training
│   │
│   ├── inference/                   # Inference engines
│   │   ├── tft_inference.py        # Existing Python inference
│   │   ├── export_to_tensorrt.py   # ⭐ NEW: PyTorch → TensorRT
│   │   ├── tensorrt_engine.py      # ⭐ NEW: Python wrapper
│   │   └── cpp/                    # ⭐ NEW: C++ TensorRT engine
│   │       ├── CMakeLists.txt
│   │       ├── tensorrt_engine.hpp
│   │       ├── tensorrt_engine.cpp
│   │       └── bindings.cpp        # Pybind11 wrapper
│   │
│   └── daemons/
│       ├── tft_inference_daemon.py      # Existing Python daemon
│       └── tft_tensorrt_daemon.py       # ⭐ NEW: TensorRT daemon (port 8002)
│
└── bin/
    ├── export_model_to_tensorrt.py      # ⭐ NEW: Conversion script
    └── benchmark_tensorrt.py            # ⭐ NEW: Performance testing
```

---

## File Naming Conventions

### PyTorch Models (Existing)
```
models/tft_model_YYYYMMDD_HHMMSS/
    model.safetensors           # PyTorch weights
    config.json
    training_info.json
```

### TensorRT Models (New)
```
models_tensorrt/tft_tensorrt_YYYYMMDD_HHMMSS/
    model.onnx                  # Source ONNX
    model_fp32.engine           # FP32 precision (baseline)
    model_fp16.engine           # FP16 precision (2-4x faster)
    model_int8.engine           # INT8 precision (4-8x faster, needs calibration)
    conversion_info.json        # Conversion metadata
```

---

## Model Metadata Files

### `conversion_info.json`
```json
{
  "source_model": "models/tft_model_20251119_080000",
  "source_checkpoint": "checkpoints/last.ckpt",
  "converted_at": "2025-11-19T08:30:00",
  "pytorch_version": "2.0.1",
  "tensorrt_version": "8.6.1",
  "cuda_version": "11.8",
  "onnx_opset": 17,
  "engines_built": {
    "fp32": {
      "file": "model_fp32.engine",
      "build_time_seconds": 120,
      "file_size_mb": 2.4
    },
    "fp16": {
      "file": "model_fp16.engine",
      "build_time_seconds": 180,
      "file_size_mb": 1.8
    },
    "int8": {
      "file": "model_int8.engine",
      "build_time_seconds": 300,
      "file_size_mb": 1.2,
      "calibration_samples": 1000
    }
  },
  "input_shapes": {
    "encoder_cont": [1, 288, 14],
    "encoder_cat": [1, 288, 2],
    "decoder_cont": [1, 96, 4],
    "decoder_cat": [1, 96, 2]
  },
  "output_shapes": {
    "predictions": [1, 96, 7]
  },
  "validation": {
    "accuracy_vs_pytorch": 0.9998,
    "mean_absolute_error": 0.0023,
    "max_absolute_error": 0.015
  }
}
```

### `benchmark_results.json`
```json
{
  "benchmarked_at": "2025-11-19T08:45:00",
  "hardware": {
    "gpu": "NVIDIA GeForce RTX 4090",
    "cuda_version": "11.8",
    "driver_version": "537.13"
  },
  "pytorch_baseline": {
    "latency_ms": {
      "mean": 52.3,
      "std": 8.4,
      "min": 45.2,
      "max": 78.1
    },
    "throughput_per_sec": 19.1,
    "gpu_utilization_pct": 24
  },
  "tensorrt_fp32": {
    "latency_ms": {
      "mean": 8.7,
      "std": 0.8,
      "min": 7.9,
      "max": 11.2
    },
    "throughput_per_sec": 114.9,
    "gpu_utilization_pct": 68,
    "speedup_vs_pytorch": 6.0
  },
  "tensorrt_fp16": {
    "latency_ms": {
      "mean": 3.2,
      "std": 0.3,
      "min": 2.9,
      "max": 4.1
    },
    "throughput_per_sec": 312.5,
    "gpu_utilization_pct": 82,
    "speedup_vs_pytorch": 16.3
  },
  "tensorrt_int8": {
    "latency_ms": {
      "mean": 1.4,
      "std": 0.2,
      "min": 1.2,
      "max": 1.9
    },
    "throughput_per_sec": 714.3,
    "gpu_utilization_pct": 91,
    "speedup_vs_pytorch": 37.4,
    "accuracy_loss_pct": 0.12
  }
}
```

---

## Service Configuration

### Python Inference Daemon (Existing)
- **Port**: 8000
- **Model Path**: `models/tft_model_*/model.safetensors`
- **Use Case**: Development, debugging, flexibility
- **Startup**: `python src/daemons/tft_inference_daemon.py --port 8000`

### TensorRT Inference Daemon (New)
- **Port**: 8002
- **Model Path**: `models_tensorrt/tft_tensorrt_*/model_fp16.engine`
- **Use Case**: Production, maximum performance
- **Startup**: `python src/daemons/tft_tensorrt_daemon.py --port 8002 --precision fp16`

### Configuration File: `daemon_config.yaml`
```yaml
inference:
  # Python inference (development)
  python:
    enabled: true
    port: 8000
    model_dir: models/
    model_pattern: tft_model_*
    use_latest: true

  # TensorRT inference (production)
  tensorrt:
    enabled: true
    port: 8002
    model_dir: models_tensorrt/
    model_pattern: tft_tensorrt_*
    use_latest: true
    precision: fp16  # Options: fp32, fp16, int8
    max_batch_size: 32
    cuda_streams: 4
```

---

## Conversion Workflow

### Step 1: Train PyTorch Model (Existing)
```bash
cd NordIQ
python src/training/tft_trainer.py --epochs 20
# Creates: models/tft_model_20251119_080000/
```

### Step 2: Export to TensorRT
```bash
cd NordIQ
python bin/export_model_to_tensorrt.py \
  --source models/tft_model_20251119_080000 \
  --output models_tensorrt/tft_tensorrt_20251119_080000 \
  --precision fp16 \
  --validate
```

**What this does**:
1. ✅ Loads PyTorch model from safetensors
2. ✅ Exports to ONNX (`model.onnx`)
3. ✅ Validates ONNX with ONNX Runtime
4. ✅ Builds TensorRT engine (`model_fp16.engine`)
5. ✅ Runs accuracy validation (PyTorch vs TensorRT)
6. ✅ Runs performance benchmark
7. ✅ Saves metadata (`conversion_info.json`, `benchmark_results.json`)
8. ✅ Copies server_mapping.json and training_info.json

### Step 3: Start TensorRT Daemon
```bash
cd NordIQ
python src/daemons/tft_tensorrt_daemon.py --port 8002
```

### Step 4: Test Both Daemons
```bash
# Python daemon (existing)
curl -X POST http://localhost:8000/predict -d @test_data.json

# TensorRT daemon (new)
curl -X POST http://localhost:8002/predict -d @test_data.json
```

---

## Safety Guardrails

### 1. Model Loading Logic

**Python Inference** (`tft_inference_daemon.py`):
```python
def load_model(model_dir: str = "models/"):
    """Only loads PyTorch .safetensors models"""
    model_path = find_latest_model(model_dir, pattern="tft_model_*")

    # Reject TensorRT models
    if "tensorrt" in str(model_path).lower():
        raise ValueError("Cannot load TensorRT model in Python daemon!")

    # Load safetensors
    model = TemporalFusionTransformer.load_from_checkpoint(...)
    return model
```

**TensorRT Inference** (`tft_tensorrt_daemon.py`):
```python
def load_engine(model_dir: str = "models_tensorrt/", precision: str = "fp16"):
    """Only loads TensorRT .engine files"""
    model_path = find_latest_model(model_dir, pattern="tft_tensorrt_*")

    # Reject PyTorch models
    if "tensorrt" not in str(model_path).lower():
        raise ValueError("Cannot load PyTorch model in TensorRT daemon!")

    # Load TensorRT engine
    engine_file = model_path / f"model_{precision}.engine"
    if not engine_file.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_file}")

    return TensorRTEngine(engine_file)
```

### 2. Model Discovery

```python
def find_latest_model(base_dir: str, pattern: str):
    """Find most recent model matching pattern"""
    models = sorted(Path(base_dir).glob(pattern), reverse=True)

    if not models:
        raise FileNotFoundError(f"No models found in {base_dir} matching {pattern}")

    latest = models[0]
    print(f"[LOAD] Using model: {latest.name}")
    return latest
```

### 3. Startup Script Updates

**`start_all.bat`** (Windows):
```batch
@echo off
echo Starting ArgusAI Services...

REM Choose inference backend
set INFERENCE_BACKEND=tensorrt

if "%INFERENCE_BACKEND%"=="python" (
    echo [INFERENCE] Using Python backend (port 8000)
    start "TFT Inference (Python)" cmd /k "cd /d "%~dp0" && conda activate py310 && python src\daemons\tft_inference_daemon.py --port 8000"
) else if "%INFERENCE_BACKEND%"=="tensorrt" (
    echo [INFERENCE] Using TensorRT backend (port 8002)
    start "TFT Inference (TensorRT)" cmd /k "cd /d "%~dp0" && conda activate py310 && python src\daemons\tft_tensorrt_daemon.py --port 8002 --precision fp16"
)

REM Start other services...
```

---

## Migration Checklist

### Phase 1: Setup (Week 1)
- [ ] Create `models_tensorrt/` directory
- [ ] Create `src/inference/` directory structure
- [ ] Install dependencies: `pip install onnx onnxruntime-gpu tensorrt pycuda`
- [ ] Verify TensorRT installation: `python -c "import tensorrt; print(tensorrt.__version__)"`

### Phase 2: Fresh Training (Week 1)
- [ ] Delete old models: `rm -rf models/*`
- [ ] Generate fresh 720-hour dataset (30 days)
- [ ] Train new model: 20 epochs
- [ ] Creates: `models/tft_model_YYYYMMDD_HHMMSS/`

### Phase 3: TensorRT Conversion (Week 2)
- [ ] Create export script: `bin/export_model_to_tensorrt.py`
- [ ] Export to ONNX
- [ ] Build FP32 engine (baseline)
- [ ] Build FP16 engine (production)
- [ ] Validate accuracy (must be >99.9% match)
- [ ] Benchmark performance

### Phase 4: TensorRT Daemon (Week 3)
- [ ] Create `src/daemons/tft_tensorrt_daemon.py`
- [ ] Test with single prediction
- [ ] Test with batch predictions
- [ ] Load test with 20 concurrent requests
- [ ] Monitor GPU utilization

### Phase 5: Production Deployment (Week 4)
- [ ] Update `start_all.bat` to use TensorRT
- [ ] Update dashboard to use port 8002
- [ ] Create backup startup scripts (Python fallback)
- [ ] Document performance improvements
- [ ] Archive Python inference (keep for debugging)

---

## Rollback Plan

If TensorRT has issues:

1. **Immediate**: Change `INFERENCE_BACKEND=python` in `start_all.bat`
2. **Restart**: Python daemon comes back online in seconds
3. **Debug**: TensorRT issues don't affect training or Python inference

---

## Future Enhancements

### INT8 Quantization (Optional)
- Collect 1000 representative samples for calibration
- Build INT8 engine (50-100x speedup)
- Validate accuracy loss (<1% acceptable)

### C++ Native Daemon (Optional)
- Replace Python FastAPI with C++ HTTP server
- Eliminate Python overhead entirely
- Expected: Additional 2-3x speedup

### Multi-GPU Support (Optional)
- Distribute inference across multiple GPUs
- Handle 10,000+ servers easily

---

**Bottom Line**: Clean separation, no confusion, easy rollback, massive performance gains!

Built by Craig Giannelli and Claude Code
