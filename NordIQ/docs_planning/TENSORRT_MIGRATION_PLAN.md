# TensorRT Migration Plan for ArgusAI TFT Engine

**Goal**: Ultra-fast inference with 10-50x speedup on NVIDIA GPUs

---

## Phase 1: Export PyTorch Model to ONNX

### Step 1.1: Prepare PyTorch Model for Export

**File**: `NordIQ/src/inference/export_to_onnx.py`

```python
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pathlib import Path

def export_tft_to_onnx(checkpoint_path: str, output_path: str):
    """
    Export trained TFT model to ONNX format.

    ONNX is the intermediate format between PyTorch and TensorRT.
    """
    # Load trained model
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Define input shapes
    batch_size = 1
    encoder_length = 288  # 24 hours context
    num_features = 14     # NordIQ Metrics

    # Create dummy inputs matching your data contract
    dummy_input = {
        'encoder_cont': torch.randn(batch_size, encoder_length, num_features),
        'encoder_cat': torch.randint(0, 100, (batch_size, encoder_length, 2)),  # server_id, status
        'decoder_cont': torch.randn(batch_size, 96, 4),  # time features
        'decoder_cat': torch.randint(0, 100, (batch_size, 96, 2)),
    }

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,  # Latest stable
        do_constant_folding=True,
        input_names=['encoder_cont', 'encoder_cat', 'decoder_cont', 'decoder_cat'],
        output_names=['predictions'],
        dynamic_axes={
            'encoder_cont': {0: 'batch'},
            'encoder_cat': {0: 'batch'},
            'decoder_cont': {0: 'batch'},
            'decoder_cat': {0: 'batch'},
            'predictions': {0: 'batch'}
        }
    )

    print(f"✅ Exported to ONNX: {output_path}")
```

### Step 1.2: Verify ONNX Model

```python
import onnx
import onnxruntime as ort

# Load and check ONNX model
onnx_model = onnx.load("tft_model.onnx")
onnx.checker.check_model(onnx_model)

# Test inference with ONNX Runtime
session = ort.InferenceSession("tft_model.onnx")
outputs = session.run(None, dummy_input)
print(f"✅ ONNX inference successful: {outputs[0].shape}")
```

---

## Phase 2: Convert ONNX to TensorRT Engine

### Step 2.1: Build TensorRT Engine (Python)

**File**: `NordIQ/src/inference/build_tensorrt_engine.py`

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def build_tensorrt_engine(onnx_path: str, engine_path: str, precision: str = 'fp16'):
    """
    Build optimized TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Where to save TensorRT engine
        precision: 'fp32', 'fp16', or 'int8'
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX")

    # Build engine configuration
    config = builder.create_builder_config()
    config.max_workspace_size = 4 << 30  # 4GB

    # Set precision mode
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        print("✅ FP16 precision enabled (2-4x faster)")
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: INT8 requires calibration dataset
        print("✅ INT8 precision enabled (4-8x faster)")

    # Optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "encoder_cont",
        min=(1, 288, 14),
        opt=(8, 288, 14),   # Optimal batch size
        max=(32, 288, 14)   # Max batch size
    )
    config.add_optimization_profile(profile)

    # Build engine
    print("🔨 Building TensorRT engine (this may take 5-10 minutes)...")
    engine = builder.build_engine(network, config)

    # Serialize and save
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"✅ TensorRT engine saved: {engine_path}")
    print(f"   Precision: {precision.upper()}")
    print(f"   Optimization complete!")
```

### Step 2.2: Alternative - Command Line Tool

```bash
# Install TensorRT (comes with CUDA Toolkit)
# Already have it with your RTX 4090 setup

# Convert ONNX to TensorRT using trtexec
trtexec \
  --onnx=tft_model.onnx \
  --saveEngine=tft_model_fp16.engine \
  --fp16 \
  --workspace=4096 \
  --minShapes=encoder_cont:1x288x14 \
  --optShapes=encoder_cont:8x288x14 \
  --maxShapes=encoder_cont:32x288x14 \
  --verbose
```

---

## Phase 3: C++ Inference Engine

### Step 3.1: TensorRT C++ Inference Class

**File**: `NordIQ/src/inference/tensorrt_engine.hpp`

```cpp
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <memory>
#include <vector>

class TFTTensorRTEngine {
public:
    TFTTensorRTEngine(const std::string& engine_path);
    ~TFTTensorRTEngine();

    // Run inference on batch of server data
    std::vector<float> predict(
        const std::vector<float>& encoder_continuous,
        const std::vector<int>& encoder_categorical,
        const std::vector<float>& decoder_continuous,
        const std::vector<int>& decoder_categorical
    );

    // Batch inference (process multiple servers at once)
    std::vector<std::vector<float>> predict_batch(
        const std::vector<std::vector<float>>& batch_encoder_cont,
        const std::vector<std::vector<int>>& batch_encoder_cat,
        const std::vector<std::vector<float>>& batch_decoder_cont,
        const std::vector<std::vector<int>>& batch_decoder_cat
    );

private:
    // TensorRT components
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    // CUDA memory buffers
    void** buffers_;
    cudaStream_t stream_;

    // Model dimensions
    int batch_size_;
    int encoder_length_;
    int num_features_;
    int prediction_horizon_;

    void allocate_buffers();
    void free_buffers();
};
```

**Implementation**: `NordIQ/src/inference/tensorrt_engine.cpp`

```cpp
#include "tensorrt_engine.hpp"
#include <iostream>
#include <cassert>

TFTTensorRTEngine::TFTTensorRTEngine(const std::string& engine_path) {
    // Load serialized engine
    std::ifstream file(engine_path, std::ios::binary);
    assert(file.good());

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Create TensorRT runtime
    runtime_ = nvinfer1::createInferRuntime(gLogger);
    assert(runtime_ != nullptr);

    // Deserialize engine
    engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
    assert(engine_ != nullptr);

    // Create execution context
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);

    // Create CUDA stream
    cudaStreamCreate(&stream_);

    // Allocate GPU memory
    allocate_buffers();

    std::cout << "✅ TensorRT engine loaded successfully" << std::endl;
}

std::vector<float> TFTTensorRTEngine::predict(
    const std::vector<float>& encoder_continuous,
    const std::vector<int>& encoder_categorical,
    const std::vector<float>& decoder_continuous,
    const std::vector<int>& decoder_categorical
) {
    // Copy input data to GPU
    cudaMemcpyAsync(buffers_[0], encoder_continuous.data(),
                    encoder_continuous.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    cudaMemcpyAsync(buffers_[1], encoder_categorical.data(),
                    encoder_categorical.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream_);

    cudaMemcpyAsync(buffers_[2], decoder_continuous.data(),
                    decoder_continuous.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);

    cudaMemcpyAsync(buffers_[3], decoder_categorical.data(),
                    decoder_categorical.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream_);

    // Execute inference
    context_->enqueueV2(buffers_, stream_, nullptr);

    // Copy output back to CPU
    int output_size = batch_size_ * prediction_horizon_ * 7; // 7 quantiles
    std::vector<float> output(output_size);

    cudaMemcpyAsync(output.data(), buffers_[4],
                    output_size * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);

    cudaStreamSynchronize(stream_);

    return output;
}

TFTTensorRTEngine::~TFTTensorRTEngine() {
    free_buffers();
    cudaStreamDestroy(stream_);
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
}
```

---

## Phase 4: FastAPI C++ Integration

### Option A: Python Wrapper (Pybind11)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensorrt_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(tensorrt_inference, m) {
    py::class_<TFTTensorRTEngine>(m, "TFTEngine")
        .def(py::init<const std::string&>())
        .def("predict", &TFTTensorRTEngine::predict)
        .def("predict_batch", &TFTTensorRTEngine::predict_batch);
}
```

**Use in Python FastAPI**:
```python
from tensorrt_inference import TFTEngine

# Load once at startup
tft_engine = TFTEngine("tft_model_fp16.engine")

@app.post("/predict")
async def predict(data: MetricsData):
    # Ultra-fast C++ inference
    predictions = tft_engine.predict(
        data.encoder_cont,
        data.encoder_cat,
        data.decoder_cont,
        data.decoder_cat
    )
    return predictions
```

### Option B: Pure C++ REST API (Ultimate Performance)

Replace Python FastAPI with C++ HTTP server (Crow, Pistache, or cpp-httplib):

```cpp
#include "crow.h"
#include "tensorrt_engine.hpp"

int main() {
    crow::SimpleApp app;
    TFTTensorRTEngine engine("tft_model_fp16.engine");

    CROW_ROUTE(app, "/predict")
        .methods("POST"_method)
        ([&engine](const crow::request& req) {
            // Parse JSON input
            auto json = crow::json::load(req.body);

            // Run inference
            auto predictions = engine.predict(
                json["encoder_cont"],
                json["encoder_cat"],
                json["decoder_cont"],
                json["decoder_cat"]
            );

            // Return JSON response
            crow::json::wvalue result;
            result["predictions"] = predictions;
            return result;
        });

    app.port(8000).multithreaded().run();
}
```

---

## Performance Benchmarks (Projected)

### Current Python Setup:
- **Latency**: 50-100ms per prediction
- **Throughput**: ~10-20 predictions/sec
- **GPU Utilization**: 15-30%

### TensorRT FP16:
- **Latency**: 3-5ms per prediction (15-30x faster)
- **Throughput**: 200-300 predictions/sec
- **GPU Utilization**: 60-80% (much better)

### TensorRT INT8 (with calibration):
- **Latency**: 1-2ms per prediction (50-100x faster)
- **Throughput**: 500-1000 predictions/sec
- **GPU Utilization**: 80-95%

### Real-World Impact for 20-Server Fleet:
| Metric | Python | TensorRT FP16 | TensorRT INT8 |
|--------|--------|---------------|---------------|
| **Time for 20 predictions** | 1-2 seconds | 60-100ms | 20-40ms |
| **Max servers (5s polling)** | 50-100 | 1,000-1,500 | 2,500-5,000 |
| **Power efficiency** | Baseline | 2x better | 3x better |

---

## Implementation Timeline

**Week 1: ONNX Export**
- [ ] Export TFT model to ONNX
- [ ] Validate ONNX with ONNX Runtime
- [ ] Benchmark ONNX vs Python (expect 2-3x speedup)

**Week 2: TensorRT Conversion**
- [ ] Build FP16 TensorRT engine
- [ ] Test inference accuracy (should be 99.9%+ match)
- [ ] Benchmark TensorRT FP16 (expect 10-20x speedup)

**Week 3: C++ Integration**
- [ ] Build Pybind11 wrapper
- [ ] Integrate with existing FastAPI daemon
- [ ] Load test with real data

**Week 4: Production Deployment**
- [ ] Add engine caching (avoid rebuild)
- [ ] Multi-stream inference for batching
- [ ] Monitoring and metrics

**Optional: INT8 Quantization**
- [ ] Collect calibration dataset
- [ ] Build INT8 engine
- [ ] Validate accuracy (target: <1% degradation)

---

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| **ONNX export fails** | Use `torch.jit.trace()` instead of export |
| **TensorRT version mismatch** | Pin CUDA 11.8 + TensorRT 8.6.1 |
| **Accuracy loss in FP16** | Validate against Python with test dataset |
| **INT8 accuracy degradation** | May need to keep some layers in FP16 |
| **Complex deployment** | Package engine with Docker |

---

## Tools & Dependencies

**Python:**
- `torch` (already have)
- `onnx` - `pip install onnx`
- `onnxruntime-gpu` - `pip install onnxruntime-gpu`
- `tensorrt` - comes with CUDA Toolkit
- `pycuda` - `pip install pycuda`

**C++:**
- TensorRT (installed with CUDA 11.8)
- CUDA Runtime
- Pybind11 or Crow (HTTP server)

**Build System:**
- CMake 3.18+
- NVCC (CUDA compiler)

---

## Alternative: LibTorch (Easier Path)

If TensorRT proves too complex, LibTorch is a solid middle ground:

```cpp
#include <torch/script.h>

auto model = torch::jit::load("tft_model.pt");
auto output = model.forward({input_tensor}).toTensor();
```

**Pros**:
- ✅ Easier to implement
- ✅ No ONNX conversion needed
- ✅ Still 2-5x faster than Python

**Cons**:
- ❌ Not as fast as TensorRT (only 2-5x vs 10-50x)
- ❌ Larger binary size
- ❌ Less optimization

---

**Recommendation**: Start with TensorRT FP16 for maximum performance. The effort is worth it for 10-30x speedup!

Built by Craig Giannelli and Claude Code
