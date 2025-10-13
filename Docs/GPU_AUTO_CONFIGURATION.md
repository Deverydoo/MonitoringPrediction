# GPU Auto-Configuration System

## Overview

Automatic GPU detection and optimal configuration for both training and inference across different hardware platforms.

## Supported GPUs

### Consumer/Workstation
- **RTX 4090** (Ada Lovelace, SM 8.9)
  - Batch size (train): 32
  - Batch size (inference): 128
  - Workers: 8
  - Tensor Cores: medium precision

- **RTX 3090** (Ampere, SM 8.6)
  - Batch size (train): 32
  - Batch size (inference): 128
  - Workers: 8
  - Tensor Cores: medium precision

### Data Center - Current Generation
- **Tesla V100** (Volta, SM 7.0)
  - Batch size (train): 64
  - Batch size (inference): 256
  - Workers: 16
  - Tensor Cores: high precision (enterprise reproducibility)

- **Tesla A100** (Ampere, SM 8.0)
  - Batch size (train): 128
  - Batch size (inference): 512
  - Workers: 32
  - Tensor Cores: high precision + TF32 support

### Data Center - Next Generation
- **H100** (Hopper, SM 9.0)
  - Batch size (train): 256
  - Batch size (inference): 1024
  - Workers: 32
  - Tensor Cores: high precision + FP8 support

- **H200** (Hopper HBM3e, SM 9.0)
  - Batch size (train): 512
  - Batch size (inference): 2048
  - Workers: 32
  - Tensor Cores: high precision + FP8 support
  - 141GB HBM3e memory

## How It Works

### 1. Detection Phase
```python
from gpu_profiles import setup_gpu

gpu = setup_gpu()
```

The system:
1. Detects GPU model name via `torch.cuda.get_device_name()`
2. Reads compute capability via `torch.cuda.get_device_capability()`
3. Matches to predefined profiles or falls back to compute capability ranges

### 2. Configuration Phase

Automatically applies:
- **Tensor Core Precision**: `torch.set_float32_matmul_precision('medium'|'high')`
  - Consumer GPUs (RTX): `'medium'` - balance speed/precision
  - Data Center GPUs (Tesla/H100): `'high'` - enterprise precision

- **cuDNN Settings**:
  - `cudnn.benchmark`: Auto-tune convolution algorithms (True for all)
  - `cudnn.deterministic`: Reproducibility (False for consumer, True for enterprise)

- **Memory Allocation**:
  - Consumer: 85% reservation (leave headroom)
  - Data Center: 90% reservation (maximize utilization)

- **Batch Sizes**: GPU-specific optimal values

- **DataLoader Workers**: CPU core allocation (8-32 depending on GPU class)

### 3. Usage in Code

#### Inference (tft_inference.py)
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
```

Batch sizes auto-selected:
```python
batch_size = self.gpu.get_batch_size('inference') if self.gpu else 64
num_workers = min(self.gpu.get_num_workers(), 4) if self.gpu else 0
```

#### Training (tft_trainer.py)
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
```

Workers auto-configured:
```python
if self.gpu:
    optimal_workers = self.gpu.get_num_workers()
else:
    optimal_workers = 2
```

## Example Output

### RTX 4090 (Your Current System)
```
[GPU] Detected: NVIDIA GeForce RTX 4090
[GPU] Compute Capability: SM 8.9
[GPU] Profile: RTX 4090
[GPU] Consumer/Workstation GPU - Ada Lovelace architecture
[GPU] Tensor Cores: Enabled (precision=medium)
[GPU] cuDNN: benchmark=True, deterministic=False
[GPU] Memory: 85% reserved
```

### H100 (Work Environment - Future)
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

## Performance Benefits

### Training Speed
- **RTX 4090**: ~20-30 min/epoch (90 servers, 720h data)
- **A100**: ~10-15 min/epoch (estimated 2x faster)
- **H100**: ~5-8 min/epoch (estimated 4x faster with FP8)

### Inference Throughput
- **RTX 4090**: ~128 predictions/batch
- **A100**: ~512 predictions/batch
- **H100**: ~1024 predictions/batch
- **H200**: ~2048 predictions/batch

## Fallback Behavior

If GPU is unknown:
1. Matches by compute capability range
2. Falls back to "Generic" profile with conservative settings:
   - Batch size (train): 16
   - Batch size (inference): 64
   - Workers: 4
   - Precision: `'highest'` (safest)

## API Reference

### setup_gpu()
```python
gpu = setup_gpu()
```
Returns `GPUDetector` instance with applied configuration.

### GPUDetector Methods
```python
gpu.get_batch_size('train')      # Get optimal training batch size
gpu.get_batch_size('inference')  # Get optimal inference batch size
gpu.get_num_workers()            # Get optimal DataLoader workers
gpu.get_config()                 # Get full configuration dict
gpu.print_summary()              # Print formatted summary
```

### GPUProfile Attributes
```python
profile.name                     # GPU model name
profile.compute_capability       # (major, minor) tuple
profile.tensor_cores             # bool
profile.matmul_precision         # 'highest' | 'high' | 'medium'
profile.recommended_batch_size_train
profile.recommended_batch_size_inference
profile.num_workers
profile.memory_fraction
```

## Testing

Test detection:
```bash
python gpu_profiles.py
```

Test inference integration:
```bash
python -c "from tft_inference import TFTInference; TFTInference()"
```

Test training integration:
```bash
python -c "from tft_trainer import TFTTrainer; TFTTrainer()"
```

## Future Enhancements

1. **Multi-GPU Support**: Detect and distribute across multiple GPUs
2. **Dynamic Batch Sizing**: Adjust batch size based on available memory
3. **FP8 Precision**: Native FP8 support for H100/H200
4. **Profile Learning**: Learn optimal settings from actual runs
5. **Cloud GPU Support**: AWS, Azure, GCP GPU instance profiles

## Notes

- Settings are applied at initialization time
- No code changes needed when switching GPUs
- Profiles optimized for financial ML workloads
- Enterprise GPUs prioritize reproducibility over speed
- Consumer GPUs prioritize speed over determinism
