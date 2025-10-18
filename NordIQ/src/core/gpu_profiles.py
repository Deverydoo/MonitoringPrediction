#!/usr/bin/env python3
"""
gpu_profiles.py - GPU Detection and Auto-Configuration

Automatically detects GPU type and applies optimal settings for:
- Tensor Core precision (float32 matmul)
- Memory allocation strategy
- Batch sizes
- Number of workers
- cuDNN settings

Supported GPUs:
- RTX 4090 (Consumer/Workstation)
- Tesla V100/A100 (Data Center)
- H100/H200 (Next-gen Data Center)
- Generic fallback
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class GPUProfile:
    """GPU configuration profile."""
    name: str
    compute_capability: tuple
    tensor_cores: bool
    matmul_precision: str  # 'highest', 'high', 'medium'
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    memory_fraction: float  # Reserve this fraction of GPU memory
    recommended_batch_size_train: int
    recommended_batch_size_inference: int
    num_workers: int
    description: str


# GPU Profile Database
GPU_PROFILES = {
    # RTX 4090 - Ada Lovelace (SM 8.9)
    'RTX 4090': GPUProfile(
        name='RTX 4090',
        compute_capability=(8, 9),
        tensor_cores=True,
        matmul_precision='medium',  # Balance speed/precision
        cudnn_benchmark=True,       # Enable auto-tuning
        cudnn_deterministic=False,  # Allow non-deterministic for speed
        memory_fraction=0.85,       # 24GB VRAM - leave headroom
        recommended_batch_size_train=32,
        recommended_batch_size_inference=128,
        num_workers=8,
        description='Consumer/Workstation GPU - Ada Lovelace architecture'
    ),

    # RTX 3090 - Ampere (SM 8.6)
    'RTX 3090': GPUProfile(
        name='RTX 3090',
        compute_capability=(8, 6),
        tensor_cores=True,
        matmul_precision='medium',
        cudnn_benchmark=True,
        cudnn_deterministic=False,
        memory_fraction=0.85,       # 24GB VRAM
        recommended_batch_size_train=32,
        recommended_batch_size_inference=128,
        num_workers=8,
        description='Consumer/Workstation GPU - Ampere architecture'
    ),

    # Tesla V100 - Volta (SM 7.0)
    'Tesla V100': GPUProfile(
        name='Tesla V100',
        compute_capability=(7, 0),
        tensor_cores=True,
        matmul_precision='high',    # Data center - prioritize precision
        cudnn_benchmark=True,
        cudnn_deterministic=True,   # Reproducibility for enterprise
        memory_fraction=0.90,       # 16GB/32GB variants
        recommended_batch_size_train=64,
        recommended_batch_size_inference=256,
        num_workers=16,
        description='Data Center GPU - Volta architecture'
    ),

    # Tesla A100 - Ampere (SM 8.0)
    'A100': GPUProfile(
        name='Tesla A100',
        compute_capability=(8, 0),
        tensor_cores=True,
        matmul_precision='high',    # Data center precision
        cudnn_benchmark=True,
        cudnn_deterministic=True,
        memory_fraction=0.90,       # 40GB/80GB variants
        recommended_batch_size_train=128,
        recommended_batch_size_inference=512,
        num_workers=32,
        description='Data Center GPU - Ampere architecture with TF32'
    ),

    # H100 - Hopper (SM 9.0)
    'H100': GPUProfile(
        name='H100',
        compute_capability=(9, 0),
        tensor_cores=True,
        matmul_precision='high',    # Hopper has FP8 support
        cudnn_benchmark=True,
        cudnn_deterministic=True,
        memory_fraction=0.90,       # 80GB HBM3
        recommended_batch_size_train=256,
        recommended_batch_size_inference=1024,
        num_workers=32,
        description='Next-gen Data Center GPU - Hopper architecture with FP8'
    ),

    # H200 - Hopper (SM 9.0)
    'H200': GPUProfile(
        name='H200',
        compute_capability=(9, 0),
        tensor_cores=True,
        matmul_precision='high',
        cudnn_benchmark=True,
        cudnn_deterministic=True,
        memory_fraction=0.90,       # 141GB HBM3e
        recommended_batch_size_train=512,
        recommended_batch_size_inference=2048,
        num_workers=32,
        description='Ultra High-Memory Data Center GPU - Hopper with HBM3e'
    ),

    # Generic fallback
    'Generic': GPUProfile(
        name='Generic GPU',
        compute_capability=(0, 0),
        tensor_cores=False,
        matmul_precision='highest', # Conservative default
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        memory_fraction=0.75,
        recommended_batch_size_train=16,
        recommended_batch_size_inference=64,
        num_workers=4,
        description='Generic GPU - conservative settings'
    )
}


class GPUDetector:
    """Detect GPU and apply optimal configuration."""

    def __init__(self):
        self.profile: Optional[GPUProfile] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_name = None
        self.compute_capability = None

        if torch.cuda.is_available():
            self._detect_gpu()
            self._apply_profile()

    def _detect_gpu(self):
        """Detect GPU model and compute capability."""
        self.gpu_name = torch.cuda.get_device_name(0)
        self.compute_capability = torch.cuda.get_device_capability(0)

        print(f"[GPU] Detected: {self.gpu_name}")
        print(f"[GPU] Compute Capability: SM {self.compute_capability[0]}.{self.compute_capability[1]}")

    def _match_profile(self) -> GPUProfile:
        """Match detected GPU to profile."""
        gpu_lower = self.gpu_name.lower()

        # Match by name patterns
        if 'rtx 4090' in gpu_lower or '4090' in gpu_lower:
            return GPU_PROFILES['RTX 4090']
        elif 'rtx 3090' in gpu_lower or '3090' in gpu_lower:
            return GPU_PROFILES['RTX 3090']
        elif 'v100' in gpu_lower:
            return GPU_PROFILES['Tesla V100']
        elif 'a100' in gpu_lower:
            return GPU_PROFILES['A100']
        elif 'h200' in gpu_lower:
            return GPU_PROFILES['H200']
        elif 'h100' in gpu_lower:
            return GPU_PROFILES['H100']

        # Fallback: match by compute capability
        cc = self.compute_capability
        if cc >= (9, 0):  # Hopper
            print("[GPU] Unknown Hopper GPU - using H100 profile")
            return GPU_PROFILES['H100']
        elif cc >= (8, 9):  # Ada Lovelace
            print("[GPU] Unknown Ada GPU - using RTX 4090 profile")
            return GPU_PROFILES['RTX 4090']
        elif cc >= (8, 0):  # Ampere
            print("[GPU] Unknown Ampere GPU - using A100 profile")
            return GPU_PROFILES['A100']
        elif cc >= (7, 0):  # Volta
            print("[GPU] Unknown Volta GPU - using V100 profile")
            return GPU_PROFILES['Tesla V100']
        else:
            print("[GPU] Unknown GPU - using generic profile")
            return GPU_PROFILES['Generic']

    def _apply_profile(self):
        """Apply GPU profile settings."""
        self.profile = self._match_profile()

        print(f"[GPU] Profile: {self.profile.name}")
        print(f"[GPU] {self.profile.description}")

        # Apply Tensor Core precision
        if self.profile.tensor_cores:
            torch.set_float32_matmul_precision(self.profile.matmul_precision)
            print(f"[GPU] Tensor Cores: Enabled (precision={self.profile.matmul_precision})")

        # Apply cuDNN settings
        torch.backends.cudnn.benchmark = self.profile.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.profile.cudnn_deterministic
        print(f"[GPU] cuDNN: benchmark={self.profile.cudnn_benchmark}, "
              f"deterministic={self.profile.cudnn_deterministic}")

        # Set memory fraction
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(self.profile.memory_fraction)
            print(f"[GPU] Memory: {self.profile.memory_fraction*100:.0f}% reserved")

    def get_batch_size(self, mode: str = 'train') -> int:
        """Get recommended batch size for current GPU."""
        if not self.profile:
            return 16 if mode == 'train' else 64

        return (self.profile.recommended_batch_size_train
                if mode == 'train'
                else self.profile.recommended_batch_size_inference)

    def get_num_workers(self) -> int:
        """Get recommended number of data loader workers."""
        if not self.profile:
            return 4
        return self.profile.num_workers

    def get_config(self) -> Dict[str, Any]:
        """Get full configuration dict."""
        if not self.profile:
            return {
                'device': 'cpu',
                'batch_size_train': 16,
                'batch_size_inference': 64,
                'num_workers': 0
            }

        return {
            'device': str(self.device),
            'gpu_name': self.gpu_name,
            'compute_capability': f"SM {self.compute_capability[0]}.{self.compute_capability[1]}",
            'profile': self.profile.name,
            'tensor_cores': self.profile.tensor_cores,
            'matmul_precision': self.profile.matmul_precision,
            'batch_size_train': self.profile.recommended_batch_size_train,
            'batch_size_inference': self.profile.recommended_batch_size_inference,
            'num_workers': self.profile.num_workers,
            'memory_fraction': self.profile.memory_fraction
        }

    def print_summary(self):
        """Print GPU configuration summary."""
        print("\n" + "="*60)
        print("GPU CONFIGURATION SUMMARY")
        print("="*60)

        config = self.get_config()
        for key, value in config.items():
            print(f"  {key:<25}: {value}")

        print("="*60 + "\n")


# Convenience function for quick setup
def setup_gpu() -> GPUDetector:
    """
    Detect and configure GPU automatically.

    Returns:
        GPUDetector instance with applied profile

    Example:
        >>> gpu = setup_gpu()
        >>> batch_size = gpu.get_batch_size('train')
        >>> num_workers = gpu.get_num_workers()
    """
    return GPUDetector()


if __name__ == "__main__":
    # Test GPU detection
    print("Testing GPU Detection and Profiling\n")
    gpu = setup_gpu()
    gpu.print_summary()

    print("\nRecommended Settings:")
    print(f"  Training batch size:   {gpu.get_batch_size('train')}")
    print(f"  Inference batch size:  {gpu.get_batch_size('inference')}")
    print(f"  DataLoader workers:    {gpu.get_num_workers()}")
