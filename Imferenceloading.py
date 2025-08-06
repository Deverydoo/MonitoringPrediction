#!/usr/bin/env python3
“””
Enhanced model loading with Safetensors and fallback support
“””

import torch
from pathlib import Path
import json
from typing import Optional, Dict, Any
from safetensors.torch import load_file

def load_tft_model_enhanced(model_path: str) -> tuple[Optional[object], Optional[Dict]]:
“””
Load TFT model with enhanced error handling and format support.

```
Args:
    model_path: Path to model directory or file
    
Returns:
    tuple: (model, metadata) or (None, None) if failed
"""
model_dir = Path(model_path)

if not model_dir.exists():
    log_message(f"❌ Model path not found: {model_path}")
    return None, None

# Try loading from Safetensors first
safetensors_path = model_dir / 'model.safetensors'
config_path = model_dir / 'model_config.json'
metadata_path = model_dir / 'training_metadata.json'

try:
    if safetensors_path.exists() and config_path.exists():
        return _load_from_safetensors(safetensors_path, config_path, metadata_path)
except Exception as e:
    log_message(f"⚠️  Safetensors loading failed: {e}")

# Fallback to PyTorch format
torch_path = model_dir / 'model.pth'
try:
    if torch_path.exists():
        return _load_from_pytorch(torch_path, metadata_path)
except Exception as e:
    log_message(f"⚠️  PyTorch loading failed: {e}")

log_message("❌ No valid model format found")
return None, None
```

def _load_from_safetensors(safetensors_path: Path, config_path: Path, metadata_path: Path):
“”“Load model from Safetensors format.”””
log_message(f”📥 Loading TFT model from Safetensors: {safetensors_path}”)

```
# Load configuration
with open(config_path, 'r') as f:
    config = json.load(f)

# Load metadata if available
metadata = {}
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

# Load model state dict
state_dict = load_file(str(safetensors_path))

# Create model instance (this is a placeholder - you'll need to recreate the TFT model)
# For now, return the state dict and config
model_info = {
    'state_dict': state_dict,
    'config': config,
    'model_type': 'TemporalFusionTransformer'
}

log_message("✅ Safetensors model loaded successfully")
return model_info, metadata
```

def _load_from_pytorch(torch_path: Path, metadata_path: Path):
“”“Load model from PyTorch format.”””
log_message(f”📥 Loading TFT model from PyTorch: {torch_path}”)

```
# Load PyTorch checkpoint
checkpoint = torch.load(torch_path, map_location='cpu')

# Load metadata if available
metadata = {}
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

# Extract model
if 'model' in checkpoint:
    model = checkpoint['model']
elif 'model_state_dict' in checkpoint:
    # State dict only - would need to reconstruct model
    model = {
        'state_dict': checkpoint['model_state_dict'],
        'config': checkpoint.get('config', {}),
        'model_type': 'TemporalFusionTransformer'
    }
else:
    # Assume entire checkpoint is the model
    model = checkpoint

log_message("✅ PyTorch model loaded successfully")
return model, metadata
```

# Enhanced TFTInference class with better model loading

class EnhancedTFTInference:
“”“Enhanced TFT inference with robust model loading.”””

```
def __init__(self, model_path: Optional[str] = None):
    self.model = None
    self.model_config = None
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find and load model
    self.model_path = self._find_model_path(model_path)
    if self.model_path:
        self._load_model_enhanced()
    else:
        log_message("⚠️  No trained model found. Please train a model first.")

def _load_model_enhanced(self):
    """Load model using enhanced loading method."""
    if not self.model_path:
        return
    
    try:
        model_info, metadata = load_tft_model_enhanced(self.model_path)
        
        if model_info is None:
            log_message("❌ Failed to load any model format")
            return
        
        # Store model information
        if isinstance(model_info, dict):
            if 'state_dict' in model_info:
                # We have a state dict - would need to reconstruct the TFT model
                self.model_state = model_info['state_dict']
                self.model_config = model_info.get('config', {})
            else:
                # We have a full model
                self.model = model_info
        else:
            # Direct model object
            self.model = model_info
        
        log_message("✅ Enhanced model loading completed")
        
    except Exception as e:
        log_message(f"❌ Enhanced model loading failed: {e}")
        self.model = None

def is_ready(self) -> bool:
    """Check if inference engine is ready."""
    return (self.model is not None or 
            (hasattr(self, 'model_state') and self.model_state is not None))

def _find_model_path(self, model_path: Optional[str]) -> Optional[str]:
    """Find the latest trained model."""
    if model_path and Path(model_path).exists():
        return model_path
    
    # Look for latest model in models directory
    models_dir = Path('./models/')
    if not models_dir.exists():
        return None
    
    # Find TFT model directories
    model_dirs = list(models_dir.glob('tft_monitoring_*'))
    if not model_dirs:
        return None
    
    # Sort by timestamp (newest first)
    model_dirs.sort(reverse=True)
    
    # Find first valid model
    for model_dir in model_dirs:
        # Check for either Safetensors or PyTorch format
        if (model_dir / 'model.safetensors').exists() or (model_dir / 'model.pth').exists():
            return str(model_dir)
    
    return None
```
