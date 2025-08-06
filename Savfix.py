#!/usr/bin/env python3
“””
Fixed model saving with proper tensor handling for Safetensors
“””

import torch
from safetensors.torch import save_file
from pathlib import Path
import json
from datetime import datetime

def save_model_safely(self):
“”“Save model using Safetensors format with proper tensor handling.”””
if self.model is None:
return

```
# Create models directory
models_dir = Path(self.config.get('models_dir', './models/'))
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_dir = models_dir / f'tft_monitoring_{timestamp}'
model_dir.mkdir(parents=True, exist_ok=True)

try:
    # Get state dict and create independent tensor copies
    state_dict = self.model.state_dict()
    
    # Create a clean state dict with independent tensors
    clean_state_dict = {}
    for key, tensor in state_dict.items():
        # Create independent copy to avoid shared memory issues
        if tensor.is_cuda:
            # For GPU tensors, move to CPU first then clone
            clean_state_dict[key] = tensor.detach().cpu().clone()
        else:
            # For CPU tensors, just clone
            clean_state_dict[key] = tensor.detach().clone()
    
    # Save with Safetensors using clean state dict
    model_path = model_dir / 'model.safetensors'
    save_file(clean_state_dict, str(model_path))
    log_message(f"💾 Model weights saved securely: {model_path}")
    
    # Alternative: Save as regular PyTorch checkpoint as backup
    torch_path = model_dir / 'model.pth'
    torch.save({
        'model_state_dict': clean_state_dict,
        'model_config': self.model.hparams,
        'timestamp': timestamp
    }, torch_path)
    log_message(f"💾 PyTorch backup saved: {torch_path}")
    
    # Save preprocessing components
    if hasattr(self.preprocessor, 'scalers') and self.preprocessor.scalers:
        scalers_path = model_dir / 'scalers.pkl'
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.preprocessor.scalers, f)
    
    if hasattr(self.preprocessor, 'encoders') and self.preprocessor.encoders:
        encoders_path = model_dir / 'encoders.pkl'
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.preprocessor.encoders, f)
    
    # Save model configuration
    config_path = model_dir / 'model_config.json'
    model_config = {
        'model_type': 'TemporalFusionTransformer',
        'framework': 'pytorch_forecasting',
        'pytorch_version': torch.__version__,
        'training_config': self.config,
        'timestamp': timestamp,
        'training_stats': self.training_stats,
        'model_hyperparams': dict(self.model.hparams) if hasattr(self.model, 'hparams') else {}
    }
    
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2, default=str)
    
    # Save training metadata
    metadata_path = model_dir / 'training_metadata.json'
    metadata = {
        'training_completed': True,
        'model_type': 'TFT',
        'framework': 'pytorch_forecasting',
        'training_time': str(datetime.now() - self.training_stats['start_time']),
        'final_epoch': self.trainer.current_epoch if self.trainer else 0,
        'best_val_loss': float(self.trainer.callback_metrics.get('val_loss', float('inf'))) if self.trainer else None,
        'model_path': str(model_path),
        'created_at': datetime.now().isoformat(),
        'safetensors_format': True
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log_message(f"✅ TFT model saved successfully: {model_dir}")
    return str(model_dir)
    
except Exception as e:
    log_message(f"❌ Failed to save model: {e}")
    # Try fallback saving method
    return self._fallback_save_model(model_dir)
```

def _fallback_save_model(self, model_dir: Path) -> str:
“”“Fallback model saving if Safetensors fails.”””
try:
log_message(“🔄 Trying fallback model saving…”)

```
    # Save just the PyTorch model without Safetensors
    torch_path = model_dir / 'model.pth'
    torch.save({
        'model': self.model,  # Save entire model
        'config': self.config,
        'timestamp': datetime.now().isoformat()
    }, torch_path)
    
    log_message(f"💾 Fallback model saved: {torch_path}")
    return str(model_dir)
    
except Exception as e:
    log_message(f"❌ Fallback save also failed: {e}")
    return None
```
