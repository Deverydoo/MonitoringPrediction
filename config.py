#!/usr/bin/env python3
"""
config.py - Clean TFT Configuration
Production-ready configuration for TFT monitoring system
"""

import os
from pathlib import Path
from typing import Dict, Any

# Core Configuration
CONFIG = {
  "framework": "pytorch_forecasting",
  "model_type": "TemporalFusionTransformer",
  "save_format": "safetensors",
  "training_dir": "./training/",
  "models_dir": "./models/",
  "checkpoints_dir": "./checkpoints/",
  "logs_dir": "./logs/",
  
  "hidden_size": 32,
  "attention_heads": 8,
  "dropout": 0.15,
  "continuous_size": 16,
  "hidden_continuous_size": 16,
  "output_size": 7,
  "loss": "quantile",
  "random_seed": 42,
  "epochs": 20,
  "batch_size": 32,
  "learning_rate": 0.01,
  "gradient_clip_val": 0.1,
  "early_stopping_patience": 8,
  "reduce_on_plateau_patience": 4,

  # Phase 2: Enhanced features
  "auto_lr_find": False,  # Set to True to automatically find learning rate
  "lr_monitor_interval": "step",  # Log LR every step
  "log_every_n_steps": 50,  # Log metrics every N steps

  # Phase 3: Advanced optimizations
  "precision": "32-true",  # Options: "32-true", "16-mixed", "bf16-mixed"
  "accumulate_grad_batches": 1,  # Gradient accumulation (1 = disabled)
  "multi_target": False,  # Set to True for multi-target prediction
  
  "prediction_horizon": 96,
  "context_length": 288,
  "min_encoder_length": 144,
  "min_prediction_length": 12,
  
  "poll_interval_minutes": 5,
  "poll_interval_seconds": 300,
  "time_span_hours": 720,
  "anomaly_ratio": 0.12,
  "servers_count": 15,
  "validation_split": 0.2,
  "normalize_features": True,
  "handle_missing_data": True,
  "feature_engineering": True,
  "use_cyclical_encoding": True,
  "num_workers": 4,
  "pin_memory": True,
  "persistent_workers": True,
  "prefetch_factor": 3,
  "mixed_precision": True,
  "model_compression": True,
  "save_checkpoints": True,
  "checkpoint_every_n_epochs": 5,
  "keep_recent_checkpoints": 5,
  "save_best_only": True,
  
  "alert_thresholds": {
    "cpu_percent": {
      "warning": 75.0,
      "critical": 90.0
    },
    "memory_percent": {
      "warning": 80.0,
      "critical": 93.0
    },
    "disk_percent": {
      "warning": 85.0,
      "critical": 95.0
    },
    "load_average": {
      "warning": 4.0,
      "critical": 8.0
    },
    "java_heap_usage": {
      "warning": 82.0,
      "critical": 94.0
    },
    "network_errors": {
      "warning": 50,
      "critical": 200
    },
    "anomaly_score": {
      "warning": 0.65,
      "critical": 0.85
    }
  },
  
  "log_level": "INFO",
  "tensorboard_enabled": True,
  "cleanup_enabled": True,
  "log_retention_days": 45,
  
  "target_metrics": [
    "cpu_percent",
    "memory_percent",
    "disk_percent",
    "load_average",
    "java_heap_usage",
    "network_errors",
    "anomaly_score"
  ],
  
  "time_features": [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
    "is_business_hours"
  ],
  
  "categorical_features": [
    "server_name",
    "status",
    "timeframe",
    "service_type",
    "datacenter",
    "environment"
  ],
  
  "visualization": {
    "enabled": True,
    "save_plots": True,
    "interactive_plots": True,
    "plot_dir": "./plots/",
    "dpi": 150,
    "figsize": [16, 12],
    "style": "dark_background",
    "color_palette": "viridis",
    "show_confidence_intervals": True,
    "show_attention_weights": True,
    "animate_predictions": False,
    "export_formats": ["png", "svg", "html"]
  }
}


def setup_directories() -> bool:
    """Create required directories."""
    try:
        for key in ["training_dir", "models_dir", "checkpoints_dir", "logs_dir"]:
            Path(CONFIG[key]).mkdir(parents=True, exist_ok=True)
        
        # Also create visualization directory if enabled
        if CONFIG["visualization"]["enabled"]:
            Path(CONFIG["visualization"]["plot_dir"]).mkdir(parents=True, exist_ok=True)
        
        return True
    except Exception:
        return False


def get_device() -> str:
    """Get available device."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# Initialize directories on import
setup_directories()