#!/usr/bin/env python3
"""
config.py - Streamlined TFT Configuration
Clean, minimal configuration focused on PyTorch Forecasting TFT training only
Removes all legacy BERT/TensorFlow/language model code
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# PyTorch imports only
try:
    import torch
    import pytorch_lightning as pl
    import pytorch_forecasting
    TORCH_AVAILABLE = True
    DEVICE_TYPE = 'CUDA' if torch.cuda.is_available() else 'CPU'
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE_TYPE = 'CPU'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlined TFT-only configuration
CONFIG = {
    # Framework (TFT only)
    "framework": "pytorch_forecasting",
    "torch_available": TORCH_AVAILABLE,
    "device_type": DEVICE_TYPE,
    
    # Project directories
    "training_dir": "./training/",
    "models_dir": "./models/",
    "checkpoints_dir": "./checkpoints/",
    "logs_dir": "./logs/",
    "data_config_dir": "./data_config/",
    
    # TFT Model architecture
    "model_type": "TemporalFusionTransformer",
    "hidden_size": 32,
    "attention_heads": 4,
    "dropout": 0.1,
    "continuous_size": 16,
    
    # Training parameters
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.03,
    "early_stopping_patience": 10,
    "mixed_precision": True if DEVICE_TYPE == 'CUDA' else False,
    "gradient_clip_val": 0.1,
    
    # Time series parameters
    "prediction_horizon": 6,      # Predict 6 steps ahead (30 minutes at 5min intervals)
    "context_length": 24,         # Use 24 steps history (2 hours at 5min intervals)
    "poll_interval_minutes": 5,   # Data collection interval
    
    # Dataset generation (metrics only)
    "metrics_samples": 200000,    # Large dataset for TFT training
    "time_span_hours": 168,       # 1 week of data by default
    "anomaly_ratio": 0.15,        # 15% anomalies
    "servers_count": 57,          # Realistic server count
    
    # Data processing
    "train_test_split": 0.8,      # 80% train, 20% validation
    "normalize_features": True,
    "handle_missing_data": True,
    "feature_engineering": True,
    
    # Performance optimization
    "num_workers": 4,
    "pin_memory": True if DEVICE_TYPE == 'CUDA' else False,
    "persistent_workers": True,
    
    # Model storage (Safetensors only)
    "save_format": "safetensors",
    "model_compression": True,
    "save_checkpoints": True,
    "checkpoint_every_n_epochs": 5,
    
    # MongoDB integration (for real data)
    "mongodb_enabled": os.environ.get('MONGODB_ENABLED', 'false').lower() == 'true',
    "mongodb_uri": os.environ.get('MONGODB_URI', ''),
    "mongodb_database": os.environ.get('MONGODB_DB', 'monitoring'),
    "mongodb_collection": os.environ.get('MONGODB_COLLECTION', 'server_metrics'),
    
    # Alert thresholds (for inference)
    "alert_thresholds": {
        "cpu_usage": {"warning": 80.0, "critical": 95.0},
        "memory_usage": {"warning": 85.0, "critical": 95.0},
        "disk_usage": {"warning": 90.0, "critical": 98.0},
        "load_average": {"warning": 5.0, "critical": 10.0},
        "java_heap_usage": {"warning": 85.0, "critical": 95.0},
        "network_errors": {"warning": 100, "critical": 500},
        "anomaly_score": {"warning": 0.7, "critical": 0.9}
    },
    
    # Cleanup settings
    "cleanup_enabled": True,
    "keep_recent_checkpoints": 3,
    "log_retention_days": 30,
    "auto_cleanup_on_training": True,
}

def detect_training_environment() -> str:
    """Detect and configure PyTorch training environment."""
    if not TORCH_AVAILABLE:
        logger.error("❌ PyTorch not available")
        return "unavailable"
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        
        logger.info(f"🎮 PyTorch CUDA: {gpu_name} ({gpu_memory}GB)")
        
        # Optimize batch size based on GPU memory
        if gpu_memory >= 40:  # H100/H200 class
            CONFIG['batch_size'] = min(CONFIG['batch_size'], 64)
        elif gpu_memory >= 20:  # RTX 4090/A100 class
            CONFIG['batch_size'] = min(CONFIG['batch_size'], 32)
        elif gpu_memory >= 12:  # RTX 3080/4070 class
            CONFIG['batch_size'] = min(CONFIG['batch_size'], 16)
        else:
            CONFIG['batch_size'] = min(CONFIG['batch_size'], 8)
        
        return "pytorch_cuda"
    
    # CPU fallback
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"💻 PyTorch CPU: {cpu_count} cores")
    CONFIG['batch_size'] = min(CONFIG['batch_size'], max(4, cpu_count // 4))
    CONFIG['mixed_precision'] = False  # Disable for CPU
    
    return "pytorch_cpu"

def setup_directories():
    """Create necessary directories."""
    dirs = [
        CONFIG["training_dir"],
        CONFIG["models_dir"],
        CONFIG["checkpoints_dir"],
        CONFIG["logs_dir"],
        CONFIG["data_config_dir"]
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def validate_tft_environment() -> bool:
    """Validate TFT training environment."""
    logger.info("🔍 Validating TFT environment...")
    
    # Check PyTorch
    if not TORCH_AVAILABLE:
        logger.error("❌ PyTorch not available")
        return False
    
    logger.info(f"✅ PyTorch: {torch.__version__}")
    
    # Check PyTorch Lightning
    try:
        logger.info(f"✅ PyTorch Lightning: {pl.__version__}")
    except Exception as e:
        logger.error(f"❌ PyTorch Lightning issue: {e}")
        return False
    
    # Check PyTorch Forecasting
    try:
        logger.info(f"✅ PyTorch Forecasting available")
    except Exception as e:
        logger.error(f"❌ PyTorch Forecasting issue: {e}")
        return False
    
    # Check datasets
    training_dir = Path(CONFIG["training_dir"])
    metrics_path = training_dir / "metrics_dataset.json"
    
    if not metrics_path.exists():
        logger.warning(f"⚠️  No metrics dataset found: {metrics_path}")
        logger.info("💡 Run dataset generation first")
    else:
        logger.info(f"✅ Metrics dataset found: {metrics_path}")
    
    # Environment summary
    env = detect_training_environment()
    logger.info(f"🚀 Environment: {env}")
    logger.info(f"📊 Batch size optimized: {CONFIG['batch_size']}")
    logger.info(f"⚡ Mixed precision: {CONFIG['mixed_precision']}")
    
    return True

def get_tft_model_config() -> Dict[str, Any]:
    """Get TFT-specific model configuration."""
    return {
        "model_type": CONFIG["model_type"],
        "hidden_size": CONFIG["hidden_size"],
        "attention_heads": CONFIG["attention_heads"],
        "dropout": CONFIG["dropout"],
        "continuous_size": CONFIG["continuous_size"],
        "prediction_horizon": CONFIG["prediction_horizon"],
        "context_length": CONFIG["context_length"],
        "learning_rate": CONFIG["learning_rate"],
        "mixed_precision": CONFIG["mixed_precision"],
        "gradient_clip_val": CONFIG["gradient_clip_val"],
    }

def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    return {
        "epochs": CONFIG["epochs"],
        "batch_size": CONFIG["batch_size"],
        "early_stopping_patience": CONFIG["early_stopping_patience"],
        "train_test_split": CONFIG["train_test_split"],
        "num_workers": CONFIG["num_workers"],
        "pin_memory": CONFIG["pin_memory"],
        "persistent_workers": CONFIG["persistent_workers"],
    }

def get_data_config() -> Dict[str, Any]:
    """Get data processing configuration."""
    return {
        "metrics_samples": CONFIG["metrics_samples"],
        "time_span_hours": CONFIG["time_span_hours"],
        "anomaly_ratio": CONFIG["anomaly_ratio"],
        "servers_count": CONFIG["servers_count"],
        "poll_interval_minutes": CONFIG["poll_interval_minutes"],
        "normalize_features": CONFIG["normalize_features"],
        "handle_missing_data": CONFIG["handle_missing_data"],
        "feature_engineering": CONFIG["feature_engineering"],
    }

def save_config(config_path: str = "./tft_config.json"):
    """Save configuration to JSON file."""
    try:
        # Create serializable config
        save_config = CONFIG.copy()
        save_config["saved_at"] = datetime.now().isoformat()
        save_config["torch_version"] = torch.__version__ if TORCH_AVAILABLE else "N/A"
        save_config["lightning_version"] = pl.__version__ if TORCH_AVAILABLE else "N/A"
        
        with open(config_path, 'w') as f:
            json.dump(save_config, f, indent=2)
        
        logger.info(f"💾 Configuration saved: {config_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save config: {e}")

def load_config(config_path: str = "./tft_config.json"):
    """Load configuration from JSON file."""
    if not Path(config_path).exists():
        logger.info(f"ℹ️  No config file found: {config_path}")
        return
    
    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        
        # Update CONFIG with loaded values
        for key, value in loaded_config.items():
            if key in CONFIG and key not in ["torch_available", "device_type", "saved_at", "torch_version", "lightning_version"]:
                CONFIG[key] = value
        
        logger.info(f"📖 Configuration loaded: {config_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")

def cleanup_old_files():
    """Remove old checkpoints and logs."""
    if not CONFIG.get("cleanup_enabled", True):
        return
    
    try:
        # Cleanup old checkpoints
        checkpoints_dir = Path(CONFIG["checkpoints_dir"])
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob("*.ckpt"))
            if len(checkpoint_files) > CONFIG["keep_recent_checkpoints"]:
                # Sort by modification time, keep recent ones
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                to_remove = checkpoint_files[CONFIG["keep_recent_checkpoints"]:]
                
                for checkpoint in to_remove:
                    checkpoint.unlink()
                
                logger.info(f"🧹 Removed {len(to_remove)} old checkpoints")
        
        # Cleanup old logs
        logs_dir = Path(CONFIG["logs_dir"])
        if logs_dir.exists():
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=CONFIG["log_retention_days"])
            
            old_logs = []
            for log_file in logs_dir.rglob("*.log"):
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    old_logs.append(log_file)
            
            for log_file in old_logs:
                log_file.unlink()
            
            if old_logs:
                logger.info(f"🧹 Removed {len(old_logs)} old log files")
                
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")

def get_system_info() -> Dict[str, Any]:
    """Get system information for diagnostics."""
    info = {
        "framework": CONFIG["framework"],
        "torch_available": TORCH_AVAILABLE,
        "device_type": DEVICE_TYPE,
        "environment": detect_training_environment(),
        "config_valid": validate_tft_environment(),
    }
    
    if TORCH_AVAILABLE:
        info.update({
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        })
        
        if torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory // (1024**3),
            })
    
    return info

# Initialize on import
setup_directories()
load_config()

if __name__ == "__main__":
    print("🚀 TFT Monitoring System Configuration")
    print("=" * 50)
    
    # Validate environment
    if validate_tft_environment():
        print("✅ TFT environment ready!")
        
        # Show system info
        info = get_system_info()
        print(f"\nSystem Info:")
        print(f"  Framework: {info['framework']}")
        print(f"  Environment: {info['environment']}")
        print(f"  Device: {info['device_type']}")
        
        if info.get('gpu_name'):
            print(f"  GPU: {info['gpu_name']} ({info['gpu_memory_gb']}GB)")
        
        print(f"\nTraining Config:")
        print(f"  Batch size: {CONFIG['batch_size']}")
        print(f"  Epochs: {CONFIG['epochs']}")
        print(f"  Mixed precision: {CONFIG['mixed_precision']}")
        print(f"  Context length: {CONFIG['context_length']}")
        print(f"  Prediction horizon: {CONFIG['prediction_horizon']}")
        
        # Save current config
        save_config()
        
    else:
        print("❌ TFT environment validation failed")
        print("💡 Check PyTorch and PyTorch Forecasting installation")