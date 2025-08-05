#!/usr/bin/env python3
"""
config.py - Complete TFT Configuration with all required functions
Includes hardware detection and environment validation
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# STATIC CONFIGURATION - No heavy dependencies
# ============================================================================

CONFIG = {
    # Framework Settings
    "framework": "pytorch_forecasting",
    "model_type": "TemporalFusionTransformer",
    "save_format": "safetensors",
    
    # Project Directories
    "training_dir": "./training/",
    "models_dir": "./models/",
    "checkpoints_dir": "./checkpoints/",
    "logs_dir": "./logs/",
    "data_config_dir": "./data_config/",
    
    # TFT Model Architecture
    "hidden_size": 32,
    "attention_heads": 4,
    "dropout": 0.1,
    "continuous_size": 16,
    "hidden_continuous_size": 8,
    "output_size": 7,  # Default quantiles
    "loss": "quantile",
    
    # Training Parameters (defaults - will be optimized at runtime)
    "epochs": 30,
    "batch_size": 32,  # Runtime will optimize based on available hardware
    "learning_rate": 0.03,
    "gradient_clip_val": 0.1,
    "early_stopping_patience": 10,
    "reduce_on_plateau_patience": 4,
    
    # Time Series Parameters
    "prediction_horizon": 6,      # Predict 6 steps ahead (30 minutes at 5min intervals)
    "context_length": 24,         # Use 24 steps history (2 hours at 5min intervals)
    "min_encoder_length": 12,     # Minimum context length
    "min_prediction_length": 1,   # Minimum prediction length
    
    # Data Collection Settings
    "poll_interval_minutes": 5,   # Data collection interval
    "poll_interval_seconds": 300, # Same in seconds
    
    # Dataset Generation Parameters
    "metrics_samples": 200000,    # Large dataset for TFT training
    "time_span_hours": 168,       # 1 week of data by default
    "anomaly_ratio": 0.15,        # 15% anomalies
    "servers_count": 57,          # Realistic server count
    "validation_split": 0.2,      # 80% train, 20% validation
    
    # Data Processing
    "normalize_features": True,
    "handle_missing_data": True,
    "feature_engineering": True,
    "use_holiday_features": False,
    "use_cyclical_encoding": True,
    
    # Performance Settings (defaults - optimized at runtime)
    "num_workers": 4,             # DataLoader workers
    "pin_memory": False,          # Will be True if CUDA available
    "persistent_workers": True,
    "prefetch_factor": 2,
    "mixed_precision": False,     # Will be True if GPU available
    
    # Model Storage Settings
    "model_compression": True,
    "save_checkpoints": True,
    "checkpoint_every_n_epochs": 5,
    "keep_recent_checkpoints": 3,
    "save_best_only": True,
    
    # MongoDB Settings (optional - for real data)
    "mongodb_enabled": os.environ.get('MONGODB_ENABLED', 'false').lower() == 'true',
    "mongodb_uri": os.environ.get('MONGODB_URI', ''),
    "mongodb_database": os.environ.get('MONGODB_DB', 'monitoring'),
    "mongodb_collection": os.environ.get('MONGODB_COLLECTION', 'server_metrics'),
    
    # Alert Thresholds (for inference)
    "alert_thresholds": {
        "cpu_usage": {"warning": 80.0, "critical": 95.0},
        "memory_usage": {"warning": 85.0, "critical": 95.0},
        "disk_usage": {"warning": 90.0, "critical": 98.0},
        "load_average": {"warning": 5.0, "critical": 10.0},
        "java_heap_usage": {"warning": 85.0, "critical": 95.0},
        "network_errors": {"warning": 100, "critical": 500},
        "anomaly_score": {"warning": 0.7, "critical": 0.9}
    },
    
    # Logging Settings
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "tensorboard_enabled": True,
    "wandb_enabled": False,
    
    # Cleanup Settings
    "cleanup_enabled": True,
    "log_retention_days": 30,
    "auto_cleanup_on_training": True,
    
    # Feature Columns (what metrics to use)
    "target_metrics": ["cpu_percent", "memory_percent", "disk_percent", "load_average"],
    "time_features": ["hour", "day_of_week", "day_of_month", "month"],
    "categorical_features": ["server_name", "status", "timeframe"],
    
    # Advanced Settings
    "use_attention_visualization": False,
    "calculate_feature_importance": True,
    "export_onnx": False,
    "quantize_model": False,
}

# ============================================================================
# ENVIRONMENT DETECTION FUNCTIONS
# ============================================================================

def validate_tft_environment() -> bool:
    """
    Validate that the TFT training environment is properly configured.
    
    Returns:
        bool: True if environment is valid
    """
    try:
        # Check PyTorch
        import torch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        
        # Check for GPU
        if torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️  No GPU available, will use CPU")
        
        # Check Lightning
        try:
            import lightning
            logger.info(f"✅ Lightning: {lightning.__version__}")
        except ImportError:
            try:
                import pytorch_lightning
                logger.warning("⚠️  Using old pytorch_lightning package")
            except ImportError:
                logger.error("❌ Neither lightning nor pytorch_lightning found")
                return False
        
        # Check PyTorch Forecasting
        try:
            import pytorch_forecasting
            logger.info("✅ PyTorch Forecasting available")
        except ImportError:
            logger.error("❌ PyTorch Forecasting not found")
            logger.info("💡 Install with: pip install pytorch-forecasting")
            return False
        
        # Check Safetensors
        try:
            from safetensors.torch import save_file, load_file
            logger.info("✅ Safetensors available")
        except ImportError:
            logger.error("❌ Safetensors not found")
            logger.info("💡 Install with: pip install safetensors")
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to validate environment: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during validation: {e}")
        return False


def detect_training_environment() -> str:
    """
    Detect the current training environment (GPU type or CPU).
    
    Returns:
        str: Description of the training environment
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Detect specific GPU types
            if "H100" in gpu_name or "H200" in gpu_name:
                return f"High-end GPU: {gpu_name} ({gpu_memory:.1f}GB)"
            elif "A100" in gpu_name:
                return f"Data center GPU: {gpu_name} ({gpu_memory:.1f}GB)"
            elif "RTX" in gpu_name or "GTX" in gpu_name:
                return f"Consumer GPU: {gpu_name} ({gpu_memory:.1f}GB)"
            elif "Tesla" in gpu_name or "V100" in gpu_name:
                return f"Professional GPU: {gpu_name} ({gpu_memory:.1f}GB)"
            else:
                return f"GPU: {gpu_name} ({gpu_memory:.1f}GB)"
        else:
            import platform
            cpu_info = platform.processor()
            return f"CPU: {cpu_info if cpu_info else 'Unknown CPU'}"
            
    except Exception as e:
        logger.warning(f"Could not detect training environment: {e}")
        return "Unknown environment"


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.
    
    Returns:
        Dict: System information including hardware and software details
    """
    info = {
        'environment': 'unknown',
        'framework': 'pytorch_forecasting',
        'gpu_available': False,
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'cpu_count': os.cpu_count() or 1,
        'python_version': None,
        'torch_version': None,
        'lightning_version': None,
    }
    
    try:
        import platform
        info['python_version'] = platform.python_version()
        info['platform'] = platform.platform()
        info['processor'] = platform.processor()
    except:
        pass
    
    try:
        import torch
        info['torch_version'] = torch.__version__
        
        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info['cuda_version'] = torch.version.cuda
            info['environment'] = f"GPU: {info['gpu_name']}"
        else:
            info['environment'] = "CPU"
    except:
        pass
    
    try:
        import lightning
        info['lightning_version'] = lightning.__version__
        info['lightning_type'] = 'unified'
    except:
        try:
            import pytorch_lightning
            info['lightning_version'] = pytorch_lightning.__version__
            info['lightning_type'] = 'legacy'
        except:
            pass
    
    return info


# ============================================================================
# CONFIGURATION MANAGEMENT FUNCTIONS
# ============================================================================

def setup_directories() -> bool:
    """
    Create all necessary project directories.
    
    Returns:
        bool: True if all directories created successfully
    """
    try:
        dirs_to_create = [
            CONFIG["training_dir"],
            CONFIG["models_dir"],
            CONFIG["checkpoints_dir"],
            CONFIG["logs_dir"],
            CONFIG["data_config_dir"]
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
        logger.info(f"✅ Created project directories")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create directories: {e}")
        return False


def get_config() -> Dict[str, Any]:
    """
    Get a copy of the current configuration.
    
    Returns:
        Dict: Copy of configuration dictionary
    """
    return CONFIG.copy()


def update_config(updates: Dict[str, Any], validate: bool = True) -> bool:
    """
    Update configuration with new values.
    
    Args:
        updates: Dictionary of configuration updates
        validate: Whether to validate the updates
        
    Returns:
        bool: True if update successful
    """
    try:
        if validate:
            # Validate critical settings
            if 'prediction_horizon' in updates:
                if updates['prediction_horizon'] < 1:
                    logger.error("Prediction horizon must be >= 1")
                    return False
                    
            if 'context_length' in updates:
                if updates['context_length'] < updates.get('prediction_horizon', CONFIG['prediction_horizon']):
                    logger.error("Context length must be >= prediction horizon")
                    return False
        
        CONFIG.update(updates)
        logger.info(f"✅ Configuration updated with {len(updates)} changes")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update config: {e}")
        return False


def save_config(filepath: Optional[str] = None, include_metadata: bool = True) -> bool:
    """
    Save configuration to JSON file.
    
    Args:
        filepath: Path to save config (default: ./tft_config.json)
        include_metadata: Whether to include metadata like timestamp
        
    Returns:
        bool: True if saved successfully
    """
    if filepath is None:
        filepath = "./tft_config.json"
    
    try:
        save_data = CONFIG.copy()
        
        if include_metadata:
            save_data['_metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'version': '2.0',
                'type': 'tft_configuration'
            }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, sort_keys=True)
            
        logger.info(f"💾 Configuration saved to: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save config: {e}")
        return False


def load_config(filepath: Optional[str] = None, merge: bool = True) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to config file (default: ./tft_config.json)
        merge: Whether to merge with existing config or replace
        
    Returns:
        Dict: Loaded configuration
    """
    if filepath is None:
        filepath = "./tft_config.json"
    
    if not Path(filepath).exists():
        logger.warning(f"Config file not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)
        
        # Remove metadata if present
        loaded_config.pop('_metadata', None)
        
        if merge:
            CONFIG.update(loaded_config)
            logger.info(f"📖 Configuration loaded: {filepath}")
        else:
            CONFIG.clear()
            CONFIG.update(loaded_config)
            logger.info(f"📖 Configuration replaced from: {filepath}")
            
        return loaded_config
        
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return {}


def get_runtime_config() -> Dict[str, Any]:
    """
    Get configuration optimized for current runtime environment.
    This function returns config but does NOT import torch.
    Runtime optimization happens in the training/inference scripts.
    
    Returns:
        Dict: Runtime configuration
    """
    runtime_config = CONFIG.copy()
    
    # Add runtime-specific paths
    runtime_config['timestamp'] = datetime.now().strftime('%Y%m%d_%H%M%S')
    runtime_config['run_id'] = f"tft_run_{runtime_config['timestamp']}"
    
    # Ensure directories exist
    setup_directories()
    
    return runtime_config


def get_model_config() -> Dict[str, Any]:
    """
    Get TFT model-specific configuration.
    
    Returns:
        Dict: Model configuration subset
    """
    model_keys = [
        'model_type', 'hidden_size', 'attention_heads', 'dropout',
        'continuous_size', 'hidden_continuous_size', 'output_size',
        'loss', 'prediction_horizon', 'context_length',
        'min_encoder_length', 'min_prediction_length'
    ]
    
    return {k: CONFIG[k] for k in model_keys if k in CONFIG}


def get_training_config() -> Dict[str, Any]:
    """
    Get training-specific configuration.
    
    Returns:
        Dict: Training configuration subset
    """
    training_keys = [
        'epochs', 'batch_size', 'learning_rate', 'gradient_clip_val',
        'early_stopping_patience', 'reduce_on_plateau_patience',
        'validation_split', 'mixed_precision', 'num_workers',
        'pin_memory', 'save_checkpoints', 'checkpoint_every_n_epochs'
    ]
    
    return {k: CONFIG[k] for k in training_keys if k in CONFIG}


def get_data_config() -> Dict[str, Any]:
    """
    Get data processing configuration.
    
    Returns:
        Dict: Data configuration subset
    """
    data_keys = [
        'metrics_samples', 'time_span_hours', 'anomaly_ratio',
        'servers_count', 'poll_interval_minutes', 'normalize_features',
        'handle_missing_data', 'feature_engineering', 'target_metrics',
        'time_features', 'categorical_features'
    ]
    
    return {k: CONFIG[k] for k in data_keys if k in CONFIG}


def validate_config() -> Tuple[bool, list]:
    """
    Validate current configuration for common issues.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues)
    """
    issues = []
    
    # Check critical parameters
    if CONFIG['prediction_horizon'] < 1:
        issues.append("prediction_horizon must be >= 1")
        
    if CONFIG['context_length'] < CONFIG['prediction_horizon']:
        issues.append("context_length should be >= prediction_horizon")
        
    if CONFIG['batch_size'] < 1:
        issues.append("batch_size must be >= 1")
        
    if CONFIG['epochs'] < 1:
        issues.append("epochs must be >= 1")
        
    if CONFIG['learning_rate'] <= 0:
        issues.append("learning_rate must be > 0")
        
    if CONFIG['anomaly_ratio'] < 0 or CONFIG['anomaly_ratio'] > 1:
        issues.append("anomaly_ratio must be between 0 and 1")
    
    # Check paths
    required_dirs = ['training_dir', 'models_dir', 'checkpoints_dir', 'logs_dir']
    for dir_key in required_dirs:
        if not CONFIG.get(dir_key):
            issues.append(f"{dir_key} is not set")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logger.warning(f"⚠️  Configuration has {len(issues)} issues")
        for issue in issues:
            logger.warning(f"   - {issue}")
    
    return is_valid, issues


def reset_to_defaults():
    """Reset configuration to defaults."""
    global CONFIG
    CONFIG = get_default_config()
    logger.info("🔄 Configuration reset to defaults")


def get_default_config() -> Dict[str, Any]:
    """
    Get a fresh copy of default configuration.
    
    Returns:
        Dict: Default configuration
    """
    return {
        "framework": "pytorch_forecasting",
        "model_type": "TemporalFusionTransformer",
        "save_format": "safetensors",
        "training_dir": "./training/",
        "models_dir": "./models/",
        "checkpoints_dir": "./checkpoints/",
        "logs_dir": "./logs/",
        "data_config_dir": "./data_config/",
        "hidden_size": 32,
        "attention_heads": 4,
        "dropout": 0.1,
        "continuous_size": 16,
        "hidden_continuous_size": 8,
        "output_size": 7,
        "loss": "quantile",
        "epochs": 30,
        "batch_size": 32,
        "learning_rate": 0.03,
        "gradient_clip_val": 0.1,
        "early_stopping_patience": 10,
        "reduce_on_plateau_patience": 4,
        "prediction_horizon": 6,
        "context_length": 24,
        "min_encoder_length": 12,
        "min_prediction_length": 1,
        "poll_interval_minutes": 5,
        "poll_interval_seconds": 300,
        "metrics_samples": 200000,
        "time_span_hours": 168,
        "anomaly_ratio": 0.15,
        "servers_count": 57,
        "validation_split": 0.2,
        "normalize_features": True,
        "handle_missing_data": True,
        "feature_engineering": True,
        "use_holiday_features": False,
        "use_cyclical_encoding": True,
        "num_workers": 4,
        "pin_memory": False,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "mixed_precision": False,
        "model_compression": True,
        "save_checkpoints": True,
        "checkpoint_every_n_epochs": 5,
        "keep_recent_checkpoints": 3,
        "save_best_only": True,
        "mongodb_enabled": os.environ.get('MONGODB_ENABLED', 'false').lower() == 'true',
        "mongodb_uri": os.environ.get('MONGODB_URI', ''),
        "mongodb_database": os.environ.get('MONGODB_DB', 'monitoring'),
        "mongodb_collection": os.environ.get('MONGODB_COLLECTION', 'server_metrics'),
        "alert_thresholds": {
            "cpu_usage": {"warning": 80.0, "critical": 95.0},
            "memory_usage": {"warning": 85.0, "critical": 95.0},
            "disk_usage": {"warning": 90.0, "critical": 98.0},
            "load_average": {"warning": 5.0, "critical": 10.0},
            "java_heap_usage": {"warning": 85.0, "critical": 95.0},
            "network_errors": {"warning": 100, "critical": 500},
            "anomaly_score": {"warning": 0.7, "critical": 0.9}
        },
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "tensorboard_enabled": True,
        "wandb_enabled": False,
        "cleanup_enabled": True,
        "log_retention_days": 30,
        "auto_cleanup_on_training": True,
        "target_metrics": ["cpu_percent", "memory_percent", "disk_percent", "load_average"],
        "time_features": ["hour", "day_of_week", "day_of_month", "month"],
        "categorical_features": ["server_name", "status", "timeframe"],
        "use_attention_visualization": False,
        "calculate_feature_importance": True,
        "export_onnx": False,
        "quantize_model": False,
    }


# ============================================================================
# INITIALIZATION
# ============================================================================

# Create directories on import
setup_directories()

# Try to load existing config if available
if Path("./tft_config.json").exists():
    load_config("./tft_config.json", merge=True)
    logger.info("📖 Configuration loaded: ./tft_config.json")


# ============================================================================
# CLI INTERFACE (when run directly)
# ============================================================================

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="TFT Configuration Manager")
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--save', type=str, help='Save config to file')
    parser.add_argument('--load', type=str, help='Load config from file')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    parser.add_argument('--reset', action='store_true', help='Reset to defaults')
    parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set a config value')
    parser.add_argument('--check-env', action='store_true', help='Check TFT environment')
    parser.add_argument('--system-info', action='store_true', help='Show system information')
    
    args = parser.parse_args()
    
    if args.show:
        print("\n🔧 Current TFT Configuration:")
        print("=" * 50)
        for key, value in CONFIG.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
    
    elif args.save:
        if save_config(args.save):
            print(f"✅ Configuration saved to {args.save}")
        else:
            print("❌ Failed to save configuration")
            sys.exit(1)
    
    elif args.load:
        if load_config(args.load):
            print(f"✅ Configuration loaded from {args.load}")
        else:
            print("❌ Failed to load configuration")
            sys.exit(1)
    
    elif args.validate:
        is_valid, issues = validate_config()
        if is_valid:
            print("✅ Configuration is valid")
        else:
            print(f"❌ Configuration has {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
            sys.exit(1)
    
    elif args.reset:
        reset_to_defaults()
        print("✅ Configuration reset to defaults")
    
    elif args.check_env:
        print("\n🔍 Checking TFT Environment:")
        print("=" * 50)
        if validate_tft_environment():
            print("\n✅ TFT environment is properly configured")
            print(f"🎮 Training environment: {detect_training_environment()}")
        else:
            print("\n❌ TFT environment has issues - see messages above")
            sys.exit(1)
    
    elif args.system_info:
        print("\n🖥️  System Information:")
        print("=" * 50)
        info = get_system_info()
        for key, value in info.items():
            if value is not None:
                print(f"{key}: {value}")
    
    elif args.set:
        key, value = args.set
        # Try to parse value as JSON first (for nested configs)
        try:
            import json
            value = json.loads(value)
        except:
            # Try to parse as number
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except:
                # Keep as string
                pass
        
        if update_config({key: value}):
            print(f"✅ Set {key} = {value}")
        else:
            print(f"❌ Failed to set {key}")
            sys.exit(1)
    
    else:
        parser.print_help()