#!/usr/bin/env python3
"""
tft_model_trainer_fixed.py - FIXED TFT Model Trainer 
UPDATED: Uses unified 'lightning' package instead of 'pytorch-lightning'
Compatible with pytorch-forecasting==1.0.0 and lightning>=2.0.0
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Configure environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

# Import the UPDATED TFT training core (with fixed Lightning imports)
try:
    from training_core import create_trainer, validate_training_environment
    from common_utils import log_message, check_models_like_trainer
except ImportError as e:
    print(f"❌ Failed to import TFT training components: {e}")
    print("💡 Make sure you have the updated training_core.py with Lightning 2.0+ support")
    print("💡 Run: python quick_lightning_fix.py")
    sys.exit(1)

# UPDATED TFT Configuration (compatible with Lightning 2.0+)
DEFAULT_TFT_CONFIG = {
    # Model architecture
    'model_type': 'TemporalFusionTransformer',
    'hidden_size': 32,
    'attention_heads': 4,
    'dropout': 0.1,
    'continuous_size': 16,
    
    # Training parameters
    'epochs': 30,
    'batch_size': 32,
    'learning_rate': 0.03,
    'early_stopping_patience': 10,
    'mixed_precision': True,
    
    # Time series parameters
    'prediction_horizon': 6,   # Predict 6 time steps ahead
    'context_length': 24,      # Use 24 time steps history
    
    # Directories (same as before for compatibility)
    'training_dir': './training/',
    'models_dir': './models/',
    'checkpoints_dir': './checkpoints/',
    'logs_dir': './logs/',
    
    # Framework settings (UPDATED for Lightning 2.0+)
    'framework': 'pytorch_forecasting',
    'lightning_version': '2.0+',  # Indicates unified Lightning package
    'torch_compile': False,  # TFT doesn't support torch.compile yet
}


class DistilledModelTrainer:
    """FIXED TFT trainer - compatible with Lightning 2.0+."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, resume_training: bool = False):
        """
        Initialize FIXED TFT trainer.
        
        Args:
            config: Training configuration dictionary  
            resume_training: Whether to resume from existing checkpoint
        """
        # Use UPDATED TFT config
        self.config = config or DEFAULT_TFT_CONFIG.copy()
        
        # Merge with defaults
        for key, value in DEFAULT_TFT_CONFIG.items():
            if key not in self.config:
                self.config[key] = value
        
        self.framework = 'pytorch_forecasting'  # Still using TFT
        
        log_message("🎯 FIXED TFT Model Trainer (Lightning 2.0+ Compatible)")
        log_message(f"📊 Using PyTorch Forecasting + unified Lightning package")
        log_message(f"🎮 Model: Temporal Fusion Transformer")
        
        if resume_training:
            latest_model = self.find_latest_model()
            if latest_model:
                log_message(f"🔄 Resume training available: {latest_model}")
            else:
                log_message("🆕 No existing TFT model found, starting fresh")
        
        # Check environment compatibility
        self._check_lightning_compatibility()
    
    def _check_lightning_compatibility(self):
        """Check Lightning package compatibility."""
        try:
            import lightning
            log_message(f"✅ Lightning (unified): {lightning.__version__}")
        except ImportError:
            log_message("❌ Lightning package not found")
            log_message("💡 Run: pip install lightning>=2.0.0")
            return
        
        try:
            import pytorch_forecasting
            log_message("✅ PyTorch Forecasting compatible")
        except ImportError:
            log_message("❌ PyTorch Forecasting not found")
            log_message("💡 Run: pip install pytorch-forecasting==1.0.0")
            return
        
        # Test critical imports
        try:
            from lightning.pytorch.callbacks import EarlyStopping
            from pytorch_forecasting import TemporalFusionTransformer
            log_message("✅ All critical imports working")
        except ImportError as e:
            log_message(f"❌ Import issue: {e}")
            log_message("💡 Run: python quick_lightning_fix.py")
    
    def train(self) -> bool:
        """
        Train using FIXED TFT implementation.
        """
        log_message("🏋️ Starting FIXED TFT model training")
        log_message(f"🔧 Framework: PyTorch Forecasting + Lightning 2.0+")
        log_message(f"📊 Training on metrics dataset")
        
        try:
            # Validate FIXED TFT environment
            if not validate_training_environment(self.config):
                log_message("❌ TFT environment validation failed")
                log_message("💡 Try running: python quick_lightning_fix.py")
                return False
            
            # Check for metrics dataset
            dataset_path = Path(self.config['training_dir']) / 'metrics_dataset.json'
            if not dataset_path.exists():
                log_message(f"❌ Metrics dataset not found: {dataset_path}")
                log_message("💡 Please run dataset generation first")
                return False
            
            # Create FIXED TFT trainer and train
            trainer = create_trainer(self.config)
            success = trainer.train()
            
            if success:
                log_message("🎉 FIXED TFT training completed successfully!")
                self._log_training_summary()
                return True
            else:
                log_message("❌ TFT training failed")
                log_message("💡 Check logs above for specific error details")
                return False
            
        except KeyboardInterrupt:
            log_message("⏸️ Training interrupted by user")
            return False
            
        except Exception as e:
            log_message(f"❌ Training failed: {str(e)}")
            log_message("💡 If this is a Lightning compatibility error:")
            log_message("   1. Run: python quick_lightning_fix.py")
            log_message("   2. Restart your Python environment")
            return False
    
    def find_latest_model(self) -> Optional[str]:
        """Find latest TFT model."""
        models_dir = Path(self.config.get('models_dir', './models/'))
        if not models_dir.exists():
            return None
        
        # Look for TFT model directories
        model_dirs = list(models_dir.glob('tft_monitoring_*'))
        if not model_dirs:
            return None
        
        # Sort by timestamp (newest first)
        model_dirs.sort(reverse=True)
        latest_model = model_dirs[0]
        
        # Verify TFT model files exist
        required_files = ['model.safetensors', 'model_config.json', 'training_metadata.json']
        if all((latest_model / f).exists() for f in required_files):
            return str(latest_model)
        
        return None
    
    def _log_training_summary(self):
        """Log TFT training summary."""
        latest_model = self.find_latest_model()
        if latest_model:
            try:
                metadata_path = Path(latest_model) / 'training_metadata.json'
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                log_message("📊 TFT Training Summary:")
                log_message(f"   Model saved: {latest_model}")
                log_message(f"   Model type: {metadata.get('model_type', 'TFT')}")
                log_message(f"   Framework: {metadata.get('framework', 'pytorch_forecasting')}")
                log_message(f"   Training time: {metadata.get('training_time', 'N/A')}")
                log_message(f"   Final epoch: {metadata.get('final_epoch', 'N/A')}")
                log_message(f"   Best validation loss: {metadata.get('best_val_loss', 'N/A')}")
                
            except Exception as e:
                log_message(f"⚠️  Could not load training summary: {e}")


def main():
    """Command-line interface for FIXED TFT training."""
    parser = argparse.ArgumentParser(
        description="FIXED TFT model trainer (Lightning 2.0+ compatible)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', '-e', type=int, default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b', type=int, default=None,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate', '-lr', type=float, default=None,
        help='Learning rate for training'
    )
    
    # TFT-specific parameters
    parser.add_argument(
        '--prediction-horizon', type=int, default=None,
        help='Number of time steps to predict ahead'
    )
    parser.add_argument(
        '--context-length', type=int, default=None,
        help='Number of historical time steps to use'
    )
    parser.add_argument(
        '--hidden-size', type=int, default=None,
        help='Hidden size for TFT model'
    )
    
    # Training options
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from latest checkpoint'
    )
    parser.add_argument(
        '--no-mixed-precision', action='store_true',
        help='Disable mixed precision training'
    )
    parser.add_argument(
        '--force-cpu', action='store_true',
        help='Force CPU training even if GPU is available'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--config-file', type=str, default=None,
        help='Load configuration from JSON file'
    )
    parser.add_argument(
        '--check-env', action='store_true',
        help='Check Lightning environment compatibility'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check environment if requested
    if args.check_env:
        print("🔍 CHECKING LIGHTNING ENVIRONMENT")
        print("=" * 50)
        
        try:
            import lightning
            print(f"✅ Lightning (unified): {lightning.__version__}")
        except ImportError:
            print("❌ Lightning not found - run: pip install lightning>=2.0.0")
            return 1
        
        try:
            import pytorch_forecasting
            print("✅ PyTorch Forecasting: Available")
        except ImportError:
            print("❌ PyTorch Forecasting not found - run: pip install pytorch-forecasting==1.0.0")
            return 1
        
        try:
            from lightning.pytorch.callbacks import EarlyStopping
            from pytorch_forecasting import TemporalFusionTransformer
            print("✅ All critical imports working")
            print("🎉 Environment is compatible!")
            return 0
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Run: python quick_lightning_fix.py")
            return 1
    
    try:
        # Load config from file if specified
        config = DEFAULT_TFT_CONFIG.copy()
        
        if args.config_file and Path(args.config_file).exists():
            log_message(f"📖 Loading config from: {args.config_file}")
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)
            config.update(file_config)
        
        # Apply command line overrides
        if args.epochs:
            config['epochs'] = args.epochs
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if args.learning_rate:
            config['learning_rate'] = args.learning_rate
        if args.prediction_horizon:
            config['prediction_horizon'] = args.prediction_horizon
        if args.context_length:
            config['context_length'] = args.context_length
        if args.hidden_size:
            config['hidden_size'] = args.hidden_size
        if args.no_mixed_precision:
            config['mixed_precision'] = False
        if args.force_cpu:
            config['force_cpu'] = True
        
        # Create and run FIXED trainer
        trainer = DistilledModelTrainer(config=config, resume_training=args.resume)
        success = trainer.train()
        
        if success:
            log_message("✅ FIXED TFT training completed successfully!")
            log_message("💡 Model saved with Safetensors format")
            log_message("💡 Ready for inference with tft_inference.py")
            sys.exit(0)
        else:
            log_message("❌ TFT training failed!")
            log_message("💡 If Lightning compatibility issues persist:")
            log_message("   python quick_lightning_fix.py")
            sys.exit(1)
            
    except Exception as e:
        log_message(f"❌ Fatal error: {str(e)}")
        if "pytorch_lightning" in str(e) or "LightningModule" in str(e):
            log_message("💡 This looks like a Lightning version conflict")
            log_message("   Run: python quick_lightning_fix.py")
        sys.exit(1)


# Module interface functions for Jupyter notebook usage
def train_tft_model(config: Optional[Dict[str, Any]] = None, resume: bool = False) -> bool:
    """
    Train FIXED TFT model - designed for Jupyter notebook usage.
    
    Args:
        config: Optional configuration dictionary
        resume: Whether to resume from existing checkpoint
        
    Returns:
        bool: True if training succeeded
    """
    trainer = DistilledModelTrainer(config=config, resume_training=resume)
    return trainer.train()


def get_default_config() -> Dict[str, Any]:
    """Get default FIXED TFT configuration."""
    return DEFAULT_TFT_CONFIG.copy()


def validate_setup() -> bool:
    """Validate FIXED TFT training setup."""
    return validate_training_environment(DEFAULT_TFT_CONFIG)


def check_lightning_environment() -> bool:
    """Check if Lightning environment is properly configured."""
    try:
        import lightning
        from lightning.pytorch.callbacks import EarlyStopping
        from pytorch_forecasting import TemporalFusionTransformer
        print("✅ Lightning environment is compatible")
        return True
    except ImportError as e:
        print(f"❌ Lightning environment issue: {e}")
        print("💡 Run: python quick_lightning_fix.py")
        return False


# Backward compatibility functions
def train() -> bool:
    """Legacy function for backward compatibility."""
    log_message("⚠️  Using legacy train() function - consider using train_tft_model()")
    return train_tft_model()


if __name__ == "__main__":
    main()