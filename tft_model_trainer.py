#!/usr/bin/env python3
"""
distilled_model_trainer.py - TFT Model Trainer (Replacement)
REPLACEMENT for the old BERT-based trainer - now uses Temporal Fusion Transformer
All duplicate training logic removed - uses new TFT training_core.py exclusively
Can be used as both importable module and command-line tool
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

# Import the NEW TFT training core (replacing old BERT core)
try:
    from training_core import create_trainer, validate_training_environment
    from common_utils import log_message, check_models_like_trainer
except ImportError as e:
    print(f"❌ Failed to import TFT training components: {e}")
    print("💡 Make sure you have the new training_core.py with TFT support")
    sys.exit(1)

# TFT Configuration (replaces old BERT config)
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
    
    # Framework settings
    'framework': 'pytorch_forecasting',
    'torch_compile': False,  # TFT doesn't support torch.compile yet
}


class DistilledModelTrainer:
    """TFT trainer - REPLACEMENT for old BERT trainer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, resume_training: bool = False):
        """
        Initialize TFT trainer (replaces BERT trainer).
        
        Args:
            config: Training configuration dictionary  
            resume_training: Whether to resume from existing checkpoint
        """
        # Use TFT config instead of old BERT config
        self.config = config or DEFAULT_TFT_CONFIG.copy()
        
        # Merge with defaults
        for key, value in DEFAULT_TFT_CONFIG.items():
            if key not in self.config:
                self.config[key] = value
        
        self.framework = 'pytorch_forecasting'  # No more BERT
        
        log_message("🎯 TFT Model Trainer Initialized (BERT Replacement)")
        log_message(f"📊 Using PyTorch Forecasting framework")
        log_message(f"🎮 Model: Temporal Fusion Transformer")
        
        if resume_training:
            latest_model = self.find_latest_model()
            if latest_model:
                log_message(f"🔄 Resume training available: {latest_model}")
            else:
                log_message("🆕 No existing TFT model found, starting fresh")
    
    def train(self) -> bool:
        """
        Train using TFT - NO MORE BERT training logic.
        """
        log_message("🏋️ Starting TFT model training (BERT replacement)")
        log_message(f"🔧 Framework: PyTorch Forecasting TFT")
        log_message(f"📊 Training on metrics dataset only (ignoring language dataset)")
        
        try:
            # Validate TFT environment (replaces old BERT validation)
            if not validate_training_environment(self.config):
                log_message("❌ TFT environment validation failed")
                return False
            
            # Check for metrics dataset
            dataset_path = Path(self.config['training_dir']) / 'metrics_dataset.json'
            if not dataset_path.exists():
                log_message(f"❌ Metrics dataset not found: {dataset_path}")
                log_message("💡 Please run dataset generation first")
                return False
            
            # Create TFT trainer and train (replaces old BERT trainer)
            trainer = create_trainer(self.config)
            success = trainer.train()
            
            if success:
                log_message("🎉 TFT training completed successfully!")
                self._log_training_summary()
                return True
            else:
                log_message("❌ TFT training failed")
                return False
            
        except KeyboardInterrupt:
            log_message("⏸️ Training interrupted by user")
            return False
            
        except Exception as e:
            log_message(f"❌ Training failed: {str(e)}")
            return False
    
    def find_latest_model(self) -> Optional[str]:
        """Find latest TFT model (replaces BERT model search)."""
        models_dir = Path(self.config.get('models_dir', './models/'))
        if not models_dir.exists():
            return None
        
        # Look for TFT model directories (instead of BERT)
        model_dirs = list(models_dir.glob('tft_monitoring_*'))
        if not model_dirs:
            return None
        
        # Sort by timestamp (newest first)
        model_dirs.sort(reverse=True)
        latest_model = model_dirs[0]
        
        # Verify TFT model files exist
        required_files = ['model.safetensors', 'config.json', 'training_metadata.json']
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
    """Command-line interface for TFT training."""
    parser = argparse.ArgumentParser(
        description="TFT model trainer for server monitoring (BERT replacement)",
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
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
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
        
        # Create and run trainer
        trainer = DistilledModelTrainer(config=config, resume_training=args.resume)
        success = trainer.train()
        
        if success:
            log_message("✅ TFT training completed successfully!")
            sys.exit(0)
        else:
            log_message("❌ TFT training failed!")
            sys.exit(1)
            
    except Exception as e:
        log_message(f"❌ Fatal error: {str(e)}")
        sys.exit(1)


# Module interface functions for Jupyter notebook usage
def train_tft_model(config: Optional[Dict[str, Any]] = None, resume: bool = False) -> bool:
    """
    Train TFT model - designed for Jupyter notebook usage.
    
    Args:
        config: Optional configuration dictionary
        resume: Whether to resume from existing checkpoint
        
    Returns:
        bool: True if training succeeded
    """
    trainer = DistilledModelTrainer(config=config, resume_training=resume)
    return trainer.train()


def get_default_config() -> Dict[str, Any]:
    """Get default TFT configuration."""
    return DEFAULT_TFT_CONFIG.copy()


def validate_setup() -> bool:
    """Validate TFT training setup."""
    return validate_training_environment(DEFAULT_TFT_CONFIG)


# Backward compatibility functions (so existing notebooks don't break)
def train() -> bool:
    """Legacy function for backward compatibility."""
    log_message("⚠️  Using legacy train() function - consider using train_tft_model()")
    return train_tft_model()


class TFTModelTrainer:
    """Alias for backward compatibility."""
    def __init__(self, *args, **kwargs):
        log_message("⚠️  TFTModelTrainer is deprecated, use DistilledModelTrainer")
        self._trainer = DistilledModelTrainer(*args, **kwargs)
    
    def train(self):
        return self._trainer.train()


if __name__ == "__main__":
    main()