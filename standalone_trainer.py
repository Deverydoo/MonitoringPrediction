#!/usr/bin/env python3
"""
standalone_trainer.py
Streamlined command-line PyTorch trainer using unified training core
Optimized for PyTorch 2.0.1 + Transformers 4.26.1
"""

import os
import sys
import argparse
from pathlib import Path

# Setup environment
os.environ['NUMEXPR_MAX_THREADS'] = '16'

# Import unified training core
try:
    from training_core import create_trainer, validate_training_environment
    from config import CONFIG
    from common_utils import log_message, analyze_existing_datasets, check_models_like_trainer
except ImportError as e:
    print(f"❌ Failed to import dependencies: {e}")
    sys.exit(1)


def main():
    """Main entry point for standalone training."""
    parser = argparse.ArgumentParser(
        description="Standalone PyTorch trainer for distilled monitoring model"
    )
    parser.add_argument(
        '--config', '-c', 
        type=str, 
        default='config.py',
        help='Configuration file path'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU training'
    )
    parser.add_argument(
        '--no-compile',
        action='store_true',
        help='Disable torch.compile'
    )
    parser.add_argument(
        '--use-dask',
        action='store_true',
        help='Enable Dask GPU acceleration'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load and customize configuration
    try:
        config = CONFIG.copy()
        
        # Apply command line overrides
        if args.epochs:
            config['epochs'] = args.epochs
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if args.learning_rate:
            config['learning_rate'] = args.learning_rate
        if args.force_cpu:
            config['force_cpu'] = True
        if args.no_compile:
            config['torch_compile'] = False
        if args.use_dask:
            config['use_dask_gpu'] = True
            
    except Exception as e:
        log_message(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Validate environment and datasets
    if not validate_training_environment(config):
        return 1
    
    # Initialize and run trainer
    log_message("🚀 STANDALONE DISTILLED MODEL TRAINER")
    log_message("=" * 50)
    
    try:
        import torch
        log_message(f"PyTorch: {torch.__version__}")
        log_message(f"CUDA available: {torch.cuda.is_available()}")
        
        # Check if Dask is requested and available
        if config.get('use_dask_gpu', False):
            try:
                import dask
                log_message(f"Dask available: {dask.__version__}")
            except ImportError:
                log_message("⚠️  Dask requested but not available")
                config['use_dask_gpu'] = False
        
        log_message(f"Configuration: {len(config)} settings loaded")
        
        # Create and run trainer
        trainer = create_trainer(config)
        success = trainer.train()
        
        if success:
            log_message("🎉 Training completed successfully!")
            
            # Verify model was saved
            if check_models_like_trainer(Path(config['models_dir'])):
                log_message("✅ Model verification passed")
            else:
                log_message("⚠️  Model verification failed")
            
            return 0
        else:
            log_message("❌ Training failed!")
            return 1
            
    except Exception as e:
        log_message(f"❌ Trainer initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())