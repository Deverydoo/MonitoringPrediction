#!/usr/bin/env python3
"""
distilled_model_trainer.py - SIMPLIFIED trainer using ONLY unified training core
All duplicate training logic removed - uses training_core.py exclusively
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Configure encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['NUMEXPR_MAX_THREADS'] = '16'

# Import ONLY the unified training core
try:
    from training_core import create_trainer, validate_training_environment
    from config import CONFIG, FRAMEWORK, FRAMEWORK_AVAILABLE, DEVICE_TYPE
    from common_utils import log_message, check_models_like_trainer
except ImportError as e:
    raise ImportError(f"Failed to import project modules: {e}")


class DistilledModelTrainer:
    """Simplified trainer - delegates ALL training to unified core."""
    
    def __init__(self, config: Dict[str, Any], resume_training: bool = False):
        """Initialize trainer with minimal setup."""
        self.config = config
        self.framework = FRAMEWORK
        
        log_message(f"📁 Using {self.framework.title()} framework")
        log_message(f"🎮 Device type: {DEVICE_TYPE}")
        
        if resume_training:
            latest_model = self.find_latest_model()
            if latest_model:
                log_message("🔄 Resume training option available")
            else:
                log_message("🆕 No existing model found, starting fresh")
    
    def train(self) -> bool:
        """Train using ONLY unified training core - no duplicate logic."""
        
        # Setup logging
        logs_dir = Path('./logs/')
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        log_message("🏋️ Starting distilled model training")
        log_message(f"🔧 Framework: {self.framework.title()}")
        
        try:
            # Validate environment using common function
            if not validate_training_environment(self.config):
                return False
            
            # Create and run unified trainer - NO other training logic
            trainer = create_trainer(self.config)
            success = trainer.train()
            
            if success:
                log_message(f"🎉 Training completed successfully!")
                return True
            else:
                log_message("❌ Training failed")
                return False
            
        except KeyboardInterrupt:
            log_message("⏸️ Training interrupted by user")
            return False
            
        except Exception as e:
            log_message(f"❌ Training failed: {str(e)}")
            return False
    
    def find_latest_model(self) -> Optional[str]:
        """Find latest model using common utilities."""
        models_dir = Path(self.config.get('models_dir', './models/'))
        if check_models_like_trainer(models_dir):
            from common_utils import get_latest_model_path
            return get_latest_model_path(models_dir)
        return None


# Backwards compatibility functions for notebook interface
def main():
    """Main training function for direct execution."""
    from config import CONFIG
    from common_utils import ensure_directory_structure
    
    # Setup environment
    ensure_directory_structure(CONFIG)
    
    # Initialize trainer
    trainer = DistilledModelTrainer(CONFIG)
    
    # Start training
    success = trainer.train()
    
    if success:
        log_message("🎉 Training completed successfully!")
        return True
    else:
        log_message("❌ Training failed!")
        return False


# Export for notebook compatibility
def create_distilled_trainer(config: Dict[str, Any]) -> DistilledModelTrainer:
    """Factory function for notebook interface."""
    return DistilledModelTrainer(config)


if __name__ == "__main__":
    main()