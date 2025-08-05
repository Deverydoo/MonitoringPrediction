#!/usr/bin/env python3
"""
test_tft_training.py - Test Script for TFT Training System
Quick test to verify the new TFT training system works with existing metrics dataset
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_environment():
    """Test if all required dependencies are available."""
    print("🔍 TESTING TFT ENVIRONMENT")
    print("=" * 40)
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda} ({torch.cuda.get_device_name(0)})")
        else:
            print("⚠️  CUDA: Not available (will use CPU)")
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    try:
        import pytorch_lightning as pl
        print(f"✅ PyTorch Lightning: {pl.__version__}")
    except ImportError:
        print("❌ PyTorch Lightning not available")
        return False
    
    try:
        import pytorch_forecasting
        print(f"✅ PyTorch Forecasting: Available")
    except ImportError:
        print("❌ PyTorch Forecasting not available")
        print("💡 Install with: pip install pytorch-forecasting")
        return False
    
    try:
        from safetensors.torch import save_file, load_file
        print("✅ Safetensors: Available")
    except ImportError:
        print("❌ Safetensors not available")
        return False
    
    return True


def test_dataset():
    """Test if the metrics dataset exists and is readable."""
    print("\n📊 TESTING DATASET")
    print("=" * 40)
    
    dataset_path = Path('./training/metrics_dataset.json')
    if not dataset_path.exists():
        print(f"❌ Metrics dataset not found: {dataset_path}")
        print("💡 Please run dataset generation first")
        return False
    
    try:
        import json
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        if 'training_samples' not in data:
            print("❌ Invalid dataset format - missing 'training_samples'")
            return False
        
        samples = data['training_samples']
        print(f"✅ Dataset loaded: {len(samples)} samples")
        
        # Check sample structure
        if samples:
            sample = samples[0]
            required_keys = ['timestamp', 'metrics']
            missing_keys = [key for key in required_keys if key not in sample]
            if missing_keys:
                print(f"⚠️  Sample missing keys: {missing_keys}")
            else:
                print("✅ Sample structure looks good")
                
            # Check metrics
            metrics = sample.get('metrics', {})
            print(f"✅ Found {len(metrics)} metrics in first sample")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        return False


def test_training_core():
    """Test if the new training core can be imported and initialized."""
    print("\n🔧 TESTING TRAINING CORE")
    print("=" * 40)
    
    try:
        from training_core import create_trainer, validate_training_environment
        print("✅ Training core imported successfully")
        
        # Test configuration
        test_config = {
            'training_dir': './training/',
            'models_dir': './models/',
            'checkpoints_dir': './checkpoints/',
            'logs_dir': './logs/',
            'epochs': 1,  # Just 1 epoch for testing
            'batch_size': 16,
            'learning_rate': 0.03,
            'prediction_horizon': 3,
            'context_length': 12,
            'mixed_precision': False  # Disable for testing
        }
        
        # Validate environment
        if validate_training_environment(test_config):
            print("✅ Training environment validation passed")
        else:
            print("❌ Training environment validation failed")
            return False
        
        # Try to create trainer
        trainer = create_trainer(test_config)
        print("✅ TFT trainer created successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import training core: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing training core: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing with a small sample."""
    print("\n🔄 TESTING DATA PREPROCESSING")
    print("=" * 40)
    
    try:
        from training_core import TFTDataPreprocessor
        
        config = {
            'prediction_horizon': 3,
            'context_length': 12
        }
        
        preprocessor = TFTDataPreprocessor(config)
        
        # Try to load and preprocess data
        df = preprocessor.load_existing_metrics_dataset('./training/')
        print(f"✅ Data preprocessing successful: {df.shape}")
        
        # Check if we can create time series dataset
        training_dataset, validation_dataset = preprocessor.create_time_series_dataset(df)
        print(f"✅ Time series datasets created")
        print(f"   Training samples: {len(training_dataset)}")
        print(f"   Validation samples: {len(validation_dataset)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def run_quick_training_test():
    """Run a very quick training test (1 epoch)."""
    print("\n🏋️ QUICK TRAINING TEST")
    print("=" * 40)
    
    try:
        from training_core import create_trainer, validate_training_environment
        
        # Quick test configuration
        test_config = {
            'training_dir': './training/',
            'models_dir': './models/',
            'checkpoints_dir': './checkpoints/',
            'logs_dir': './logs/',
            'epochs': 1,          # Just 1 epoch
            'batch_size': 8,      # Small batch
            'learning_rate': 0.03,
            'prediction_horizon': 2,  # Short horizon
            'context_length': 8,      # Short context
            'mixed_precision': False,
            'early_stopping_patience': 1
        }
        
        print("⚠️  Running 1 epoch training test...")
        print("   This may take a few minutes...")
        
        # Create and run trainer
        trainer = create_trainer(test_config)
        success = trainer.train()
        
        if success:
            print("✅ Quick training test completed successfully!")
            return True
        else:
            print("❌ Quick training test failed")
            return False
            
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False


def main():
    """Run all tests."""
    print("🧪 TFT TRAINING SYSTEM TEST")
    print("=" * 50)
    
    # Run tests
    tests = [
        ("Environment", test_environment),
        ("Dataset", test_dataset),
        ("Training Core", test_training_core),
        ("Data Preprocessing", test_data_preprocessing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} test failed")
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
    
    print(f"\n📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TFT training system is ready.")
        
        # Ask if user wants to run quick training test
        try:
            response = input("\n🏋️ Run quick training test (1 epoch)? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                if run_quick_training_test():
                    print("🎉 Quick training test successful!")
                    print("💡 You can now run full training with:")
                    print("   python distilled_model_trainer.py")
                else:
                    print("❌ Quick training test failed")
        except KeyboardInterrupt:
            print("\n⏸️ Test interrupted by user")
    else:
        print("❌ Some tests failed. Please fix issues before training.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())