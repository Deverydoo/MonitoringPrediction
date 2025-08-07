#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union, List
import json

# Import our modules
from config import CONFIG
from metrics_generator import generate_dataset as gen_data
from tft_trainer import train_model as train_tft
from tft_inference import predict as run_prediction

def setup() -> bool:
    """Validate environment setup."""
    print("🔍 Validating environment...")
    
    try:
        import torch
        import lightning
        import pytorch_forecasting
        
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Lightning: {lightning.__version__}")
        print(f"✅ PyTorch Forecasting: Available")
        
        if torch.cuda.is_available():
            print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  Using CPU")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install requirements:")
        print("   pip install torch lightning pytorch-forecasting")
        return False

def generate_dataset(hours: Optional[int] = None) -> bool:
    """
    Generate training dataset.
    
    Args:
        hours: Hours of data to generate (default from CONFIG)
        
    Returns:
        True if successful, False otherwise
    """
    if hours is None:
        hours = CONFIG.get('time_span_hours', 336)
        
    try:
        result = gen_data(hours=hours)
        if result:
            print(f"✅ Generated {hours} hours of data")
            return True
        else:
            print("❌ Dataset generation failed")
            return False
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return False

def train(epochs: Optional[int] = None, config: Optional[Dict] = None) -> Optional[str]:
    """
    Train the TFT model.
    
    Args:
        epochs: Number of training epochs
        config: Override configuration
        
    Returns:
        Path to trained model or None if failed
    """
    try:
        training_config = config or {}
        if epochs:
            training_config['epochs'] = epochs
        model_path = train_tft(training_config)
        print(f"✅ Model trained and saved to: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def predict(data: Optional[Union[List, str]] = None) -> Optional[Dict]:
    """
    Run predictions on data.
    
    Args:
        data: Input data (list of dicts) or path to JSON file
        
    Returns:
        Prediction results or None if failed
    """
    if data is None:
        # Generate sample data for testing
        data = []
        for i in range(30):  # 30 minutes of data
            data.append({
                "timestamp": f"2025-01-01T{12 + i//60:02d}:{i%60:02d}:00Z",
                "server_id": "server-001", 
                "cpu_percent": 45.0 + np.random.normal(0, 10),
                "memory_percent": 60.0 + np.random.normal(0, 15), 
                "disk_io_read": 100.0 + np.random.normal(0, 30),
                "disk_io_write": 80.0 + np.random.normal(0, 25),
                "network_bytes_sent": 1000 + np.random.normal(0, 200),
                "network_bytes_recv": 800 + np.random.normal(0, 150),
                "error_count": max(0, int(np.random.poisson(2))),
                "load_average": 1.5 + np.random.normal(0, 0.3)
            })
    elif isinstance(data, str):
        # Load from file
        try:
            with open(data) as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading data file: {e}")
            return None
    
    # Call the actual prediction function from inference module
    try:
        return run_prediction(data)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None

def status():
    """Check system status."""
    print("🔍 System Status Check")
    print("=" * 30)
    
    # Check if dataset exists
    dataset_file = Path("training/metrics_dataset.json")
    if dataset_file.exists():
        print("✅ Dataset: Available")
        try:
            import json
            with open(dataset_file) as f:
                data = json.load(f)
                # Handle both old and new dataset formats
                if isinstance(data, list):
                    print(f"   Records: {len(data):,}")
                elif isinstance(data, dict):
                    samples = data.get("training_samples", data.get("samples", []))
                    metadata = data.get("metadata", {})
                    print(f"   Records: {len(samples):,}")
                    if metadata:
                        hours = metadata.get("time_span_hours", "Unknown")
                        servers = metadata.get("servers_count", "Unknown")
                        print(f"   Time span: {hours} hours")
                        print(f"   Servers: {servers}")
        except Exception as e:
            print(f"   Records: Could not read ({e})")
    else:
        print("❌ Dataset: Missing")
        
    # Check if models exist - look for actual model directory pattern
    models_dir = Path("models")
    if models_dir.exists():
        # Look for TFT model directories (both old and new patterns)
        model_patterns = ["tft_model_*", "tft_monitoring_*"]
        model_dirs = []
        for pattern in model_patterns:
            model_dirs.extend(list(models_dir.glob(pattern)))
        
        if model_dirs:
            print("✅ Model: Available")
            # Sort by modification time (newest first)
            model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_model = model_dirs[0]
            print(f"   Latest: {latest_model.name}")
            print(f"   Total models: {len(model_dirs)}")
            
            # Check model format
            safetensors_file = latest_model / "model.safetensors"
            pytorch_file = latest_model / "pytorch_model.bin"
            checkpoint_files = list(latest_model.glob("*.ckpt"))
            config_file = latest_model / "config.json"
            metadata_file = latest_model / "metadata.json"
            
            if safetensors_file.exists():
                print("   Format: Safetensors ✅")
                if config_file.exists():
                    print("   Config: Available ✅")
                if metadata_file.exists():
                    print("   Metadata: Available ✅")
            elif pytorch_file.exists():
                print("   Format: PyTorch checkpoint")
            elif checkpoint_files:
                print("   Format: Lightning checkpoint") 
            else:
                print("   Format: Unknown")
                
        else:
            print("❌ Model: Not trained")
    else:
        print("❌ Model: Models directory missing")
        
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            print(f"✅ GPU: {gpu_name}")
        else:
            print("⚠️  GPU: Not available (using CPU)")
    except ImportError:
        print("❌ PyTorch: Not installed")

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='TFT Server Monitoring System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate mock dataset')
    gen_parser.add_argument('--hours', type=int, default=336, help='Hours of data to generate')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train TFT model')
    train_parser.add_argument('--epochs', type=int, help='Training epochs')
    
    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Run predictions')
    pred_parser.add_argument('--input', help='Input JSON file')
    
    # Status command
    subparsers.add_parser('status', help='Check system status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'generate':
        print(f"🔄 Generating {args.hours} hours of mock data...")
        result = generate_dataset(args.hours)
        if result:
            print("✅ Dataset generated successfully")
        return 0 if result else 1
        
    elif args.command == 'train':
        print("🚀 Training TFT model...")
        result = train(args.epochs)
        return 0 if result else 1
        
    elif args.command == 'predict':
        print("🔮 Running predictions...")
        data = None
        if args.input:
            import json
            try:
                with open(args.input) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"❌ Error loading input file: {e}")
                return 1
        
        result = predict(data)
        if result and 'predictions' in result:
            print("\n📈 Predictions (next 30 minutes):")
            for metric, values in result['predictions'].items():
                current = values[0]
                future = values[-1] 
                trend = "📈" if future > current else "📉" if future < current else "➡️"
                print(f"   {metric}: {current:.1f} {trend} {future:.1f}")
                
        if result.get('alerts'):
            print(f"\n⚠️  {len(result['alerts'])} alerts generated:")
            for alert in result['alerts'][:3]:
                icon = "🔴" if alert['severity'] == 'critical' else "🟡"
                print(f"   {icon} {alert['metric']}: {alert['value']:.1f} (step {alert['steps_ahead']})")
        else:
            print("\n✅ No alerts - system healthy")
            
        return 0
        
    elif args.command == 'status':
        status()
        return 0
        
    return 1

# Allow module to be imported and used directly
if __name__ == "__main__":
    sys.exit(main())