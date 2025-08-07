#!/usr/bin/env python3
"""
main.py - FIXED Main Interface
Consistent with the fixed data pipeline
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# Import our fixed modules
from metrics_generator import generate_dataset
from tft_trainer import train_model
from tft_inference import predict


def setup() -> bool:
    """Validate environment setup."""
    print("🔍 Validating environment...")
    
    try:
        import torch
        import lightning
        import pytorch_forecasting
        import safetensors
        
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Lightning: {lightning.__version__}")
        print(f"✅ PyTorch Forecasting: Available")
        print(f"✅ Safetensors: Available")
        
        if torch.cuda.is_available():
            print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ Using CPU")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def status():
    """Check system status with correct structure."""
    print("🔍 System Status")
    print("=" * 30)
    
    # Check dataset
    dataset_path = Path("./training/metrics_dataset.json")
    if dataset_path.exists():
        print("✅ Dataset: Available")
        try:
            with open(dataset_path) as f:
                data = json.load(f)
            
            # Use correct keys from fixed generator
            records = data.get('records', [])
            metadata = data.get('metadata', {})
            
            print(f"   Records: {len(records):,}")
            if metadata:
                print(f"   Servers: {metadata.get('servers_count', 'Unknown')}")
                print(f"   Hours: {metadata.get('time_span_hours', 'Unknown')}")
                print(f"   Total samples: {metadata.get('total_samples', 'Unknown')}")
        except Exception as e:
            print(f"   Error reading dataset: {e}")
    else:
        print("❌ Dataset: Not found")
    
    # Check models
    models_dir = Path("./models")
    if models_dir.exists():
        model_dirs = list(models_dir.glob("tft_model_*"))
        if model_dirs:
            print("✅ Models: Available")
            latest = sorted(model_dirs)[-1]
            print(f"   Latest: {latest.name}")
            
            safetensors_file = latest / "model.safetensors"
            pytorch_file = latest / "model.pth"
            if safetensors_file.exists():
                print("   Format: Safetensors ✅")
            elif pytorch_file.exists():
                print("   Format: PyTorch")
            else:
                print("   Format: Unknown")
        else:
            print("❌ Models: None found")
    else:
        print("❌ Models: Directory missing")
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ GPU: Not available")
    except ImportError:
        print("❌ PyTorch: Not installed")


def train(dataset_path: str = "./training/metrics_dataset.json", epochs: Optional[int] = None) -> Optional[str]:
    """Train model with error handling."""
    try:
        return train_model(dataset_path, epochs)
    except Exception as e:
        print(f"❌ Training error: {e}")
        return None


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='TFT Monitoring System')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    subparsers.add_parser('setup', help='Check environment setup')
    subparsers.add_parser('status', help='Check system status')
    
    gen_parser = subparsers.add_parser('generate', help='Generate dataset')
    gen_parser.add_argument('--hours', type=int, default=24, help='Hours of data')
    
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    train_parser.add_argument('--dataset', type=str, help='Dataset path')
    
    pred_parser = subparsers.add_parser('predict', help='Run predictions')
    pred_parser.add_argument('--input', type=str, help='Input data file')
    pred_parser.add_argument('--horizon', type=int, default=6, help='Prediction horizon')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == 'setup':
        success = setup()
        return 0 if success else 1
    
    elif args.command == 'status':
        status()
        return 0
    
    elif args.command == 'generate':
        print(f"🔄 Generating {args.hours} hours of data...")
        success = generate_dataset(hours=args.hours, output_file="./training/metrics_dataset.json")
        if success:
            print("✅ Dataset generated successfully")
        return 0 if success else 1
    
    elif args.command == 'train':
        print("🚀 Training TFT model...")
        dataset_path = args.dataset or "./training/metrics_dataset.json"
        model_path = train(dataset_path, args.epochs)
        if model_path:
            print(f"✅ Training successful: {model_path}")
            return 0
        else:
            print("❌ Training failed")
            return 1
    
    elif args.command == 'predict':
        print("🔮 Running predictions...")
        
        data = None
        if args.input:
            try:
                with open(args.input) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"❌ Failed to load input: {e}")
                return 1
        
        results = predict(data, horizon=args.horizon)
        
        print("\n📈 Predictions:")
        for metric, values in results['predictions'].items():
            current = values[0]
            future = values[-1]
            trend = "📈" if future > current else "📉" if future < current else "➡️"
            print(f"  {metric}: {current:.1f} {trend} {future:.1f}")
        
        if results.get('alerts'):
            print(f"\n⚠️ {len(results['alerts'])} alerts:")
            for alert in results['alerts'][:3]:
                icon = "🔴" if alert['severity'] == 'critical' else "🟡"
                print(f"  {icon} {alert['metric']}: {alert['value']:.1f} in {alert['minutes_ahead']} min")
        else:
            print("\n✅ No alerts - system healthy")
        
        return 0
    
    return 1


if __name__ == "__main__":
    sys.exit(main())

# Export functions for notebook use
__all__ = ['setup', 'status', 'generate_dataset', 'train', 'predict']