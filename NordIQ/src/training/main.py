#!/usr/bin/env python3
"""
main.py - Main Interface for NordIQ AI Systems
Streamlined CLI for dataset creation, training, and prediction pipeline

Version: 1.1.0
Copyright © 2025 NordIQ AI, LLC. All rights reserved.
"""

# Setup Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
sys.path.insert(0, str(Path(__file__).parent.parent / "generators"))
sys.path.insert(0, str(Path(__file__).parent))  # For tft_trainer

import argparse
from typing import Optional

# Import our modules
from metrics_generator import generate_dataset
from tft_trainer import train_model
# from tft_inference import predict  # TODO: Fix this import
from core.config import MODEL_CONFIG, METRICS_CONFIG, API_CONFIG
from linborg_schema import LINBORG_METRICS, NUM_LINBORG_METRICS, validate_linborg_metrics


def setup() -> bool:
    """Validate environment setup."""
    print("🔍 Validating environment...")
    print("=" * 50)

    try:
        import torch
        import lightning
        import pytorch_forecasting
        import safetensors
        import pandas as pd

        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ Lightning: {lightning.__version__}")
        print(f"✅ PyTorch Forecasting: {pytorch_forecasting.__version__}")
        print(f"✅ Pandas: {pd.__version__}")

        # Check for Parquet support
        try:
            import pyarrow
            print(f"✅ PyArrow (Parquet): {pyarrow.__version__}")
        except ImportError:
            print("⚠️ PyArrow missing - Parquet support unavailable")
            print("   Install with: pip install pyarrow")

        # GPU info
        device = "GPU" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            print(f"🔥 Device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            print(f"💻 Device: {device}")

        # Version info
        try:
            with open('VERSION', 'r') as f:
                version = f.read().strip()
            print(f"📦 System Version: {version}")
        except:
            print("📦 System Version: Unknown")

        print()
        print("Environment validation complete!")
        return True

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  conda activate py310")
        print("  pip install -r requirements.txt")
        return False


def status():
    """Check system status - datasets, models, config."""
    print("🔍 System Status")
    print("=" * 50)

    training_dir = Path(MODEL_CONFIG.get("training_data_dir", "./training/"))
    models_dir = Path(MODEL_CONFIG.get("model_save_dir", "./models/"))

    # Check for datasets (Parquet preferred)
    parquet_files = list(training_dir.glob("*.parquet")) if training_dir.exists() else []
    csv_files = list(training_dir.glob("*.csv")) if training_dir.exists() else []

    if parquet_files:
        print(f"✅ Datasets (Parquet): {len(parquet_files)} file(s)")
        latest_parquet = max(parquet_files, key=lambda p: p.stat().st_mtime)
        print(f"   Latest: {latest_parquet.name}")

        try:
            import pandas as pd
            df = pd.read_parquet(latest_parquet)
            print(f"   Rows: {len(df):,}")
            if 'server_name' in df.columns:
                print(f"   Servers: {df['server_name'].nunique()}")
            if 'profile' in df.columns:
                print(f"   Profiles: {df['profile'].nunique()} ({', '.join(sorted(df['profile'].unique()))})")
            if 'timestamp' in df.columns:
                time_span_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
                print(f"   Time span: {time_span_hours:.1f} hours ({time_span_hours/24:.1f} days)")

            # Check for LINBORG metrics (using centralized schema)
            present, missing = validate_linborg_metrics(df.columns)
            if len(missing) == 0:
                print(f"   Metrics: {NUM_LINBORG_METRICS} LINBORG metrics ✅")
            elif len(present) > 0:
                print(f"   Metrics: {len(present)}/{NUM_LINBORG_METRICS} LINBORG metrics (partial)")
                print(f"   Missing: {', '.join(missing[:5])}...")
            else:
                # Legacy metrics check
                if 'cpu_pct' in df.columns:
                    print(f"   Metrics: Legacy format (needs regeneration)")
        except Exception as e:
            print(f"   Info unavailable: {e}")
    elif csv_files:
        print(f"⚠️ Datasets (CSV only): {len(csv_files)} file(s)")
        print("   Consider regenerating as Parquet for better performance")
    else:
        print("❌ Datasets: None found")
        print(f"   Generate with: python main.py generate --hours 720 --servers 20")

    # Check models
    if models_dir.exists():
        model_dirs = list(models_dir.glob("tft_model_*"))
        if model_dirs:
            print(f"✅ Models: {len(model_dirs)} trained model(s)")
            latest = max(model_dirs, key=lambda p: p.stat().st_mtime)
            print(f"   Latest: {latest.name}")

            if (latest / "model.safetensors").exists():
                print("   Format: Safetensors ✅")
            elif (latest / "model.pth").exists():
                print("   Format: PyTorch (legacy)")

            # Check training info
            import json
            training_info_file = latest / "training_info.json"
            if training_info_file.exists():
                with open(training_info_file) as f:
                    info = json.load(f)
                print(f"   Epochs: {info.get('epochs', 'unknown')}")
                print(f"   Train Loss: {info.get('train_loss', {}).get('final', 'N/A')}")
        else:
            print("❌ Models: None found")
            print(f"   Train with: python main.py train --epochs 20")
    else:
        print("❌ Models: Directory missing")

    # Device info
    try:
        import torch
        device = "GPU" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            print(f"🔥 Device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            print(f"💻 Device: {device}")
    except ImportError:
        print("❌ PyTorch: Not installed")

    # API Config
    print(f"\n📡 API Configuration:")
    print(f"   Daemon: {API_CONFIG.get('daemon_url', 'Not configured')}")
    print(f"   Metrics Generator: {API_CONFIG.get('metrics_generator_url', 'Not configured')}")

    print()


def train(dataset_path: Optional[str] = None, epochs: Optional[int] = None,
          per_server: bool = False, incremental: bool = True) -> Optional[str]:
    """Train model with incremental training support - delegates to tft_trainer."""
    try:
        # Use config defaults if not specified
        if dataset_path is None:
            dataset_path = MODEL_CONFIG.get("training_data_dir", "./training/")

        if epochs is None:
            epochs = MODEL_CONFIG.get("max_epochs", 20)

        return train_model(dataset_path, epochs, per_server=per_server, incremental=incremental)
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='TFT Monitoring System v1.0.0 - Complete ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate environment
  python main.py setup

  # Check system status
  python main.py status

  # Generate training data (30 days, 20 servers)
  python main.py generate --hours 720 --servers 20

  # Train model (20 epochs)
  python main.py train --epochs 20

  # Run predictions
  python main.py predict --input data.parquet --model models/tft_model_latest

Full pipeline:
  python main.py setup
  python main.py generate --hours 720 --servers 20
  python main.py train --epochs 20
  python main.py status
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Setup command
    subparsers.add_parser('setup', help='Validate environment and dependencies')

    # Status command
    subparsers.add_parser('status', help='Show system status (datasets, models, device)')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate training dataset')
    gen_parser.add_argument('--hours', type=int,
                           default=720,  # 30 days recommended
                           help='Hours of data to generate (default: 720 = 30 days)')
    gen_parser.add_argument('--servers', type=int,
                           default=20,
                           help='Number of servers (default: 20)')
    gen_parser.add_argument('--output', type=str,
                           default=MODEL_CONFIG.get('training_data_dir', './training/'),
                           help='Output directory (default: ./training/)')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train TFT model')
    train_parser.add_argument('--epochs', type=int,
                             default=MODEL_CONFIG.get('max_epochs', 20),
                             help='Training epochs (default: 20)')
    train_parser.add_argument('--dataset', type=str,
                             help='Dataset directory or file path')
    train_parser.add_argument('--per-server', action='store_true',
                             help='Train separate model per server (not recommended)')

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Run predictions')
    pred_parser.add_argument('--input', type=str,
                            required=True,
                            help='Input data file (Parquet or CSV)')
    pred_parser.add_argument('--model', type=str,
                            help='Model directory path')
    pred_parser.add_argument('--horizon', type=int,
                            default=MODEL_CONFIG.get('max_prediction_length', 96),
                            help='Prediction horizon in timesteps (default: 96 = 8 hours)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute commands
    if args.command == 'setup':
        return 0 if setup() else 1

    elif args.command == 'status':
        status()
        return 0

    elif args.command == 'generate':
        print(f"🔄 Generating {args.hours}h of data for {args.servers} servers...")
        print(f"   Output: {args.output}")
        print()
        success = generate_dataset(hours=args.hours, num_servers=args.servers)
        if success:
            print(f"\n✅ Dataset generated successfully!")
            print(f"   Next step: python main.py train --epochs 20")
        return 0 if success else 1

    elif args.command == 'train':
        print("🚀 Training TFT model...")
        if args.epochs:
            print(f"   Epochs: {args.epochs}")
        if args.dataset:
            print(f"   Dataset: {args.dataset}")
        print()

        model_path = train(args.dataset, args.epochs, args.per_server)
        if model_path:
            print(f"\n✅ Model saved: {model_path}")
            print(f"   Start services: start_all.bat (Windows) or ./start_all.sh (Linux/Mac)")
            return 0
        return 1

    elif args.command == 'predict':
        print("🔮 Running predictions...")
        try:
            results = predict(data_path=args.input, model_path=args.model,
                            horizon=args.horizon)

            if results:
                print("\n📈 Prediction complete")
                if isinstance(results, dict) and 'predictions' in results:
                    for metric, values in results['predictions'].items():
                        print(f"  {metric}: {len(values)} forecasts")
                return 0
            else:
                print("❌ Prediction failed")
                return 1

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())


# Export functions for notebook/module use
__all__ = ['setup', 'status', 'train', 'generate_dataset', 'train_model', 'predict']
