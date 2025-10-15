#!/usr/bin/env python3
"""
main.py - Optimized Main Interface
Streamlined for Parquet-first data pipeline with config integration
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Import our modules
from metrics_generator import generate_dataset
from tft_trainer import train_model
from tft_inference import predict
from config import CONFIG
from linborg_schema import LINBORG_METRICS, NUM_LINBORG_METRICS, validate_linborg_metrics


def setup() -> bool:
    """Validate environment setup."""
    print("🔍 Validating environment...")

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

        # GPU info
        device = "GPU" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            print(f"🔥 Device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            print(f"💻 Device: {device}")

        return True

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def status():
    """Check system status - Parquet-first approach."""
    print("🔍 System Status")
    print("=" * 50)

    training_dir = Path(CONFIG.get("training_dir", "./training/"))
    models_dir = Path(CONFIG.get("models_dir", "./models/"))

    # Check for datasets (Parquet preferred)
    parquet_files = list(training_dir.glob("*.parquet"))
    csv_files = list(training_dir.glob("*.csv"))

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
        else:
            print("❌ Models: None found")
    else:
        print("❌ Models: Directory missing")

    # Device info (consolidated from setup)
    try:
        import torch
        device = "GPU" if torch.cuda.is_available() else "CPU"
        if torch.cuda.is_available():
            print(f"🔥 Device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            print(f"💻 Device: {device}")
    except ImportError:
        print("❌ PyTorch: Not installed")


def train(dataset_path: Optional[str] = None, epochs: Optional[int] = None,
          per_server: bool = False) -> Optional[str]:
    """Train model - delegates to tft_trainer."""
    try:
        # Use config defaults if not specified
        if dataset_path is None:
            dataset_path = CONFIG.get("training_dir", "./training/")

        return train_model(dataset_path, epochs, per_server=per_server)
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='TFT Monitoring System - Optimized CLI',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Setup command
    subparsers.add_parser('setup', help='Validate environment and dependencies')

    # Status command
    subparsers.add_parser('status', help='Show system status (datasets, models, device)')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate training dataset')
    gen_parser.add_argument('--hours', type=int,
                           default=CONFIG.get('time_span_hours', 24),
                           help='Hours of data to generate')
    gen_parser.add_argument('--servers', type=int,
                           default=CONFIG.get('servers_count', 15),
                           help='Number of servers')
    gen_parser.add_argument('--output', type=str,
                           default=CONFIG.get('training_dir', './training/'),
                           help='Output directory')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train TFT model')
    train_parser.add_argument('--epochs', type=int,
                             default=CONFIG.get('epochs', 20),
                             help='Training epochs')
    train_parser.add_argument('--dataset', type=str,
                             help='Dataset directory or file path')
    train_parser.add_argument('--per-server', action='store_true',
                             help='Train separate model per server')

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Run predictions')
    pred_parser.add_argument('--input', type=str,
                            help='Input data file (Parquet or CSV)')
    pred_parser.add_argument('--model', type=str,
                            help='Model directory path')
    pred_parser.add_argument('--horizon', type=int,
                            default=CONFIG.get('prediction_horizon', 96),
                            help='Prediction horizon (timesteps)')

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
        # Note: generate_dataset now outputs Parquet by default
        success = generate_dataset(hours=args.hours)
        return 0 if success else 1

    elif args.command == 'train':
        print("🚀 Training TFT model...")
        model_path = train(args.dataset, args.epochs, args.per_server)
        if model_path:
            print(f"✅ Model saved: {model_path}")
            return 0
        return 1

    elif args.command == 'predict':
        print("🔮 Running predictions...")
        try:
            # Simplified predict call - tft_inference handles file loading
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
__all__ = ['setup', 'status', 'train']