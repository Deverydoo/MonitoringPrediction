#!/usr/bin/env python3
"""
tft_trainer.py - Fixed TFT Model Training Module
Resolves the "'int' object is not subscriptable" error
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch
import pandas as pd
import numpy as np
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from safetensors.torch import save_file

from config import CONFIG
import pandas as pd


class TFTTrainer:
    """Fixed TFT trainer that handles data preparation correctly."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def prepare_data(self) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Load and prepare data for TFT training with proper error handling."""
        print("📊 Loading dataset...")
        
        # Load metrics dataset
        dataset_path = Path(self.config["training_dir"]) / "metrics_dataset.json"
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        samples = data["training_samples"]
        print(f"✅ Loaded {len(samples):,} samples")
        
        # Convert to DataFrame with proper error handling
        records = []
        for sample in samples:
            try:
                record = {
                    "timestamp": pd.to_datetime(sample["timestamp"]),
                    "server_name": sample["server_name"],
                    "status": sample["status"],
                    **sample["metrics"]
                }
                records.append(record)
            except Exception as e:
                print(f"⚠️  Skipping malformed sample: {e}")
                continue
        
        if not records:
            raise ValueError("No valid records found in dataset")
        
        df = pd.DataFrame(records)
        
        # CRITICAL FIX: Ensure continuous time series per server
        print("📊 Creating proper time series structure...")
        df = df.sort_values(["server_name", "timestamp"])
        
        # Create proper time index for each server
        server_dfs = []
        min_required = self.config["context_length"] + self.config["prediction_horizon"] + 5
        
        for server in df["server_name"].unique():
            server_df = df[df["server_name"] == server].copy()
            server_df = server_df.sort_values("timestamp").reset_index(drop=True)
            
            # Create continuous time index
            server_df["time_idx"] = range(len(server_df))
            server_df["series_id"] = server
            
            if len(server_df) >= min_required:
                server_dfs.append(server_df)
            else:
                print(f"⚠️  Skipping {server}: only {len(server_df)} points (need {min_required})")
        
        if not server_dfs:
            raise ValueError(f"No servers have enough data points! Need at least {min_required}")
        
        # Combine all valid servers
        df = pd.concat(server_dfs, ignore_index=True)
        print(f"✅ Using {len(server_dfs)} servers with sufficient data")
        
        # Add time features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        
        # CRITICAL FIX: Much more conservative length adjustment
        max_time_per_server = df.groupby("series_id")["time_idx"].max()
        min_max_time = max_time_per_server.min()
        
        # Very conservative settings to ensure data remains
        max_prediction_length = min(3, min_max_time // 10)  # Very small prediction horizon
        max_encoder_length = min(12, min_max_time // 3)     # Smaller encoder length
        
        # Ensure minimum values
        max_prediction_length = max(1, max_prediction_length)
        max_encoder_length = max(6, max_encoder_length)
        
        print(f"📊 Conservative lengths - Encoder: {max_encoder_length}, Prediction: {max_prediction_length}")
        print(f"📊 Min samples per server: {min_max_time}, Required: {max_encoder_length + max_prediction_length}")
        
        # Create training/validation split - much more conservative
        training_cutoff = min_max_time - max_prediction_length - 5
        
        # Define features
        target = "cpu_percent"  # Primary target
        
        # Only use features that exist
        available_cols = set(df.columns)
        time_varying_unknown_reals = []
        for col in ["cpu_percent", "memory_percent", "disk_percent", "load_average"]:
            if col in available_cols:
                time_varying_unknown_reals.append(col)
        
        time_varying_known_reals = ["time_idx", "hour", "day_of_week", "month"]
        
        print(f"🎯 Target: {target}")
        print(f"📊 Features: {len(time_varying_unknown_reals)} numeric columns")
        
        try:
            # Create training dataset with very conservative settings
            training = TimeSeriesDataSet(
                df[df["time_idx"] <= training_cutoff],
                time_idx="time_idx",
                target=target,
                group_ids=["series_id"],
                max_encoder_length=max_encoder_length,
                max_prediction_length=max_prediction_length,
                min_encoder_length=max_encoder_length // 3,  # Much smaller minimum
                min_prediction_length=1,
                time_varying_known_reals=["time_idx"],  # Only time_idx for now
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
                add_relative_time_idx=False,  # Disable to reduce complexity
                add_target_scales=False,      # Disable to reduce complexity
                add_encoder_length=False,     # Disable to reduce complexity
                allow_missing_timesteps=True
            )
            
            # Create validation dataset
            validation = TimeSeriesDataSet.from_dataset(
                training,
                df[df["time_idx"] > training_cutoff],
                predict=True,
                stop_randomization=True
            )
            
            print(f"✅ Training set: {len(training)} sequences")
            print(f"✅ Validation set: {len(validation)} sequences")
            
            if len(training) == 0:
                raise ValueError("Training dataset is empty! Check data filtering.")
            
            return training, validation
            
        except Exception as e:
            print(f"❌ Dataset creation failed: {e}")
            self._debug_data(df, training_cutoff)
            raise
    
    def _debug_data(self, df, training_cutoff):
        """Debug data issues."""
        print(f"🔍 Debug Info:")
        print(f"   Total rows: {len(df)}")
        print(f"   Unique servers: {df['series_id'].nunique()}")
        print(f"   Training cutoff: {training_cutoff}")
        print(f"   Max time_idx: {df['time_idx'].max()}")
        print(f"   Min time_idx: {df['time_idx'].min()}")
        print(f"   Training rows: {len(df[df['time_idx'] <= training_cutoff])}")
        print(f"   Validation rows: {len(df[df['time_idx'] > training_cutoff])}")
        print(f"   Sample time_idx values: {sorted(df['time_idx'].unique())[:10]}...")
    
    def train(self, epochs: Optional[int] = None) -> str:
        """Train TFT model with better error handling."""
        print("🏋️ Starting TFT training...")
        start_time = datetime.now()
        
        try:
            # Prepare data
            training_dataset, validation_dataset = self.prepare_data()
            
            # Create data loaders with conservative settings
            batch_size = min(self.config["batch_size"], 16)  # Smaller batches
            
            train_dataloader = training_dataset.to_dataloader(
                train=True,
                batch_size=batch_size,
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=False,
                shuffle=True
            )
            
            val_dataloader = validation_dataset.to_dataloader(
                train=False,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=False
            )
            
            # Create model with conservative settings
            print("🤖 Creating TFT model...")
            self.model = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate=0.03,
                hidden_size=16,  # Smaller for stability
                attention_head_size=4,
                dropout=0.1,
                hidden_continuous_size=8,
                loss=QuantileLoss(),
                log_interval=10,
                reduce_on_plateau_patience=4
            )
            
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"✅ Model created: {param_count:,} parameters")
            
            # Setup trainer with minimal config
            max_epochs = epochs or min(self.config["epochs"], 10)  # Limit epochs
            
            trainer = Trainer(
                max_epochs=max_epochs,
                gradient_clip_val=0.1,
                enable_progress_bar=True,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                precision="32",  # Use full precision for stability
                callbacks=[
                    EarlyStopping(
                        monitor="val_loss",
                        patience=5,
                        mode="min"
                    )
                ],
                enable_checkpointing=False
            )
            # Train model
            print(f"🚀 Training for {max_epochs} epochs...")
            trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
            
            # Save model
            model_dir = self._save_model()
            
            training_time = datetime.now() - start_time
            print(f"🎉 Training completed in {training_time}")
            print(f"💾 Model saved to: {model_dir}")
            
            return str(model_dir)
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("\n💡 Common fixes:")
            print("   1. Generate more data: generate_dataset(hours=720)")
            print("   2. Check data quality in the dataset")
            print("   3. Reduce batch size or model complexity")
            raise
    
    def _save_model(self) -> Path:
        """Save trained model with proper handling of shared tensors."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(self.config["models_dir"]) / f"tft_model_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Method 1: Create independent tensor copies (recommended)
            state_dict = self.model.state_dict()
            clean_state_dict = {}
            
            for key, tensor in state_dict.items():
                # Create independent copy to avoid shared memory
                if tensor.is_cuda:
                    clean_state_dict[key] = tensor.detach().cpu().clone()
                else:
                    clean_state_dict[key] = tensor.detach().clone()
            
            # Save with Safetensors using clean state dict
            model_path = model_dir / "model.safetensors"
            save_file(clean_state_dict, str(model_path))
            print(f"✅ Model weights saved: {model_path}")
            
            # Save configuration
            config_path = model_dir / "config.json"
            with open(config_path, "w") as f:
                json.dump({
                    "model_type": "TemporalFusionTransformer",
                    "framework": "pytorch_forecasting", 
                    "created_at": timestamp,
                    "training_config": self.config
                }, f, indent=2)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump({
                    "training_completed": True,
                    "model_type": "TFT",
                    "framework": "pytorch_forecasting",
                    "created_at": datetime.now().isoformat(),
                    "model_path": str(model_path),
                    "safetensors_format": True
                }, f, indent=2)
            
            print(f"✅ Model saved successfully: {model_dir}")
            return model_dir
            
        except Exception as e:
            print(f"❌ Safetensors save failed: {e}")
            # Fallback: Save as regular PyTorch checkpoint
            return self._fallback_save_model(model_dir)
    
    def _fallback_save_model(self, model_dir: Path) -> Path:
        """Fallback: Save as regular PyTorch checkpoint."""
        try:
            print("🔄 Using PyTorch checkpoint fallback...")
            
            # Save as regular PyTorch checkpoint
            checkpoint_path = model_dir / "model.ckpt" 
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'model_class': 'TemporalFusionTransformer'
            }, checkpoint_path)
            
            print(f"✅ Fallback model saved: {checkpoint_path}")
            return model_dir
            
        except Exception as e:
            print(f"❌ Fallback save also failed: {e}")
            return None


def train_model(epochs: Optional[int] = None) -> str:
    """Train TFT model - module interface."""
    trainer = TFTTrainer()
    return trainer.train(epochs)


def diagnose_dataset():
    """Diagnose dataset issues."""
    print("\n🔍 Dataset Diagnosis")
    print("=" * 50)
    
    dataset_path = Path(CONFIG["training_dir"]) / "metrics_dataset.json"
    
    if not dataset_path.exists():
        print("❌ No dataset found!")
        return
    
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        samples = data["training_samples"]
        metadata = data.get("metadata", {})
        
        print(f"📊 Total samples: {len(samples):,}")
        print(f"🖥️  Servers: {metadata.get('servers_count', 'unknown')}")
        print(f"⏱️  Time span: {metadata.get('time_span_hours', 'unknown')} hours")
        
        # Analyze server distribution
        if samples:
            df = pd.DataFrame(samples)
            server_counts = df['server_name'].value_counts()
            print(f"\n📈 Samples per server:")
            print(f"   Min: {server_counts.min()}")
            print(f"   Max: {server_counts.max()}")
            print(f"   Mean: {server_counts.mean():.1f}")
            
            # Check minimum requirements
            min_required = CONFIG["context_length"] + CONFIG["prediction_horizon"] + 5
            insufficient = server_counts[server_counts < min_required]
            
            if len(insufficient) > 0:
                print(f"\n⚠️  {len(insufficient)} servers have insufficient data")
                print(f"   Required: {min_required} samples minimum")
                print("\n💡 Solutions:")
                print("   1. Generate more data: generate_dataset(hours=720)")
                print("   2. Reduce context_length in config.py")
                print("   3. Reduce prediction_horizon in config.py")
            else:
                print("\n✅ All servers have sufficient data for training")
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--diagnose", action="store_true", help="Diagnose dataset")
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_dataset()
        return 0
    
    try:
        model_dir = train_model(args.epochs)
        print(f"✅ Model saved to: {model_dir}")
        return 0
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n💡 Run diagnosis: python tft_trainer.py --diagnose")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())