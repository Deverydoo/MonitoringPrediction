#!/usr/bin/env python3
"""
tft_trainer.py - Fixed TFT Model Trainer
Updated to work with the actual metrics_generator.py output format
Prioritizes parquet files over JSON for better performance
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from safetensors.torch import save_file

from config import CONFIG


class TFTTrainer:
    """TFT trainer that works with metrics_generator.py output format."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
    
    def load_dataset(self, dataset_dir: str = "./training/") -> pd.DataFrame:
        """Load dataset, preferring parquet over JSON format."""
        training_path = Path(dataset_dir)
        
        # Debug: Show what files exist
        print(f"🔍 Looking for dataset in: {training_path.absolute()}")
        if training_path.exists():
            files = list(training_path.glob("*"))
            print(f"📁 Files found: {[f.name for f in files]}")
        else:
            print(f"❌ Directory doesn't exist: {training_path}")
        
        # Try parquet first (better performance) 
        parquet_path = training_path / 'metrics_dataset.parquet'
        if parquet_path.exists():
            print(f"📊 Loading parquet dataset: {parquet_path}")
            df = pd.read_parquet(parquet_path)
            print(f"✅ Loaded {len(df):,} records from parquet")
            return self._prepare_dataframe(df)
        
        # Try parquet with different name
        server_parquet = training_path / 'server_metrics.parquet'
        if server_parquet.exists():
            print(f"📊 Loading server_metrics parquet: {server_parquet}")
            df = pd.read_parquet(server_parquet)
            print(f"✅ Loaded {len(df):,} records from parquet")
            return self._prepare_dataframe(df)
        
        # Fallback to JSON
        json_path = training_path / 'metrics_dataset.json'
        if json_path.exists():
            print(f"📊 Loading JSON dataset: {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Handle the actual structure from metrics_generator.py
            if 'records' in data:
                records = data['records']
            elif 'training_samples' in data:  # Fallback for old format
                records = data['training_samples']
            else:
                raise ValueError(f"Dataset format not recognized. Keys: {list(data.keys())}")
            
            df = pd.DataFrame(records)
            print(f"✅ Loaded {len(df):,} records from JSON")
            return self._prepare_dataframe(df)
        
        # Check for any parquet file
        parquet_files = list(training_path.glob("*.parquet"))
        if parquet_files:
            parquet_file = parquet_files[0]
            print(f"📊 Loading found parquet file: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            print(f"✅ Loaded {len(df):,} records from parquet")
            return self._prepare_dataframe(df)
        
        # Check for any JSON file
        json_files = list(training_path.glob("*.json"))
        if json_files:
            json_file = json_files[0]
            print(f"📊 Loading found JSON file: {json_file}")
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if 'records' in data:
                records = data['records']
            elif 'training_samples' in data:
                records = data['training_samples']
            else:
                raise ValueError(f"Dataset format not recognized. Keys: {list(data.keys())}")
            
            df = pd.DataFrame(records)
            print(f"✅ Loaded {len(df):,} records from JSON")
            return self._prepare_dataframe(df)
        
        raise FileNotFoundError(f"No dataset files found in {training_path.absolute()}\nLooked for: *.parquet, *.json")
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for TFT training."""
        print("🔧 Preparing data for TFT training...")
        print(f"📊 Original columns: {list(df.columns)}")
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Map server identifier column
        if 'server_id' in df.columns:
            group_col = 'server_id'
        elif 'server_name' in df.columns:
            group_col = 'server_name'
        else:
            raise ValueError(f"No server identifier found. Available columns: {list(df.columns)}")
        
        print(f"📊 Using server column: {group_col}")
        
        # Sort by server and time
        df = df.sort_values([group_col, 'timestamp']).reset_index(drop=True)
        
        # Create time index for each server
        df['time_idx'] = df.groupby(group_col).cumcount()
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Map column names to standard format
        column_mapping = {
            'cpu_pct': 'cpu_percent',
            'mem_pct': 'memory_percent',
            'disk_io_mb_s': 'disk_percent',  # Using disk I/O as proxy for disk usage
            'net_in_mb_s': 'load_average',   # Using network as proxy for load
            'state': 'status',
            'server_name': 'server_id'  # Standardize to server_id
        }
        
        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
                print(f"📊 Mapped {old_col} -> {new_col}")
        
        # Ensure status column exists and is categorical string
        if 'status' not in df.columns:
            if 'state' in df.columns:
                df['status'] = df['state']
            else:
                df['status'] = 'normal'  # Default value
        df['status'] = df['status'].fillna('normal').astype(str)
        
        # Define required metrics (after mapping)
        required_metrics = ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average']
        
        # Check which metrics we have after mapping
        available_metrics = [m for m in required_metrics if m in df.columns]
        missing_metrics = [m for m in required_metrics if m not in df.columns]
        
        print(f"📊 Available metrics: {available_metrics}")
        if missing_metrics:
            print(f"⚠️ Missing metrics: {missing_metrics}")
            
            # Try to use alternative columns
            if 'cpu_percent' not in df.columns and 'cpu_pct' in df.columns:
                df['cpu_percent'] = df['cpu_pct']
            if 'memory_percent' not in df.columns and 'mem_pct' in df.columns:
                df['memory_percent'] = df['mem_pct']
            if 'disk_percent' not in df.columns:
                # Use any available disk/io metric
                disk_candidates = [col for col in df.columns if 'disk' in col.lower() or 'io' in col.lower()]
                if disk_candidates:
                    df['disk_percent'] = df[disk_candidates[0]]
                    print(f"📊 Using {disk_candidates[0]} as disk_percent")
                else:
                    df['disk_percent'] = 50.0  # Default value
                    print("⚠️ No disk metric found, using default value")
            
            if 'load_average' not in df.columns:
                # Use any available network/load metric
                load_candidates = [col for col in df.columns if any(x in col.lower() for x in ['net', 'load', 'latency'])]
                if load_candidates:
                    df['load_average'] = df[load_candidates[0]] / 100  # Scale down network values
                    print(f"📊 Using {load_candidates[0]} as load_average")
                else:
                    df['load_average'] = 1.0  # Default value
                    print("⚠️ No load metric found, using default value")
        
        # Fill any NaN values and ensure numeric types
        for col in required_metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 50.0)
        
        # Ensure server_id column exists for grouping
        if 'server_id' not in df.columns:
            df['server_id'] = df[group_col]
        
        print(f"✅ Data prepared: {df.shape}")
        print(f"📊 Final columns: {list(df.columns)}")
        return df
    
    def create_datasets(self, df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Create training and validation TimeSeriesDataSets."""
        print("📊 Creating TimeSeriesDataSets...")
        
        # Check minimum series length
        min_length = df.groupby('server_id')['time_idx'].count().min()
        print(f"📊 Min series length: {min_length}")
        
        # Adjust parameters based on data availability
        max_encoder_length = min(self.config['context_length'], min_length // 3)
        max_prediction_length = min(self.config['prediction_horizon'], min_length // 10)
        
        if max_encoder_length < 6 or max_prediction_length < 1:
            raise ValueError(f"Insufficient data: encoder_length={max_encoder_length}, pred_length={max_prediction_length}")
        
        # Training cutoff (80% for training)
        training_cutoff = int(min_length * 0.8)
        
        print(f"📊 Using encoder length: {max_encoder_length}, prediction length: {max_prediction_length}")
        print(f"📊 Training cutoff: {training_cutoff}")
        
        # Define features
        time_varying_unknown_reals = ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average']
        time_varying_known_reals = ['hour', 'day_of_week', 'month', 'is_weekend']
        time_varying_unknown_categoricals = ['status']
        
        # Create training dataset
        training = TimeSeriesDataSet(
            df[df['time_idx'] <= training_cutoff],
            time_idx='time_idx',
            target='cpu_percent',  # Primary target
            group_ids=['server_id'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            min_encoder_length=max_encoder_length // 2,
            min_prediction_length=1,
            time_varying_unknown_reals=time_varying_unknown_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            target_normalizer=GroupNormalizer(groups=['server_id'], transformation='softplus'),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )
        
        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df[df['time_idx'] > training_cutoff],
            predict=True,
            stop_randomization=True
        )
        
        print(f"✅ Training samples: {len(training)}")
        print(f"✅ Validation samples: {len(validation)}")
        
        return training, validation
    
    def train(self, dataset_path: str = "./training/") -> Optional[str]:
        """Train the TFT model."""
        print("🏋️ Starting TFT training...")
        
        try:
            # Load and prepare data
            df = self.load_dataset(dataset_path)
            training_dataset, validation_dataset = self.create_datasets(df)
            
            # Create data loaders
            train_dataloader = training_dataset.to_dataloader(
                train=True,
                batch_size=self.config['batch_size'],
                num_workers=0  # Avoid multiprocessing issues
            )
            val_dataloader = validation_dataset.to_dataloader(
                train=False,
                batch_size=self.config['batch_size'] * 2,
                num_workers=0
            )
            
            # Create TFT model
            self.model = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate=self.config['learning_rate'],
                hidden_size=self.config['hidden_size'],
                attention_head_size=self.config['attention_heads'],
                dropout=self.config['dropout'],
                hidden_continuous_size=self.config['hidden_continuous_size'],
                loss=QuantileLoss(),
                reduce_on_plateau_patience=self.config['reduce_on_plateau_patience']
            )
            
            print(f"✅ Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
            # Setup trainer
            callbacks = []
            if self.config.get('early_stopping_patience', 0) > 0:
                callbacks.append(
                    EarlyStopping(
                        monitor='val_loss',
                        patience=self.config['early_stopping_patience'],
                        mode='min'
                    )
                )
            
            trainer = Trainer(
                max_epochs=self.config['epochs'],
                gradient_clip_val=self.config.get('gradient_clip_val', 0.1),
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_checkpointing=False,
                logger=False,
                enable_progress_bar=True,
                callbacks=callbacks
            )
            
            print(f"🚀 Training for {self.config['epochs']} epochs...")
            trainer.fit(self.model, train_dataloader, val_dataloader)
            
            print("✅ Training completed successfully!")
            
            # Save model
            model_dir = self._save_model()
            return str(model_dir)
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_model(self) -> Path:
        """Save model with Safetensors format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(self.config['models_dir']) / f"tft_model_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save with Safetensors
            model_path = model_dir / "model.safetensors"
            state_dict = self.model.state_dict()
            
            # Create clean state dict for Safetensors
            clean_state_dict = {}
            for key, tensor in state_dict.items():
                if tensor.is_cuda:
                    clean_state_dict[key] = tensor.detach().cpu().clone()
                else:
                    clean_state_dict[key] = tensor.detach().clone()
            
            save_file(clean_state_dict, str(model_path))
            print(f"💾 Safetensors model saved: {model_path}")
            
            # Save configuration
            config_path = model_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'model_type': 'TemporalFusionTransformer',
                    'framework': 'pytorch_forecasting',
                    'created_at': timestamp,
                    'training_config': {
                        'epochs': self.config['epochs'],
                        'batch_size': self.config['batch_size'],
                        'learning_rate': self.config['learning_rate'],
                        'hidden_size': self.config['hidden_size']
                    }
                }, f, indent=2)
            
            # Save training metadata
            metadata_path = model_dir / "training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'training_completed': True,
                    'model_type': 'TFT',
                    'framework': 'pytorch_forecasting',
                    'created_at': datetime.now().isoformat(),
                    'safetensors_format': True,
                    'model_path': str(model_path)
                }, f, indent=2)
            
            print(f"✅ Model saved to: {model_dir}")
            return model_dir
            
        except Exception as e:
            print(f"❌ Model save failed: {e}")
            return model_dir


def train_model(dataset_path: str = "./training/", epochs: Optional[int] = None) -> Optional[str]:
    """Module interface for training."""
    config = CONFIG.copy()
    if epochs is not None:
        config['epochs'] = epochs
    
    trainer = TFTTrainer(config)
    return trainer.train(dataset_path)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument("--dataset", type=str, default="./training/",
                       help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, help="Training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")
    
    args = parser.parse_args()
    
    # Override config if specified
    config = CONFIG.copy()
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    
    trainer = TFTTrainer(config)
    model_path = trainer.train(args.dataset)
    
    if model_path:
        print(f"🎉 Training successful: {model_path}")
        return 0
    else:
        print("❌ Training failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())