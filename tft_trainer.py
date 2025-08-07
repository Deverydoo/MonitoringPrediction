#!/usr/bin/env python3
"""
tft_trainer.py - FIXED TFT Trainer
Works with the actual dataset format from metrics_generator.py
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


class TFTTrainer:
    """TFT trainer that works with the actual dataset format."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'epochs': 10,
            'batch_size': 16,
            'learning_rate': 0.03,
            'prediction_horizon': 6,
            'context_length': 24
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
    
    def load_and_prepare_data(self, dataset_path: str) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Load dataset with CORRECT key mapping."""
        print("📊 Loading dataset...")
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Use correct key from metrics_generator.py
        records = data.get('records', [])
        if not records:
            raise ValueError(f"No records found. Available keys: {list(data.keys())}")
            
        print(f"✅ Loaded {len(records):,} records")
        
        # Convert to DataFrame with the ACTUAL structure
        df = pd.DataFrame(records)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # CRITICAL FIX: Use 'server_id' (actual key) not 'server_name'
        print(f"📊 Available columns: {list(df.columns)}")
        print(f"📊 Sample record keys: {list(records[0].keys())}")
        
        # Verify required columns exist
        required_cols = ['server_id', 'timestamp', 'cpu_percent', 'memory_percent', 'disk_percent', 'load_average']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Sort by server and time (using correct column name)
        df = df.sort_values(['server_id', 'timestamp']).reset_index(drop=True)
        
        # Create time_idx for each server group
        df['time_idx'] = df.groupby('server_id').cumcount()
        
        # Verify data quality
        series_lengths = df.groupby('server_id')['time_idx'].count()
        min_length = series_lengths.min()
        max_length = series_lengths.max()
        
        print(f"📊 Series lengths - Min: {min_length}, Max: {max_length}")
        print(f"📊 Unique servers: {df['server_id'].nunique()}")
        
        if min_length < 50:
            raise ValueError(f"Need longer time series. Min length: {min_length}, need at least 50")
        
        # Adjust parameters based on actual data
        max_prediction_length = min(self.config['prediction_horizon'], min_length // 10)
        max_encoder_length = min(self.config['context_length'], min_length // 3)
        
        # Ensure we have enough data for train/val split
        training_cutoff = int(min_length * 0.8)
        
        print(f"📊 Using encoder length: {max_encoder_length}, prediction length: {max_prediction_length}")
        print(f"📊 Training cutoff: {training_cutoff}")
        
        # Define features that actually exist in the data
        time_varying_unknown_reals = ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average']
        
        # Add time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        time_varying_known_reals = ['hour', 'day_of_week']
        
        # Use status as categorical (exists in the data)
        time_varying_unknown_categoricals = ['status']
        
        # Create training dataset with CORRECT group_ids
        training = TimeSeriesDataSet(
            df[df['time_idx'] <= training_cutoff],
            time_idx='time_idx',
            target='cpu_percent',  # Primary target
            group_ids=['server_id'],  # FIXED: Use actual column name
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            min_encoder_length=max_encoder_length // 2,
            min_prediction_length=1,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            target_normalizer=GroupNormalizer(groups=['server_id']),
            allow_missing_timesteps=True,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
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
        
        # Verify we have meaningful data
        if len(training) == 0:
            raise ValueError("No training samples created! Check data preprocessing.")
        if len(validation) == 0:
            print("⚠️ No validation samples - using training data for validation")
            validation = training
        
        return training, validation
    
    def train(self, dataset_path: str = "./training/metrics_dataset.json") -> Optional[str]:
        """Train the TFT model with proper data handling."""
        print("🏋️ Starting TFT training...")
        
        try:
            # Load and prepare data
            training_dataset, validation_dataset = self.load_and_prepare_data(dataset_path)
            
            # Create data loaders with reduced batch size for small datasets
            actual_batch_size = min(self.config['batch_size'], len(training_dataset))
            print(f"📊 Using batch size: {actual_batch_size}")
            
            train_dataloader = training_dataset.to_dataloader(
                train=True,
                batch_size=actual_batch_size,
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=False  # Avoid memory issues
            )
            
            val_dataloader = validation_dataset.to_dataloader(
                train=False,
                batch_size=actual_batch_size,
                num_workers=0,
                pin_memory=False
            )
            
            print(f"📊 Train batches: {len(train_dataloader)}")
            print(f"📊 Val batches: {len(val_dataloader)}")
            
            # Create model
            self.model = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate=self.config['learning_rate'],
                hidden_size=16,  # Small model for test data
                attention_head_size=2,
                dropout=0.1,
                hidden_continuous_size=8,
                loss=QuantileLoss(),
                reduce_on_plateau_patience=3
            )
            
            print(f"✅ Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
            
            # Setup trainer with minimal configuration
            trainer = Trainer(
                max_epochs=self.config['epochs'],
                gradient_clip_val=0.1,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_checkpointing=False,  # Disable to avoid issues
                logger=False,  # Disable to avoid issues
                enable_progress_bar=True,
                num_sanity_val_steps=0,  # Skip validation sanity check
                callbacks=[]  # No callbacks to avoid conflicts
            )
            
            # Train the model
            print(f"🚀 Training for {self.config['epochs']} epochs...")
            trainer.fit(self.model, train_dataloader, val_dataloader)
            
            print("✅ Training completed successfully!")
            
            # Save model
            model_dir = self._save_model()
            print(f"✅ Model saved to: {model_dir}")
            
            return str(model_dir)
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_model(self) -> Path:
        """Save trained model with Safetensors."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"./models/tft_model_{timestamp}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save with Safetensors
            model_path = model_dir / "model.safetensors"
            state_dict = self.model.state_dict()
            
            # Clean state dict for Safetensors
            clean_state_dict = {}
            for key, tensor in state_dict.items():
                if tensor.is_cuda:
                    clean_state_dict[key] = tensor.detach().cpu().clone()
                else:
                    clean_state_dict[key] = tensor.detach().clone()
            
            save_file(clean_state_dict, str(model_path))
            print(f"💾 Safetensors model saved: {model_path}")
            
            # Save config
            config_path = model_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'model_type': 'TemporalFusionTransformer',
                    'created_at': timestamp,
                    'training_config': self.config
                }, f, indent=2)
            
            # Save metadata
            metadata_path = model_dir / "training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'training_completed': True,
                    'model_type': 'TFT',
                    'framework': 'pytorch_forecasting',
                    'created_at': datetime.now().isoformat(),
                    'safetensors_format': True
                }, f, indent=2)
            
            return model_dir
            
        except Exception as e:
            print(f"❌ Safetensors save failed: {e}")
            # Fallback to PyTorch format
            torch_path = model_dir / "model.pth"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config
            }, torch_path)
            print(f"💾 Fallback PyTorch save: {torch_path}")
            return model_dir


def train_model(dataset_path: str = "./training/metrics_dataset.json", 
               epochs: Optional[int] = None) -> Optional[str]:
    """Module interface for training."""
    config = {
        'epochs': epochs or 10,
        'batch_size': 16,
        'learning_rate': 0.03,
        'prediction_horizon': 6,
        'context_length': 24
    }
    trainer = TFTTrainer(config)
    return trainer.train(dataset_path)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument("--dataset", type=str, default="./training/metrics_dataset.json",
                       help="Path to dataset file")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': 0.03
    }
    
    trainer = TFTTrainer(config)
    model_path = trainer.train(args.dataset)
    
    if model_path:
        print(f"✅ Training successful: {model_path}")
        return 0
    else:
        print("❌ Training failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())