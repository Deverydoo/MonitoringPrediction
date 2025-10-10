#!/usr/bin/env python3
"""
tft_trainer.py - Fixed TFT Model Trainer
Updated to work with the actual metrics_generator.py output format
Prioritizes parquet files over JSON for better performance
"""

import json
import argparse
import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import time
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from safetensors.torch import save_file

from config import CONFIG


def set_random_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f" Random seed set to {seed} for reproducibility")


class TrainingProgressCallback(Callback):
    """Phase 2: Enhanced progress reporting with ETA and metrics tracking."""

    def __init__(self):
        super().__init__()
        self.train_start_time = None
        self.epoch_start_time = None
        self.best_val_loss = float('inf')

    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        self.train_start_time = time.time()
        print("\n" + "="*60)
        print("[START] TRAINING STARTED")
        print("="*60)

    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each epoch."""
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each epoch with enhanced metrics."""
        if self.epoch_start_time is None:
            return

        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.train_start_time
        current_epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs

        # Calculate ETA
        avg_epoch_time = total_time / current_epoch
        remaining_epochs = max_epochs - current_epoch
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_minutes = eta_seconds / 60

        # Get current metrics
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss', metrics.get('loss', 0))
        val_loss = metrics.get('val_loss', 0)

        # Track best validation loss
        if val_loss > 0 and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            improvement = "[BEST] NEW BEST"
        else:
            improvement = ""

        # Print enhanced progress
        print(f"\n[INFO] Epoch {current_epoch}/{max_epochs} completed in {epoch_time:.1f}s")
        print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {improvement}")
        print(f"   Progress: [{current_epoch}/{max_epochs}] {current_epoch/max_epochs*100:.1f}%")
        print(f"   ETA: {eta_minutes:.1f} min | Elapsed: {total_time/60:.1f} min")

    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""
        if self.train_start_time is None:
            return

        total_time = time.time() - self.train_start_time
        print("\n" + "="*60)
        print("[OK] TRAINING COMPLETE")
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best val loss: {self.best_val_loss:.4f}")
        print("="*60 + "\n")


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
        print(f"[SEARCH] Looking for dataset in: {training_path.absolute()}")
        if training_path.exists():
            files = list(training_path.glob("*"))
            print(f"[DIR] Files found: {[f.name for f in files]}")
        else:
            print(f"[ERROR] Directory doesn't exist: {training_path}")

        # PRIORITY 1: Try partitioned parquet directory (from metrics_generator.py)
        parquet_dir = training_path / 'server_metrics_parquet'
        if parquet_dir.exists() and parquet_dir.is_dir():
            print(f"[LOAD] Loading partitioned parquet dataset: {parquet_dir}")
            try:
                # Read all partitions at once (pandas handles this automatically)
                df = pd.read_parquet(parquet_dir)
                print(f"[OK] Loaded {len(df):,} records from partitioned parquet")
                return self._prepare_dataframe(df)
            except Exception as e:
                print(f"[WARNING]  Failed to load partitioned parquet: {e}")
                # Continue to next option

        # PRIORITY 2: Try single parquet files (faster than JSON)
        parquet_candidates = [
            'server_metrics.parquet',  # From metrics_generator.py CSV mode
            'metrics_dataset.parquet',  # Legacy name
            'demo_dataset.parquet'      # From demo_data_generator.py
        ]

        for parquet_name in parquet_candidates:
            parquet_path = training_path / parquet_name
            if parquet_path.exists():
                print(f"[INFO] Loading parquet dataset: {parquet_path}")
                df = pd.read_parquet(parquet_path)
                print(f"[OK] Loaded {len(df):,} records from parquet")
                return self._prepare_dataframe(df)

        # PRIORITY 3: Try any parquet file in directory
        parquet_files = list(training_path.glob("*.parquet"))
        if parquet_files:
            parquet_file = sorted(parquet_files)[0]  # Use first alphabetically
            print(f"[INFO] Loading found parquet file: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            print(f"[OK] Loaded {len(df):,} records from parquet")
            return self._prepare_dataframe(df)

        # PRIORITY 4: Try CSV files (faster than JSON for large datasets)
        csv_candidates = [
            'server_metrics.csv',
            'metrics_dataset.csv',
            'demo_dataset.csv'
        ]

        for csv_name in csv_candidates:
            csv_path = training_path / csv_name
            if csv_path.exists():
                print(f"[FILE] Loading CSV dataset: {csv_path}")
                df = pd.read_csv(csv_path)
                print(f"[OK] Loaded {len(df):,} records from CSV")
                return self._prepare_dataframe(df)

        # PRIORITY 5: Fallback to JSON (slowest, legacy support)
        json_path = training_path / 'metrics_dataset.json'
        if json_path.exists():
            print(f"[INFO] Loading JSON dataset: {json_path} ([WARNING]  slow for large datasets)")
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
            print(f"[OK] Loaded {len(df):,} records from JSON")
            return self._prepare_dataframe(df)

        # PRIORITY 6: Try any JSON file
        json_files = list(training_path.glob("*.json"))
        # Exclude metadata files
        json_files = [f for f in json_files if 'metadata' not in f.name.lower()]

        if json_files:
            json_file = sorted(json_files)[0]
            print(f"[INFO] Loading found JSON file: {json_file} ([WARNING]  slow for large datasets)")
            with open(json_file, 'r') as f:
                data = json.load(f)

            if 'records' in data:
                records = data['records']
            elif 'training_samples' in data:
                records = data['training_samples']
            else:
                raise ValueError(f"Dataset format not recognized. Keys: {list(data.keys())}")

            df = pd.DataFrame(records)
            print(f"[OK] Loaded {len(df):,} records from JSON")
            return self._prepare_dataframe(df)

        raise FileNotFoundError(
            f"No dataset files found in {training_path.absolute()}\n"
            f"Looked for:\n"
            f"  - Parquet (partitioned): server_metrics_parquet/\n"
            f"  - Parquet (single): *.parquet\n"
            f"  - CSV: *.csv\n"
            f"  - JSON: *.json"
        )
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for TFT training."""
        print("[PREP] Preparing data for TFT training...")
        print(f"[INFO] Original columns: {list(df.columns)}")
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Map server identifier column
        if 'server_id' in df.columns:
            group_col = 'server_id'
        elif 'server_name' in df.columns:
            group_col = 'server_name'
        else:
            raise ValueError(f"No server identifier found. Available columns: {list(df.columns)}")
        
        print(f"[INFO] Using server column: {group_col}")
        
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
            'latency_ms': 'load_average',    # Using latency as proxy for load
            'state': 'status'
        }

        # Apply column mapping
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
                print(f"[INFO] Mapped {old_col} -> {new_col}")

        # Encode server_name to numeric server_id
        if 'server_name' in df.columns and 'server_id' not in df.columns:
            # Create categorical encoding for server names
            df['server_id'] = pd.Categorical(df['server_name']).codes.astype(str)
            print(f"[INFO] Encoded {len(df['server_name'].unique())} server_names to server_id")
        
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
        
        print(f"[INFO] Available metrics: {available_metrics}")
        if missing_metrics:
            print(f"[WARNING] Missing metrics: {missing_metrics}")
            
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
                    print(f"[INFO] Using {disk_candidates[0]} as disk_percent")
                else:
                    df['disk_percent'] = 50.0  # Default value
                    print("[WARNING] No disk metric found, using default value")
            
            if 'load_average' not in df.columns:
                # Use any available network/load metric
                load_candidates = [col for col in df.columns if any(x in col.lower() for x in ['net', 'load', 'latency'])]
                if load_candidates:
                    df['load_average'] = df[load_candidates[0]] / 100  # Scale down network values
                    print(f"[INFO] Using {load_candidates[0]} as load_average")
                else:
                    df['load_average'] = 1.0  # Default value
                    print("[WARNING] No load metric found, using default value")
        
        # Fill any NaN values and ensure numeric types
        for col in required_metrics:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 50.0)
        
        # Ensure server_id column exists for grouping
        if 'server_id' not in df.columns:
            df['server_id'] = df[group_col]
        
        print(f"[OK] Data prepared: {df.shape}")
        print(f"[INFO] Final columns: {list(df.columns)}")
        return df
    
    def create_datasets(self, df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Create training and validation TimeSeriesDataSets."""
        print("[INFO] Creating TimeSeriesDataSets...")

        # Check minimum series length
        min_length = df.groupby('server_id')['time_idx'].count().min()
        print(f"[INFO] Min series length: {min_length}")

        # Adjust parameters based on data availability
        max_encoder_length = min(self.config['context_length'], min_length // 3)
        max_prediction_length = min(self.config['prediction_horizon'], min_length // 10)

        if max_encoder_length < 6 or max_prediction_length < 1:
            raise ValueError(f"Insufficient data: encoder_length={max_encoder_length}, pred_length={max_prediction_length}")

        # Phase 2: Configurable validation split
        validation_split = self.config.get('validation_split', 0.2)
        training_cutoff = int(min_length * (1 - validation_split))
        
        print(f"[INFO] Using encoder length: {max_encoder_length}, prediction length: {max_prediction_length}")
        print(f"[INFO] Validation split: {validation_split:.1%} | Training cutoff: {training_cutoff}")
        
        # Define features
        time_varying_unknown_reals = ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average']
        time_varying_known_reals = ['hour', 'day_of_week', 'month', 'is_weekend']
        time_varying_unknown_categoricals = ['status']

        # Phase 3: Multi-target prediction support
        multi_target = self.config.get('multi_target', False)
        if multi_target:
            # Use all metrics as targets
            target_metrics = self.config.get('target_metrics', ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average'])
            # Filter to only metrics that exist in the data
            available_targets = [m for m in target_metrics if m in df.columns]
            target = available_targets if len(available_targets) > 1 else 'cpu_percent'
            print(f"[INFO] Multi-target mode: {available_targets if isinstance(target, list) else [target]}")
        else:
            target = 'cpu_percent'  # Single target (default)
            print(f"[INFO] Single-target mode: {target}")

        # Create training dataset
        training = TimeSeriesDataSet(
            df[df['time_idx'] <= training_cutoff],
            time_idx='time_idx',
            target=target,  # Phase 3: single or multi-target
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
        
        print(f"[OK] Training samples: {len(training)}")
        print(f"[OK] Validation samples: {len(validation)}")
        
        return training, validation
    
    def find_learning_rate(self, dataset_path: str = "./training/") -> Optional[float]:
        """Find optimal learning rate using PyTorch Lightning Tuner.

        Phase 2: Learning rate finder to automatically determine the best LR.
        Returns the suggested learning rate or None if finder fails.
        """
        print("[SEARCH] Finding optimal learning rate...")

        try:
            # Load and prepare data
            df = self.load_dataset(dataset_path)
            training_dataset, validation_dataset = self.create_datasets(df)

            # Create dataloader
            train_dataloader = training_dataset.to_dataloader(
                train=True,
                batch_size=self.config['batch_size'],
                num_workers=2,
                pin_memory=False
            )

            # Create model
            model = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate=0.001,  # Initial LR, will be tuned
                hidden_size=self.config['hidden_size'],
                attention_head_size=self.config['attention_heads'],
                dropout=self.config['dropout'],
                hidden_continuous_size=self.config['hidden_continuous_size'],
                loss=QuantileLoss(),
                reduce_on_plateau_patience=self.config['reduce_on_plateau_patience']
            )

            # Create minimal trainer for LR finding
            trainer = Trainer(
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                logger=False,
                enable_checkpointing=False
            )

            # Run LR finder
            from lightning.pytorch.tuner import Tuner
            tuner = Tuner(trainer)
            lr_finder = tuner.lr_find(
                model,
                train_dataloaders=train_dataloader,
                min_lr=1e-5,
                max_lr=1.0,
                num_training=100
            )

            # Get suggestion
            if lr_finder and hasattr(lr_finder, 'suggestion'):
                suggested_lr = lr_finder.suggestion()
                print(f"[TIP] Suggested learning rate: {suggested_lr:.6f}")

                # Plot if possible
                try:
                    fig = lr_finder.plot(suggest=True)
                    plot_path = Path(self.config['logs_dir']) / 'lr_finder.png'
                    plot_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(plot_path)
                    print(f"[TRAIN] LR finder plot saved: {plot_path}")
                except Exception as e:
                    print(f"[WARNING]  Could not save LR plot: {e}")

                return suggested_lr
            else:
                print("[WARNING]  LR finder did not return a suggestion")
                return None

        except Exception as e:
            print(f"[WARNING]  LR finder failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def train(self, dataset_path: str = "./training/", find_lr: bool = False) -> Optional[str]:
        """Train the TFT model with Phase 2 optimizations.

        Args:
            dataset_path: Path to training data
            find_lr: If True, run learning rate finder before training

        Phase 2 Features:
        - Learning rate finder (optional)
        - Learning rate monitoring
        - Configurable validation split
        - Enhanced progress reporting
        """
        print("[TRAIN] Starting TFT training (Phase 2 optimized)...")

        # Set random seed for reproducibility
        seed = self.config.get('random_seed', 42)
        set_random_seed(seed)

        try:
            # Phase 2: Optional learning rate finder
            learning_rate = self.config['learning_rate']
            if find_lr:
                suggested_lr = self.find_learning_rate(dataset_path)
                if suggested_lr:
                    learning_rate = suggested_lr
                    print(f"[INFO] Using suggested LR: {learning_rate:.6f}")
                else:
                    print(f"[WARNING]  Using config LR: {learning_rate}")

            # Load and prepare data
            df = self.load_dataset(dataset_path)
            training_dataset, validation_dataset = self.create_datasets(df)

            # Optimal worker configuration for data loading
            optimal_workers = 4 if torch.cuda.is_available() else 2
            use_pin_memory = torch.cuda.is_available()

            print(f" Data loading: {optimal_workers} workers, pin_memory={use_pin_memory}")

            # Create data loaders with multi-threading
            train_dataloader = training_dataset.to_dataloader(
                train=True,
                batch_size=self.config['batch_size'],
                num_workers=optimal_workers,
                pin_memory=use_pin_memory,
                persistent_workers=True if optimal_workers > 0 else False
            )
            val_dataloader = validation_dataset.to_dataloader(
                train=False,
                batch_size=self.config['batch_size'] * 2,
                num_workers=optimal_workers,
                pin_memory=use_pin_memory,
                persistent_workers=True if optimal_workers > 0 else False
            )

            # Create TFT model with Phase 2 learning rate
            self.model = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate=learning_rate,  # Phase 2: use found LR if available
                hidden_size=self.config['hidden_size'],
                attention_head_size=self.config['attention_heads'],
                dropout=self.config['dropout'],
                hidden_continuous_size=self.config['hidden_continuous_size'],
                loss=QuantileLoss(),
                reduce_on_plateau_patience=self.config['reduce_on_plateau_patience']
            )

            print(f"[OK] Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")

            # Setup checkpointing
            checkpoint_callback = ModelCheckpoint(
                dirpath=Path(self.config['checkpoints_dir']),
                filename='tft-{epoch:02d}-{val_loss:.4f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min',
                save_last=True,
                verbose=True
            )
            print(f"[SAVE] Checkpointing enabled: {self.config['checkpoints_dir']}")

            # Setup TensorBoard logging
            logger = TensorBoardLogger(
                save_dir=self.config['logs_dir'],
                name='tft_training',
                version=datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            print(f"[INFO] TensorBoard logging: {logger.log_dir}")

            # Phase 2: Learning rate monitoring
            lr_monitor = LearningRateMonitor(logging_interval='step')
            print("[TRAIN] Learning rate monitoring enabled")

            # Phase 2: Enhanced progress reporting
            progress_callback = TrainingProgressCallback()
            print("[INFO] Enhanced progress reporting enabled")

            # Setup callbacks
            callbacks = [checkpoint_callback, lr_monitor, progress_callback]
            if self.config.get('early_stopping_patience', 0) > 0:
                early_stop = EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['early_stopping_patience'],
                    mode='min',
                    verbose=True
                )
                callbacks.append(early_stop)
                print(f"[STOP] Early stopping: patience={self.config['early_stopping_patience']}")

            # Phase 3: Mixed precision training
            precision = self.config.get('precision', '32-true')
            if precision != '32-true':
                print(f"[FAST] Mixed precision: {precision}")

            # Phase 3: Gradient accumulation
            accumulate_batches = self.config.get('accumulate_grad_batches', 1)
            if accumulate_batches > 1:
                effective_batch = self.config['batch_size'] * accumulate_batches
                print(f"[LOAD] Gradient accumulation: {accumulate_batches} batches (effective batch: {effective_batch})")

            # Create trainer with Phase 1 + 2 + 3 optimizations
            trainer = Trainer(
                max_epochs=self.config['epochs'],
                gradient_clip_val=self.config.get('gradient_clip_val', 0.1),
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                enable_checkpointing=True,
                logger=logger,
                enable_progress_bar=True,
                log_every_n_steps=50,
                callbacks=callbacks,
                # Phase 3: Advanced optimizations
                precision=precision,
                accumulate_grad_batches=accumulate_batches
            )
            
            print(f"[START] Training for {self.config['epochs']} epochs...")
            trainer.fit(self.model, train_dataloader, val_dataloader)
            
            print("[OK] Training completed successfully!")
            
            # Save model
            model_dir = self._save_model()
            return str(model_dir)
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
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
            print(f"[SAVE] Safetensors model saved: {model_path}")
            
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
            
            print(f"[OK] Model saved to: {model_dir}")
            return model_dir
            
        except Exception as e:
            print(f"[ERROR] Model save failed: {e}")
            return model_dir


def train_model(dataset_path: str = "./training/",
                epochs: Optional[int] = None,
                per_server: bool = False) -> Optional[str]:
    """
    Module interface for training.

    Args:
        dataset_path: Path to training data directory
        epochs: Number of training epochs (overrides config)
        per_server: Train separate model per server (default: False - single fleet model)

    Returns:
        Path to trained model directory, or None if failed
    """
    config = CONFIG.copy()
    if epochs is not None:
        config['epochs'] = epochs

    if per_server:
        # Per-server training mode
        print("[PREP] Per-server training mode enabled")
        print("[WARNING]  Note: This will train multiple models (one per server)")
        # For now, just train single model - per-server implementation is future work
        print("[WARNING]  Per-server mode not yet implemented - training fleet-wide model")
        # TODO: Implement per-server model training

    trainer = TFTTrainer(config)
    return trainer.train(dataset_path)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Train TFT model")
    parser.add_argument("--dataset", type=str, default="./training/",
                       help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, help="Training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--per-server", action="store_true",
                       help="Train separate model per server (experimental)")

    args = parser.parse_args()

    # Call train_model function
    model_path = train_model(
        dataset_path=args.dataset,
        epochs=args.epochs,
        per_server=args.per_server
    )

    if model_path:
        print(f"[SUCCESS] Training successful: {model_path}")
        return 0
    else:
        print("[ERROR] Training failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())