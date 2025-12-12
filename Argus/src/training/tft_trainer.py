#!/usr/bin/env python3
"""
tft_trainer.py - ArgusAI TFT Model Trainer
Updated to work with the actual metrics_generator.py output format
Prioritizes parquet files over JSON for better performance

Built by Craig Giannelli and Claude Code
"""

# Setup Python path for imports
import sys
from pathlib import Path
# Add src/ to path (parent of this file's parent = src/)
sys.path.insert(0, str(Path(__file__).parent.parent))

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
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import QuantileLoss
from safetensors.torch import save_file

from core.config import MODEL_CONFIG
from core.server_encoder import ServerEncoder
from core.data_validator import DataValidator, CONTRACT_VERSION, VALID_STATES
from core.gpu_profiles import setup_gpu
from core.nordiq_metrics import NORDIQ_METRICS, validate_nordiq_metrics


def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
                      If False (default), use cudnn.benchmark for faster training
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Fully deterministic (slower)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[SEED] Random seed {seed} (deterministic mode - slower)")
    else:
        # Faster training with cudnn auto-tuner
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print(f"[SEED] Random seed {seed} (benchmark mode - faster)")


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
        self.config = config or MODEL_CONFIG

        # Auto-detect GPU and apply optimal profile
        if torch.cuda.is_available():
            self.gpu = setup_gpu()
            self.device = self.gpu.device
            # Use GPU-optimal batch size if not specified
            if 'batch_size' not in self.config or self.config['batch_size'] == 32:
                self.config['batch_size'] = self.gpu.get_batch_size('train')
                print(f"[GPU] Auto-configured batch size: {self.config['batch_size']}")
        else:
            self.gpu = None
            self.device = torch.device('cpu')

        self.model = None
    
    def load_dataset(self, dataset_dir: str = "./training/", chunk_id: str = None) -> pd.DataFrame:
        """Load dataset, preferring time-chunked parquet for memory efficiency.

        Args:
            dataset_dir: Path to training data directory
            chunk_id: Specific time chunk to load (e.g., '20251201_08'). If None, loads all data.

        Returns:
            DataFrame ready for training
        """
        training_path = Path(dataset_dir)

        # Debug: Show what files exist
        print(f"[SEARCH] Looking for dataset in: {training_path.absolute()}")
        if training_path.exists():
            files = list(training_path.glob("*"))
            print(f"[DIR] Files found: {[f.name for f in files]}")
        else:
            print(f"[ERROR] Directory doesn't exist: {training_path}")

        # PRIORITY 1: Time-chunked parquet (memory-efficient streaming)
        partitioned_dir = training_path / 'server_metrics_partitioned'
        if partitioned_dir.exists() and partitioned_dir.is_dir():
            manifest_path = partitioned_dir / 'chunk_manifest.json'
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                print(f"[LOAD] Found time-chunked dataset: {len(manifest['chunks'])} chunks")

                if chunk_id:
                    # Load specific chunk only
                    chunk_dir = partitioned_dir / f"time_chunk={chunk_id}"
                    if chunk_dir.exists():
                        print(f"[CHUNK] Loading chunk: {chunk_id}")
                        df = pd.read_parquet(chunk_dir)
                        print(f"[OK] Loaded {len(df):,} records from chunk {chunk_id}")
                        return self._prepare_dataframe(df)
                    else:
                        print(f"[WARNING] Chunk {chunk_id} not found, loading all data")

                # Load all chunks (fallback for small datasets or full training)
                print(f"[LOAD] Loading all {len(manifest['chunks'])} time chunks...")
                try:
                    import pyarrow.parquet as pq
                    table = pq.read_table(
                        partitioned_dir,
                        use_threads=True,
                        memory_map=True
                    )
                    df = table.to_pandas()
                    # Drop the partition column if present
                    if 'time_chunk' in df.columns:
                        df = df.drop(columns=['time_chunk'])
                    print(f"[OK] Loaded {len(df):,} records from time-chunked parquet")
                    return self._prepare_dataframe(df)
                except Exception as e:
                    print(f"[WARNING] PyArrow load failed: {e}, trying pandas...")
                    df = pd.read_parquet(partitioned_dir)
                    if 'time_chunk' in df.columns:
                        df = df.drop(columns=['time_chunk'])
                    print(f"[OK] Loaded {len(df):,} records from time-chunked parquet (pandas)")
                    return self._prepare_dataframe(df)

        # PRIORITY 2: Legacy partitioned parquet (by profile)
        parquet_dir = training_path / 'server_metrics_parquet'
        if parquet_dir.exists() and parquet_dir.is_dir():
            print(f"[LOAD] Loading profile-partitioned parquet dataset: {parquet_dir}")
            try:
                import pyarrow.parquet as pq
                table = pq.read_table(
                    parquet_dir,
                    use_threads=True,
                    memory_map=True
                )
                df = table.to_pandas()
                print(f"[OK] Loaded {len(df):,} records from partitioned parquet (PyArrow)")
                return self._prepare_dataframe(df)
            except ImportError:
                df = pd.read_parquet(parquet_dir)
                print(f"[OK] Loaded {len(df):,} records from partitioned parquet (pandas)")
                return self._prepare_dataframe(df)
            except Exception as e:
                print(f"[WARNING] Failed to load partitioned parquet: {e}")

        # PRIORITY 3: Try single parquet files
        parquet_candidates = [
            'server_metrics.parquet',
            'metrics_dataset.parquet',
            'demo_dataset.parquet'
        ]

        for parquet_name in parquet_candidates:
            parquet_path = training_path / parquet_name
            if parquet_path.exists():
                print(f"[INFO] Loading parquet dataset: {parquet_path}")
                df = pd.read_parquet(parquet_path)
                print(f"[OK] Loaded {len(df):,} records from parquet")
                return self._prepare_dataframe(df)

        # PRIORITY 4: Try any parquet file in directory
        parquet_files = list(training_path.glob("*.parquet"))
        if parquet_files:
            parquet_file = sorted(parquet_files)[0]
            print(f"[INFO] Loading found parquet file: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            print(f"[OK] Loaded {len(df):,} records from parquet")
            return self._prepare_dataframe(df)

        # PRIORITY 5: Try CSV files
        csv_candidates = ['server_metrics.csv', 'metrics_dataset.csv', 'demo_dataset.csv']
        for csv_name in csv_candidates:
            csv_path = training_path / csv_name
            if csv_path.exists():
                print(f"[FILE] Loading CSV dataset: {csv_path}")
                df = pd.read_csv(csv_path)
                print(f"[OK] Loaded {len(df):,} records from CSV")
                return self._prepare_dataframe(df)

        # PRIORITY 6: Fallback to JSON
        json_path = training_path / 'metrics_dataset.json'
        if json_path.exists():
            print(f"[INFO] Loading JSON dataset: {json_path}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            records = data.get('records') or data.get('training_samples', [])
            df = pd.DataFrame(records)
            print(f"[OK] Loaded {len(df):,} records from JSON")
            return self._prepare_dataframe(df)

        raise FileNotFoundError(
            f"No dataset files found in {training_path.absolute()}\n"
            f"Looked for:\n"
            f"  - Time-chunked: server_metrics_partitioned/\n"
            f"  - Profile-partitioned: server_metrics_parquet/\n"
            f"  - Single parquet: *.parquet\n"
            f"  - CSV: *.csv\n"
            f"  - JSON: *.json"
        )

    def get_chunk_manifest(self, dataset_dir: str = "./training/") -> Optional[dict]:
        """Get chunk manifest for streaming training.

        Returns manifest dict with chunk list, or None if not chunked.
        """
        training_path = Path(dataset_dir)
        manifest_path = training_path / 'server_metrics_partitioned' / 'chunk_manifest.json'

        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return None

    def load_chunk(self, dataset_dir: str, chunk_id: str) -> pd.DataFrame:
        """Load a specific time chunk for streaming training.

        Args:
            dataset_dir: Path to training data directory
            chunk_id: Time chunk ID (e.g., '20251201_08')

        Returns:
            DataFrame for the specified chunk
        """
        training_path = Path(dataset_dir)
        chunk_dir = training_path / 'server_metrics_partitioned' / f'time_chunk={chunk_id}'

        if not chunk_dir.exists():
            raise FileNotFoundError(f"Chunk {chunk_id} not found at {chunk_dir}")

        df = pd.read_parquet(chunk_dir)
        print(f"[CHUNK] Loaded {len(df):,} records from chunk {chunk_id}")
        return self._prepare_dataframe(df)
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for TFT training with fleet-level features for cascading failure detection."""
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

        # =============================================================================
        # FLEET-LEVEL FEATURES FOR CASCADING FAILURE DETECTION
        # =============================================================================
        # These features allow the model to learn cross-server correlations and detect
        # environment-wide issues even when individual servers appear "green"
        print("[FLEET] Computing fleet-level features for cascading failure detection...")

        # Group by timestamp to get fleet-wide metrics at each time point
        fleet_metrics = df.groupby('timestamp').agg({
            'cpu_user_pct': ['mean', 'std', 'max'],
            'mem_used_pct': ['mean', 'std', 'max'],
            'cpu_iowait_pct': ['mean', 'std', 'max'],
            'load_average': ['mean', 'std', 'max'],
        })

        # Flatten column names
        fleet_metrics.columns = [f'fleet_{col[0]}_{col[1]}' for col in fleet_metrics.columns]
        fleet_metrics = fleet_metrics.reset_index()

        # Count servers in warning/critical states at each timestamp
        # High CPU (>80%)
        servers_high_cpu = df.groupby('timestamp')['cpu_user_pct'].apply(
            lambda x: (x > 80).sum()
        ).reset_index(name='fleet_servers_high_cpu')

        # High memory (>85%)
        servers_high_mem = df.groupby('timestamp')['mem_used_pct'].apply(
            lambda x: (x > 85).sum()
        ).reset_index(name='fleet_servers_high_mem')

        # High IO wait (>15%)
        servers_high_iowait = df.groupby('timestamp')['cpu_iowait_pct'].apply(
            lambda x: (x > 15).sum()
        ).reset_index(name='fleet_servers_high_iowait')

        # Total servers at each timestamp (for normalization)
        total_servers = df.groupby('timestamp')[group_col].nunique().reset_index(name='fleet_total_servers')

        # Merge fleet metrics back to main dataframe
        df = df.merge(fleet_metrics, on='timestamp', how='left')
        df = df.merge(servers_high_cpu, on='timestamp', how='left')
        df = df.merge(servers_high_mem, on='timestamp', how='left')
        df = df.merge(servers_high_iowait, on='timestamp', how='left')
        df = df.merge(total_servers, on='timestamp', how='left')

        # Compute percentage of fleet in stress state (normalized 0-1)
        df['fleet_pct_high_cpu'] = df['fleet_servers_high_cpu'] / df['fleet_total_servers'].clip(lower=1)
        df['fleet_pct_high_mem'] = df['fleet_servers_high_mem'] / df['fleet_total_servers'].clip(lower=1)
        df['fleet_pct_high_iowait'] = df['fleet_servers_high_iowait'] / df['fleet_total_servers'].clip(lower=1)

        # Compute deviation from fleet average (per-server anomaly signal)
        # Positive = server is above fleet average, Negative = below
        df['cpu_vs_fleet'] = df['cpu_user_pct'] - df['fleet_cpu_user_pct_mean']
        df['mem_vs_fleet'] = df['mem_used_pct'] - df['fleet_mem_used_pct_mean']
        df['iowait_vs_fleet'] = df['cpu_iowait_pct'] - df['fleet_cpu_iowait_pct_mean']

        # Fill any NaN values in fleet features
        fleet_feature_cols = [col for col in df.columns if col.startswith('fleet_') or col.endswith('_vs_fleet')]
        for col in fleet_feature_cols:
            df[col] = df[col].fillna(0)

        print(f"[FLEET] Added {len(fleet_feature_cols)} fleet-level features")
        print(f"[FLEET] Features: {fleet_feature_cols[:5]}... (showing first 5)")

        # Encode server_name to hash-based server_id (production-ready)
        if 'server_name' in df.columns:
            # Use hash-based encoder for stable, production-ready encoding
            self.server_encoder = ServerEncoder()
            self.server_encoder.create_mapping(df['server_name'].unique().tolist())

            # Encode server names
            df['server_id'] = df['server_name'].apply(self.server_encoder.encode)
            print(f"[OK] Encoded {len(df['server_name'].unique())} server_names using hash-based encoding")
            print(f"   Sample mappings: {dict(list(self.server_encoder.name_to_id.items())[:3])}")
        
        # Ensure status column exists and is categorical string
        if 'status' not in df.columns:
            df['status'] = 'normal'  # Default value
        df['status'] = df['status'].fillna('normal').astype(str)

        # Validate NordIQ Metrics Framework metrics (using centralized schema)
        available_metrics, missing_metrics = validate_nordiq_metrics(df.columns)

        print(f"[INFO] Available NordIQ Metrics Framework metrics: {len(available_metrics)}/{len(NORDIQ_METRICS)}")
        if missing_metrics:
            print(f"[WARNING] Missing NordIQ Metrics Framework metrics: {missing_metrics}")
            raise ValueError(
                f"Data is missing {len(missing_metrics)} required NordIQ Metrics Framework metrics.\n"
                f"Please regenerate training data with metrics_generator.py to include all 14 metrics."
            )

        # Fill any NaN values and ensure numeric types
        for col in NORDIQ_METRICS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Use median for filling, or reasonable defaults
                if col.endswith('_pct'):
                    default_val = 50.0  # Percentage metrics
                elif col in ['back_close_wait', 'front_close_wait']:
                    default_val = 2  # Connection counts
                elif col == 'uptime_days':
                    default_val = 25  # Typical uptime
                elif col == 'load_average':
                    default_val = 2.0
                else:
                    default_val = 5.0  # Network metrics

                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else default_val)
        
        # Ensure server_id column exists for grouping
        if 'server_id' not in df.columns:
            df['server_id'] = df[group_col]
        
        # Validate against data contract
        validator = DataValidator(strict=False)
        is_valid, errors, warnings = validator.validate_schema(df)

        if not is_valid:
            print("[ERROR] Data violates contract!")
            for error in errors:
                print(f"   ERROR: {error}")

        if warnings:
            print(f"[WARNING] {len(warnings)} validation warning(s)")
            for warning in warnings[:3]:  # Show first 3
                print(f"   {warning}")

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
        # Explicitly cast to int to avoid float issues on different platforms
        max_encoder_length = int(min(self.config['context_length'], min_length // 3))
        max_prediction_length = int(min(self.config['prediction_horizon'], min_length // 10))

        if max_encoder_length < 6 or max_prediction_length < 1:
            raise ValueError(f"Insufficient data: encoder_length={max_encoder_length}, pred_length={max_prediction_length}")

        # Phase 2: Configurable validation split
        validation_split = self.config.get('validation_split', 0.2)
        training_cutoff = int(min_length * (1 - validation_split))
        
        print(f"[INFO] Using encoder length: {max_encoder_length}, prediction length: {max_prediction_length}")
        print(f"[INFO] Validation split: {validation_split:.1%} | Training cutoff: {training_cutoff}")
        
        # Define features - use centralized NordIQ Metrics Framework schema
        time_varying_unknown_reals = NORDIQ_METRICS.copy()
        time_varying_known_reals = ['hour', 'day_of_week', 'month', 'is_weekend']
        time_varying_unknown_categoricals = ['status']

        # Add fleet-level features for cascading failure detection
        # These are "known" at prediction time because they're computed from current fleet state
        fleet_features = [
            # Fleet averages (known at prediction time from current state)
            'fleet_cpu_user_pct_mean', 'fleet_cpu_user_pct_std', 'fleet_cpu_user_pct_max',
            'fleet_mem_used_pct_mean', 'fleet_mem_used_pct_std', 'fleet_mem_used_pct_max',
            'fleet_cpu_iowait_pct_mean', 'fleet_cpu_iowait_pct_std', 'fleet_cpu_iowait_pct_max',
            'fleet_load_average_mean', 'fleet_load_average_std', 'fleet_load_average_max',
            # Fleet stress indicators
            'fleet_pct_high_cpu', 'fleet_pct_high_mem', 'fleet_pct_high_iowait',
            # Per-server deviation from fleet (anomaly signals)
            'cpu_vs_fleet', 'mem_vs_fleet', 'iowait_vs_fleet',
        ]

        # Add fleet features that exist in the dataframe
        available_fleet_features = [f for f in fleet_features if f in df.columns]
        if available_fleet_features:
            time_varying_unknown_reals.extend(available_fleet_features)
            print(f"[FLEET] Added {len(available_fleet_features)} fleet features to model input")
        else:
            print("[WARNING] No fleet features found - cascading failure detection disabled")

        # CRITICAL: Profile as static categorical for transfer learning
        static_categoricals = []
        if 'profile' in df.columns:
            static_categoricals.append('profile')
            df['profile'] = df['profile'].fillna('generic').astype(str)
            print(f"[TRANSFER] Profile feature enabled - model will learn per-profile patterns")
            print(f"[TRANSFER] Profiles detected: {sorted(df['profile'].unique())}")
        else:
            print("[WARNING] No profile column - transfer learning disabled")

        # Phase 3: Multi-target prediction support with NordIQ Metrics Framework metrics
        multi_target = self.config.get('multi_target', False)
        if multi_target:
            # Use key NordIQ Metrics Framework metrics as targets (subset for manageability)
            target_metrics = self.config.get('target_metrics', [
                'cpu_user_pct', 'cpu_iowait_pct', 'mem_used_pct',
                'swap_used_pct', 'load_average'
            ])
            # Filter to only metrics that exist in the data
            available_targets = [m for m in target_metrics if m in df.columns]
            target = available_targets if len(available_targets) > 1 else 'cpu_user_pct'
            print(f"[INFO] Multi-target mode: {available_targets if isinstance(target, list) else [target]}")

            # Multi-target requires MultiNormalizer
            if isinstance(target, list):
                target_normalizer = MultiNormalizer(
                    [GroupNormalizer(groups=['server_id'], transformation='softplus') for _ in target]
                )
            else:
                target_normalizer = GroupNormalizer(groups=['server_id'], transformation='softplus')
        else:
            # Single target: use cpu_user as primary indicator (most directly actionable)
            target = 'cpu_user_pct'
            target_normalizer = GroupNormalizer(groups=['server_id'], transformation='softplus')
            print(f"[INFO] Single-target mode: {target}")

        # Create categorical encoders that support unknown categories (production-ready)
        categorical_encoders = {
            'server_id': NaNLabelEncoder(add_nan=True),  # Allow unknown servers
            'status': NaNLabelEncoder(add_nan=True)       # Allow unknown statuses
        }

        # Add profile encoder if available
        if 'profile' in df.columns:
            categorical_encoders['profile'] = NaNLabelEncoder(add_nan=True)  # Allow unknown profiles

        # Create training dataset with profile-based transfer learning
        training = TimeSeriesDataSet(
            df[df['time_idx'] <= training_cutoff],
            time_idx='time_idx',
            target=target,  # Phase 3: single or multi-target
            group_ids=['server_id'],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            min_encoder_length=int(max_encoder_length // 2),
            min_prediction_length=1,
            static_categoricals=static_categoricals,  # PROFILE: Enables transfer learning
            time_varying_unknown_reals=time_varying_unknown_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            categorical_encoders=categorical_encoders,  # Support unknown categories
            target_normalizer=target_normalizer,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        if static_categoricals:
            print(f"[TRANSFER] Model configured with profile-based transfer learning")
            print(f"[TRANSFER] New servers will predict based on their profile patterns")
        print(f"[INFO] Model configured to accept unknown servers and statuses")
        
        # Create validation dataset
        # Use predict=False to get overlapping samples for proper validation
        # (predict=True would only give 1 sample per server)
        validation = TimeSeriesDataSet.from_dataset(
            training,
            df[df['time_idx'] > training_cutoff],
            predict=False,
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

    def find_latest_checkpoint(self) -> Optional[Tuple[Path, int]]:
        """Find the latest model checkpoint for incremental training.

        Returns:
            Tuple of (checkpoint_path, total_epochs_completed) or None if no checkpoint found
        """
        models_dir = Path(self.config['models_dir'])

        if not models_dir.exists():
            print("[INFO] No models directory found - starting fresh training")
            return None

        # Find all model directories
        model_dirs = sorted(models_dir.glob('tft_model_*'), key=lambda p: p.stat().st_mtime)

        if not model_dirs:
            print("[INFO] No existing models found - starting fresh training")
            return None

        # Get the most recent model
        latest_model_dir = model_dirs[-1]

        # Check for checkpoint files
        checkpoint_dir = Path(self.config['checkpoints_dir'])
        if not checkpoint_dir.exists():
            print(f"[INFO] No checkpoints directory - will use model from {latest_model_dir.name}")
            return None

        # Find the last checkpoint (last.ckpt is saved by ModelCheckpoint)
        last_ckpt = checkpoint_dir / "last.ckpt"

        if last_ckpt.exists():
            print(f"[RESUME] Found checkpoint: {last_ckpt}")

            # Try to load training info to get epoch count
            training_info_path = latest_model_dir / "training_info.json"
            total_epochs = 0
            if training_info_path.exists():
                try:
                    with open(training_info_path, 'r') as f:
                        info = json.load(f)
                        total_epochs = info.get('total_epochs_completed', info.get('epochs', 0))
                        print(f"[RESUME] Previous training completed {total_epochs} epochs")
                except Exception as e:
                    print(f"[WARNING] Could not read training info: {e}")

            return (last_ckpt, total_epochs)
        else:
            print(f"[INFO] No last checkpoint found in {checkpoint_dir}")
            return None

    def _log_system_info(self) -> None:
        """Log comprehensive system information for training reproducibility."""
        print("\n" + "=" * 70)
        print("SYSTEM INFORMATION - Training Environment")
        print("=" * 70)

        # Platform information
        import platform
        import sys
        print(f"Platform:        {platform.system()} {platform.release()}")
        print(f"Python:          {platform.python_version()}")

        # Detect Linux distribution
        if platform.system() == 'Linux':
            try:
                import distro
                print(f"Distribution:    {distro.name()} {distro.version()}")
            except ImportError:
                pass

        # PyTorch and CUDA information
        import torch
        print(f"PyTorch:         {torch.__version__}")
        print(f"CUDA Available:  {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA Version:    {torch.version.cuda}")
            print(f"GPU:             {torch.cuda.get_device_name(0)}")
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU Memory:      {gpu_mem_gb:.1f} GB")

            # GPU utilization (if nvidia-smi available)
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                                       '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    gpu_util, mem_used, mem_total = result.stdout.strip().split(',')
                    print(f"GPU Utilization: {gpu_util.strip()}%")
                    print(f"GPU Mem Used:    {mem_used.strip()} MB / {mem_total.strip()} MB")
            except:
                pass

        # Key dependencies
        print(f"\nKey Dependencies:")
        try:
            import lightning
            print(f"  Lightning:     {lightning.__version__}")
        except:
            pass

        try:
            import pytorch_forecasting
            print(f"  PytorchFrcst:  {pytorch_forecasting.__version__}")
        except:
            pass

        try:
            import pandas as pd
            print(f"  Pandas:        {pd.__version__}")
        except:
            pass

        try:
            import numpy as np
            print(f"  NumPy:         {np.__version__}")
        except:
            pass

        # Training configuration
        print(f"\nTraining Configuration:")
        print(f"  Batch Size:    {self.config['batch_size']}")
        print(f"  Learning Rate: {self.config['learning_rate']}")
        print(f"  Epochs:        {self.config['epochs']}")
        print(f"  Hidden Size:   {self.config['hidden_size']}")
        print(f"  Attention:     {self.config['attention_heads']} heads")
        print(f"  Context:       {self.config['context_length']} timesteps")
        print(f"  Horizon:       {self.config['prediction_horizon']} timesteps")

        print("=" * 70 + "\n")

    def train(self, dataset_path: str = "./training/", find_lr: bool = False, incremental: bool = True) -> Optional[str]:
        """Train the TFT model with incremental training support.

        Args:
            dataset_path: Path to training data
            find_lr: If True, run learning rate finder before training
            incremental: If True, resume from latest checkpoint (default: True)

        Features:
        - Incremental training: Each session adds epochs to existing model
        - Learning rate finder (optional)
        - Learning rate monitoring
        - Configurable validation split
        - Enhanced progress reporting
        """
        print("[TRAIN] Starting TFT training (incremental mode enabled)...")

        # Log system information for training reproducibility
        self._log_system_info()

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

            # GPU-optimal worker configuration for data loading
            if self.gpu:
                optimal_workers = self.gpu.get_num_workers()
                use_pin_memory = True
            else:
                optimal_workers = 2
                use_pin_memory = False

            # CRITICAL: Handle platform-specific multiprocessing limitations
            # Platform-specific behavior with PyTorch 2.0.1:
            # - Small datasets (<100k samples): Use single worker to avoid overhead (all platforms)
            # - Windows: Multiprocessing OK, but NO persistent_workers (causes hangs)
            # - Linux/RedHat/Mac: Full optimization with persistent_workers
            val_samples = len(validation_dataset)
            import platform
            system = platform.system()
            is_windows = system == 'Windows'
            is_linux = system == 'Linux'  # Includes RedHat, Ubuntu, CentOS, etc.

            # Detect Linux distribution if available
            if is_linux:
                try:
                    import distro
                    distro_name = distro.name()
                    platform_display = f"Linux ({distro_name})"
                except ImportError:
                    platform_display = "Linux"
            else:
                platform_display = system

            # Use single worker for small datasets regardless of platform
            if val_samples < 100000:
                print(f"[INFO] Small validation set ({val_samples:,} samples) - using single-worker mode")
                optimal_workers = 0
                use_pin_memory = False
                use_persistent = False
            # Large dataset on Windows: enable multiprocessing but disable persistent workers
            elif is_windows:
                print(f"[INFO] Platform: {platform_display}")
                print(f"[INFO] Large validation set ({val_samples:,} samples)")
                print(f"[INFO] Using {optimal_workers} workers WITHOUT persistent_workers (Windows compatibility)")
                use_pin_memory = True
                use_persistent = False  # Critical: persistent_workers causes hangs on Windows
            # Large dataset on Linux/Mac: full optimization with persistent workers
            else:
                print(f"[INFO] Platform: {platform_display}")
                print(f"[INFO] Large validation set ({val_samples:,} samples)")
                print(f"[INFO] Using {optimal_workers} workers WITH persistent_workers (optimal performance)")
                use_pin_memory = True
                use_persistent = True

            print(f"[INFO] Data loading: {optimal_workers} workers, pin_memory={use_pin_memory}, persistent={use_persistent}")

            # Create data loaders with appropriate worker configuration
            # prefetch_factor improves GPU utilization by keeping batches ready
            prefetch = self.config.get('prefetch_factor', 3) if optimal_workers > 0 else None

            train_dataloader = training_dataset.to_dataloader(
                train=True,
                batch_size=self.config['batch_size'],
                num_workers=optimal_workers,
                pin_memory=use_pin_memory,
                persistent_workers=use_persistent if optimal_workers > 0 else False,
                prefetch_factor=prefetch
            )
            val_dataloader = validation_dataset.to_dataloader(
                train=False,
                batch_size=self.config['batch_size'] * 2,
                num_workers=optimal_workers,
                pin_memory=use_pin_memory,
                persistent_workers=use_persistent if optimal_workers > 0 else False,
                prefetch_factor=prefetch
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
            # NOTE: We explicitly do NOT pass ckpt_path to trainer.fit() to prevent auto-resume
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
                accumulate_grad_batches=accumulate_batches,
                # Critical: Skip sanity validation when val set is tiny (prevents hang)
                num_sanity_val_steps=0
            )

            # Check for existing checkpoint for incremental training
            checkpoint_info = None
            previous_epochs = 0
            ckpt_path = None

            if incremental:
                checkpoint_info = self.find_latest_checkpoint()
                if checkpoint_info:
                    ckpt_path, previous_epochs = checkpoint_info
                    print(f"[INCREMENTAL] Resuming from checkpoint")
                    print(f"[INCREMENTAL] Previous epochs completed: {previous_epochs}")
                    print(f"[INCREMENTAL] Will train {self.config['epochs']} additional epochs")
                    print(f"[INCREMENTAL] Total epochs after this session: {previous_epochs + self.config['epochs']}")
                else:
                    print("[INFO] No checkpoint found - starting fresh training")
            else:
                print("[INFO] Incremental mode disabled - training from scratch")

            # Train the model
            print(f"[START] Training for {self.config['epochs']} epochs...")
            trainer.fit(
                self.model,
                train_dataloader,
                val_dataloader,
                ckpt_path=ckpt_path  # None for fresh, checkpoint path for incremental
            )

            # Track total epochs across sessions
            self.total_epochs_completed = previous_epochs + self.config['epochs']
            
            print("[OK] Training completed successfully!")

            # Save model
            model_dir = self._save_model()
            return str(model_dir)

        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _get_streaming_checkpoint_path(self) -> Path:
        """Get path for streaming checkpoint file."""
        return Path(self.config['models_dir']) / 'streaming_checkpoint.pt'

    def _format_eta(self, seconds: float) -> str:
        """Format ETA in human-readable format.

        Args:
            seconds: Estimated seconds remaining

        Returns:
            Formatted string like "2h 30m" or "45m 12s"
        """
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def _save_streaming_checkpoint(self, epoch: int, chunk_idx: int, best_loss: float,
                                    chunk_order: list, training_start: float,
                                    epoch_loss: float, chunk_count: int,
                                    chunk_times: list = None, total_chunks: int = 0,
                                    chunks_completed: int = 0) -> None:
        """Save streaming training checkpoint for resume capability.

        Saves:
        - Model weights
        - Current epoch and chunk index
        - Best loss seen so far
        - Chunk order for current epoch (to maintain shuffle consistency)
        - Training start time (for ETA calculations)
        - Accumulated epoch loss and chunk count
        - Chunk timing history for ETA estimates
        - Progress tracking metrics
        """
        checkpoint_path = self._get_streaming_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': epoch,
            'chunk_idx': chunk_idx,
            'best_loss': best_loss,
            'chunk_order': chunk_order,
            'training_start': training_start,
            'epoch_loss': epoch_loss,
            'chunk_count': chunk_count,
            'config': self.config,
            'saved_at': datetime.now().isoformat(),
            # New fields for progress tracking
            'chunk_times': chunk_times or [],
            'total_chunks': total_chunks,
            'chunks_completed': chunks_completed,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"[CHECKPOINT] Saved at epoch {epoch+1}, chunk {chunk_idx+1} -> {checkpoint_path}")

    def _load_streaming_checkpoint(self) -> Optional[dict]:
        """Load streaming checkpoint if it exists.

        Returns checkpoint dict or None if no checkpoint exists.
        """
        checkpoint_path = self._get_streaming_checkpoint_path()

        if not checkpoint_path.exists():
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print(f"\n{'='*60}")
            print(f"[CHECKPOINT] RESUMING FROM SAVED CHECKPOINT")
            print(f"{'='*60}")
            print(f"[CHECKPOINT] Saved at: {checkpoint.get('saved_at', 'unknown')}")
            print(f"[CHECKPOINT] Epoch {checkpoint['epoch']+1}, Chunk {checkpoint['chunk_idx']+1}")
            print(f"[CHECKPOINT] Best loss so far: {checkpoint['best_loss']:.4f}")

            # Show progress info if available
            if 'chunks_completed' in checkpoint and 'total_chunks' in checkpoint:
                pct = (checkpoint['chunks_completed'] / checkpoint['total_chunks']) * 100 if checkpoint['total_chunks'] > 0 else 0
                print(f"[CHECKPOINT] Progress: {checkpoint['chunks_completed']}/{checkpoint['total_chunks']} chunks ({pct:.1f}%)")

            # Show timing info if available
            if 'chunk_times' in checkpoint and checkpoint['chunk_times']:
                avg_time = sum(checkpoint['chunk_times']) / len(checkpoint['chunk_times'])
                print(f"[CHECKPOINT] Avg chunk time: {avg_time:.1f}s (from {len(checkpoint['chunk_times'])} samples)")

            print(f"{'='*60}\n")
            return checkpoint
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            return None

    def _clear_streaming_checkpoint(self) -> None:
        """Remove streaming checkpoint after successful training completion."""
        checkpoint_path = self._get_streaming_checkpoint_path()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"[CHECKPOINT] Cleared checkpoint file")

    def train_streaming(self, dataset_path: str = "./training/", chunks_per_epoch: int = None,
                        checkpoint_every: int = 5) -> Optional[str]:
        """
        Memory-efficient streaming training that processes time chunks sequentially.

        CRITICAL FOR LARGE DATASETS:
        - Loads one time chunk at a time (chunk size from config, default 2 hours)
        - Each chunk trains for 1 sub-epoch
        - Memory usage stays bounded (~2-4 GB per chunk instead of 130+ GB)
        - Full dataset is still seen each epoch (all chunks processed)
        - Checkpoints every N chunks for resume capability

        Args:
            dataset_path: Path to training data (must be time-chunked format)
            chunks_per_epoch: Limit chunks per epoch (None = all chunks)
            checkpoint_every: Save checkpoint every N chunks (default: 5)

        Memory Profile (2-hour chunks, 90 servers):
        - 90 servers × 2 hours × 5-sec = ~130K rows per chunk
        - ~130K rows × 50 cols × 8 bytes = ~50MB raw data per chunk
        - TimeSeriesDataSet overhead: ~3-5x = ~150-250MB per chunk
        - Total memory: ~2-4 GB (vs 130+ GB for full dataset)

        Resume Capability:
        - Checkpoints saved every N chunks (configurable, default 5)
        - If interrupted, restart with same command to resume
        - Checkpoint includes: model weights, epoch, chunk index, best loss
        """
        print("[TRAIN] Starting STREAMING training mode (memory-efficient)...")
        print("=" * 70)

        # Log system info
        self._log_system_info()

        # Set random seed
        seed = self.config.get('random_seed', 42)
        set_random_seed(seed)

        try:
            # Get chunk manifest
            manifest = self.get_chunk_manifest(dataset_path)
            if not manifest:
                print("[ERROR] No time-chunked dataset found!")
                print("[INFO] Run metrics_generator.py to create time-chunked parquet")
                print("[INFO] Falling back to standard training...")
                return self.train(dataset_path)

            all_chunks = manifest['chunks']
            n_chunks = len(all_chunks)

            if chunks_per_epoch:
                n_chunks = min(chunks_per_epoch, n_chunks)
                all_chunks = all_chunks[:n_chunks]

            total_epochs = self.config['epochs']
            total_chunks_overall = n_chunks * total_epochs

            print(f"[STREAM] Found {len(manifest['chunks'])} time chunks ({manifest['chunk_hours']} hours each)")
            print(f"[STREAM] Will process {n_chunks} chunks per epoch")
            print(f"[STREAM] Total epochs: {total_epochs}")
            print(f"[STREAM] Total chunks to process: {total_chunks_overall}")
            print(f"[STREAM] Checkpoint every {checkpoint_every} chunks")
            print(f"[STREAM] Memory mode: ~1 chunk in memory at a time")
            print("=" * 70)

            # Check for existing checkpoint to resume from
            checkpoint = self._load_streaming_checkpoint()
            start_epoch = 0
            start_chunk_idx = 0
            best_loss = float('inf')
            chunk_order = None
            training_start = time.time()
            epoch_loss = 0.0
            chunk_count = 0

            # Progress tracking
            total_chunks_all_epochs = total_chunks_overall  # Use the pre-calculated total
            chunks_completed_total = 0
            chunk_times = []  # Rolling history of chunk processing times

            # Initialize model from first chunk to get architecture
            print(f"\n[INIT] Initializing model from first chunk...")
            first_chunk_df = self.load_chunk(dataset_path, all_chunks[0])
            training_dataset, _ = self.create_datasets(first_chunk_df)

            learning_rate = self.config['learning_rate']
            self.model = TemporalFusionTransformer.from_dataset(
                training_dataset,
                learning_rate=learning_rate,
                hidden_size=self.config['hidden_size'],
                attention_head_size=self.config['attention_heads'],
                dropout=self.config['dropout'],
                hidden_continuous_size=self.config['hidden_continuous_size'],
                loss=QuantileLoss(),
                reduce_on_plateau_patience=self.config['reduce_on_plateau_patience']
            )
            print(f"[OK] Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")

            # Load checkpoint state if resuming
            if checkpoint:
                print(f"\n[RESUME] Restoring from checkpoint...")
                self.model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint['epoch']
                start_chunk_idx = checkpoint['chunk_idx'] + 1  # Start from next chunk
                best_loss = checkpoint['best_loss']
                chunk_order = checkpoint['chunk_order']
                epoch_loss = checkpoint['epoch_loss']
                chunk_count = checkpoint['chunk_count']

                # Restore progress tracking
                chunk_times = checkpoint.get('chunk_times', [])
                chunks_completed_total = checkpoint.get('chunks_completed', start_epoch * n_chunks + start_chunk_idx)

                # If we finished an epoch, move to next
                if start_chunk_idx >= n_chunks:
                    start_epoch += 1
                    start_chunk_idx = 0
                    chunk_order = None  # Will reshuffle
                    epoch_loss = 0.0
                    chunk_count = 0

                # Calculate and display resume progress
                overall_pct = (chunks_completed_total / total_chunks_all_epochs) * 100
                remaining_chunks = total_chunks_all_epochs - chunks_completed_total

                print(f"[RESUME] Continuing from epoch {start_epoch+1}, chunk {start_chunk_idx+1}")
                print(f"[RESUME] Best loss: {best_loss:.4f}")
                print(f"[RESUME] Overall progress: {chunks_completed_total}/{total_chunks_all_epochs} chunks ({overall_pct:.1f}%)")

                if chunk_times:
                    avg_chunk_time = sum(chunk_times) / len(chunk_times)
                    eta_seconds = remaining_chunks * avg_chunk_time
                    eta_minutes = eta_seconds / 60
                    eta_hours = eta_minutes / 60
                    if eta_hours >= 1:
                        print(f"[RESUME] Estimated time remaining: {eta_hours:.1f} hours ({remaining_chunks} chunks @ {avg_chunk_time:.1f}s avg)")
                    else:
                        print(f"[RESUME] Estimated time remaining: {eta_minutes:.1f} minutes ({remaining_chunks} chunks @ {avg_chunk_time:.1f}s avg)")

            # Free first chunk from memory
            del first_chunk_df, training_dataset
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Setup logging
            logger = TensorBoardLogger(
                save_dir=self.config['logs_dir'],
                name='tft_streaming',
                version=datetime.now().strftime("%Y%m%d_%H%M%S")
            )

            # Streaming training loop
            for epoch in range(start_epoch, total_epochs):
                epoch_start = time.time()

                # Reset epoch stats if starting fresh epoch
                if epoch > start_epoch or start_chunk_idx == 0:
                    epoch_loss = 0.0
                    chunk_count = 0

                print(f"\n{'='*60}")
                print(f"[EPOCH {epoch+1}/{total_epochs}] Starting streaming epoch...")
                print(f"{'='*60}")

                # Use saved chunk order if resuming mid-epoch, otherwise shuffle
                if chunk_order is None or epoch > start_epoch:
                    chunk_order = all_chunks.copy()
                    np.random.shuffle(chunk_order)
                    start_chunk_idx = 0  # Reset for new epochs

                for chunk_idx in range(start_chunk_idx if epoch == start_epoch else 0, n_chunks):
                    chunk_id = chunk_order[chunk_idx]
                    chunk_start = time.time()

                    # Calculate overall progress
                    current_chunk_global = epoch * n_chunks + chunk_idx
                    overall_pct = (chunks_completed_total / total_chunks_all_epochs) * 100

                    # Calculate ETA based on average chunk time
                    if chunk_times:
                        avg_chunk_time = sum(chunk_times[-50:]) / len(chunk_times[-50:])  # Use last 50 for rolling avg
                        remaining_chunks = total_chunks_all_epochs - chunks_completed_total
                        eta_seconds = remaining_chunks * avg_chunk_time
                        eta_str = self._format_eta(eta_seconds)
                    else:
                        avg_chunk_time = 0
                        eta_str = "calculating..."

                    # Load this chunk with progress info
                    print(f"\n[CHUNK {chunk_idx+1}/{n_chunks}] Loading {chunk_id}...")
                    print(f"[PROGRESS] Overall: {chunks_completed_total}/{total_chunks_all_epochs} ({overall_pct:.1f}%) | ETA: {eta_str}")

                    try:
                        chunk_df = self.load_chunk(dataset_path, chunk_id)
                    except Exception as e:
                        print(f"[WARNING] Failed to load chunk {chunk_id}: {e}")
                        continue

                    # Create datasets for this chunk
                    try:
                        train_ds, val_ds = self.create_datasets(chunk_df)
                    except Exception as e:
                        print(f"[WARNING] Failed to create dataset from chunk {chunk_id}: {e}")
                        del chunk_df
                        gc.collect()
                        continue

                    # Create dataloaders
                    train_dl = train_ds.to_dataloader(
                        train=True,
                        batch_size=self.config['batch_size'],
                        num_workers=0,  # Single worker for streaming
                        pin_memory=torch.cuda.is_available()
                    )
                    val_dl = val_ds.to_dataloader(
                        train=False,
                        batch_size=self.config['batch_size'] * 2,
                        num_workers=0
                    )

                    # Train for 1 sub-epoch on this chunk
                    trainer = Trainer(
                        max_epochs=1,
                        gradient_clip_val=self.config.get('gradient_clip_val', 0.1),
                        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                        devices=1,
                        enable_checkpointing=False,  # Manual checkpointing
                        logger=logger,
                        enable_progress_bar=True,
                        log_every_n_steps=50,
                        num_sanity_val_steps=0
                    )

                    trainer.fit(self.model, train_dl, val_dl)

                    # Get chunk loss
                    chunk_loss = trainer.callback_metrics.get('val_loss', 0)
                    if isinstance(chunk_loss, torch.Tensor):
                        chunk_loss = chunk_loss.item()
                    epoch_loss += chunk_loss
                    chunk_count += 1

                    # Track timing
                    chunk_time = time.time() - chunk_start
                    chunk_times.append(chunk_time)
                    chunks_completed_total += 1

                    # Update progress display
                    new_overall_pct = (chunks_completed_total / total_chunks_all_epochs) * 100
                    avg_chunk_time = sum(chunk_times[-50:]) / len(chunk_times[-50:])
                    remaining_chunks = total_chunks_all_epochs - chunks_completed_total
                    eta_seconds = remaining_chunks * avg_chunk_time
                    eta_str = self._format_eta(eta_seconds)

                    print(f"[CHUNK] {chunk_id} done in {chunk_time:.1f}s | Loss: {chunk_loss:.4f}")
                    print(f"[PROGRESS] {chunks_completed_total}/{total_chunks_all_epochs} ({new_overall_pct:.1f}%) | Avg: {avg_chunk_time:.1f}s/chunk | ETA: {eta_str}")

                    # Free memory
                    del chunk_df, train_ds, val_ds, train_dl, val_dl, trainer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Save checkpoint every N chunks
                    if (chunk_idx + 1) % checkpoint_every == 0:
                        self._save_streaming_checkpoint(
                            epoch=epoch,
                            chunk_idx=chunk_idx,
                            best_loss=best_loss,
                            chunk_order=chunk_order,
                            training_start=training_start,
                            epoch_loss=epoch_loss,
                            chunk_count=chunk_count,
                            chunk_times=chunk_times[-100:],  # Keep last 100 for checkpoint
                            total_chunks=total_chunks_all_epochs,
                            chunks_completed=chunks_completed_total
                        )

                # Epoch summary
                avg_epoch_loss = epoch_loss / max(chunk_count, 1)
                epoch_time = time.time() - epoch_start

                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    improvement = " [BEST]"
                else:
                    improvement = ""

                # Calculate overall progress
                epochs_completed = epoch + 1
                overall_pct = (chunks_completed_total / total_chunks_all_epochs) * 100

                print(f"\n{'='*60}")
                print(f"[EPOCH {epoch+1}/{total_epochs}] COMPLETE{improvement}")
                print(f"{'='*60}")
                print(f"   Epoch time:  {epoch_time/60:.1f} min")
                print(f"   Avg loss:    {avg_epoch_loss:.4f}")
                print(f"   Best loss:   {best_loss:.4f}")
                print(f"   Progress:    {chunks_completed_total}/{total_chunks_all_epochs} chunks ({overall_pct:.1f}%)")

                # Calculate remaining time using chunk-based ETA
                if chunk_times:
                    avg_chunk_time = sum(chunk_times[-50:]) / len(chunk_times[-50:])
                    remaining_chunks = total_chunks_all_epochs - chunks_completed_total
                    eta_seconds = remaining_chunks * avg_chunk_time
                    eta_str = self._format_eta(eta_seconds)
                    print(f"   ETA:         {eta_str} ({remaining_chunks} chunks @ {avg_chunk_time:.1f}s avg)")
                print(f"{'='*60}")

                # Save checkpoint at end of epoch
                self._save_streaming_checkpoint(
                    epoch=epoch,
                    chunk_idx=n_chunks - 1,
                    best_loss=best_loss,
                    chunk_order=chunk_order,
                    training_start=training_start,
                    epoch_loss=epoch_loss,
                    chunk_count=chunk_count,
                    chunk_times=chunk_times[-100:],
                    total_chunks=total_chunks_all_epochs,
                    chunks_completed=chunks_completed_total
                )

                # Reset for next epoch
                chunk_order = None

            # Training complete
            total_time = time.time() - training_start
            avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0

            print(f"\n{'='*70}")
            print(f"[OK] STREAMING TRAINING COMPLETE")
            print(f"{'='*70}")
            print(f"   Total time:       {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
            print(f"   Chunks processed: {chunks_completed_total}")
            print(f"   Epochs completed: {total_epochs}")
            print(f"   Best loss:        {best_loss:.4f}")
            print(f"   Avg chunk time:   {avg_chunk_time:.1f}s")
            print(f"{'='*70}")

            # Clear checkpoint after successful completion
            self._clear_streaming_checkpoint()

            # Save model
            self.total_epochs_completed = total_epochs
            model_dir = self._save_model()
            return str(model_dir)

        except Exception as e:
            print(f"[ERROR] Streaming training failed: {e}")
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
            
            # Save training info with contract compliance and incremental tracking
            training_info_path = model_dir / "training_info.json"
            with open(training_info_path, 'w') as f:
                json.dump({
                    'trained_at': datetime.now().isoformat(),
                    'training_completed': True,
                    'model_type': 'TemporalFusionTransformer',
                    'framework': 'pytorch_forecasting',
                    'safetensors_format': True,
                    'epochs': self.config['epochs'],  # Epochs in this session
                    'total_epochs_completed': getattr(self, 'total_epochs_completed', self.config['epochs']),  # Total across all sessions
                    'unique_states': VALID_STATES,  # From contract
                    'state_encoder_size': len(VALID_STATES),
                    'data_contract_version': CONTRACT_VERSION,
                    'model_path': str(model_path)
                }, f, indent=2)

            # Save server mapping (CRITICAL for inference)
            if hasattr(self, 'server_encoder'):
                mapping_path = model_dir / "server_mapping.json"
                self.server_encoder.save_mapping(mapping_path)
                print(f"[OK] Server mapping saved with model")
            else:
                print(f"[WARNING] No server encoder found - this may cause issues in inference")

            # Save TimeSeriesDataSet parameters including categorical encoders
            # This is CRITICAL for inference to work properly
            if hasattr(self.model, 'dataset_parameters'):
                dataset_params_path = model_dir / "dataset_parameters.pkl"
                import pickle
                with open(dataset_params_path, 'wb') as f:
                    pickle.dump(self.model.dataset_parameters, f)
                print(f"[OK] Dataset parameters (including encoders) saved")
            else:
                print(f"[WARNING] Model doesn't have dataset_parameters - encoders not saved!")

            print(f"[OK] Model saved to: {model_dir}")
            print(f"[OK] Contract version: {CONTRACT_VERSION}")
            return model_dir
            
        except Exception as e:
            print(f"[ERROR] Model save failed: {e}")
            return model_dir


def train_model(dataset_path: str = "./training/",
                epochs: Optional[int] = None,
                per_server: bool = False,
                incremental: bool = True,
                streaming: bool = False) -> Optional[str]:
    """
    Module interface for training with incremental and streaming support.

    Args:
        dataset_path: Path to training data directory
        epochs: Number of training epochs to add (overrides config)
        per_server: Train separate model per server (default: False - single fleet model)
        incremental: Resume from latest checkpoint (default: True)
        streaming: Use memory-efficient streaming mode (default: False)

    Returns:
        Path to trained model directory, or None if failed

    Training Modes:
        - Standard: Loads full dataset into memory (fast, but memory-heavy)
        - Streaming: Loads 8-hour chunks one at a time (slower, but ~10x less memory)

    Incremental Training:
        - When incremental=True (default), finds latest checkpoint and resumes
        - Epochs parameter adds to existing training (e.g., 1 epoch/week = continuous learning)
        - Total epoch count is tracked across sessions in training_info.json
        - Example: Run with epochs=1 weekly to add 1 epoch/week to existing model
    """
    config = MODEL_CONFIG.copy()
    if epochs is not None:
        config['epochs'] = epochs

    if per_server:
        print("[PREP] Per-server training mode enabled")
        print("[WARNING] Per-server mode not yet implemented - training fleet-wide model")

    trainer = TFTTrainer(config)

    if streaming:
        return trainer.train_streaming(dataset_path)
    else:
        return trainer.train(dataset_path, incremental=incremental)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Train TFT model with incremental and streaming training support",
        epilog="""
Examples:
  python tft_trainer.py --epochs 1 --incremental  # Add 1 epoch to existing model
  python tft_trainer.py --streaming               # Memory-efficient streaming mode
  python tft_trainer.py --fresh --epochs 3        # Fresh training, 3 epochs
        """
    )
    parser.add_argument("--dataset", type=str, default="./training/",
                       help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, help="Epochs to train (adds to existing if --incremental)")
    parser.add_argument("--batch-size", type=int, help="Batch size (overrides config)")
    parser.add_argument("--per-server", action="store_true",
                       help="Train separate model per server (experimental)")
    parser.add_argument("--incremental", action="store_true", default=True,
                       help="Resume from latest checkpoint (default: True)")
    parser.add_argument("--fresh", action="store_true",
                       help="Start fresh training (ignore checkpoints)")
    parser.add_argument("--streaming", action="store_true",
                       help="Memory-efficient streaming mode (loads 8-hour chunks)")

    args = parser.parse_args()

    # Determine incremental mode
    incremental = args.incremental and not args.fresh

    # Call train_model function
    model_path = train_model(
        dataset_path=args.dataset,
        epochs=args.epochs,
        streaming=args.streaming,
        per_server=args.per_server,
        incremental=incremental
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