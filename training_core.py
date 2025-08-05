#!/usr/bin/env python3
"""
training_core.py - TFT-based Training Core (Replacement)
Temporal Fusion Transformer training system for server metrics prediction
Replaces BERT/TensorFlow with PyTorch Forecasting TFT
Optimized for PyTorch 2.0.1 + PyTorch Lightning 2.0.2
"""

import os
import json
import pickle
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Ensure optimal thread usage
os.environ['NUMEXPR_MAX_THREADS'] = '16'

# PyTorch ecosystem imports
try:
    import torch
    import torch.nn as nn
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    
    # PyTorch Forecasting imports
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    
    # Safetensors for secure model storage
    from safetensors.torch import save_file, load_file
    
    # Data processing
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
except ImportError as e:
    raise ImportError(f"Failed to import required dependencies: {e}")

# Import project modules
from common_utils import (
    load_dataset_file, log_message, get_dataset_paths, get_optimal_workers
)


class TFTDataPreprocessor:
    """Preprocessor for existing metrics dataset to TFT format."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        
    def load_existing_metrics_dataset(self, training_dir: str) -> pd.DataFrame:
        """Load the existing metrics_dataset.json and convert to TFT format."""
        log_message("📊 Loading existing metrics dataset...")
        
        dataset_paths = get_dataset_paths(Path(training_dir))
        metrics_path = dataset_paths['metrics_dataset']
        
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics dataset not found: {metrics_path}")
        
        # Load the existing dataset
        metrics_data = load_dataset_file(metrics_path)
        if not metrics_data or 'training_samples' not in metrics_data:
            raise ValueError("Invalid metrics dataset format")
        
        log_message(f"✅ Found {len(metrics_data['training_samples'])} samples")
        
        # Convert to time series format
        return self._convert_to_time_series(metrics_data['training_samples'])
    
    def _convert_to_time_series(self, samples: List[Dict]) -> pd.DataFrame:
        """Convert existing sample format to time series DataFrame."""
        log_message("🔄 Converting samples to time series format...")
        
        # Group samples by server/series to create time series
        series_data = {}
        
        for sample in samples:
            timestamp = sample.get('timestamp')
            server_name = sample.get('server_name', 'default_server')
            status = sample.get('status', 'normal')
            metrics = sample.get('metrics', {})
            
            if server_name not in series_data:
                series_data[server_name] = []
            
            # Create time series point
            point = {
                'timestamp': pd.to_datetime(timestamp),
                'series_id': server_name,
                'status': status,
                **metrics  # Add all metrics directly
            }
            series_data[server_name].append(point)
        
        # Combine all series into single DataFrame
        all_points = []
        for series_id, points in series_data.items():
            # Sort by timestamp
            points.sort(key=lambda x: x['timestamp'])
            # Add time index
            for i, point in enumerate(points):
                point['time_idx'] = i
            all_points.extend(points)
        
        df = pd.DataFrame(all_points)
        log_message(f"✅ Created time series with {len(df)} points across {len(series_data)} series")
        
        return self._engineer_features(df)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for TFT training."""
        log_message("🔧 Engineering features for TFT...")
        
        # Sort by series and timestamp
        df = df.sort_values(['series_id', 'timestamp']).reset_index(drop=True)
        
        # Re-create time index properly
        df['time_idx'] = df.groupby('series_id').cumcount()
        
        # Encode categorical variables
        status_encoder = LabelEncoder()
        df['status_encoded'] = status_encoder.fit_transform(df['status'].fillna('normal'))
        self.encoders['status'] = status_encoder
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Identify numeric metrics columns (exclude metadata columns)
        metadata_cols = {'timestamp', 'series_id', 'status', 'status_encoded', 'time_idx', 'hour', 'day_of_week', 'month'}
        metric_cols = [col for col in df.columns if col not in metadata_cols and df[col].dtype in ['int64', 'float64']]
        
        log_message(f"📊 Found {len(metric_cols)} metric columns: {metric_cols[:5]}...")
        
        # Fill missing values and ensure numeric types
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        
        # Create target variables (next step predictions for key metrics)
        key_targets = []
        for col in ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average']:
            if col in df.columns:
                key_targets.append(col)
                df[f'{col}_target'] = df.groupby('series_id')[col].shift(-1)
        
        # If no standard targets found, use first few numeric columns
        if not key_targets:
            key_targets = metric_cols[:4]  # Use first 4 metrics as targets
            for col in key_targets:
                df[f'{col}_target'] = df.groupby('series_id')[col].shift(-1)
        
        # Remove rows with missing targets (last row of each series)
        df = df.dropna(subset=[f'{key_targets[0]}_target'])
        
        log_message(f"✅ Feature engineering complete. Shape: {df.shape}")
        log_message(f"🎯 Target variables: {key_targets}")
        
        return df
    
    def create_time_series_dataset(self, df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Create PyTorch Forecasting TimeSeriesDataSet."""
        log_message("📊 Creating TimeSeriesDataSet for TFT...")
        
        # Define prediction and context lengths
        max_prediction_length = self.config.get('prediction_horizon', 6)  # 6 time steps ahead
        max_encoder_length = self.config.get('context_length', 24)        # 24 time steps history
        
        # Adjust lengths based on available data
        min_series_length = df.groupby('series_id')['time_idx'].count().min()
        if min_series_length < max_encoder_length + max_prediction_length:
            max_encoder_length = min(max_encoder_length, min_series_length // 2)
            max_prediction_length = min(max_prediction_length, min_series_length - max_encoder_length)
            log_message(f"⚠️  Adjusted lengths - Encoder: {max_encoder_length}, Prediction: {max_prediction_length}")
        
        # Split train/validation by time
        training_cutoff = df['time_idx'].quantile(0.8)
        
        # Define dataset parameters
        time_varying_known_reals = ['hour', 'day_of_week', 'month']
        
        # Get all numeric columns as time-varying unknown reals
        metadata_cols = {'timestamp', 'series_id', 'status', 'status_encoded', 'time_idx', 'hour', 'day_of_week', 'month'}
        target_cols = {col for col in df.columns if col.endswith('_target')}
        
        time_varying_unknown_reals = [
            col for col in df.columns 
            if col not in metadata_cols and col not in target_cols and df[col].dtype in ['int64', 'float64']
        ]
        
        # Primary target (first available target column)
        target_col = [col for col in df.columns if col.endswith('_target')][0]
        
        log_message(f"🎯 Primary target: {target_col}")
        log_message(f"📊 Time-varying reals: {len(time_varying_unknown_reals)} columns")
        
        # Create training dataset
        training = TimeSeriesDataSet(
            df[df['time_idx'] <= training_cutoff],
            time_idx='time_idx',
            target=target_col,
            group_ids=['series_id'],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            time_varying_known_categoricals=[],
            time_varying_unknown_categoricals=['status_encoded'],
            categorical_encoders={'status_encoded': 'auto'},
            target_normalizer=GroupNormalizer(groups=['series_id'], transformation='softplus'),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        
        # Create validation dataset
        validation = TimeSeriesDataSet.from_dataset(
            training, 
            df[df['time_idx'] > training_cutoff], 
            predict=True, 
            stop_randomization=True
        )
        
        log_message(f"✅ TimeSeriesDataSet created:")
        log_message(f"   Training samples: {len(training)}")
        log_message(f"   Validation samples: {len(validation)}")
        log_message(f"   Encoder length: {max_encoder_length}")
        log_message(f"   Prediction length: {max_prediction_length}")
        
        return training, validation


class TFTTrainer:
    """TFT trainer replacing the old BERT-based trainer."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.trainer = None
        self.preprocessor = TFTDataPreprocessor(config)
        
        # Training state
        self.training_stats = {
            'start_time': None,
            'best_loss': float('inf'),
            'total_epochs': 0,
        }
    
    def train(self) -> bool:
        """Main training method - replaces old BERT training."""
        log_message("🏋️ Starting TFT training (replacing BERT)...")
        self.training_stats['start_time'] = datetime.now()
        
        try:
            # Load existing metrics dataset
            training_dir = self.config.get('training_dir', './training/')
            df = self.preprocessor.load_existing_metrics_dataset(training_dir)
            
            # Create time series datasets
            training_dataset, validation_dataset = self.preprocessor.create_time_series_dataset(df)
            
            # Create data loaders
            batch_size = self.config.get('batch_size', 32)
            train_dataloader = training_dataset.to_dataloader(
                train=True, 
                batch_size=batch_size,
                num_workers=get_optimal_workers()
            )
            val_dataloader = validation_dataset.to_dataloader(
                train=False, 
                batch_size=batch_size * 2,
                num_workers=get_optimal_workers()
            )
            
            # Create TFT model
            self.model = self._create_tft_model(training_dataset)
            
            # Setup PyTorch Lightning trainer
            self.trainer = self._setup_trainer()
            
            # Train model
            log_message("🚀 Starting TFT training loop...")
            self.trainer.fit(
                self.model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader
            )
            
            # Save model
            self._save_model_securely()
            
            # Log completion
            training_time = datetime.now() - self.training_stats['start_time']
            log_message(f"🎉 TFT training completed in {training_time}")
            
            return True
            
        except Exception as e:
            log_message(f"❌ TFT training failed: {str(e)}")
            log_message(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _create_tft_model(self, training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
        """Create TFT model from dataset."""
        log_message("🤖 Creating Temporal Fusion Transformer...")
        
        # Model configuration
        model_config = {
            'learning_rate': self.config.get('learning_rate', 0.03),
            'hidden_size': self.config.get('hidden_size', 16),
            'attention_head_size': self.config.get('attention_heads', 4), 
            'dropout': self.config.get('dropout', 0.1),
            'hidden_continuous_size': self.config.get('continuous_size', 8),
            'loss': QuantileLoss(),
            'log_interval': 10,
            'reduce_on_plateau_patience': 4,
        }
        
        # Create model from dataset
        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            **model_config
        )
        
        log_message(f"✅ TFT model created with {sum(p.numel() for p in tft.parameters()):,} parameters")
        return tft
    
    def _setup_trainer(self) -> pl.Trainer:
        """Setup PyTorch Lightning trainer."""
        log_message("⚡ Setting up PyTorch Lightning trainer...")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=self.config.get('checkpoints_dir', './checkpoints/'),
                filename='tft-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                save_top_k=3,
                mode='min',
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('early_stopping_patience', 10),
                mode='min',
            ),
        ]
        
        # Logger
        logger = TensorBoardLogger(
            save_dir=self.config.get('logs_dir', './logs/'),
            name='tft_training'
        )
        
        # Trainer configuration
        trainer_config = {
            'max_epochs': self.config.get('epochs', 30),
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'callbacks': callbacks,
            'logger': logger,
            'gradient_clip_val': 0.1,
            'enable_progress_bar': True,
            'enable_model_summary': True,
        }
        
        # Mixed precision for GPU
        if torch.cuda.is_available() and self.config.get('mixed_precision', True):
            trainer_config['precision'] = 16
            log_message("🚀 Mixed precision training enabled")
        
        trainer = pl.Trainer(**trainer_config)
        log_message("✅ PyTorch Lightning trainer configured")
        
        return trainer
    
    def _save_model_securely(self):
        """Save model using Safetensors format."""
        if self.model is None:
            return
        
        # Create models directory
        models_dir = Path(self.config.get('models_dir', './models/'))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = models_dir / f'tft_monitoring_{timestamp}'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save model state dict with Safetensors
            model_path = model_dir / 'model.safetensors'
            save_file(self.model.state_dict(), str(model_path))
            log_message(f"💾 Model weights saved: {model_path}")
            
            # Save preprocessing components
            if self.preprocessor.scalers:
                scalers_path = model_dir / 'scalers.pkl'
                with open(scalers_path, 'wb') as f:
                    pickle.dump(self.preprocessor.scalers, f)
            
            if self.preprocessor.encoders:
                encoders_path = model_dir / 'encoders.pkl'
                with open(encoders_path, 'wb') as f:
                    pickle.dump(self.preprocessor.encoders, f)
            
            # Save model configuration
            config_path = model_dir / 'config.json'
            model_config = {
                'model_type': 'TemporalFusionTransformer',
                'framework': 'pytorch_forecasting',
                'pytorch_version': torch.__version__,
                'training_config': self.config,
                'timestamp': timestamp,
                'training_stats': self.training_stats
            }
            
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2, default=str)
            
            # Save training metadata (compatible with existing system)
            metadata_path = model_dir / 'training_metadata.json'
            metadata = {
                'training_completed': True,
                'model_type': 'TFT',
                'framework': 'pytorch_forecasting',
                'training_time': str(datetime.now() - self.training_stats['start_time']),
                'final_epoch': self.trainer.current_epoch if self.trainer else 0,
                'best_val_loss': float(self.trainer.callback_metrics.get('val_loss', float('inf'))),
                'model_path': str(model_path),
                'created_at': datetime.now().isoformat()
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log_message(f"✅ TFT model saved successfully: {model_dir}")
            
        except Exception as e:
            log_message(f"❌ Failed to save model: {e}")


# Factory functions to maintain compatibility with existing code
def create_trainer(config: Dict[str, Any]) -> TFTTrainer:
    """Factory function to create TFT trainer - replaces old BERT trainer."""
    return TFTTrainer(config)


def validate_training_environment(config: Dict[str, Any]) -> bool:
    """Validate TFT training environment - replaces old validation."""
    log_message("🔍 Validating TFT training environment...")
    
    # Check PyTorch version
    try:
        import torch
        log_message(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        log_message("❌ PyTorch not available")
        return False
    
    # Check PyTorch Lightning
    try:
        import pytorch_lightning as pl
        log_message(f"✅ PyTorch Lightning: {pl.__version__}")
    except ImportError:
        log_message("❌ PyTorch Lightning not available")
        return False
    
    # Check PyTorch Forecasting
    try:
        import pytorch_forecasting
        log_message(f"✅ PyTorch Forecasting available")
    except ImportError:
        log_message("❌ PyTorch Forecasting not available")
        return False
    
    # Check dataset availability
    training_dir = Path(config.get('training_dir', './training/'))
    dataset_paths = get_dataset_paths(training_dir)
    
    if not dataset_paths['metrics_dataset'].exists():
        log_message(f"❌ Metrics dataset not found: {dataset_paths['metrics_dataset']}")
        return False
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        log_message(f"✅ GPU available: {gpu_name} ({gpu_memory}GB)")
    else:
        log_message("⚠️  No GPU available, using CPU")
    
    log_message("✅ TFT training environment validated")
    return True


# Legacy compatibility - these functions are called by existing trainer scripts
class MonitoringModel:
    """Legacy compatibility wrapper - no longer used but prevents import errors."""
    def __init__(self, *args, **kwargs):
        log_message("⚠️  MonitoringModel is deprecated, using TFT instead")


class TrainingEnvironment:
    """Legacy compatibility wrapper."""
    def __init__(self, config):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_message("⚠️  TrainingEnvironment is deprecated, using TFT trainer instead")


if __name__ == "__main__":
    # Test the TFT training core
    test_config = {
        'training_dir': './training/',
        'models_dir': './models/',
        'checkpoints_dir': './checkpoints/',
        'logs_dir': './logs/',
        'epochs': 20,
        'batch_size': 32,
        'learning_rate': 0.03,
        'prediction_horizon': 6,
        'context_length': 24,
        'mixed_precision': True
    }
    
    if validate_training_environment(test_config):
        trainer = create_trainer(test_config)
        success = trainer.train()
        if success:
            log_message("✅ TFT training test completed successfully")
        else:
            log_message("❌ TFT training test failed")
    else:
        log_message("❌ Environment validation failed")