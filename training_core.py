#!/usr/bin/env python3
"""
training_core.py - Fixed TFT Training Core for Lightning 2.0+
UPDATED: Uses 'lightning' package instead of deprecated 'pytorch_lightning'
Compatible with pytorch-forecasting==1.0.0 and lightning>=2.0.0
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

# PyTorch ecosystem imports - FIXED FOR LIGHTNING 2.0+
try:
    import torch
    import torch.nn as nn
    
    # FIXED: Use 'lightning' instead of 'pytorch_lightning'
    import lightning as L
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    
    # PyTorch Forecasting imports
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    
    # Safetensors for secure model storage
    from safetensors.torch import save_file, load_file
    
    # Data processing
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    LIGHTNING_AVAILABLE = True
    
except ImportError as e:
    LIGHTNING_AVAILABLE = False
    import_error = str(e)
    # Provide helpful error message
    if "pytorch_lightning" in str(e):
        print("❌ Old pytorch_lightning import detected. Use 'lightning' package instead.")
        print("💡 Install with: pip install lightning>=2.0.0")
    else:
        print(f"❌ Import error: {e}")

# Import project modules
from common_utils import (
    load_dataset_file, log_message, get_dataset_paths, get_optimal_workers
)


class TFTDataPreprocessor:
    """Enhanced preprocessor with fixed categorical handling for Lightning 2.0+."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scalers = {}
        self.encoders = {}
        
    def load_existing_metrics_dataset(self, training_dir: str) -> pd.DataFrame:
        """Load metrics dataset from JSON format."""
        log_message("📊 Loading metrics dataset...")
        
        training_path = Path(training_dir)
        json_path = training_path / 'metrics_dataset.json'
        
        if not json_path.exists():
            raise FileNotFoundError(f"Metrics dataset not found: {json_path}")
        
        # Load JSON dataset
        dataset_paths = get_dataset_paths(training_path)
        metrics_data = load_dataset_file(dataset_paths['metrics_dataset'])
        
        if not metrics_data or 'training_samples' not in metrics_data:
            raise ValueError("Invalid metrics dataset format")
        
        log_message(f"✅ Loaded metrics dataset: {json_path.name}")
        log_message(f"✅ Found {len(metrics_data['training_samples'])} samples")
        
        # Check file size
        file_size_mb = json_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 100:
            log_message(f"💡 Large dataset detected ({file_size_mb:.1f} MB)")
            log_message("💡 Consider converting to Parquet format for better performance")
        
        # Convert to time series format
        return self._convert_json_to_time_series(metrics_data['training_samples'])
    
    def _convert_json_to_time_series(self, samples: List[Dict]) -> pd.DataFrame:
        """Convert JSON samples to time series DataFrame."""
        log_message("🔄 Converting JSON samples to time series format...")
        
        # Process samples efficiently
        records = []
        for sample in samples:
            record = {
                'timestamp': sample.get('timestamp'),
                'server_name': sample.get('server_name', 'unknown'),
                'status': sample.get('status', 'normal'),
                'timeframe': sample.get('timeframe', 'unknown'),
                'severity': sample.get('severity', 'low')
            }
            
            # Flatten metrics
            metrics = sample.get('metrics', {})
            for metric_name, metric_value in metrics.items():
                record[metric_name] = metric_value
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Group by server to create proper time series
        log_message("📊 Organizing time series by server...")
        
        server_dataframes = []
        for server_name in df['server_name'].unique():
            server_df = df[df['server_name'] == server_name].copy()
            server_df = server_df.sort_values('timestamp').reset_index(drop=True)
            server_df['series_id'] = server_name
            server_df['time_idx'] = range(len(server_df))
            server_dataframes.append(server_df)
        
        # Combine all servers
        final_df = pd.concat(server_dataframes, ignore_index=True)
        log_message(f"✅ Created time series: {final_df.shape}")
        
        return self._engineer_features(final_df)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for TFT training with fixed categorical handling."""
        log_message("🔧 Engineering features for TFT...")
        
        # Sort by series and timestamp
        df = df.sort_values(['series_id', 'timestamp']).reset_index(drop=True)
        
        # Re-create time index properly for each series
        df['time_idx'] = df.groupby('series_id').cumcount()
        
        # Keep categorical variables as strings (TFT expects strings)
        df['status'] = df['status'].fillna('normal').astype(str)
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Identify numeric metrics columns
        metadata_cols = {
            'timestamp', 'series_id', 'status', 'time_idx', 
            'hour', 'day_of_week', 'month', 'server_name', 
            'timeframe', 'severity'
        }
        
        metric_cols = [
            col for col in df.columns 
            if col not in metadata_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        log_message(f"📊 Found {len(metric_cols)} metric columns")
        
        # Fill missing values and ensure numeric types
        for col in metric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        
        # Create target variables for key metrics
        target_metrics = []
        for col in ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average']:
            if col in df.columns:
                target_metrics.append(col)
        
        # If standard metrics not found, use first numeric columns
        if not target_metrics:
            target_metrics = metric_cols[:4] if len(metric_cols) >= 4 else metric_cols
        
        # Create target columns (shifted values for prediction)
        for col in target_metrics:
            df[f'{col}_target'] = df.groupby('series_id')[col].shift(-1)
        
        # Remove rows with missing targets (last row of each series)
        if target_metrics:
            df = df.dropna(subset=[f'{target_metrics[0]}_target'])
        
        log_message(f"✅ Feature engineering complete. Shape: {df.shape}")
        log_message(f"🎯 Target variables: {target_metrics}")
        
        return df
    
    def create_time_series_dataset(self, df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Create PyTorch Forecasting TimeSeriesDataSet."""
        log_message("📊 Creating TimeSeriesDataSet for TFT...")
        
        # Define prediction and context lengths
        max_prediction_length = self.config.get('prediction_horizon', 6)
        max_encoder_length = self.config.get('context_length', 24)
        
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
        
        # Get numeric columns as time-varying unknown reals
        metadata_cols = {
            'timestamp', 'series_id', 'status', 'time_idx', 
            'hour', 'day_of_week', 'month', 'server_name',
            'timeframe', 'severity'  
        }
        target_cols = {col for col in df.columns if col.endswith('_target')}
        
        time_varying_unknown_reals = [
            col for col in df.columns 
            if (col not in metadata_cols and 
                col not in target_cols and 
                pd.api.types.is_numeric_dtype(df[col]))
        ]
        
        # Use status as string categorical
        time_varying_unknown_categoricals = ['status']
        
        # Primary target (first available target column)
        target_col = [col for col in df.columns if col.endswith('_target')][0]
        
        log_message(f"🎯 Primary target: {target_col}")
        log_message(f"📊 Time-varying reals: {len(time_varying_unknown_reals)} columns")
        log_message(f"📊 Categorical variables: {time_varying_unknown_categoricals}")
        
        try:
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
                time_varying_unknown_categoricals=time_varying_unknown_categoricals,
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
            
        except Exception as e:
            log_message(f"❌ Failed to create TimeSeriesDataSet: {e}")
            raise


class TFTTrainer:
    """Fixed TFT trainer for Lightning 2.0+."""
    
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
        """Main training method with Lightning 2.0+ compatibility."""
        log_message("🏋️ Starting TFT training with Lightning 2.0+...")
        self.training_stats['start_time'] = datetime.now()
        
        try:
            # Validate environment first
            if not self._validate_environment():
                return False
            
            # Load dataset
            training_dir = self.config.get('training_dir', './training/')
            df = self.preprocessor.load_existing_metrics_dataset(training_dir)
            
            # Create time series datasets
            training_dataset, validation_dataset = self.preprocessor.create_time_series_dataset(df)
            
            # Create data loaders with optimal settings
            batch_size = self.config.get('batch_size', 32)
            num_workers = min(get_optimal_workers(), 4)  # Limit workers for stability
            
            train_dataloader = training_dataset.to_dataloader(
                train=True, 
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            val_dataloader = validation_dataset.to_dataloader(
                train=False, 
                batch_size=batch_size * 2,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            # Create TFT model
            self.model = self._create_tft_model(training_dataset)
            
            # Setup Lightning trainer
            self.trainer = self._setup_trainer()
            
            # Train model with error handling
            log_message("🚀 Starting TFT training loop...")
            try:
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
                
            except Exception as training_error:
                log_message(f"❌ Training loop failed: {training_error}")
                log_message(f"Error details: {str(training_error)}")
                return False
            
        except Exception as e:
            log_message(f"❌ TFT training failed: {str(e)}")
            log_message(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _validate_environment(self) -> bool:
        """Validate training environment compatibility for Lightning 2.0+."""
        log_message("🔍 Validating TFT training environment...")
        
        # Check PyTorch
        try:
            log_message(f"✅ PyTorch: {torch.__version__}")
        except Exception as e:
            log_message(f"❌ PyTorch issue: {e}")
            return False
        
        # Check Lightning (not pytorch_lightning)
        try:
            import lightning as L
            log_message(f"✅ Lightning: {L.__version__}")
        except Exception as e:
            log_message(f"❌ Lightning issue: {e}")
            log_message("💡 Install with: pip install lightning>=2.0.0")
            return False
        
        # Check PyTorch Forecasting
        try:
            import pytorch_forecasting
            log_message(f"✅ PyTorch Forecasting available")
        except Exception as e:
            log_message(f"❌ PyTorch Forecasting issue: {e}")
            return False
        
        # Check Safetensors
        try:
            from safetensors.torch import save_file, load_file
            log_message("✅ Safetensors available")
        except ImportError:
            log_message("❌ Safetensors not available")
            return False
        
        # Check dataset availability
        training_dir = Path(self.config.get('training_dir', './training/'))
        dataset_path = training_dir / 'metrics_dataset.json'
        
        if not dataset_path.exists():
            log_message(f"❌ No metrics dataset found: {dataset_path}")
            log_message("💡 Run dataset generation first")
            return False
        else:
            log_message(f"✅ Dataset found: {dataset_path}")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            log_message(f"✅ GPU available: {gpu_name} ({gpu_memory}GB)")
        else:
            log_message("⚠️  No GPU available, using CPU")
        
        log_message("✅ TFT training environment validated")
        return True
    
    def _create_tft_model(self, training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
        """Create TFT model compatible with Lightning 2.0+."""
        log_message("🤖 Creating Temporal Fusion Transformer...")
        
        # Model configuration with conservative settings for compatibility
        model_config = {
            'learning_rate': self.config.get('learning_rate', 0.03),
            'hidden_size': self.config.get('hidden_size', 16),  # Smaller for stability
            'attention_head_size': self.config.get('attention_heads', 4), 
            'dropout': self.config.get('dropout', 0.1),
            'hidden_continuous_size': self.config.get('continuous_size', 8),  # Smaller for stability
            'loss': QuantileLoss(),
            'log_interval': 10,
            'reduce_on_plateau_patience': 4,
        }
        
        try:
            # Create model from dataset
            tft = TemporalFusionTransformer.from_dataset(
                training_dataset,
                **model_config
            )
            
            param_count = sum(p.numel() for p in tft.parameters())
            log_message(f"✅ TFT model created with {param_count:,} parameters")
            
            return tft
            
        except Exception as e:
            log_message(f"❌ Failed to create TFT model: {e}")
            raise
    
    def _setup_trainer(self) -> Trainer:
        """Setup Lightning 2.0+ Trainer."""
        log_message("⚡ Setting up Lightning 2.0+ trainer...")
        
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
            'callbacks': callbacks,
            'logger': logger,
            'gradient_clip_val': 0.1,
            'enable_progress_bar': True,
            'enable_model_summary': True,
        }
        
        # Set accelerator based on availability
        if torch.cuda.is_available():
            trainer_config['accelerator'] = 'gpu'
            trainer_config['devices'] = 1
            
            # Mixed precision for GPU
            if self.config.get('mixed_precision', True):
                trainer_config['precision'] = 16
                log_message("🚀 Mixed precision training enabled")
        else:
            trainer_config['accelerator'] = 'cpu'
            trainer_config['devices'] = 1
        
        try:
            trainer = Trainer(**trainer_config)
            log_message("✅ Lightning 2.0+ trainer configured")
            return trainer
            
        except Exception as e:
            log_message(f"❌ Failed to create trainer: {e}")
            raise
    
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
            config_path = model_dir / 'model_config.json'
            model_config = {
                'model_type': 'TemporalFusionTransformer',
                'framework': 'pytorch_forecasting',
                'pytorch_version': torch.__version__,
                'lightning_version': L.__version__,
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
    """Factory function to create TFT trainer."""
    if not LIGHTNING_AVAILABLE:
        raise ImportError("Lightning 2.0+ is required. Install with: pip install lightning>=2.0.0")
    return TFTTrainer(config)


def validate_training_environment(config: Dict[str, Any]) -> bool:
    """Validate TFT training environment."""
    if not LIGHTNING_AVAILABLE:
        log_message("❌ Lightning 2.0+ not available")
        return False
    trainer = TFTTrainer(config)
    return trainer._validate_environment()


if __name__ == "__main__":
    # Test script
    print("🚀 Fixed TFT Training Core (Lightning 2.0+ Compatible)")
    print("=" * 60)
    
    # Check Lightning availability
    try:
        import lightning as L
        print(f"✅ Lightning {L.__version__} detected")
    except ImportError:
        print("❌ Lightning not found")
        print("💡 Install with: pip install lightning>=2.0.0")
        sys.exit(1)
    
    # Check for old pytorch_lightning
    try:
        import pytorch_lightning
        print("⚠️  WARNING: Old pytorch_lightning package detected")
        print("   This may cause conflicts. Consider uninstalling it.")
    except ImportError:
        print("✅ No old pytorch_lightning package found (good!)")
    
    print("\n📦 Required packages:")
    print("   pip install torch==2.0.1")
    print("   pip install lightning>=2.0.0")
    print("   pip install pytorch-forecasting==1.0.0")
    print("   pip install safetensors>=0.3.0")
    