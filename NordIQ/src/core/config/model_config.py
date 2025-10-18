"""
TFT Model Configuration - Training Hyperparameters

SINGLE SOURCE OF TRUTH for all TFT model settings, training parameters,
and data processing configuration.

If you want to change batch size, learning rate, context length, etc.,
change it HERE and ONLY here. All training scripts import from this file.
"""

from pathlib import Path

# =============================================================================
# Model Architecture
# =============================================================================

MODEL_CONFIG = {
    # Framework and model type
    'framework': 'pytorch_forecasting',
    'model_type': 'TemporalFusionTransformer',
    'save_format': 'safetensors',  # Safer than pickle

    # Architecture dimensions
    'hidden_size': 32,                  # Hidden layer size (32 = production, 16 = faster training)
    'attention_heads': 8,                # Multi-head attention (8 = production, 4 = faster)
    'dropout': 0.15,                     # Dropout rate for regularization
    'continuous_size': 16,               # Continuous feature embedding size
    'hidden_continuous_size': 16,        # Hidden continuous layer size
    'output_size': 7,                    # Output quantiles (p10, p20, p30, p40, p50, p60, p70, p80, p90)
    'loss': 'quantile',                  # Loss function (quantile for probabilistic forecasting)

    # Time series dimensions (NordIQ Metrics Framework 14-metric system)
    'prediction_horizon': 96,            # 96 timesteps × 5min = 8 hours ahead
    'context_length': 288,               # 288 timesteps × 5min = 24 hours lookback
    'min_encoder_length': 144,           # Minimum encoder length (12 hours)
    'min_prediction_length': 12,         # Minimum prediction length (1 hour)

    # =============================================================================
    # Training Configuration
    # =============================================================================

    # Training epochs and batch size
    'epochs': 20,                        # PRODUCTION: 20 epochs, FAST TEST: 3 epochs
    'batch_size': 32,                    # PRODUCTION: 32, LOW MEMORY: 16, HIGH MEMORY: 64
    'learning_rate': 0.01,               # Initial learning rate
    'gradient_clip_val': 0.1,            # Gradient clipping to prevent exploding gradients

    # Early stopping and learning rate reduction
    'early_stopping_patience': 8,        # Stop if no improvement for 8 epochs
    'reduce_on_plateau_patience': 4,     # Reduce LR if no improvement for 4 epochs
    'reduce_lr_factor': 0.5,             # Multiply LR by 0.5 when reducing
    'min_lr': 1e-6,                      # Minimum learning rate

    # Learning rate finder (optional)
    'auto_lr_find': False,               # Set to True to automatically find learning rate
    'lr_monitor_interval': 'step',       # Log LR every 'step' or 'epoch'
    'log_every_n_steps': 50,             # Log metrics every N steps

    # =============================================================================
    # Optimization and Performance
    # =============================================================================

    # Precision and optimization
    'precision': '32-true',              # Options: '32-true' (default), '16-mixed' (faster), 'bf16-mixed' (A100 GPUs)
    'accumulate_grad_batches': 1,        # Gradient accumulation (1 = disabled, 4 = effective batch_size × 4)
    'multi_target': False,               # Set to True for multi-target prediction

    # Data loader settings (CRITICAL for performance)
    'num_workers': 4,                    # PRODUCTION: 4-8, LOW CPU: 0-2
    'pin_memory': True,                  # Faster GPU transfer (True if using GPU)
    'persistent_workers': True,          # Keep workers alive between epochs (faster)
    'prefetch_factor': 3,                # Number of batches to prefetch per worker

    # Mixed precision training (faster on modern GPUs)
    'mixed_precision': True,             # Enable mixed precision (FP16/FP32)

    # =============================================================================
    # Data Processing
    # =============================================================================

    # Time intervals
    'poll_interval_minutes': 5,          # Data collection interval (NordIQ Metrics Framework = 5 minutes)
    'poll_interval_seconds': 300,        # Same as above in seconds
    'time_span_hours': 720,              # Total data span (720h = 30 days)

    # Data validation and preprocessing
    'validation_split': 0.2,             # 20% of data for validation
    'normalize_features': True,          # Z-score normalization
    'handle_missing_data': True,         # Interpolate missing values
    'feature_engineering': True,         # Create derived features
    'use_cyclical_encoding': True,       # Sin/cos encoding for time features

    # Fleet configuration
    'servers_count': 20,                 # Number of servers in fleet
    'anomaly_ratio': 0.12,               # Expected % of anomalous data points

    # =============================================================================
    # NordIQ Metrics Framework Metrics (14 metrics)
    # =============================================================================

    # Target metrics to predict (14 NordIQ Metrics Framework metrics)
    'target_metrics': [
        # CPU metrics (5)
        'cpu_user_pct',
        'cpu_sys_pct',
        'cpu_iowait_pct',      # CRITICAL - "System troubleshooting 101"
        'cpu_idle_pct',
        'java_cpu_pct',

        # Memory metrics (2)
        'mem_used_pct',
        'swap_used_pct',

        # Disk metrics (1)
        'disk_usage_pct',

        # Network metrics (2)
        'net_in_mb_s',
        'net_out_mb_s',

        # Connection metrics (2)
        'back_close_wait',
        'front_close_wait',

        # System metrics (2)
        'load_average',
        'uptime_days'
    ],

    # Time features (cyclical encoding)
    'time_features': [
        'hour',                  # Hour of day (0-23)
        'day_of_week',          # Day of week (0-6)
        'day_of_month',         # Day of month (1-31)
        'month',                # Month (1-12)
        'quarter',              # Quarter (1-4)
        'is_weekend',           # Boolean weekend indicator
        'is_business_hours'     # Boolean business hours indicator (9am-5pm)
    ],

    # Categorical features (embeddings)
    'categorical_features': [
        'server_name',          # Server hostname (ppml0001, ppdb001, etc.)
        'server_profile',       # Profile type (ML Compute, Database, etc.)
        'status',               # Operational status
        'timeframe',            # Time period identifier
        'datacenter',           # Datacenter location
        'environment'           # Production, staging, etc.
    ],

    # =============================================================================
    # Checkpointing and Model Saving
    # =============================================================================

    # Model checkpointing
    'save_checkpoints': True,            # Save checkpoints during training
    'checkpoint_every_n_epochs': 5,      # Save checkpoint every N epochs
    'keep_recent_checkpoints': 5,        # Keep only N most recent checkpoints
    'save_best_only': True,              # Only save checkpoints that improve validation loss
    'model_compression': True,           # Compress saved models (smaller file size)

    # =============================================================================
    # Directories
    # =============================================================================

    # Output directories (relative to project root)
    'training_dir': './training/',       # Training data directory
    'models_dir': './models/',           # Saved models directory
    'checkpoints_dir': './checkpoints/', # Training checkpoints
    'logs_dir': './logs/',               # Training logs

    # =============================================================================
    # Logging and Monitoring
    # =============================================================================

    'log_level': 'INFO',                 # Logging level (DEBUG, INFO, WARNING, ERROR)
    'tensorboard_enabled': True,         # Enable TensorBoard logging
    'cleanup_enabled': True,             # Clean up old logs and checkpoints
    'log_retention_days': 45,            # Keep logs for N days

    # =============================================================================
    # Reproducibility
    # =============================================================================

    'random_seed': 42,                   # Random seed for reproducibility

    # =============================================================================
    # Advanced Features (Phase 3)
    # =============================================================================

    'visualization': {
        'enabled': True,                 # Enable visualizations
        'save_plots': True,              # Save plots to disk
        'interactive_plots': True,       # Create interactive HTML plots
        'plot_dir': './plots/',          # Plot output directory
        'dpi': 150,                      # Plot resolution
        'figsize': [16, 12],             # Plot size (width, height)
        'style': 'dark_background',      # Matplotlib style
        'color_palette': 'viridis',      # Color palette
        'show_confidence_intervals': True,   # Show prediction confidence intervals
        'show_attention_weights': True,      # Show attention mechanism weights
        'animate_predictions': False,        # Create animated prediction plots
        'export_formats': ['png', 'svg', 'html']  # Export formats
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def setup_directories() -> bool:
    """
    Create required directories if they don't exist.

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        for key in ['training_dir', 'models_dir', 'checkpoints_dir', 'logs_dir']:
            Path(MODEL_CONFIG[key]).mkdir(parents=True, exist_ok=True)

        # Create visualization directory if enabled
        if MODEL_CONFIG['visualization']['enabled']:
            Path(MODEL_CONFIG['visualization']['plot_dir']).mkdir(parents=True, exist_ok=True)

        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False


def get_device() -> str:
    """
    Get available device for training (CUDA or CPU).

    Returns:
        str: 'cuda' if GPU available, 'cpu' otherwise
    """
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate configuration values for consistency.

    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []

    # Check batch size
    if MODEL_CONFIG['batch_size'] < 1:
        errors.append("batch_size must be >= 1")

    # Check epochs
    if MODEL_CONFIG['epochs'] < 1:
        errors.append("epochs must be >= 1")

    # Check learning rate
    if not (1e-6 <= MODEL_CONFIG['learning_rate'] <= 1.0):
        errors.append("learning_rate must be between 1e-6 and 1.0")

    # Check time dimensions
    if MODEL_CONFIG['context_length'] < MODEL_CONFIG['min_encoder_length']:
        errors.append("context_length must be >= min_encoder_length")

    # Check prediction horizon
    if MODEL_CONFIG['prediction_horizon'] < MODEL_CONFIG['min_prediction_length']:
        errors.append("prediction_horizon must be >= min_prediction_length")

    # Check validation split
    if not (0.0 < MODEL_CONFIG['validation_split'] < 1.0):
        errors.append("validation_split must be between 0 and 1")

    return (len(errors) == 0, errors)


# Auto-setup directories on import
setup_directories()

# Validate configuration
is_valid, errors = validate_config()
if not is_valid:
    print("⚠️  Configuration validation warnings:")
    for error in errors:
        print(f"  - {error}")
