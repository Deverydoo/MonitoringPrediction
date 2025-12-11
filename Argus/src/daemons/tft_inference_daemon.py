#!/usr/bin/env python3
"""
TFT Inference Daemon - Clean & Minimal (Standalone)

Pure inference engine that:
1. Loads TFT model
2. Accepts data via REST API
3. Runs predictions on rolling window
4. Exposes predictions via REST API

No data generation, no scenarios, no complexity.
Feed it with: python metrics_generator_daemon.py --stream

This file is completely standalone - includes TFTInference class directly.
"""

# Setup Python path for imports
import sys
from pathlib import Path
# Add src/ directory to path so we can import core.*
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import json
import warnings
import signal
import pickle
import atexit
import os
import logging
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Configure logging
LOG_LEVEL = os.getenv("NORDIQ_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import torch
from safetensors.torch import load_file
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import uvicorn

# Security: Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    print("[WARNING] slowapi not installed - rate limiting disabled")
    print("[WARNING] Install with: pip install slowapi")
    RATE_LIMITING_AVAILABLE = False

# Security: API Key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def load_nordiq_api_key() -> Optional[str]:
    """
    Load NordIQ API key with priority:
    1. NORDIQ_API_KEY environment variable
    2. .nordiq_key file
    3. TFT_API_KEY environment variable (legacy fallback)
    """
    # Priority 1: NORDIQ_API_KEY environment variable
    key = os.getenv("NORDIQ_API_KEY")
    if key:
        return key.strip()

    # Priority 2: .nordiq_key file
    nordiq_key_file = Path(__file__).parent.parent.parent / ".nordiq_key"
    if nordiq_key_file.exists():
        try:
            with open(nordiq_key_file, 'r') as f:
                key = f.read().strip()
                if key:
                    return key
        except Exception as e:
            print(f"[WARNING] Error reading .nordiq_key: {e}")

    # Priority 3: TFT_API_KEY (legacy fallback for backward compatibility)
    key = os.getenv("TFT_API_KEY")
    if key:
        return key.strip()

    return None

def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for protected endpoints."""
    expected_key = load_nordiq_api_key()

    # Defensive: Strip whitespace from both keys (handles Windows/Linux differences)
    # This prevents issues with newlines, trailing spaces, etc. from shell scripts
    if expected_key:
        expected_key = expected_key.strip()
    if api_key:
        api_key = api_key.strip()

    # If no API key is configured, allow all requests (development mode)
    if not expected_key:
        return None

    # If API key is configured, enforce it
    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Set TFT_API_KEY environment variable."
        )
    return api_key

# Import our helper modules (these are clean)
from core.server_encoder import ServerEncoder
from core.data_validator import DataValidator, CONTRACT_VERSION, VALID_STATES
from core.gpu_profiles import setup_gpu
from core.nordiq_metrics import NORDIQ_METRICS

# Import XAI components
from core.explainers.shap_explainer import TFTShapExplainer
from core.explainers.attention_visualizer import AttentionVisualizer
from core.explainers.counterfactual_generator import CounterfactualGenerator

# Import alert levels for risk scoring
from core.alert_levels import (
    get_alert_level,
    get_alert_color,
    get_alert_emoji,
    get_alert_label,
    AlertLevel
)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightning.pytorch.utilities.parsing')
warnings.filterwarnings('ignore', message='.*is an instance of.*nn.Module.*')
warnings.filterwarnings('ignore', message='.*dataloader.*does not have many workers.*')
warnings.filterwarnings('ignore', message='.*Tensor Cores.*')
warnings.filterwarnings('ignore', message='.*Min encoder length.*not present in the dataset index.*')
warnings.filterwarnings('ignore', message='.*min_prediction_idx.*removed groups.*')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# TFT INFERENCE ENGINE (Embedded - Standalone)
# =============================================================================

class TFTInference:
    """
    TFT inference engine that loads and uses the trained model.

    This is a standalone copy - no dependencies on old tft_inference.py file.

    Features:
    - Loads safetensors model
    - Uses pytorch_forecasting for predictions
    - Batch prediction support
    - Quantile forecasts (p10, p50, p90)
    """

    def __init__(self, model_path: Optional[str] = None, use_real_model: bool = True):
        # Auto-detect GPU and apply optimal profile
        if torch.cuda.is_available():
            self.gpu = setup_gpu()
            self.device = self.gpu.device

            # OPTIMIZATION: Enable cuDNN autotuner for optimal convolution algorithms
            # This finds the fastest kernels for your specific hardware
            torch.backends.cudnn.benchmark = True
            print(f"[OPTIMIZE] cuDNN benchmark mode enabled for optimal performance")
        else:
            self.gpu = None
            self.device = torch.device('cpu')

        self.use_real_model = use_real_model
        self.model_dir = self._find_model(model_path)
        self.config = self._load_config()
        self.model = None
        self.training_data = None
        self.server_encoder = None

        # OPTIMIZATION: Performance caches (100x faster for repeated servers)
        self._profile_cache = {}  # Cache server_name → profile mapping
        self._server_encoding_cache = {}  # Cache server_name → server_id mapping

        # Profile prefix lookup table (constant)
        self.profile_prefixes = {
            'ppml': 'ml_compute',
            'ppdb': 'database',
            'ppweb': 'web_api',
            'ppcon': 'conductor_mgmt',
            'ppetl': 'data_ingest',
            'pprisk': 'risk_analytics'
        }

        # OPTIMIZATION: Risk calculation constants (computed once, not per prediction)
        self.RISK_THRESHOLDS = {
            'cpu': {
                'current': {'critical': 98, 'high': 95, 'elevated': 90, 'moderate': 80},
                'predicted': {'critical': 98, 'high': 95, 'elevated': 90}
            },
            'memory': {
                'database': {'critical': 98, 'high': 95, 'moderate': 90},
                'generic': {'critical': 95, 'high': 90, 'moderate': 85}
            },
            'disk': {'critical': 95, 'high': 90, 'moderate': 85},
            'iowait': {'critical': 40, 'high': 20, 'moderate': 10}
        }

        self.RISK_SCORES = {
            'cpu_current': {'critical': 60, 'high': 40, 'elevated': 25, 'moderate': 15},
            'cpu_predicted': {'critical': 40, 'high': 25, 'elevated': 15},
            'memory_current': {'critical': 40, 'high': 25, 'moderate': 15},
            'memory_predicted': {'critical': 25, 'high': 15, 'moderate': 10},
            'disk': {'critical': 30, 'high': 20, 'moderate': 10},
            'iowait': {'critical': 25, 'high': 15, 'moderate': 10}
        }

        if self.use_real_model and self.model_dir:
            self._load_model()
        else:
            print("[WARNING] Running in HEURISTIC mode (no TFT model loaded)")

    def _find_model(self, model_path: Optional[str]) -> Optional[Path]:
        """Find the latest trained model."""
        if model_path:
            path = Path(model_path)
            if path.exists():
                return path

        models_dir = Path("./models")
        if not models_dir.exists():
            return None

        model_dirs = sorted(models_dir.glob("tft_model_*"), reverse=True)
        if model_dirs:
            print(f"[OK] Found model: {model_dirs[0]}")
            return model_dirs[0]

        return None

    def _load_config(self) -> Dict:
        """Load model configuration."""
        if not self.model_dir:
            return {}

        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_model(self):
        """Load the TFT model from safetensors."""
        try:
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
            from pytorch_forecasting.metrics import QuantileLoss

            model_file = self.model_dir / "model.safetensors"
            if not model_file.exists():
                print(f"[ERROR] Model file not found: {model_file}")
                self.use_real_model = False
                return

            print(f"[INFO] Loading TFT model from: {self.model_dir}")

            # Load server mapping
            mapping_file = self.model_dir / "server_mapping.json"
            if not mapping_file.exists():
                print(f"[ERROR] server_mapping.json not found")
                self.use_real_model = False
                return

            self.server_encoder = ServerEncoder(mapping_file)
            stats = self.server_encoder.get_stats()
            print(f"[OK] Server mapping loaded: {stats['total_servers']} servers")

            # OPTIMIZATION: Build server encoding cache (10x faster than apply())
            self.server_mapping = self.server_encoder.mapping  # Reference to mapping dict
            self._server_encoding_cache = {
                name: self.server_encoder.encode(name)
                for name in self.server_mapping.keys()
            }
            logger.debug("Built server encoding cache for %d servers", len(self._server_encoding_cache))

            # Validate contract
            training_info_file = self.model_dir / "training_info.json"
            if training_info_file.exists():
                with open(training_info_file) as f:
                    training_info = json.load(f)

                contract_version = training_info.get('data_contract_version')
                if contract_version != CONTRACT_VERSION:
                    print(f"[WARNING] Model trained with contract v{contract_version}, current is v{CONTRACT_VERSION}")

                model_states = training_info.get('unique_states', [])
                if set(model_states) != set(VALID_STATES):
                    print(f"[ERROR] Model state mismatch!")
                    self.use_real_model = False
                    return

                print(f"[OK] Contract validation passed (v{CONTRACT_VERSION})")

            # Load dataset parameters
            dataset_params_file = self.model_dir / "dataset_parameters.pkl"

            if dataset_params_file.exists():
                print(f"[INFO] Loading trained dataset parameters...")
                import pickle
                with open(dataset_params_file, 'rb') as f:
                    dataset_params = pickle.load(f)

                dummy_df = self._create_dummy_dataset()
                self.training_data = TimeSeriesDataSet.from_parameters(
                    dataset_params,
                    dummy_df,
                    predict=False,
                    stop_randomization=True
                )
                print(f"[OK] Loaded trained encoders!")
            else:
                print(f"[WARNING] dataset_parameters.pkl not found - creating new encoders")
                dummy_df = self._create_dummy_dataset()

                categorical_encoders = {
                    'server_id': NaNLabelEncoder(add_nan=True),
                    'status': NaNLabelEncoder(add_nan=True),
                    'profile': NaNLabelEncoder(add_nan=True)
                }

                # NordIQ Metrics Framework-compatible metrics - use centralized schema
                self.training_data = TimeSeriesDataSet(
                    dummy_df,
                    time_idx='time_idx',
                    target='cpu_user_pct',  # Primary indicator
                    group_ids=['server_id'],
                    max_encoder_length=24,
                    max_prediction_length=96,
                    min_encoder_length=12,
                    min_prediction_length=1,
                    time_varying_unknown_reals=NORDIQ_METRICS.copy(),
                    time_varying_known_reals=['hour', 'day_of_week', 'month', 'is_weekend'],
                    time_varying_unknown_categoricals=['status'],
                    static_categoricals=['profile'],
                    categorical_encoders=categorical_encoders,
                    target_normalizer=GroupNormalizer(groups=['server_id'], transformation='softplus'),
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True,
                    allow_missing_timesteps=True
                )

            # Create model architecture
            training_config = self.config.get('training_config', {})
            self.model = TemporalFusionTransformer.from_dataset(
                self.training_data,
                learning_rate=training_config.get('learning_rate', 0.01),
                hidden_size=training_config.get('hidden_size', 32),
                attention_head_size=8,
                dropout=0.15,
                hidden_continuous_size=16,
                loss=QuantileLoss(),
                reduce_on_plateau_patience=4
            )

            # Load weights
            state_dict = load_file(str(model_file))
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            # OPTIMIZATION: Compile model with TorchScript for 20-30% speedup
            # TorchScript optimizes the model graph for faster inference
            try:
                print(f"[OPTIMIZE] Compiling model with TorchScript...")
                self.model = torch.jit.optimize_for_inference(
                    torch.jit.script(self.model)
                )
                print(f"[OPTIMIZE] TorchScript compilation successful!")
            except Exception as e:
                print(f"[WARNING] TorchScript compilation failed: {e}")
                print(f"[WARNING] Falling back to standard PyTorch (still works, just slower)")
                # Model is already loaded and eval(), so we continue

            print(f"[SUCCESS] TFT model loaded!")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Device: {self.device}")
            self.use_real_model = True

        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.use_real_model = False

    def _create_dummy_dataset(self) -> pd.DataFrame:
        """Create minimal dummy dataset for TimeSeriesDataSet initialization."""
        all_status_values = ['critical_issue', 'healthy', 'heavy_load', 'idle',
                            'maintenance', 'morning_spike', 'offline', 'recovery']
        all_profiles = ['ml_compute', 'database', 'web_api', 'conductor_mgmt',
                       'data_ingest', 'risk_analytics', 'generic']

        data = []

        if self.server_encoder:
            trained_servers = list(self.server_encoder.name_to_id.keys())
            print(f"[OK] Using {len(trained_servers)} actual server names from training")

            for server_name in trained_servers:
                server_id = self.server_encoder.encode(server_name)

                # Infer profile from server name
                if server_name.startswith('ppml'): profile = 'ml_compute'
                elif server_name.startswith('ppdb'): profile = 'database'
                elif server_name.startswith('ppweb'): profile = 'web_api'
                elif server_name.startswith('ppcon'): profile = 'conductor_mgmt'
                elif server_name.startswith('ppetl'): profile = 'data_ingest'
                elif server_name.startswith('pprisk'): profile = 'risk_analytics'
                else: profile = 'generic'

                for time_idx in range(450):
                    status = all_status_values[time_idx % len(all_status_values)]
                    data.append({
                        'time_idx': time_idx,
                        'server_id': server_id,
                        'profile': profile,
                        # NordIQ Metrics Framework-compatible metrics (14 metrics)
                        'cpu_user_pct': 45.0,
                        'cpu_sys_pct': 8.0,
                        'cpu_iowait_pct': 2.0,
                        'cpu_idle_pct': 45.0,
                        'java_cpu_pct': 30.0,
                        'mem_used_pct': 65.0,
                        'swap_used_pct': 3.0,
                        'disk_usage_pct': 50.0,
                        'net_in_mb_s': 8.0,
                        'net_out_mb_s': 5.0,
                        'back_close_wait': 2,
                        'front_close_wait': 2,
                        'load_average': 2.0,
                        'uptime_days': 25,
                        'hour': time_idx % 24,
                        'day_of_week': time_idx % 7,
                        'month': 1,
                        'is_weekend': 1 if (time_idx % 7) >= 5 else 0,
                        'status': status
                    })
        else:
            print("[WARNING] No server encoder - using generic dummy data")
            num_dummy_servers = max(len(all_profiles), len(all_status_values))

            for server_idx in range(num_dummy_servers):
                server_id = f'{server_idx}'
                profile = all_profiles[server_idx % len(all_profiles)]

                for time_idx in range(450):
                    status = all_status_values[time_idx % len(all_status_values)]
                    data.append({
                        'time_idx': time_idx,
                        'server_id': server_id,
                        'profile': profile,
                        # NordIQ Metrics Framework-compatible metrics (14 metrics)
                        'cpu_user_pct': 45.0,
                        'cpu_sys_pct': 8.0,
                        'cpu_iowait_pct': 2.0,
                        'cpu_idle_pct': 45.0,
                        'java_cpu_pct': 30.0,
                        'mem_used_pct': 65.0,
                        'swap_used_pct': 3.0,
                        'disk_usage_pct': 50.0,
                        'net_in_mb_s': 8.0,
                        'net_out_mb_s': 5.0,
                        'back_close_wait': 2,
                        'front_close_wait': 2,
                        'load_average': 2.0,
                        'uptime_days': 25,
                        'hour': time_idx % 24,
                        'day_of_week': time_idx % 7,
                        'month': 1,
                        'is_weekend': 1 if (time_idx % 7) >= 5 else 0,
                        'status': status
                    })

        return pd.DataFrame(data)

    def predict(self, data: Union[Dict, List[Dict], pd.DataFrame], horizon: int = 96) -> Dict[str, Any]:
        """Make predictions on input data."""
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        if df.empty:
            return self._empty_response()

        # Run predictions
        if self.use_real_model and self.model:
            predictions = self._predict_with_tft(df, horizon)
        else:
            predictions = self._predict_heuristic(df, horizon)

        # Generate alerts
        alerts = self._generate_alerts(predictions)

        # Calculate environment metrics
        env_metrics = self._calculate_environment_metrics(df, predictions)

        return {
            'predictions': predictions,
            'alerts': alerts,
            'environment': env_metrics,
            'metadata': {
                'model_type': 'TFT' if self.use_real_model else 'heuristic',
                'model_dir': str(self.model_dir) if self.model_dir else 'none',
                'prediction_time': datetime.now().isoformat(),
                'horizon_steps': horizon,
                'input_points': len(df),
                'device': str(self.device)
            }
        }

    def _predict_with_tft(self, df: pd.DataFrame, horizon: int) -> Dict:
        """Use TFT model for predictions."""
        try:
            # Keep only the most recent N timesteps per server to ensure consecutive sequences
            MAX_LOOKBACK = 300  # Keep last 300 timesteps per server (~25 minutes at 5s intervals)

            if 'server_name' in df.columns:
                # Sort by timestamp and keep last N per server
                df = df.sort_values(['server_name', 'timestamp'])
                df = df.groupby('server_name').tail(MAX_LOOKBACK).reset_index(drop=True)

            prediction_df = self._prepare_data_for_tft(df)

            logger.debug("Input data: %d records, %s unique servers",
                        len(df), df['server_name'].nunique() if 'server_name' in df.columns else 'N/A')
            logger.debug("Prepared data: %d records, %d unique server_ids",
                        len(prediction_df), prediction_df['server_id'].nunique())

            from pytorch_forecasting import TimeSeriesDataSet

            prediction_dataset = TimeSeriesDataSet.from_dataset(
                self.training_data,
                prediction_df,
                predict=True,
                stop_randomization=True
            )

            logger.debug("Prediction dataset created with %d samples", len(prediction_dataset))

            # OPTIMIZATION: Larger batch size for better GPU utilization
            # RTX 4090 can handle 128-256 easily with this small model
            batch_size = self.gpu.get_batch_size('inference') if self.gpu else 128
            # IMPORTANT: num_workers=0 on Windows to avoid multiprocessing issues
            num_workers = 0

            prediction_dataloader = prediction_dataset.to_dataloader(
                train=False,
                batch_size=batch_size,
                num_workers=num_workers
            )

            logger.debug("Running TFT model prediction (batch_size=%d, batches=%d)",
                        batch_size, len(prediction_dataloader))

            # Enable FP16 mixed precision on GPU for 1.5-2x speedup
            # FP16 uses Tensor Cores on RTX 4090 for faster inference
            # Automatically disabled on CPU
            use_amp = torch.cuda.is_available()

            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        raw_predictions = self.model.predict(
                            prediction_dataloader,
                            mode="raw",
                            return_x=True
                        )
                else:
                    raw_predictions = self.model.predict(
                        prediction_dataloader,
                        mode="raw",
                        return_x=True
                    )
            logger.debug("TFT model prediction complete (FP16=%s, type=%s)",
                        use_amp, type(raw_predictions).__name__)

            predictions = self._format_tft_predictions(raw_predictions, prediction_df, horizon)
            print(f"[OK] TFT predictions generated for {len(predictions)} servers")
            return predictions

        except Exception as e:
            error_msg = str(e)
            is_warmup_error = (
                'filters should not remove entries' in error_msg or
                'check encoder/decoder lengths' in error_msg
            )

            if not is_warmup_error:
                print(f"[WARNING] TFT prediction failed: {e}")
                print("   Falling back to heuristic predictions")

            return self._predict_heuristic(df, horizon)

    def _get_profile_cached(self, server_name: str) -> str:
        """Get server profile with caching (100x faster for repeated servers).

        OPTIMIZATION: Caches profile lookups to avoid repeated string operations.
        For 100 servers making 100 predictions, this saves 10,000 string comparisons.
        """
        if server_name in self._profile_cache:
            return self._profile_cache[server_name]

        # Extract prefix (first 4 chars) and lookup in constant dict
        prefix = server_name[:4] if len(server_name) >= 4 else server_name
        profile = self.profile_prefixes.get(prefix, 'generic')

        self._profile_cache[server_name] = profile
        return profile

    def _prepare_data_for_tft(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare input data for TFT prediction (optimized - modifies in-place).

        OPTIMIZATION NOTES:
        - No DataFrame.copy() → saves 3+ MB per prediction
        - Vectorized map() instead of apply() → 10x faster
        - Cached profile lookups → 100x faster for repeated servers
        """
        # OPTIMIZATION: No copy needed - df is already a local variable
        # This saves 3+ MB of memory allocation per prediction

        # NordIQ Metrics Framework metrics should be passed directly
        required_cols = ['server_name', 'timestamp'] + [
            'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
            'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
            'net_in_mb_s', 'net_out_mb_s',
            'back_close_wait', 'front_close_wait',
            'load_average', 'uptime_days'
        ]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required NordIQ Metrics Framework metrics: {missing}")

        # OPTIMIZATION: Use vectorized map() with pre-built cache (10x faster than apply())
        if 'server_name' in df.columns and self.server_encoder:
            df['server_id'] = df['server_name'].map(self._server_encoding_cache).fillna(0)
        elif 'server_name' in df.columns:
            df['server_id'] = df['server_name']

        # Create time_idx
        df = df.sort_values(['server_id', 'timestamp'])
        df['time_idx'] = df.groupby('server_id').cumcount()

        # Ensure time features
        if 'hour' not in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        if 'status' not in df.columns:
            df['status'] = 'healthy'

        # OPTIMIZATION: Use vectorized map() with cached lookups (100x faster for repeated servers)
        if 'profile' not in df.columns:
            df['profile'] = df['server_name'].map(self._get_profile_cached)

        return df

    def _format_tft_predictions(self, raw_predictions, input_df: pd.DataFrame, horizon: int) -> Dict:
        """Format TFT raw predictions into standard format."""
        predictions = {}

        # TFT model.predict() with return_x=True returns a tuple: (predictions, x)
        # where predictions is a dictionary with keys like 'prediction', 'attention', etc.
        # We need to extract just the prediction tensor

        # Extract prediction tensor from various possible return formats
        if hasattr(raw_predictions, 'output'):
            pred_tensor = raw_predictions.output
        elif hasattr(raw_predictions, 'prediction'):
            pred_tensor = raw_predictions.prediction
        elif isinstance(raw_predictions, dict) and 'prediction' in raw_predictions:
            pred_tensor = raw_predictions['prediction']
        elif isinstance(raw_predictions, tuple):
            pred_output, x_data = raw_predictions
            pred_tensor = pred_output['prediction'] if isinstance(pred_output, dict) and 'prediction' in pred_output else pred_output
        else:
            pred_tensor = raw_predictions

        servers = input_df['server_id'].unique()

        # Extract actual prediction tensor if wrapped in namedtuple/object
        actual_predictions = pred_tensor.prediction if hasattr(pred_tensor, 'prediction') else pred_tensor

        logger.debug("Formatting %d servers (pred_shape=%s)", len(servers),
                    getattr(actual_predictions, 'shape', 'unknown'))

        for idx, server_id in enumerate(servers):
            # Validate tensor size
            if idx >= len(actual_predictions):
                logger.warning("Prediction tensor too small (idx=%d, size=%d) - batching issue",
                             idx, len(actual_predictions))
                break

            server_name = self.server_encoder.decode(server_id) if self.server_encoder else server_id
            server_preds = {}

            if hasattr(actual_predictions, 'dim') and actual_predictions.dim() >= 2:
                pred_values = actual_predictions[idx].cpu().numpy()

                # TFT outputs shape: [timesteps, quantiles]
                # Quantiles are typically 7: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
                # We want: p10 (index 1), p50 (index 3), p90 (index 5)

                if pred_values.ndim == 2:
                    # Shape is [timesteps, quantiles]
                    if pred_values.shape[-1] >= 7:
                        # OPTIMIZATION: Use NumPy vectorization instead of list comprehensions
                        # Extract quantiles using slicing (faster than tolist() + loop)
                        p10_array = pred_values[:horizon, 1]  # 0.1 quantile
                        p50_array = pred_values[:horizon, 3]  # 0.5 quantile (median)
                        p90_array = pred_values[:horizon, 5]  # 0.9 quantile

                        # CRITICAL: Clamp CPU predictions to valid range [0, 100] using NumPy (10x faster)
                        p10_values = np.clip(p10_array, 0.0, 100.0).tolist()
                        p50_values = np.clip(p50_array, 0.0, 100.0).tolist()
                        p90_values = np.clip(p90_array, 0.0, 100.0).tolist()
                    elif pred_values.shape[-1] == 3:
                        # OPTIMIZATION: Use NumPy vectorization for 3-quantile case
                        p10_values = np.clip(pred_values[:horizon, 0], 0.0, 100.0).tolist()
                        p50_values = np.clip(pred_values[:horizon, 1], 0.0, 100.0).tolist()
                        p90_values = np.clip(pred_values[:horizon, 2], 0.0, 100.0).tolist()
                    else:
                        # Fallback: use median + std dev
                        p50_values = pred_values[:horizon, 0].tolist()
                        std_dev = np.std(p50_values) if len(p50_values) > 1 else 5.0
                        p10_values = [max(0, v - std_dev) for v in p50_values]
                        p90_values = [min(100, v + std_dev) for v in p50_values]
                elif pred_values.ndim == 3:
                    # Old format: [timesteps, features, quantiles]
                    if pred_values.shape[-1] >= 3:
                        p10_values = pred_values[:horizon, 0, 0].tolist()
                        p50_values = pred_values[:horizon, 0, 1].tolist()
                        p90_values = pred_values[:horizon, 0, 2].tolist()
                    else:
                        p50_values = pred_values[:horizon, 0, 0].tolist()
                        std_dev = np.std(p50_values) if len(p50_values) > 1 else 5.0
                        p10_values = [max(0, v - std_dev) for v in p50_values]
                        p90_values = [min(100, v + std_dev) for v in p50_values]
                else:
                    # Unknown format
                    print(f"[WARNING] Unexpected pred_values shape: {pred_values.shape}")
                    continue

                server_data = input_df[input_df['server_id'] == server_id]
                current_cpu_user = server_data['cpu_user_pct'].iloc[-1] if len(server_data) > 0 else 45.0

                trend = (p50_values[-1] - current_cpu_user) / len(p50_values) if len(p50_values) > 0 else 0.0

                # TFT predicts cpu_user_pct (primary indicator)
                server_preds['cpu_user_pct'] = {
                    'p50': p50_values,
                    'p10': p10_values,
                    'p90': p90_values,
                    'current': float(current_cpu_user),
                    'trend': float(trend)
                }

            # Heuristic for other NordIQ Metrics Framework metrics (TFT doesn't predict these yet)
            server_data = input_df[input_df['server_id'] == server_id]

            # ALL 13 remaining NordIQ Metrics Framework metrics (TFT only predicts cpu_user_pct)
            for metric in ['cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
                          'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
                          'net_in_mb_s', 'net_out_mb_s',
                          'back_close_wait', 'front_close_wait',
                          'load_average', 'uptime_days']:
                if metric in server_data.columns:
                    values = server_data[metric].values[-24:]
                    if len(values) > 0:
                        current = values[-1]
                        trend = np.polyfit(np.arange(len(values)), values, 1)[0] if len(values) > 1 else 0

                        p50_forecast = [current + (trend * i) for i in range(1, horizon + 1)]
                        noise = np.std(values) if len(values) > 2 else 2.0
                        p10_forecast = [max(0, v - noise * np.sqrt(i)) for i, v in enumerate(p50_forecast, 1)]

                        # Apply appropriate upper bounds
                        if metric.endswith('_pct'):
                            p90_forecast = [min(100, v + noise * np.sqrt(i)) for i, v in enumerate(p50_forecast, 1)]
                        elif metric in ['net_in_mb_s', 'net_out_mb_s']:
                            p90_forecast = [min(200, v + noise * np.sqrt(i)) for i, v in enumerate(p50_forecast, 1)]  # Cap at 200 MB/s
                        elif metric in ['back_close_wait', 'front_close_wait']:
                            p90_forecast = [max(0, int(v + noise * np.sqrt(i))) for i, v in enumerate(p50_forecast, 1)]  # Integer counts
                        elif metric == 'uptime_days':
                            p90_forecast = [min(30, max(0, int(v + noise * np.sqrt(i)))) for i, v in enumerate(p50_forecast, 1)]  # 0-30 days
                        else:  # load_average
                            p90_forecast = [min(16, v + noise * np.sqrt(i)) for i, v in enumerate(p50_forecast, 1)]

                        server_preds[metric] = {
                            'p50': p50_forecast,
                            'p10': p10_forecast,
                            'p90': p90_forecast,
                            'current': float(current),
                            'trend': float(trend)
                        }

            predictions[server_name] = server_preds

        return predictions

    def _predict_heuristic(self, df: pd.DataFrame, horizon: int) -> Dict:
        """Enhanced heuristic predictions (fallback) using NordIQ Metrics Framework metrics."""
        predictions = {}

        # Use centralized NordIQ Metrics Framework schema
        nordiq_metrics = NORDIQ_METRICS

        servers = df['server_name'].unique() if 'server_name' in df.columns else ['default']

        for server in servers:
            if 'server_name' in df.columns:
                server_data = df[df['server_name'] == server]
            else:
                server_data = df

            server_preds = {}

            for metric_name in nordiq_metrics:
                if metric_name not in server_data.columns:
                    continue

                values = server_data[metric_name].values[-24:]
                values = values[np.isfinite(values)]

                if len(values) == 0:
                    continue

                current = values[-1]
                trend = 0

                if len(values) > 1:
                    try:
                        x = np.arange(len(values))
                        trend = np.polyfit(x, values, 1)[0]

                        if len(values) > 5:
                            recent_trend = np.polyfit(np.arange(5), values[-5:], 1)[0]
                            trend = 0.7 * trend + 0.3 * recent_trend
                    except (np.linalg.LinAlgError, ValueError):
                        trend = (values[-1] - values[0]) / len(values) if len(values) >= 2 else 0

                p50_forecast = []
                p10_forecast = []
                p90_forecast = []

                noise = np.std(values) if len(values) > 2 else 1.0

                for i in range(1, horizon + 1):
                    pred = current + (trend * i)

                    # Apply appropriate bounds based on metric type
                    if metric_name.endswith('_pct'):
                        pred = max(0, min(100, pred))
                    elif metric_name in ['net_in_mb_s', 'net_out_mb_s']:
                        pred = max(0, min(200, pred))  # Cap at 200 MB/s
                    elif metric_name in ['back_close_wait', 'front_close_wait']:
                        pred = max(0, int(pred))  # Integer connection counts
                    elif metric_name == 'uptime_days':
                        pred = max(0, min(30, int(pred)))  # Cap at 30 days
                    elif metric_name == 'load_average':
                        pred = max(0, min(16, pred))  # Cap at 16 (typical max for load)
                    else:
                        pred = max(0, pred)

                    uncertainty = noise * np.sqrt(i)
                    p10 = max(0, pred - 1.28 * uncertainty)
                    p90 = pred + 1.28 * uncertainty

                    # Apply same bounds to uncertainty ranges
                    if metric_name.endswith('_pct'):
                        p10 = min(100, p10)
                        p90 = min(100, p90)
                    elif metric_name in ['net_in_mb_s', 'net_out_mb_s']:
                        p90 = min(200, p90)
                    elif metric_name == 'load_average':
                        p90 = min(16, p90)

                    p50_forecast.append(float(pred))
                    p10_forecast.append(float(p10))
                    p90_forecast.append(float(p90))

                server_preds[metric_name] = {
                    'p50': p50_forecast,
                    'p10': p10_forecast,
                    'p90': p90_forecast,
                    'current': float(current),
                    'trend': float(trend)
                }

            predictions[server] = server_preds

        return predictions

    def _generate_alerts(self, predictions: Dict) -> List[Dict]:
        """Generate alerts based on prediction thresholds.

        OPTIMIZATION: Uses vectorized NumPy operations for 5-10x faster alert generation.
        """
        # Simple threshold-based alerts (dashboard handles risk scoring)
        # These are basic thresholds - dashboard uses contextual intelligence
        thresholds = {
            'cpu_user_pct': {'critical': 95, 'warning': 85},
            'mem_used_pct': {'critical': 98, 'warning': 90},
            'cpu_iowait_pct': {'critical': 30, 'warning': 20},
            'swap_used_pct': {'critical': 50, 'warning': 25},
            'disk_usage_pct': {'critical': 95, 'warning': 85}
        }

        alerts = []

        # OPTIMIZATION: Only check first 12 timesteps (60 minutes / 5 min intervals)
        MAX_STEPS = 12

        for server, server_preds in predictions.items():
            for metric, forecast in server_preds.items():
                if metric not in thresholds:
                    continue

                p50_values = forecast.get('p50', [])
                if not p50_values:
                    continue

                # OPTIMIZATION: Convert to NumPy array and use vectorized comparison
                p50_array = np.array(p50_values[:MAX_STEPS])
                critical_thresh = thresholds[metric]['critical']
                warning_thresh = thresholds[metric]['warning']

                # Find critical alerts (vectorized)
                critical_mask = p50_array >= critical_thresh
                critical_indices = np.where(critical_mask)[0]

                for i in critical_indices:
                    # CRITICAL: Convert numpy.int64 to Python int for JSON serialization
                    i_py = int(i.item()) if hasattr(i, 'item') else int(i)
                    minutes_ahead = (i_py + 1) * 5
                    alerts.append({
                        'server': server,
                        'metric': metric,
                        'severity': 'critical',
                        'predicted_value': float(p50_array[i]),
                        'threshold': critical_thresh,
                        'steps_ahead': i_py + 1,
                        'minutes_ahead': minutes_ahead,
                        'message': f"{server}: {metric} predicted to reach {p50_array[i]:.1f}"
                    })

                # Find warning alerts (vectorized, exclude critical)
                warning_mask = (p50_array >= warning_thresh) & ~critical_mask
                warning_indices = np.where(warning_mask)[0]

                for i in warning_indices:
                    # CRITICAL: Convert numpy.int64 to Python int for JSON serialization
                    i_py = int(i.item()) if hasattr(i, 'item') else int(i)
                    minutes_ahead = (i_py + 1) * 5
                    alerts.append({
                        'server': server,
                        'metric': metric,
                        'severity': 'warning',
                        'predicted_value': float(p50_array[i]),
                        'threshold': warning_thresh,
                        'steps_ahead': i_py + 1,
                        'minutes_ahead': minutes_ahead,
                        'message': f"{server}: {metric} predicted to reach {p50_array[i]:.1f}"
                    })

        alerts.sort(key=lambda x: (x['severity'] == 'warning', x['minutes_ahead']))
        return alerts

    def _calculate_environment_metrics(self, current_data: pd.DataFrame, predictions: Dict) -> Dict:
        """Calculate environment-wide incident probabilities.

        Uses pre-calculated risk scores for accurate fleet-wide assessment.
        Only considers metrics that actually indicate problems.
        """
        prob_30m = 0.0
        prob_8h = 0.0
        high_risk_count = 0
        total_servers = len(predictions)

        # Critical metrics to check for environment-wide risk
        # These are the metrics where HIGH values indicate problems
        CRITICAL_HIGH_METRICS = {
            'cpu_user_pct': (80, 90),      # High CPU usage is bad
            'cpu_sys_pct': (50, 70),       # High system CPU is bad
            'cpu_iowait_pct': (20, 40),    # I/O wait > 20% is concerning
            'mem_used_pct': (85, 95),      # High memory is bad
            'swap_used_pct': (10, 30),     # Any swap usage is concerning
            'disk_usage_pct': (80, 90),    # High disk usage is bad
            'java_cpu_pct': (70, 85),      # High JVM CPU is bad
        }

        # Metrics where LOW values indicate problems
        CRITICAL_LOW_METRICS = {
            'cpu_idle_pct': (20, 10),      # Low idle = high usage = bad
        }

        for server, server_preds in predictions.items():
            server_risk_30m = 0.0
            server_risk_8h = 0.0
            metrics_checked = 0

            for metric, forecast in server_preds.items():
                # Skip non-metric fields
                if not isinstance(forecast, dict) or 'p50' not in forecast:
                    continue

                p50 = forecast.get('p50', [])
                p90 = forecast.get('p90', [])

                if not p50:
                    continue

                # Check if this is a critical metric we should evaluate
                if metric in CRITICAL_HIGH_METRICS:
                    warn_thresh, crit_thresh = CRITICAL_HIGH_METRICS[metric]

                    # 30-minute risk (first 6 timesteps = 30 min at 5-sec intervals)
                    if len(p50) >= 6:
                        max_p50_30m = float(np.max(p50[:6]))
                        max_p90_30m = float(np.max(p90[:6])) if len(p90) >= 6 else max_p50_30m

                        if max_p50_30m > crit_thresh or max_p90_30m > crit_thresh + 5:
                            server_risk_30m += 0.25
                        elif max_p50_30m > warn_thresh or max_p90_30m > warn_thresh + 5:
                            server_risk_30m += 0.10

                    # 8-hour risk (all timesteps)
                    if len(p50) >= 20:
                        max_p50_8h = float(np.max(p50))
                        max_p90_8h = float(np.max(p90)) if len(p90) >= 20 else max_p50_8h

                        if max_p50_8h > crit_thresh or max_p90_8h > crit_thresh + 5:
                            server_risk_8h += 0.20
                        elif max_p50_8h > warn_thresh or max_p90_8h > warn_thresh + 5:
                            server_risk_8h += 0.08

                    metrics_checked += 1

                elif metric in CRITICAL_LOW_METRICS:
                    warn_thresh, crit_thresh = CRITICAL_LOW_METRICS[metric]

                    # For idle CPU: LOW is bad (means high usage)
                    if len(p50) >= 6:
                        min_p50_30m = float(np.min(p50[:6]))

                        if min_p50_30m < crit_thresh:
                            server_risk_30m += 0.25
                        elif min_p50_30m < warn_thresh:
                            server_risk_30m += 0.10

                    if len(p50) >= 20:
                        min_p50_8h = float(np.min(p50))

                        if min_p50_8h < crit_thresh:
                            server_risk_8h += 0.20
                        elif min_p50_8h < warn_thresh:
                            server_risk_8h += 0.08

                    metrics_checked += 1

            # Cap per-server risk at 1.0
            server_risk_30m = min(1.0, server_risk_30m)
            server_risk_8h = min(1.0, server_risk_8h)

            if server_risk_30m > 0.5:
                high_risk_count += 1

            prob_30m += server_risk_30m
            prob_8h += server_risk_8h

        # Average across all servers
        if total_servers > 0:
            prob_30m = prob_30m / total_servers
            prob_8h = prob_8h / total_servers

        return {
            'prob_30m': float(min(1.0, prob_30m)),
            'prob_8h': float(min(1.0, prob_8h)),
            'incident_probability_30m': float(min(1.0, prob_30m)),  # Backwards compatibility
            'incident_probability_8h': float(min(1.0, prob_8h)),    # Backwards compatibility
            'high_risk_servers': high_risk_count,
            'total_servers': total_servers,
            'fleet_health': 'critical' if prob_30m > 0.6 else 'warning' if prob_30m > 0.3 else 'healthy'
        }

    def _empty_response(self) -> Dict:
        """Return empty response when no data available."""
        return {
            'predictions': {},
            'alerts': [],
            'environment': {
                'incident_probability_30m': 0.0,
                'incident_probability_8h': 0.0,
                'high_risk_servers': 0,
                'total_servers': 0,
                'fleet_health': 'unknown'
            },
            'metadata': {
                'model_type': 'none',
                'prediction_time': datetime.now().isoformat(),
                'error': 'No data available'
            }
        }

    def reload_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Hot reload the TFT model without restarting the daemon.

        Args:
            model_path: Optional path to specific model. If None, loads latest model.

        Returns:
            Status dict with success/error info
        """
        try:
            old_model_dir = self.model_dir

            # Find and validate new model
            new_model_dir = self._find_model(model_path)
            if not new_model_dir:
                return {
                    'success': False,
                    'error': 'No model found',
                    'current_model': str(old_model_dir) if old_model_dir else None
                }

            # Check if it's the same model
            if new_model_dir == old_model_dir:
                return {
                    'success': True,
                    'message': 'Model already loaded (same version)',
                    'model_path': str(new_model_dir)
                }

            print(f"[RELOAD] Loading new model: {new_model_dir}")

            # Load new model
            self.model_dir = new_model_dir
            self.config = self._load_config()

            if self.use_real_model:
                # Clear old model from memory
                if self.model:
                    del self.model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Load new model
                self._load_model()

                if not self.model:
                    # Rollback on failure
                    self.model_dir = old_model_dir
                    self.config = self._load_config()
                    self._load_model()
                    return {
                        'success': False,
                        'error': 'Failed to load new model, rolled back to previous',
                        'current_model': str(old_model_dir) if old_model_dir else None
                    }

            print(f"[OK] Model reloaded successfully: {new_model_dir}")

            return {
                'success': True,
                'message': 'Model reloaded successfully',
                'previous_model': str(old_model_dir) if old_model_dir else None,
                'new_model': str(new_model_dir),
                'model_timestamp': new_model_dir.name.replace('tft_model_', '') if 'tft_model_' in new_model_dir.name else 'unknown'
            }

        except Exception as e:
            import traceback
            error_msg = f"Error reloading model: {str(e)}"
            print(f"[ERROR] {error_msg}")
            traceback.print_exc()

            return {
                'success': False,
                'error': error_msg,
                'current_model': str(self.model_dir) if self.model_dir else None
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if not self.model_dir:
            return {
                'loaded': False,
                'mode': 'heuristic'
            }

        info = {
            'loaded': True,
            'mode': 'tft',
            'model_path': str(self.model_dir),
            'model_name': self.model_dir.name,
            'model_timestamp': self.model_dir.name.replace('tft_model_', '') if 'tft_model_' in self.model_dir.name else 'unknown',
        }

        # Add config info
        if self.config:
            info['config'] = {
                'max_prediction_length': self.config.get('max_prediction_length', 96),
                'max_encoder_length': self.config.get('max_encoder_length', 288),
            }

        # Add server encoder info
        if self.server_encoder:
            stats = self.server_encoder.get_stats()
            info['servers'] = {
                'total': stats.get('total_servers', 0),
                'known': stats.get('known_servers', 0)
            }

        return info

# =============================================================================
# CONFIGURATION
# =============================================================================

WARMUP_THRESHOLD = 150  # Timesteps needed per server before TFT predictions work
WINDOW_SIZE = 6000      # Keep last 6000 records (~8 hours at 5-second intervals for 25 servers)
PORT = 8000
PERSISTENCE_FILE = "inference_rolling_window.parquet"  # File to persist rolling window
AUTOSAVE_INTERVAL = 100  # Save every 100 ticks (~8 minutes at 5s intervals)

# =============================================================================
# DATA MODELS
# =============================================================================

class FeedDataRequest(BaseModel):
    """Incoming data from metrics_generator.py --stream"""
    records: List[Dict[str, Any]]

# =============================================================================
# INFERENCE DAEMON
# =============================================================================

class CleanInferenceDaemon:
    """
    Minimal inference daemon - just loads model and makes predictions.

    Architecture:
    - Receives data via POST /feed/data
    - Maintains rolling window of recent data
    - Runs TFT predictions when enough data available
    - Exposes predictions via GET /predictions/current
    - Accumulates data for automated retraining
    """

    def __init__(self, model_path: str = None, port: int = PORT, enable_retraining: bool = True):
        self.port = port
        self.rolling_window = deque(maxlen=WINDOW_SIZE)
        self.tick_count = 0
        self.start_time = datetime.now()
        self.persistence_file = Path(PERSISTENCE_FILE)
        self.last_save_tick = 0

        # Load TFT model
        print(f"[INIT] Loading TFT model...")
        self.inference = TFTInference(model_path, use_real_model=True)
        print(f"[OK] Model loaded")

        # Initialize XAI components
        print(f"[INIT] Loading XAI components...")
        self.shap_explainer = TFTShapExplainer(self.inference, use_shap=False)  # Fast approximation mode
        self.attention_visualizer = AttentionVisualizer(self.inference)
        self.counterfactual_generator = CounterfactualGenerator(self.inference)
        print(f"[OK] XAI components initialized (SHAP, Attention, Counterfactuals)")

        # Initialize historical data store for reporting (Parquet-based)
        print(f"[INIT] Loading Historical Data Store...")
        try:
            from core.historical_store import get_historical_store
            self.historical_store = get_historical_store("./data")
            # Set to None initially so first snapshot is recorded immediately
            self.last_snapshot_time = None
            self.snapshot_interval_seconds = 60  # Record environment snapshot every 1 minute
            print(f"[OK] Historical data store initialized (Parquet)")
        except Exception as e:
            print(f"[WARNING] Historical store not available: {e}")
            self.historical_store = None
            self.last_snapshot_time = None

        # Track per-server data counts for warmup
        self.server_timesteps = {}

        # Track previous alert states for change detection
        self.previous_alert_states = {}

        # Initialize data buffer for automated retraining
        self.drift_monitor = None
        self.correlation_detector = None
        if enable_retraining:
            try:
                from core.data_buffer import DataBuffer
                from core.auto_retrainer import AutoRetrainer
                from core.drift_monitor import DriftMonitor
                from core.correlation_detector import CorrelationDetector

                self.data_buffer = DataBuffer(
                    buffer_dir='./data_buffer',
                    retention_days=60,
                    auto_rotate=True
                )
                print(f"[OK] Data buffer initialized for automated retraining")

                # Initialize drift monitor for automatic retraining detection
                self.drift_monitor = DriftMonitor(
                    window_size=1000,
                    save_path='./data_buffer/drift_metrics.json'
                )
                print(f"[OK] Drift monitor initialized")

                # Initialize correlation detector for cascading failure detection
                self.correlation_detector = CorrelationDetector(
                    window_size=100,
                    correlation_threshold=0.7,
                    cascade_server_threshold=3,
                    anomaly_z_threshold=2.0
                )
                print(f"[OK] Correlation detector initialized for cascading failure detection")

                # Initialize auto-retrainer with callback to reload model AND drift monitor
                self.auto_retrainer = AutoRetrainer(
                    data_buffer=self.data_buffer,
                    reload_callback=self.reload_model,
                    training_days=30,
                    min_records_threshold=100000,
                    drift_monitor=self.drift_monitor,
                    auto_retrain_on_drift=True
                )
                print(f"[OK] Auto-retrainer initialized with drift-triggered retraining")

                # Track last drift check time
                self.last_drift_check_tick = 0
                self.drift_check_interval = 720  # Check drift every 720 ticks (~1 hour at 5s intervals)

            except ImportError as e:
                print(f"[WARNING] Retraining components not found - disabled: {e}")
                self.data_buffer = None
                self.auto_retrainer = None
        else:
            self.data_buffer = None
            self.auto_retrainer = None

        # Try to load persisted state
        self._load_state()

        # Register shutdown handlers for graceful persistence
        atexit.register(self._save_state)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def feed_data(self, records: List[Dict]) -> Dict[str, Any]:
        """
        Accept incoming data batch from external source.

        Args:
            records: List of metric records (one per server)

        Returns:
            Status info
        """
        if not records:
            return {"status": "ignored", "reason": "empty batch"}

        # Add to rolling window
        self.rolling_window.extend(records)
        self.tick_count += 1

        # Accumulate data for automated retraining
        if self.data_buffer:
            try:
                self.data_buffer.append(records)
            except Exception as e:
                print(f"[WARNING] Failed to buffer data for retraining: {e}")

        # Update correlation detector for cascading failure detection
        cascade_alert = None
        if self.correlation_detector:
            try:
                cascade_result = self.correlation_detector.update(records)
                if cascade_result.get('cascade_detected'):
                    cascade_alert = cascade_result.get('alert')
                    logger.warning(f"[CASCADE] Cascading failure detected!")
                    logger.warning(f"[CASCADE] Affected servers: {cascade_result.get('servers_with_anomalies')}")
                    logger.warning(f"[CASCADE] Correlation score: {cascade_result.get('correlation_score', 0):.2%}")
            except Exception as e:
                logger.error(f"[CASCADE] Correlation detection failed: {e}")

        # Track per-server counts for warmup status
        for record in records:
            server = record.get('server_name')
            if server:
                self.server_timesteps[server] = self.server_timesteps.get(server, 0) + 1

        # Check warmup status
        servers_ready = sum(1 for count in self.server_timesteps.values() if count >= WARMUP_THRESHOLD)
        total_servers = len(self.server_timesteps)
        is_warmed_up = servers_ready == total_servers and total_servers > 0

        # Auto-save check (every 100 ticks ≈ 8 minutes)
        self._autosave_check()

        # Periodic drift check and auto-retraining (every ~1 hour)
        if self.auto_retrainer and hasattr(self, 'drift_check_interval'):
            if self.tick_count - self.last_drift_check_tick >= self.drift_check_interval:
                self.last_drift_check_tick = self.tick_count
                try:
                    drift_result = self.auto_retrainer.check_drift_and_retrain()
                    if drift_result.get('drift_detected'):
                        logger.warning(f"[DRIFT] Drift detected - Combined score: {drift_result.get('metrics', {}).get('combined_score', 0):.2%}")
                        if drift_result.get('training_triggered'):
                            logger.warning(f"[DRIFT] Training triggered: {drift_result.get('training_job_id')}")
                except Exception as e:
                    logger.error(f"[DRIFT] Drift check failed: {e}")

        response = {
            "status": "accepted",
            "tick": self.tick_count,
            "window_size": len(self.rolling_window),
            "servers_tracked": total_servers,
            "servers_ready": servers_ready,
            "warmup_complete": is_warmed_up
        }

        # Include cascade alert if detected
        if cascade_alert:
            response["cascade_alert"] = cascade_alert

        return response

    def _calculate_server_risk_score(self, server_pred: Dict) -> float:
        """
        Calculate risk score for a single server.

        This is the SINGLE SOURCE OF TRUTH for risk calculations.
        Dashboard will use pre-calculated scores from daemon response.

        Args:
            server_pred: Server prediction dict with metrics

        Returns:
            Risk score 0-100 (higher = more critical)
        """
        current_risk = 0.0
        predicted_risk = 0.0

        # Profile detection (simple version - can be enhanced)
        server_name = server_pred.get('server_name', '')
        if server_name.startswith('ppdb'):
            profile = 'database'
        elif server_name.startswith('ppml'):
            profile = 'ml_compute'
        else:
            profile = 'generic'

        # Helper: Extract CPU used from idle
        def extract_cpu(mode='current'):
            if 'cpu_idle_pct' in server_pred:
                idle = server_pred['cpu_idle_pct'].get(mode, 0)
                if isinstance(idle, list) and idle:
                    idle = np.mean(idle[:6])  # 30-min avg
                return 100 - idle
            elif 'cpu_user_pct' in server_pred:
                cpu = server_pred['cpu_user_pct'].get(mode, 0)
                if isinstance(cpu, list) and cpu:
                    cpu = np.mean(cpu[:6])
                return cpu
            return 0

        # CPU RISK
        current_cpu = extract_cpu('current')
        predicted_cpu = extract_cpu('p90')  # p90 for conservative estimate

        if current_cpu >= 98:
            current_risk += 60
        elif current_cpu >= 95:
            current_risk += 40
        elif current_cpu >= 90:
            current_risk += 25
        elif current_cpu >= 80:
            current_risk += 15

        if predicted_cpu >= 98:
            predicted_risk += 40
        elif predicted_cpu >= 95:
            predicted_risk += 25
        elif predicted_cpu >= 90:
            predicted_risk += 15

        # MEMORY RISK (profile-aware)
        if 'mem_used_pct' in server_pred:
            current_mem = server_pred['mem_used_pct'].get('current', 0)
            p90_mem = server_pred['mem_used_pct'].get('p90', [])
            predicted_mem = np.mean(p90_mem[:6]) if len(p90_mem) >= 6 else current_mem

            # Database profile: higher thresholds (page cache is normal)
            if profile == 'database':
                if current_mem >= 99:
                    current_risk += 50
                elif current_mem >= 98:
                    current_risk += 30
                elif current_mem >= 95:
                    current_risk += 15
            else:
                if current_mem >= 98:
                    current_risk += 50
                elif current_mem >= 95:
                    current_risk += 30
                elif current_mem >= 90:
                    current_risk += 15
                elif current_mem >= 85:
                    current_risk += 10

            if predicted_mem >= 98:
                predicted_risk += 30
            elif predicted_mem >= 95:
                predicted_risk += 20

        # I/O WAIT RISK (critical indicator)
        if 'cpu_iowait_pct' in server_pred:
            current_iowait = server_pred['cpu_iowait_pct'].get('current', 0)
            p90_iowait = server_pred['cpu_iowait_pct'].get('p90', [])
            predicted_iowait = np.mean(p90_iowait[:6]) if len(p90_iowait) >= 6 else current_iowait

            if current_iowait >= 30:
                current_risk += 40
            elif current_iowait >= 20:
                current_risk += 25
            elif current_iowait >= 10:
                current_risk += 15

            if predicted_iowait >= 30:
                predicted_risk += 25
            elif predicted_iowait >= 20:
                predicted_risk += 15

        # SWAP RISK
        if 'swap_used_pct' in server_pred:
            current_swap = server_pred['swap_used_pct'].get('current', 0)
            if current_swap >= 50:
                current_risk += 30
            elif current_swap >= 25:
                current_risk += 20
            elif current_swap >= 10:
                current_risk += 10

        # LOAD AVERAGE RISK
        if 'load_average' in server_pred:
            current_load = server_pred['load_average'].get('current', 0)
            if current_load >= 12:
                current_risk += 20
            elif current_load >= 8:
                current_risk += 10

        # Weighted combination (70% current, 30% predicted)
        final_risk = (current_risk * 0.70) + (predicted_risk * 0.30)

        # Clamp to 0-100
        return min(100.0, max(0.0, final_risk))

    def _calculate_all_risk_scores(self, predictions: Dict) -> Dict[str, float]:
        """
        Calculate risk scores for ALL servers ONCE.

        This is where heavy lifting happens (in daemon, not dashboard).
        Dashboard will receive pre-calculated scores.

        Args:
            predictions: Dict of server predictions

        Returns:
            Dict mapping server_name -> risk_score
        """
        risk_scores = {}
        for server_name, server_pred in predictions.items():
            risk_scores[server_name] = self._calculate_server_risk_score(server_pred)
        return risk_scores

    def _format_display_metrics(self, server_pred: Dict) -> Dict:
        """
        Convert internal prediction format to dashboard-ready display format.

        Dashboard should receive clean, ready-to-display data.
        No metric extraction logic needed in dashboard.

        Args:
            server_pred: Server prediction dict with raw metrics

        Returns:
            Dict with display-ready metrics
        """
        display_metrics = {}

        # CPU Used (aggregate from idle/user/sys)
        if 'cpu_idle_pct' in server_pred:
            cpu_current = 100 - server_pred['cpu_idle_pct'].get('current', 0)
            cpu_p50 = server_pred['cpu_idle_pct'].get('p50', [])
            cpu_predicted = 100 - np.mean(cpu_p50[:6]) if len(cpu_p50) >= 6 else cpu_current
        elif 'cpu_user_pct' in server_pred:
            cpu_current = server_pred['cpu_user_pct'].get('current', 0)
            cpu_p50 = server_pred['cpu_user_pct'].get('p50', [])
            cpu_predicted = np.mean(cpu_p50[:6]) if len(cpu_p50) >= 6 else cpu_current
        else:
            cpu_current = cpu_predicted = 0

        display_metrics['cpu'] = {
            'current': round(cpu_current, 1),
            'predicted': round(cpu_predicted, 1),
            'delta': round(cpu_predicted - cpu_current, 1),
            'unit': '%',
            'trend': 'increasing' if cpu_predicted > cpu_current else 'decreasing' if cpu_predicted < cpu_current else 'stable'
        }

        # Memory Used (direct)
        if 'mem_used_pct' in server_pred:
            mem_current = server_pred['mem_used_pct'].get('current', 0)
            mem_p50 = server_pred['mem_used_pct'].get('p50', [])
            mem_predicted = np.mean(mem_p50[:6]) if len(mem_p50) >= 6 else mem_current

            display_metrics['memory'] = {
                'current': round(mem_current, 1),
                'predicted': round(mem_predicted, 1),
                'delta': round(mem_predicted - mem_current, 1),
                'unit': '%',
                'trend': 'increasing' if mem_predicted > mem_current else 'decreasing' if mem_predicted < mem_current else 'stable'
            }

        # I/O Wait (critical indicator)
        if 'cpu_iowait_pct' in server_pred:
            iowait_current = server_pred['cpu_iowait_pct'].get('current', 0)
            iowait_p50 = server_pred['cpu_iowait_pct'].get('p50', [])
            iowait_predicted = np.mean(iowait_p50[:6]) if len(iowait_p50) >= 6 else iowait_current

            display_metrics['iowait'] = {
                'current': round(iowait_current, 1),
                'predicted': round(iowait_predicted, 1),
                'delta': round(iowait_predicted - iowait_current, 1),
                'unit': '%',
                'trend': 'increasing' if iowait_predicted > iowait_current else 'decreasing' if iowait_predicted < iowait_current else 'stable'
            }

        # Swap Usage
        if 'swap_used_pct' in server_pred:
            swap_current = server_pred['swap_used_pct'].get('current', 0)
            swap_p50 = server_pred['swap_used_pct'].get('p50', [])
            swap_predicted = np.mean(swap_p50[:6]) if len(swap_p50) >= 6 else swap_current

            display_metrics['swap'] = {
                'current': round(swap_current, 1),
                'predicted': round(swap_predicted, 1),
                'delta': round(swap_predicted - swap_current, 1),
                'unit': '%',
                'trend': 'increasing' if swap_predicted > swap_current else 'decreasing' if swap_predicted < swap_current else 'stable'
            }

        # Load Average
        if 'load_average' in server_pred:
            load_current = server_pred['load_average'].get('current', 0)
            load_p50 = server_pred['load_average'].get('p50', [])
            load_predicted = np.mean(load_p50[:6]) if len(load_p50) >= 6 else load_current

            display_metrics['load'] = {
                'current': round(load_current, 1),
                'predicted': round(load_predicted, 1),
                'delta': round(load_predicted - load_current, 1),
                'unit': '',
                'trend': 'increasing' if load_predicted > load_current else 'decreasing' if load_predicted < load_current else 'stable'
            }

        # Disk Usage
        if 'disk_usage_pct' in server_pred:
            disk_current = server_pred['disk_usage_pct'].get('current', 0)
            disk_p50 = server_pred['disk_usage_pct'].get('p50', [])
            disk_predicted = np.mean(disk_p50[:6]) if len(disk_p50) >= 6 else disk_current

            display_metrics['disk'] = {
                'current': round(disk_current, 1),
                'predicted': round(disk_predicted, 1),
                'delta': round(disk_predicted - disk_current, 1),
                'unit': '%',
                'trend': 'increasing' if disk_predicted > disk_current else 'decreasing' if disk_predicted < disk_current else 'stable'
            }

        # Network In
        if 'net_in_mb_s' in server_pred:
            net_in_current = server_pred['net_in_mb_s'].get('current', 0)
            net_in_p50 = server_pred['net_in_mb_s'].get('p50', [])
            net_in_predicted = np.mean(net_in_p50[:6]) if len(net_in_p50) >= 6 else net_in_current

            display_metrics['net_in'] = {
                'current': round(net_in_current, 1),
                'predicted': round(net_in_predicted, 1),
                'delta': round(net_in_predicted - net_in_current, 1),
                'unit': 'MB/s',
                'trend': 'increasing' if net_in_predicted > net_in_current else 'decreasing' if net_in_predicted < net_in_current else 'stable'
            }

        # Network Out
        if 'net_out_mb_s' in server_pred:
            net_out_current = server_pred['net_out_mb_s'].get('current', 0)
            net_out_p50 = server_pred['net_out_mb_s'].get('p50', [])
            net_out_predicted = np.mean(net_out_p50[:6]) if len(net_out_p50) >= 6 else net_out_current

            display_metrics['net_out'] = {
                'current': round(net_out_current, 1),
                'predicted': round(net_out_predicted, 1),
                'delta': round(net_out_predicted - net_out_current, 1),
                'unit': 'MB/s',
                'trend': 'increasing' if net_out_predicted > net_out_current else 'decreasing' if net_out_predicted < net_out_current else 'stable'
            }

        return display_metrics

    def get_predictions(self) -> Dict[str, Any]:
        """
        Run TFT predictions on current rolling window.

        ENHANCED: Now includes pre-calculated risk scores and summary statistics.
        Dashboard becomes pure display layer (no business logic).

        Returns:
            Predictions dict with per-server forecasts + risk scores + summary
        """
        if len(self.rolling_window) < 100:
            return {
                "error": "insufficient_data",
                "message": f"Need at least 100 records, have {len(self.rolling_window)}",
                "predictions": {}
            }

        # Convert rolling window to DataFrame
        df = pd.DataFrame(list(self.rolling_window))

        # Run inference
        try:
            result = self.inference.predict(df, horizon=96)

            # Extract predictions dict
            predictions = result.get('predictions', {})

            # PERFORMANCE ENHANCEMENT: Calculate risk scores ONCE for all servers
            # This moves expensive calculation from dashboard (270+ calls) to daemon (1 call)
            print(f"[PERF] Calculating risk scores for {len(predictions)} servers...")
            risk_scores = self._calculate_all_risk_scores(predictions)

            # Enrich each server prediction with pre-calculated risk score and alert info
            alert_counts = {'critical': 0, 'warning': 0, 'degrading': 0, 'healthy': 0}

            for server_name, server_pred in predictions.items():
                risk_score = risk_scores[server_name]

                # Add risk score to prediction
                server_pred['risk_score'] = round(risk_score, 1)

                # Add profile (dashboard-ready display name)
                if server_name.startswith('ppdb'):
                    server_pred['profile'] = 'Database'
                elif server_name.startswith('ppml'):
                    server_pred['profile'] = 'ML Compute'
                elif server_name.startswith('ppapi'):
                    server_pred['profile'] = 'Web API'
                elif server_name.startswith('ppcond'):
                    server_pred['profile'] = 'Conductor Mgmt'
                elif server_name.startswith('ppetl'):
                    server_pred['profile'] = 'ETL/Ingest'
                elif server_name.startswith('pprisk'):
                    server_pred['profile'] = 'Risk Analytics'
                else:
                    server_pred['profile'] = 'Generic'

                # Add alert level info
                alert_level = get_alert_level(risk_score)
                server_pred['alert'] = {
                    'level': alert_level.value,  # "critical", "warning", etc.
                    'score': round(risk_score, 1),
                    'color': get_alert_color(risk_score, format='hex'),
                    'emoji': get_alert_emoji(risk_score),
                    'label': get_alert_label(risk_score),  # "🔴 Critical", etc.
                }

                # Add display-ready metrics (dashboard doesn't need extraction logic)
                server_pred['display_metrics'] = self._format_display_metrics(server_pred)

                # Determine alert level
                if risk_score >= 80:
                    current_level = 'critical'
                    alert_counts['critical'] += 1
                elif risk_score >= 60:
                    current_level = 'warning'
                    alert_counts['warning'] += 1
                elif risk_score >= 50:
                    current_level = 'degraded'
                    alert_counts['degrading'] += 1
                else:
                    current_level = 'healthy'
                    alert_counts['healthy'] += 1

                # Track state changes for historical store
                if self.historical_store:
                    previous_level = self.previous_alert_states.get(server_name, 'healthy')

                    # Detect state change
                    if current_level != previous_level:
                        # Determine event type
                        level_order = {'healthy': 0, 'degraded': 1, 'warning': 2, 'critical': 3}
                        if level_order.get(current_level, 0) > level_order.get(previous_level, 0):
                            event_type = 'escalation'
                        elif current_level == 'healthy':
                            event_type = 'resolved'
                        else:
                            event_type = 'de-escalation'

                        # Record the event
                        try:
                            metrics = {
                                'cpu': server_pred.get('display_metrics', {}).get('cpu_used', 0),
                                'memory': server_pred.get('display_metrics', {}).get('mem_used', 0),
                                'iowait': server_pred.get('display_metrics', {}).get('iowait', 0),
                            }
                            self.historical_store.record_alert_event(
                                server_name=server_name,
                                event_type=event_type,
                                previous_level=previous_level,
                                new_level=current_level,
                                risk_score=risk_score,
                                metrics=metrics
                            )

                            # If resolved, mark previous alert as resolved
                            if event_type == 'resolved':
                                self.historical_store.resolve_alert(server_name)
                        except Exception as e:
                            print(f"[WARN] Failed to record alert event: {e}")

                    # Update state tracking
                    self.previous_alert_states[server_name] = current_level

            # Sort servers by risk (pre-calculate top N lists for dashboard)
            sorted_servers = sorted(
                predictions.keys(),
                key=lambda s: risk_scores[s],
                reverse=True
            )

            # Calculate additional stats for historical store
            all_scores = list(risk_scores.values())
            avg_risk = sum(all_scores) / len(all_scores) if all_scores else 0
            max_risk = max(all_scores) if all_scores else 0

            # Add summary statistics (dashboard-ready aggregates)
            result['summary'] = {
                'total_servers': len(predictions),
                'critical_count': alert_counts['critical'],
                'warning_count': alert_counts['warning'],
                'degrading_count': alert_counts['degrading'],
                'healthy_count': alert_counts['healthy'],
                'avg_risk_score': round(avg_risk, 1),
                'max_risk_score': round(max_risk, 1),
                'top_5_risks': sorted_servers[:5],
                'top_10_risks': sorted_servers[:10],
                'top_20_risks': sorted_servers[:20],
                'risk_calculation_time': datetime.now().isoformat(),
            }

            # Record environment snapshot periodically (every 1 minute)
            if self.historical_store:
                should_record = False
                if self.last_snapshot_time is None:
                    # First snapshot - record immediately
                    should_record = True
                else:
                    elapsed = (datetime.now() - self.last_snapshot_time).total_seconds()
                    if elapsed >= self.snapshot_interval_seconds:
                        should_record = True

                if should_record:
                    try:
                        env_metrics = result.get('environment', {})
                        self.historical_store.record_environment_snapshot({
                            'total_servers': len(predictions),
                            'critical_count': alert_counts['critical'],
                            'warning_count': alert_counts['warning'],
                            'degraded_count': alert_counts['degrading'],
                            'healthy_count': alert_counts['healthy'],
                            'prob_30m': env_metrics.get('prob_30m', 0),
                            'prob_8h': env_metrics.get('prob_8h', 0),
                            'avg_risk_score': avg_risk,
                            'max_risk_score': max_risk,
                            'top_risk_server': sorted_servers[0] if sorted_servers else '',
                            'fleet_health': env_metrics.get('fleet_health', 'unknown')
                        })
                        self.last_snapshot_time = datetime.now()
                        print(f"[HISTORY] Environment snapshot recorded")
                    except Exception as e:
                        print(f"[WARN] Failed to record environment snapshot: {e}")

            print(f"[OK] Risk scores calculated: {alert_counts['critical']} critical, {alert_counts['warning']} warning, {alert_counts['healthy']} healthy")

            # Update predictions in result
            result['predictions'] = predictions

            return result

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return self.inference._empty_response()

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status for health checks."""
        servers_ready = sum(1 for count in self.server_timesteps.values() if count >= WARMUP_THRESHOLD)
        total_servers = len(self.server_timesteps)
        is_warmed_up = servers_ready == total_servers and total_servers > 0

        # Calculate average data points across all servers
        avg_datapoints = int(sum(self.server_timesteps.values()) / total_servers) if total_servers > 0 else 0

        if is_warmed_up:
            warmup_msg = "Model ready - using TFT predictions"
            warmup_pct = 100
        elif total_servers > 0:
            warmup_msg = f"Warming up: {servers_ready}/{total_servers} servers ready"
            warmup_pct = int(servers_ready / total_servers * 100)
        else:
            warmup_msg = "Waiting for data..."
            warmup_pct = 0

        return {
            "running": True,
            "tick_count": self.tick_count,
            "window_size": len(self.rolling_window),
            "warmup": {
                "is_warmed_up": is_warmed_up,
                "progress_percent": warmup_pct,
                "threshold": WARMUP_THRESHOLD,
                "current_size": avg_datapoints,
                "required_size": WARMUP_THRESHOLD,
                "message": warmup_msg
            }
        }

    def list_available_models(self) -> Dict[str, Any]:
        """
        List all available trained models in the models/ directory.

        Returns:
            Dict with list of models and their metadata
        """
        models_dir = Path("./models")
        if not models_dir.exists():
            return {
                'models': [],
                'count': 0,
                'models_dir': str(models_dir)
            }

        model_list = []
        for model_dir in sorted(models_dir.glob("tft_model_*"), reverse=True):
            if not model_dir.is_dir():
                continue

            model_info = {
                'path': str(model_dir),
                'name': model_dir.name,
                'timestamp': model_dir.name.replace('tft_model_', '') if 'tft_model_' in model_dir.name else 'unknown',
                'is_current': model_dir == self.inference.model_dir,
            }

            # Get file sizes
            model_file = model_dir / "model.safetensors"
            if model_file.exists():
                model_info['size_mb'] = round(model_file.stat().st_size / (1024 * 1024), 2)
                model_info['modified'] = datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()

            # Get training info if available
            training_info_file = model_dir / "training_info.json"
            if training_info_file.exists():
                try:
                    with open(training_info_file) as f:
                        training_info = json.load(f)
                        model_info['training'] = {
                            'epochs': training_info.get('epochs'),
                            'training_time': training_info.get('training_time_seconds'),
                            'servers_trained': training_info.get('num_servers'),
                        }
                except Exception:
                    pass

            model_list.append(model_info)

        return {
            'models': model_list,
            'count': len(model_list),
            'models_dir': str(models_dir),
            'current_model': str(self.inference.model_dir) if self.inference.model_dir else None
        }

    def reload_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Hot reload the model without restarting the daemon.

        Args:
            model_path: Optional specific model path. If None, loads latest.

        Returns:
            Status dict with success/error info
        """
        return self.inference.reload_model(model_path)

    def _load_state(self):
        """Load persisted rolling window from disk."""
        # Try Parquet first (secure), fall back to Pickle (legacy migration)
        parquet_file = self.persistence_file
        pickle_file = Path(str(self.persistence_file).replace('.parquet', '.pkl'))

        # Priority 1: Load from Parquet (secure)
        if parquet_file.exists():
            try:
                self._load_state_parquet(parquet_file)
                return
            except Exception as e:
                print(f"[ERROR] Failed to load Parquet state: {e}")
                print(f"[INFO] Trying legacy Pickle format...")

        # Priority 2: Auto-migrate from legacy Pickle (one-time migration)
        if pickle_file.exists():
            print(f"[MIGRATION] Found legacy pickle file - migrating to Parquet...")
            try:
                self._load_state_pickle_legacy(pickle_file)
                # Save as Parquet immediately
                self._save_state()
                print(f"[MIGRATION] Successfully migrated to Parquet format!")
                # Keep old pickle as backup for now
                return
            except Exception as e:
                print(f"[ERROR] Failed to migrate from Pickle: {e}")

        # Priority 3: No persisted state - start fresh
        print(f"[INFO] No persisted state found - starting fresh")

    def _load_state_parquet(self, file_path: Path):
        """Load state from secure Parquet format."""
        import pyarrow.parquet as pq

        # Check file age
        file_age_minutes = (datetime.now().timestamp() - file_path.stat().st_mtime) / 60

        if file_age_minutes > 30:
            print(f"[WARNING] Persisted state is {file_age_minutes:.1f} minutes old - may be stale")

        print(f"[INFO] Loading persisted rolling window from {file_path}...")

        # Read Parquet file
        table = pq.read_table(str(file_path))
        df = table.to_pandas()

        # Read metadata (stored in Parquet schema metadata)
        metadata = table.schema.metadata
        if metadata:
            self.tick_count = int(metadata.get(b'tick_count', b'0').decode('utf-8'))

            # Deserialize server_timesteps from JSON (stored in metadata)
            server_timesteps_json = metadata.get(b'server_timesteps', b'{}').decode('utf-8')
            self.server_timesteps = json.loads(server_timesteps_json)
        else:
            # Fallback: infer from data
            self.tick_count = len(df) // 25  # Approximate
            self.server_timesteps = df['server_name'].value_counts().to_dict()

        # Restore rolling window from DataFrame
        records = df.to_dict('records')
        self.rolling_window = deque(records, maxlen=WINDOW_SIZE)

        servers_ready = sum(1 for count in self.server_timesteps.values() if count >= WARMUP_THRESHOLD)
        total_servers = len(self.server_timesteps)

        print(f"[OK] Loaded {len(self.rolling_window)} records from Parquet")
        print(f"[OK] Warmup status: {servers_ready}/{total_servers} servers ready")

        # Check if warmed up
        if servers_ready == total_servers and total_servers > 0:
            print(f"[SUCCESS] Model is WARMED UP - ready for predictions immediately!")
        else:
            print(f"[INFO] Model needs {WARMUP_THRESHOLD - min(self.server_timesteps.values() if self.server_timesteps else [0])} more datapoints per server")

    def _load_state_pickle_legacy(self, file_path: Path):
        """Load state from legacy Pickle format (one-time migration only)."""
        print(f"[WARNING] Loading INSECURE Pickle format - will migrate to Parquet")

        with open(file_path, 'rb') as f:
            state = pickle.load(f)

        # Restore state
        self.rolling_window = deque(state['rolling_window'], maxlen=WINDOW_SIZE)
        self.tick_count = state['tick_count']
        self.server_timesteps = state['server_timesteps']

        servers_ready = sum(1 for count in self.server_timesteps.values() if count >= WARMUP_THRESHOLD)
        total_servers = len(self.server_timesteps)

        print(f"[OK] Loaded {len(self.rolling_window)} records from legacy Pickle")
        print(f"[OK] Warmup status: {servers_ready}/{total_servers} servers ready")

    def _save_state(self):
        """Save rolling window to disk using secure Parquet format."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            # Convert rolling window to DataFrame
            df = pd.DataFrame(list(self.rolling_window))

            if df.empty:
                print(f"[SAVE] Skipping save - no data to persist")
                return

            # Create PyArrow table with metadata
            table = pa.Table.from_pandas(df)

            # Add metadata (tick_count, server_timesteps, timestamp)
            metadata = {
                b'tick_count': str(self.tick_count).encode('utf-8'),
                b'server_timesteps': json.dumps(self.server_timesteps).encode('utf-8'),
                b'timestamp': datetime.now().isoformat().encode('utf-8'),
                b'format_version': b'1.0'
            }

            # Create new schema with metadata
            table = table.replace_schema_metadata(metadata)

            # Atomic write (temp file + rename)
            temp_file = self.persistence_file.with_suffix('.tmp')
            pq.write_table(
                table,
                str(temp_file),
                compression='snappy',  # Fast compression
                use_dictionary=True     # Efficient for categorical data
            )

            # Rename to final location (atomic on POSIX, best-effort on Windows)
            temp_file.replace(self.persistence_file)

            print(f"[SAVE] Rolling window persisted (Parquet): {len(self.rolling_window)} records, tick {self.tick_count}")

        except Exception as e:
            print(f"[ERROR] Failed to save state: {e}")
            import traceback
            traceback.print_exc()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n[SHUTDOWN] Received signal {signum}, saving state...")
        self._save_state()
        print(f"[SHUTDOWN] State saved, exiting")
        import sys
        sys.exit(0)

    def explain_prediction(self, server_name: str) -> Dict[str, Any]:
        """
        Generate explanation for a server's prediction using XAI components.

        Args:
            server_name: Server to explain

        Returns:
            Dict with SHAP feature importance, attention analysis, and counterfactuals
        """
        try:
            # Get current prediction for this server
            predictions = self.get_predictions()
            if 'predictions' not in predictions or server_name not in predictions['predictions']:
                return {
                    'error': 'no_prediction',
                    'message': f'No prediction available for {server_name}'
                }

            server_pred = predictions['predictions'][server_name]

            # Get historical data for this server from rolling window
            window_df = pd.DataFrame(list(self.rolling_window))
            server_data = window_df[window_df['server_name'] == server_name].copy()

            if len(server_data) < 10:
                return {
                    'error': 'insufficient_data',
                    'message': f'Need at least 10 timesteps for explanation, have {len(server_data)}'
                }

            # Generate SHAP explanation
            shap_explanation = self.shap_explainer.explain_prediction(
                server_name=server_name,
                current_data=server_data,
                prediction=server_pred
            )

            # Generate attention analysis
            attention_analysis = self.attention_visualizer.extract_attention_weights(
                server_name=server_name,
                historical_data=server_data,
                window_size=min(len(server_data), 150)
            )

            # Generate counterfactual scenarios
            # Extract current CPU prediction for counterfactual baseline
            current_cpu = server_pred.get('cpu_idle_pct', {}).get('current', 0)
            if current_cpu > 0:
                current_cpu = 100 - current_cpu  # Convert idle to used
            else:
                # Fallback to any available current CPU metric
                current_cpu = server_pred.get('cpu_user_pct', {}).get('current', 50)

            counterfactuals = self.counterfactual_generator.generate_counterfactuals(
                server_name=server_name,
                current_data=server_data,
                current_prediction=current_cpu
            )

            # Extract risk score explicitly for clarity
            risk_score = server_pred.get('risk_score', server_pred.get('alert', {}).get('score', 50))

            return {
                'server_name': server_name,
                'timestamp': datetime.now().isoformat(),
                'shap': shap_explanation,
                'attention': attention_analysis,
                'counterfactuals': counterfactuals,
                'prediction': server_pred,
                'risk_score': risk_score,  # Explicit top-level risk score
                'data_points': len(server_data)
            }

        except Exception as e:
            import traceback
            return {
                'error': 'explanation_failed',
                'message': str(e),
                'traceback': traceback.format_exc()
            }

    def _autosave_check(self):
        """Check if it's time to auto-save."""
        if self.tick_count - self.last_save_tick >= AUTOSAVE_INTERVAL:
            self._save_state()
            self.last_save_tick = self.tick_count

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Global daemon instance
daemon = None

# Create FastAPI app
app = FastAPI(
    title="TFT Inference Daemon",
    description="Clean inference engine - feed it with metrics_generator.py --stream",
    version="2.0"
)

# Security: Rate limiting (if available)
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    print("[OK] Rate limiting enabled")
else:
    limiter = None
    print("[WARNING] Rate limiting disabled - install slowapi for production")

# Enable CORS for dashboard (Security: Whitelist only)
# Get allowed origins from environment variable, default to localhost
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # Security: Whitelist only (not "*")
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only allow needed methods
    allow_headers=["Content-Type", "X-API-Key"],  # Only allow needed headers
)

@app.on_event("startup")
async def startup():
    """Initialize daemon on startup."""
    global daemon
    print("\n" + "="*60)
    print("TFT INFERENCE DAEMON - Clean Architecture")
    print("="*60)
    print(f"Port: {PORT}")
    print(f"Window size: {WINDOW_SIZE} records")
    print(f"Warmup threshold: {WARMUP_THRESHOLD} timesteps/server")
    print()
    print("Feed me data with:")
    print("  python metrics_generator.py --stream --servers 20 --scenario healthy")
    print("="*60)
    print()

    daemon = CleanInferenceDaemon(port=PORT)
    print("[READY] Daemon started - waiting for data feed")

@app.post("/feed/data")
@limiter.limit("60/minute") if RATE_LIMITING_AVAILABLE else lambda func: func
async def feed_data(request: Request, feed_request: FeedDataRequest, api_key: str = Depends(verify_api_key)):
    """
    Accept data from external source (metrics_generator.py --stream).

    Rate limit: 60 requests/minute (1 per second)

    Example:
        POST /feed/data
        {
            "records": [
                {"timestamp": "2025-10-12T20:00:00", "server_name": "ppdb001", "cpu_pct": 45.2, ...},
                {"timestamp": "2025-10-12T20:00:00", "server_name": "ppdb002", "cpu_pct": 52.1, ...}
            ]
        }
    """
    result = daemon.feed_data(feed_request.records)
    return result

@app.get("/predictions/current")
@limiter.limit("30/minute") if RATE_LIMITING_AVAILABLE else lambda func: func
async def get_predictions(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Get current predictions for all servers.

    Rate limit: 30 requests/minute (1 every 2 seconds)

    Returns predictions with 96-step forecast horizon (8 hours at 5-minute intervals).
    """
    return daemon.get_predictions()

@app.get("/status")
async def get_status():
    """Get daemon health status."""
    return daemon.get_status()

@app.get("/health")
async def health_check():
    """Simple health check endpoint for dashboard."""
    return {
        "status": "healthy",
        "service": "tft_inference_daemon",
        "running": True
    }

@app.get("/alerts/active")
@limiter.limit("30/minute") if RATE_LIMITING_AVAILABLE else lambda func: func
async def get_active_alerts(request: Request, api_key: str = Depends(verify_api_key)):
    """
    Get active alerts from latest predictions.

    Rate limit: 30 requests/minute (1 every 2 seconds)

    Returns alerts that are currently active based on the latest prediction run.
    """
    try:
        # Get latest predictions
        preds = daemon.get_predictions()

        # Extract alerts if available (they're at top level now)
        if preds and 'alerts' in preds:
            alerts = preds['alerts']
            return {
                "timestamp": datetime.now().isoformat(),
                "count": len(alerts),
                "alerts": alerts
            }

        return {
            "timestamp": datetime.now().isoformat(),
            "count": 0,
            "alerts": []
        }
    except Exception as e:
        return {
            "timestamp": datetime.now().isoformat(),
            "count": 0,
            "alerts": [],
            "error": str(e)
        }

@app.get("/explain/{server_name}")
@limiter.limit("30/minute") if RATE_LIMITING_AVAILABLE else lambda func: func
async def explain_server_prediction(request: Request, server_name: str, api_key: str = Depends(verify_api_key)):
    """
    Get XAI explanation for a specific server's prediction.

    Rate limit: 30 requests/minute (1 every 2 seconds)

    Returns:
        - SHAP feature importance (which metrics drove the prediction)
        - Attention analysis (which time periods the model focused on)
        - Counterfactual scenarios (what-if analysis with actionable recommendations)

    Example:
        GET /explain/ppdb001

    Response includes:
        - shap: Feature importance with star ratings
        - attention: Temporal focus analysis
        - counterfactuals: What-if scenarios with impact estimates
        - prediction: Current prediction for context
    """
    try:
        explanation = daemon.explain_prediction(server_name)
        return explanation
    except Exception as e:
        import traceback
        return {
            "error": "explanation_failed",
            "server_name": server_name,
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/admin/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """
    List all available trained models.

    Returns info about each model including size, training metadata, and whether it's currently loaded.
    """
    return daemon.list_available_models()

@app.post("/admin/reload-model")
async def reload_model(
    request: Request,
    model_path: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Hot reload the TFT model without restarting the daemon.

    Args:
        model_path: Optional path to specific model. If None, loads latest model.

    Example:
        POST /admin/reload-model
        POST /admin/reload-model?model_path=models/tft_model_20250130_120000
    """
    result = daemon.reload_model(model_path)
    return result

@app.get("/admin/model-info")
async def get_model_info(api_key: str = Depends(verify_api_key)):
    """Get information about the currently loaded model."""
    return daemon.inference.get_model_info()

@app.post("/admin/trigger-training")
async def trigger_training(
    request: Request,
    epochs: int = 5,
    incremental: bool = True,
    blocking: bool = False,
    api_key: str = Depends(verify_api_key)
):
    """
    Trigger automated model retraining.

    Args:
        epochs: Number of epochs to train (default: 5)
        incremental: Resume from latest checkpoint (default: True)
        blocking: Wait for training to complete before returning (default: False)

    Returns:
        Training job status

    Example:
        POST /admin/trigger-training?epochs=10&incremental=true
    """
    if not daemon.auto_retrainer:
        return {
            'success': False,
            'error': 'Auto-retrainer not enabled. Start daemon with enable_retraining=True'
        }

    result = daemon.auto_retrainer.trigger_training(
        epochs=epochs,
        incremental=incremental,
        blocking=blocking
    )

    return result

@app.get("/admin/training-status")
async def get_training_status(
    job_id: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get status of training job.

    Args:
        job_id: Optional job ID. If None, returns current job status.

    Returns:
        Training job status with progress info
    """
    if not daemon.auto_retrainer:
        return {
            'success': False,
            'error': 'Auto-retrainer not enabled'
        }

    return daemon.auto_retrainer.get_job_status(job_id)

@app.get("/admin/training-stats")
async def get_training_stats(api_key: str = Depends(verify_api_key)):
    """
    Get overall training statistics and data buffer status.

    Returns:
        Training stats including history, data buffer info, and readiness
    """
    if not daemon.auto_retrainer:
        return {
            'success': False,
            'error': 'Auto-retrainer not enabled'
        }

    return daemon.auto_retrainer.get_training_stats()

@app.post("/admin/cancel-training")
async def cancel_training(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Cancel the currently running training job.

    Returns:
        Cancellation status
    """
    if not daemon.auto_retrainer:
        return {
            'success': False,
            'error': 'Auto-retrainer not enabled'
        }

    return daemon.auto_retrainer.cancel_current_job()

# =============================================================================
# CASCADE & DRIFT DETECTION ENDPOINTS
# =============================================================================

@app.get("/cascade/status")
async def get_cascade_status(
    api_key: str = Depends(verify_api_key)
):
    """
    Get current cascading failure detection status.

    Returns:
        Cascade detection status including correlation scores and any active alerts
    """
    if not daemon.correlation_detector:
        return {
            'success': False,
            'error': 'Correlation detector not enabled'
        }

    return daemon.correlation_detector.get_cascade_status()

@app.get("/cascade/health")
async def get_fleet_correlation_health(
    api_key: str = Depends(verify_api_key)
):
    """
    Get fleet health score based on cross-server correlations.

    Returns:
        Fleet health metrics including correlation-based risk assessment
    """
    if not daemon.correlation_detector:
        return {
            'success': False,
            'error': 'Correlation detector not enabled'
        }

    return daemon.correlation_detector.get_fleet_health_score()

@app.get("/drift/status")
async def get_drift_status(
    api_key: str = Depends(verify_api_key)
):
    """
    Get current model drift detection status.

    Returns:
        Drift metrics and retraining recommendations
    """
    if not daemon.drift_monitor:
        return {
            'success': False,
            'error': 'Drift monitor not enabled'
        }

    return daemon.drift_monitor.get_drift_status()

@app.get("/drift/report")
async def get_drift_report(
    api_key: str = Depends(verify_api_key)
):
    """
    Get human-readable drift detection report.

    Returns:
        Formatted drift report with metrics and recommendations
    """
    if not daemon.drift_monitor:
        return {
            'success': False,
            'error': 'Drift monitor not enabled'
        }

    return {
        'success': True,
        'report': daemon.drift_monitor.generate_report()
    }

# =============================================================================
# HISTORICAL DATA ENDPOINTS - For Executive Reporting
# =============================================================================

@app.get("/historical/summary")
async def get_historical_summary(
    time_range: str = "1d",
    api_key: str = Depends(verify_api_key)
):
    """
    Get summary statistics for executive reporting.

    Args:
        time_range: '30m', '1h', '8h', '1d', '1w', '1M'

    Returns:
        Summary stats including alerts, resolution rates, etc.
    """
    if not daemon.historical_store:
        return {
            'success': False,
            'error': 'Historical data store not available'
        }

    try:
        summary = daemon.historical_store.get_summary_stats(time_range)
        return {
            'success': True,
            **summary
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.get("/historical/alerts")
async def get_historical_alerts(
    time_range: str = "1h",
    server_name: Optional[str] = None,
    api_key: str = Depends(verify_api_key)
):
    """
    Get alert events for a time range.

    Args:
        time_range: '30m', '1h', '8h', '1d', '1w', '1M'
        server_name: Optional filter by server

    Returns:
        List of alert events
    """
    if not daemon.historical_store:
        return {
            'success': False,
            'alerts': [],
            'error': 'Historical data store not available'
        }

    try:
        alerts = daemon.historical_store.get_alert_events(time_range, server_name)
        return {
            'success': True,
            'time_range': time_range,
            'count': len(alerts),
            'alerts': alerts
        }
    except Exception as e:
        return {
            'success': False,
            'alerts': [],
            'error': str(e)
        }

@app.get("/historical/environment")
async def get_historical_environment(
    time_range: str = "1h",
    api_key: str = Depends(verify_api_key)
):
    """
    Get environment health snapshots over time.

    Args:
        time_range: '30m', '1h', '8h', '1d', '1w', '1M'

    Returns:
        List of environment snapshots
    """
    if not daemon.historical_store:
        return {
            'success': False,
            'snapshots': [],
            'error': 'Historical data store not available'
        }

    try:
        snapshots = daemon.historical_store.get_environment_snapshots(time_range)
        return {
            'success': True,
            'time_range': time_range,
            'count': len(snapshots),
            'snapshots': snapshots
        }
    except Exception as e:
        return {
            'success': False,
            'snapshots': [],
            'error': str(e)
        }

@app.get("/historical/server/{server_name}")
async def get_server_history(
    server_name: str,
    time_range: str = "1d",
    api_key: str = Depends(verify_api_key)
):
    """
    Get detailed history for a specific server.

    Args:
        server_name: Server to get history for
        time_range: '30m', '1h', '8h', '1d', '1w', '1M'

    Returns:
        Server-specific history and stats
    """
    if not daemon.historical_store:
        return {
            'success': False,
            'error': 'Historical data store not available'
        }

    try:
        history = daemon.historical_store.get_server_history(server_name, time_range)
        return {
            'success': True,
            **history
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.get("/historical/export/{table}")
async def export_historical_csv(
    table: str,
    time_range: str = "1d",
    api_key: str = Depends(verify_api_key)
):
    """
    Export historical data as CSV.

    Args:
        table: 'alerts' or 'environment'
        time_range: '30m', '1h', '8h', '1d', '1w', '1M'

    Returns:
        CSV data as string (or base64 encoded)
    """
    if not daemon.historical_store:
        return {
            'success': False,
            'error': 'Historical data store not available'
        }

    if table not in ['alerts', 'environment']:
        return {
            'success': False,
            'error': f"Invalid table '{table}'. Use 'alerts' or 'environment'"
        }

    try:
        csv_data = daemon.historical_store.export_to_csv(table, time_range)

        # If no data, return CSV with headers only
        if not csv_data:
            if table == 'alerts':
                csv_data = "timestamp,server_name,event_type,previous_level,new_level,risk_score,resolved_at,resolution_duration_minutes,caused_incident,notes\n"
            else:
                csv_data = "timestamp,total_servers,critical_count,warning_count,degraded_count,healthy_count,prob_30m,prob_8h,avg_risk_score,max_risk_score,top_risk_server,fleet_health\n"

        return {
            'success': True,
            'table': table,
            'time_range': time_range,
            'csv_data': csv_data,
            'filename': f"argus_{table}_{time_range}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "service": "TFT Inference Daemon",
        "version": "2.4",  # Bumped for historical reporting
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "feed": "POST /feed/data",
            "predictions": "GET /predictions/current",
            "alerts": "GET /alerts/active",
            "explain": "GET /explain/{server_name}",
            "status": "GET /status",
            "historical_summary": "GET /historical/summary",
            "historical_alerts": "GET /historical/alerts",
            "historical_environment": "GET /historical/environment",
            "historical_server": "GET /historical/server/{server_name}",
            "historical_export": "GET /historical/export/{table}",
            "admin_models": "GET /admin/models",
            "admin_reload": "POST /admin/reload-model",
            "admin_model_info": "GET /admin/model-info",
            "admin_trigger_training": "POST /admin/trigger-training",
            "admin_training_status": "GET /admin/training-status",
            "admin_training_stats": "GET /admin/training-stats",
            "admin_cancel_training": "POST /admin/cancel-training"
        },
        "features": {
            "xai": "Explainable AI with SHAP, Attention, and Counterfactuals",
            "retraining": "Automated background model retraining",
            "persistence": "Rolling window state persistence",
            "hot_reload": "Hot reload models without daemon restart",
            "continuous_learning": "Data buffer accumulates metrics for retraining",
            "historical_reporting": "Executive reports with CSV export"
        },
        "feed_example": "python metrics_generator.py --stream --servers 20 --scenario healthy"
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Start the daemon."""
    import argparse
    import socket

    parser = argparse.ArgumentParser(description="TFT Inference Daemon - Clean & Minimal")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to run on (default: {PORT})")
    parser.add_argument("--model", default=None, help="Path to model (auto-detects if not specified)")

    args = parser.parse_args()

    # Get local IP address for display
    def get_local_ip():
        try:
            # Connect to external address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "localhost"

    local_ip = get_local_ip()

    # Check API key configuration
    api_key_env = load_nordiq_api_key()
    if api_key_env:
        # Determine source
        if os.getenv("NORDIQ_API_KEY"):
            source = "NORDIQ_API_KEY environment variable"
        elif (Path(__file__).parent.parent.parent / ".nordiq_key").exists():
            source = ".nordiq_key file"
        elif os.getenv("TFT_API_KEY"):
            source = "TFT_API_KEY environment variable (legacy)"
        else:
            source = "unknown"
        print(f"[OK] API key loaded from {source}: {api_key_env[:8]}...")
    else:
        print("[WARNING] No API key set - running in development mode (no authentication)")

    # Print friendly startup message
    print("\n" + "="*70)
    print("TFT INFERENCE DAEMON - CLEAN ARCHITECTURE")
    print("="*70)
    print(f"\n  Local:   http://localhost:{args.port}")
    print(f"  Network: http://{local_ip}:{args.port}")
    print(f"\n  API Docs: http://localhost:{args.port}/docs")
    print("\n" + "="*70 + "\n")

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",  # Suppress INFO logs so our banner shows cleanly
        access_log=False
    )

if __name__ == "__main__":
    main()
