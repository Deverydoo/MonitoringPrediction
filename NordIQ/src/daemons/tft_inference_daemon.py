#!/usr/bin/env python3
"""
NordIQ AI Systems - TFT Inference Daemon
Nordic precision, AI intelligence

Copyright (c) 2025 NordIQ AI, LLC. All rights reserved.
Developed by Craig Giannelli

This software is licensed under the Business Source License 1.1.
See LICENSE file for details.

TFT Inference Daemon - Clean & Minimal (Standalone)

Pure inference engine that:
1. Loads TFT model
2. Accepts data via REST API
3. Runs predictions on rolling window
4. Exposes predictions via REST API

No data generation, no scenarios, no complexity.
Feed it with: python metrics_generator_daemon.py --stream

This file is completely standalone - includes TFTInference class directly.
No dependencies on the messy old tft_inference.py file.
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
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

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

def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for protected endpoints."""
    expected_key = os.getenv("TFT_API_KEY")

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
        else:
            self.gpu = None
            self.device = torch.device('cpu')

        self.use_real_model = use_real_model
        self.model_dir = self._find_model(model_path)
        self.config = self._load_config()
        self.model = None
        self.training_data = None
        self.server_encoder = None

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
            print(f"[OK] Server mapping loaded: {self.server_encoder.get_stats()['total_servers']} servers")

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

            print(f"[DEBUG] Input data: {len(df)} records, {df['server_name'].nunique() if 'server_name' in df.columns else 'N/A'} unique servers")
            print(f"[DEBUG] Prepared data: {len(prediction_df)} records, {prediction_df['server_id'].nunique()} unique server_ids")

            from pytorch_forecasting import TimeSeriesDataSet

            prediction_dataset = TimeSeriesDataSet.from_dataset(
                self.training_data,
                prediction_df,
                predict=True,
                stop_randomization=True
            )

            print(f"[DEBUG] Prediction dataset created with {len(prediction_dataset)} samples")

            batch_size = self.gpu.get_batch_size('inference') if self.gpu else 64
            # IMPORTANT: num_workers=0 on Windows to avoid multiprocessing issues
            num_workers = 0

            prediction_dataloader = prediction_dataset.to_dataloader(
                train=False,
                batch_size=batch_size,
                num_workers=num_workers
            )

            print(f"[DEBUG] Running TFT model prediction...")
            print(f"[DEBUG] Batch size: {batch_size}, Dataloader batches: {len(prediction_dataloader)}")
            with torch.no_grad():
                raw_predictions = self.model.predict(
                    prediction_dataloader,
                    mode="raw",
                    return_x=True
                )
            print(f"[DEBUG] TFT model prediction complete")
            print(f"[DEBUG] raw_predictions type after predict: {type(raw_predictions)}")

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

    def _prepare_data_for_tft(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare input data for TFT prediction with NordIQ Metrics Framework metrics."""
        prediction_df = df.copy()

        # NordIQ Metrics Framework metrics should be passed directly - no mapping needed
        # Data from metrics_generator.py already has correct column names
        required_cols = ['server_name', 'timestamp'] + [
            'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct', 'cpu_idle_pct', 'java_cpu_pct',
            'mem_used_pct', 'swap_used_pct', 'disk_usage_pct',
            'net_in_mb_s', 'net_out_mb_s',
            'back_close_wait', 'front_close_wait',
            'load_average', 'uptime_days'
        ]

        missing = [col for col in required_cols if col not in prediction_df.columns]
        if missing:
            raise ValueError(f"Missing required NordIQ Metrics Framework metrics: {missing}")

        # Convert server_name to server_id
        if 'server_name' in prediction_df.columns and self.server_encoder:
            prediction_df['server_id'] = prediction_df['server_name'].apply(
                self.server_encoder.encode
            )
        elif 'server_name' in prediction_df.columns:
            prediction_df['server_id'] = prediction_df['server_name']

        # Create time_idx
        prediction_df = prediction_df.sort_values(['server_id', 'timestamp'])
        prediction_df['time_idx'] = prediction_df.groupby('server_id').cumcount()

        # Ensure time features
        if 'hour' not in prediction_df.columns:
            prediction_df['timestamp'] = pd.to_datetime(prediction_df['timestamp'])
            prediction_df['hour'] = prediction_df['timestamp'].dt.hour
            prediction_df['day_of_week'] = prediction_df['timestamp'].dt.dayofweek
            prediction_df['month'] = prediction_df['timestamp'].dt.month
            prediction_df['is_weekend'] = (prediction_df['day_of_week'] >= 5).astype(int)

        if 'status' not in prediction_df.columns:
            prediction_df['status'] = 'healthy'

        # Infer profile from server_name
        def get_profile(server_name):
            if server_name.startswith('ppml'): return 'ml_compute'
            if server_name.startswith('ppdb'): return 'database'
            if server_name.startswith('ppweb'): return 'web_api'
            if server_name.startswith('ppcon'): return 'conductor_mgmt'
            if server_name.startswith('ppetl'): return 'data_ingest'
            if server_name.startswith('pprisk'): return 'risk_analytics'
            return 'generic'

        if 'profile' not in prediction_df.columns:
            prediction_df['profile'] = prediction_df['server_name'].apply(get_profile)

        return prediction_df

    def _format_tft_predictions(self, raw_predictions, input_df: pd.DataFrame, horizon: int) -> Dict:
        """Format TFT raw predictions into standard format."""
        predictions = {}

        # TFT model.predict() with return_x=True returns a tuple: (predictions, x)
        # where predictions is a dictionary with keys like 'prediction', 'attention', etc.
        # We need to extract just the prediction tensor

        print(f"[DEBUG] raw_predictions type: {type(raw_predictions)}")

        # Check if it's a Prediction object (has 'output' attribute) FIRST
        if hasattr(raw_predictions, 'output'):
            pred_tensor = raw_predictions.output
            print(f"[DEBUG] Extracted .output from Prediction object")
        elif hasattr(raw_predictions, 'prediction'):
            pred_tensor = raw_predictions.prediction
            print(f"[DEBUG] Extracted .prediction attribute")
        elif isinstance(raw_predictions, dict) and 'prediction' in raw_predictions:
            pred_tensor = raw_predictions['prediction']
            print(f"[DEBUG] Extracted from dict['prediction']")
        elif isinstance(raw_predictions, tuple):
            # Fallback: unpack tuple if needed
            pred_output, x_data = raw_predictions
            if isinstance(pred_output, dict) and 'prediction' in pred_output:
                pred_tensor = pred_output['prediction']
            else:
                pred_tensor = pred_output
            print(f"[DEBUG] Unpacked from tuple")
        else:
            pred_tensor = raw_predictions
            print(f"[DEBUG] Using raw_predictions directly")

        servers = input_df['server_id'].unique()

        print(f"[DEBUG] Formatting predictions: {len(servers)} servers")
        print(f"[DEBUG] pred_tensor type: {type(pred_tensor)}")
        print(f"[DEBUG] pred_tensor has .shape: {hasattr(pred_tensor, 'shape')}")
        if hasattr(pred_tensor, 'shape'):
            print(f"[DEBUG] pred_tensor.shape: {pred_tensor.shape}")
        print(f"[DEBUG] pred_tensor has __len__: {hasattr(pred_tensor, '__len__')}")
        if hasattr(pred_tensor, '__len__'):
            print(f"[DEBUG] len(pred_tensor): {len(pred_tensor)}")
        print(f"[DEBUG] pred_tensor has __getitem__: {hasattr(pred_tensor, '__getitem__')}")

        # Introspect Output object attributes
        if hasattr(pred_tensor, '__dict__'):
            print(f"[DEBUG] pred_tensor.__dict__ keys: {list(pred_tensor.__dict__.keys())}")
        else:
            print(f"[DEBUG] pred_tensor dir(): {[x for x in dir(pred_tensor) if not x.startswith('_')]}")

        # Try to access the actual prediction data
        if hasattr(pred_tensor, 'prediction'):
            print(f"[DEBUG] pred_tensor.prediction exists, type: {type(pred_tensor.prediction)}")
            if hasattr(pred_tensor.prediction, 'shape'):
                print(f"[DEBUG] pred_tensor.prediction.shape: {pred_tensor.prediction.shape}")

        # Check if it's a namedtuple or similar
        if hasattr(pred_tensor, '_fields'):
            print(f"[DEBUG] pred_tensor is namedtuple with fields: {pred_tensor._fields}")

        # CRITICAL FIX: If pred_tensor is an Output namedtuple, extract the actual prediction tensor
        if hasattr(pred_tensor, 'prediction'):
            actual_predictions = pred_tensor.prediction
            print(f"[FIX] Using pred_tensor.prediction (shape: {actual_predictions.shape})")
        else:
            actual_predictions = pred_tensor
            print(f"[FIX] Using pred_tensor directly")

        for idx, server_id in enumerate(servers):
            # Check against the actual prediction tensor, not the namedtuple length
            if idx >= len(actual_predictions):
                print(f"[WARNING] Pred tensor too small: idx={idx}, tensor_len={len(actual_predictions)}, stopping early")
                print(f"[WARNING] This indicates a batching or prediction dataset issue")
                break

            if self.server_encoder:
                server_name = self.server_encoder.decode(server_id)
            else:
                server_name = server_id

            server_preds = {}

            if hasattr(actual_predictions, 'dim') and actual_predictions.dim() >= 2:
                pred_values = actual_predictions[idx].cpu().numpy()

                # Debug shape
                print(f"[DEBUG] Server {idx} pred_values.shape: {pred_values.shape}")

                # TFT outputs shape: [timesteps, quantiles]
                # Quantiles are typically 7: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
                # We want: p10 (index 1), p50 (index 3), p90 (index 5)

                if pred_values.ndim == 2:
                    # Shape is [timesteps, quantiles]
                    if pred_values.shape[-1] >= 7:
                        # Full quantile set: use indices 1, 3, 5 for p10, p50, p90
                        p10_values = pred_values[:horizon, 1].tolist()  # 0.1 quantile
                        p50_values = pred_values[:horizon, 3].tolist()  # 0.5 quantile (median)
                        p90_values = pred_values[:horizon, 5].tolist()  # 0.9 quantile

                        # CRITICAL: Clamp CPU predictions to valid range [0, 100]
                        p10_values = [max(0.0, min(100.0, v)) for v in p10_values]
                        p50_values = [max(0.0, min(100.0, v)) for v in p50_values]
                        p90_values = [max(0.0, min(100.0, v)) for v in p90_values]
                    elif pred_values.shape[-1] == 3:
                        # Only 3 quantiles: [p10, p50, p90]
                        p10_values = pred_values[:horizon, 0].tolist()
                        p50_values = pred_values[:horizon, 1].tolist()
                        p90_values = pred_values[:horizon, 2].tolist()

                        # Clamp to valid range [0, 100]
                        p10_values = [max(0.0, min(100.0, v)) for v in p10_values]
                        p50_values = [max(0.0, min(100.0, v)) for v in p50_values]
                        p90_values = [max(0.0, min(100.0, v)) for v in p90_values]
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
        """Generate alerts based on prediction thresholds."""
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

        for server, server_preds in predictions.items():
            for metric, forecast in server_preds.items():
                if metric not in thresholds:
                    continue

                p50_values = forecast.get('p50', [])

                for i, value in enumerate(p50_values):
                    minutes_ahead = (i + 1) * 5

                    if minutes_ahead > 60:
                        break

                    if value >= thresholds[metric]['critical']:
                        alerts.append({
                            'server': server,
                            'metric': metric,
                            'severity': 'critical',
                            'predicted_value': value,
                            'threshold': thresholds[metric]['critical'],
                            'steps_ahead': i + 1,
                            'minutes_ahead': minutes_ahead,
                            'message': f"{server}: {metric} predicted to reach {value:.1f}"
                        })
                    elif value >= thresholds[metric]['warning']:
                        alerts.append({
                            'server': server,
                            'metric': metric,
                            'severity': 'warning',
                            'predicted_value': value,
                            'threshold': thresholds[metric]['warning'],
                            'steps_ahead': i + 1,
                            'minutes_ahead': minutes_ahead,
                            'message': f"{server}: {metric} predicted to reach {value:.1f}"
                        })

        alerts.sort(key=lambda x: (x['severity'] == 'warning', x['minutes_ahead']))
        return alerts

    def _calculate_environment_metrics(self, current_data: pd.DataFrame, predictions: Dict) -> Dict:
        """Calculate environment-wide incident probabilities."""
        prob_30m = 0.0
        prob_8h = 0.0
        high_risk_count = 0
        total_servers = len(predictions)

        for server, server_preds in predictions.items():
            risk_30m = 0.0
            risk_8h = 0.0

            for metric, forecast in server_preds.items():
                p50 = forecast.get('p50', [])
                p90 = forecast.get('p90', [])

                # 30-minute risk: only flag if p50 > 90 OR p90 > 98 (severe conditions)
                if len(p50) >= 6:
                    max_p50 = max(p50[:6])
                    max_p90 = max(p90[:6]) if len(p90) >= 6 else 0

                    if max_p50 > 90 or max_p90 > 98:
                        risk_30m += 0.35
                    elif max_p50 > 85 or max_p90 > 95:
                        risk_30m += 0.15

                # 8-hour risk: only flag if p50 > 88 OR p90 > 96
                if len(p50) >= 96:
                    max_p50 = max(p50)
                    max_p90 = max(p90) if len(p90) >= 96 else 0

                    if max_p50 > 88 or max_p90 > 96:
                        risk_8h += 0.25
                    elif max_p50 > 82 or max_p90 > 92:
                        risk_8h += 0.10

            if risk_30m > 0.6:
                high_risk_count += 1

            prob_30m += min(1.0, risk_30m)
            prob_8h += min(1.0, risk_8h)

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
            'fleet_health': 'critical' if prob_30m > 0.7 else 'warning' if prob_30m > 0.4 else 'healthy'
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

        # Track per-server data counts for warmup
        self.server_timesteps = {}

        # Initialize data buffer for automated retraining
        if enable_retraining:
            try:
                from data_buffer import DataBuffer
                self.data_buffer = DataBuffer(
                    buffer_dir='./data_buffer',
                    retention_days=60,
                    auto_rotate=True
                )
                print(f"[OK] Data buffer initialized for automated retraining")
            except ImportError:
                print(f"[WARNING] data_buffer.py not found - retraining disabled")
                self.data_buffer = None
        else:
            self.data_buffer = None

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

        return {
            "status": "accepted",
            "tick": self.tick_count,
            "window_size": len(self.rolling_window),
            "servers_tracked": total_servers,
            "servers_ready": servers_ready,
            "warmup_complete": is_warmed_up
        }

    def _calculate_server_risk_score(self, server_pred: Dict) -> float:
        """
        Calculate risk score for a single server (daemon version - no Streamlit deps).

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

                # Count by severity
                if risk_score >= 80:
                    alert_counts['critical'] += 1
                elif risk_score >= 60:
                    alert_counts['warning'] += 1
                elif risk_score >= 50:
                    alert_counts['degrading'] += 1
                else:
                    alert_counts['healthy'] += 1

            # Sort servers by risk (pre-calculate top N lists for dashboard)
            sorted_servers = sorted(
                predictions.keys(),
                key=lambda s: risk_scores[s],
                reverse=True
            )

            # Add summary statistics (dashboard-ready aggregates)
            result['summary'] = {
                'total_servers': len(predictions),
                'critical_count': alert_counts['critical'],
                'warning_count': alert_counts['warning'],
                'degrading_count': alert_counts['degrading'],
                'healthy_count': alert_counts['healthy'],
                'top_5_risks': sorted_servers[:5],
                'top_10_risks': sorted_servers[:10],
                'top_20_risks': sorted_servers[:20],
                'risk_calculation_time': datetime.now().isoformat(),
            }

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

            return {
                'server_name': server_name,
                'timestamp': datetime.now().isoformat(),
                'shap': shap_explanation,
                'attention': attention_analysis,
                'counterfactuals': counterfactuals,
                'prediction': server_pred,
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

@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "service": "TFT Inference Daemon",
        "version": "2.1",  # Bumped for XAI integration
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "feed": "POST /feed/data",
            "predictions": "GET /predictions/current",
            "alerts": "GET /alerts/active",
            "explain": "GET /explain/{server_name}",  # New XAI endpoint
            "status": "GET /status"
        },
        "features": {
            "xai": "Explainable AI with SHAP, Attention, and Counterfactuals",
            "retraining": "Automated data buffer for model retraining",
            "persistence": "Rolling window state persistence"
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
    api_key_env = os.getenv("TFT_API_KEY")
    if api_key_env:
        print(f"[OK] API key loaded from environment: {api_key_env[:8]}...")
    else:
        print("[WARNING] No TFT_API_KEY set - running in development mode (no authentication)")

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
