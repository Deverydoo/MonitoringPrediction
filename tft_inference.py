#!/usr/bin/env python3
"""
tft_inference.py - TFT Model Inference with Daemon Mode
Supports: CLI mode, Daemon mode with REST/WebSocket API, Real TFT predictions
"""

import json
import argparse
import asyncio
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from collections import deque
from enum import Enum

import pandas as pd
import torch
import numpy as np
from safetensors.torch import load_file
from server_encoder import ServerEncoder
from data_validator import DataValidator, CONTRACT_VERSION, VALID_STATES
from gpu_profiles import setup_gpu
# ScenarioDemoGenerator removed - use metrics_generator.py --stream instead

# Suppress Lightning checkpoint warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightning.pytorch.utilities.parsing')
warnings.filterwarnings('ignore', message='.*is an instance of.*nn.Module.*')
warnings.filterwarnings('ignore', message='.*dataloader.*does not have many workers.*')
warnings.filterwarnings('ignore', message='.*Tensor Cores.*')

# Suppress TensorFlow info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages


# =============================================================================
# DATA GENERATION - REMOVED
# =============================================================================
# Built-in data generator removed. Use external metrics_generator.py --stream instead.
# This creates a clean separation: daemon = inference only, generator = data only.
#
# To feed data to the daemon:
#   python metrics_generator.py --stream --servers 20 --scenario healthy
#
# =============================================================================
# TFT INFERENCE ENGINE
# =============================================================================

class TFTInference:
    """
    TFT inference engine that actually loads and uses the trained model.

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
        self.server_encoder = None  # Will be loaded from model

        if self.use_real_model and self.model_dir:
            self._load_model()
        else:
            print("[WARNING] Running in HEURISTIC mode (no TFT model loaded)")
            print("   Use --model <path> to load trained TFT model")

    def _find_model(self, model_path: Optional[str]) -> Optional[Path]:
        """Find the latest trained model."""
        if model_path:
            path = Path(model_path)
            if path.exists():
                return path

        # Look for latest model
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
        """
        Load the actual TFT model from safetensors.

        Loads:
        1. Model architecture from training configuration
        2. Weights from safetensors
        3. Creates dummy TimeSeriesDataSet for prediction interface
        """
        try:
            from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
            from pytorch_forecasting.data import GroupNormalizer
            from pytorch_forecasting.metrics import QuantileLoss
            import pandas as pd

            model_file = self.model_dir / "model.safetensors"

            if not model_file.exists():
                print(f"[ERROR] Model file not found: {model_file}")
                self.use_real_model = False
                return

            print(f"[INFO] Loading TFT model from: {self.model_dir}")

            # Step 1: Load server mapping
            mapping_file = self.model_dir / "server_mapping.json"
            if not mapping_file.exists():
                print(f"[ERROR] server_mapping.json not found in {self.model_dir}")
                print("   Model was not trained with contract-compliant encoding")
                self.use_real_model = False
                return

            self.server_encoder = ServerEncoder(mapping_file)
            print(f"[OK] Server mapping loaded: {self.server_encoder.get_stats()['total_servers']} servers")

            # Step 2: Validate contract compatibility
            training_info_file = self.model_dir / "training_info.json"
            if training_info_file.exists():
                with open(training_info_file) as f:
                    training_info = json.load(f)

                contract_version = training_info.get('data_contract_version')
                if contract_version != CONTRACT_VERSION:
                    print(f"[WARNING] Model trained with contract v{contract_version}, "
                          f"current is v{CONTRACT_VERSION}")

                model_states = training_info.get('unique_states', [])
                if set(model_states) != set(VALID_STATES):
                    print(f"[ERROR] Model state mismatch!")
                    print(f"   Model has: {model_states}")
                    print(f"   Contract expects: {VALID_STATES}")
                    self.use_real_model = False
                    return

                print(f"[OK] Contract validation passed (v{CONTRACT_VERSION})")
            else:
                print(f"[WARNING] training_info.json not found - skipping validation")

            # Step 3: Try to load saved dataset parameters (includes trained encoders!)
            dataset_params_file = self.model_dir / "dataset_parameters.pkl"

            if dataset_params_file.exists():
                # Load the trained dataset parameters including categorical encoders
                print(f"[INFO] Loading trained dataset parameters (including encoders)...")
                import pickle
                with open(dataset_params_file, 'rb') as f:
                    dataset_params = pickle.load(f)

                # Create TimeSeriesDataSet from saved parameters
                dummy_df = self._create_dummy_dataset()
                self.training_data = TimeSeriesDataSet.from_parameters(
                    dataset_params,
                    dummy_df,
                    predict=False,
                    stop_randomization=True
                )
                print(f"[OK] Loaded trained encoders - all servers will be recognized!")
            else:
                # Fallback: Create new encoders (will cause "unknown" warnings)
                print(f"[WARNING] dataset_parameters.pkl not found - creating new encoders")
                print(f"   This may cause 'unknown class' warnings during inference")
                print(f"   Retrain model to save encoders properly")

                dummy_df = self._create_dummy_dataset()

                # Step 2: Create TimeSeriesDataSet matching training config
                # CRITICAL: Must use NaNLabelEncoder with add_nan=True to match training architecture
                from pytorch_forecasting.data import NaNLabelEncoder

                categorical_encoders = {
                    'server_id': NaNLabelEncoder(add_nan=True),  # Adds unknown category (+1 dimension)
                    'status': NaNLabelEncoder(add_nan=True),     # Adds unknown category (+1 dimension)
                    'profile': NaNLabelEncoder(add_nan=True)     # Adds unknown category (+1 dimension)
                }

                self.training_data = TimeSeriesDataSet(
                    dummy_df,
                    time_idx='time_idx',
                    target='cpu_percent',  # Primary target
                    group_ids=['server_id'],
                    max_encoder_length=24,  # 2 hours context
                    max_prediction_length=96,  # 8 hours ahead
                    min_encoder_length=12,
                    min_prediction_length=1,
                    time_varying_unknown_reals=['cpu_percent', 'memory_percent', 'disk_percent', 'load_average'],
                    time_varying_known_reals=['hour', 'day_of_week', 'month', 'is_weekend'],
                    time_varying_unknown_categoricals=['status'],
                    static_categoricals=['profile'],  # CRITICAL: Profile for transfer learning
                    categorical_encoders=categorical_encoders,  # CRITICAL: Must match training!
                    target_normalizer=GroupNormalizer(groups=['server_id'], transformation='softplus'),
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True,
                    allow_missing_timesteps=True
                )

            # Step 3: Create model architecture from dataset
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

            # Step 4: Load weights from safetensors
            state_dict = load_file(str(model_file))
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            print(f"[SUCCESS] TFT model loaded successfully!")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Device: {self.device}")
            self.use_real_model = True

        except ImportError as e:
            print(f"[ERROR] pytorch_forecasting not available: {e}")
            print("   Install with: pip install pytorch-forecasting")
            self.use_real_model = False
        except Exception as e:
            print(f"[ERROR] Error loading model: {e}")
            import traceback
            traceback.print_exc()
            self.use_real_model = False

    def _create_dummy_dataset(self) -> pd.DataFrame:
        """Create minimal dummy dataset for TimeSeriesDataSet initialization.

        IMPORTANT: Must use ACTUAL server names from training, not generic IDs!
        The loaded encoders expect the exact server names and hash IDs from training.
        """
        # All possible status values from training data
        all_status_values = [
            'critical_issue', 'healthy', 'heavy_load', 'idle',
            'maintenance', 'morning_spike', 'offline', 'recovery'
        ]

        # All possible profile values (CRITICAL: matches training!)
        all_profiles = [
            'ml_compute', 'database', 'web_api', 'conductor_mgmt',
            'data_ingest', 'risk_analytics', 'generic'
        ]

        # CRITICAL FIX: Use actual server names from loaded server_encoder
        # If server_encoder is available, get the actual trained server names
        if self.server_encoder:
            # Get all trained server names from the encoder
            trained_servers = list(self.server_encoder.name_to_id.keys())
            print(f"[OK] Using {len(trained_servers)} actual server names from training")

            data = []
            for server_name in trained_servers:
                # Encode to get the hash ID used in training
                server_id = self.server_encoder.encode(server_name)

                # Infer profile from server name
                if server_name.startswith('ppml'): profile = 'ml_compute'
                elif server_name.startswith('ppdb'): profile = 'database'
                elif server_name.startswith('ppweb'): profile = 'web_api'
                elif server_name.startswith('ppcon'): profile = 'conductor_mgmt'
                elif server_name.startswith('ppetl'): profile = 'data_ingest'
                elif server_name.startswith('pprisk'): profile = 'risk_analytics'
                elif server_name.startswith('ppgen'): profile = 'generic'
                else: profile = 'generic'

                # CRITICAL: Need enough time steps to satisfy encoder/decoder requirements
                # Based on config.py: context_length=288, prediction_horizon=96
                # Training adjusts to min(context_length, min_length//3)
                # So we need at least 400+ timesteps to ensure we have enough history
                for time_idx in range(450):  # Plenty of history for encoder + prediction
                    # Cycle through status values to ensure all are present
                    status = all_status_values[time_idx % len(all_status_values)]

                    data.append({
                        'time_idx': time_idx,
                        'server_id': server_id,  # Hash ID from training
                        'profile': profile,  # CRITICAL: Static categorical for transfer learning
                        'cpu_percent': 50.0,
                        'memory_percent': 60.0,
                        'disk_percent': 40.0,
                        'load_average': 2.0,
                        'hour': time_idx % 24,
                        'day_of_week': time_idx % 7,
                        'month': 1,
                        'is_weekend': 1 if (time_idx % 7) >= 5 else 0,
                        'status': status
                    })
        else:
            # Fallback: No server encoder loaded (shouldn't happen with proper model)
            print("[WARNING] No server encoder - using generic dummy data")
            data = []
            num_dummy_servers = max(len(all_profiles), len(all_status_values))

            for server_idx in range(num_dummy_servers):
                server_id = f'{server_idx}'
                profile = all_profiles[server_idx % len(all_profiles)]

                # CRITICAL: Need enough time steps to satisfy encoder/decoder requirements
                for time_idx in range(450):  # Plenty of history for encoder + prediction
                    status = all_status_values[time_idx % len(all_status_values)]
                    data.append({
                        'time_idx': time_idx,
                        'server_id': server_id,
                        'profile': profile,
                        'cpu_percent': 50.0,
                        'memory_percent': 60.0,
                        'disk_percent': 40.0,
                        'load_average': 2.0,
                        'hour': time_idx % 24,
                        'day_of_week': time_idx % 7,
                        'month': 1,
                        'is_weekend': 1 if (time_idx % 7) >= 5 else 0,
                        'status': status
                    })

        return pd.DataFrame(data)

    def predict(self, data: Union[Dict, List[Dict], pd.DataFrame],
                horizon: int = 96) -> Dict[str, Any]:
        """
        Make predictions on input data.

        Args:
            data: Input metrics (dict, list of dicts, or DataFrame)
            horizon: Prediction horizon (default 96 = 8 hours @ 5min intervals)

        Returns:
            Predictions with alerts and metadata
        """
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        if df.empty:
            return self._empty_response()

        # Run predictions (real TFT or heuristic)
        if self.use_real_model and self.model:
            predictions = self._predict_with_tft(df, horizon)
        else:
            predictions = self._predict_heuristic(df, horizon)

        # Generate alerts
        alerts = self._generate_alerts(predictions)

        # Calculate environment-wide metrics
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
        """
        Use TFT model for predictions on ALL servers (known and unknown).

        With add_nan=True in training, the model handles unknown servers natively
        by routing them to a learned "unknown" category that captures average
        server behavior patterns.

        Args:
            df: DataFrame with server metrics (must have required columns)
            horizon: Prediction horizon (default 96 steps)

        Returns:
            Dict with per-server predictions including quantiles
        """
        try:
            # Step 1: Prepare data for TFT
            prediction_df = self._prepare_data_for_tft(df)

            # Step 2: Create prediction dataset
            from pytorch_forecasting import TimeSeriesDataSet

            prediction_dataset = TimeSeriesDataSet.from_dataset(
                self.training_data,
                prediction_df,
                predict=True,
                stop_randomization=True
            )

            # Step 3: Create dataloader with GPU-optimal settings
            batch_size = self.gpu.get_batch_size('inference') if self.gpu else 64
            num_workers = min(self.gpu.get_num_workers(), 4) if self.gpu else 0  # Cap at 4 for inference

            prediction_dataloader = prediction_dataset.to_dataloader(
                train=False,
                batch_size=batch_size,
                num_workers=num_workers
            )

            # Step 4: Run predictions
            with torch.no_grad():
                raw_predictions = self.model.predict(
                    prediction_dataloader,
                    mode="raw",
                    return_x=True
                )

            # Step 5: Extract and format predictions
            predictions = self._format_tft_predictions(raw_predictions, prediction_df, horizon)

            print(f"[OK] TFT predictions generated for {len(predictions)} servers")
            return predictions

        except Exception as e:
            # Suppress traceback if it's a warmup-related error (not enough data)
            error_msg = str(e)
            is_warmup_error = (
                'filters should not remove entries' in error_msg or
                'check encoder/decoder lengths' in error_msg
            )

            if is_warmup_error:
                # Silently fall back during warmup - this is expected behavior
                pass
            else:
                # Log unexpected errors with full traceback
                print(f"[WARNING] TFT prediction failed: {e}")
                print("   Falling back to heuristic predictions")
                import traceback
                traceback.print_exc()

            return self._predict_heuristic(df, horizon)

    def _prepare_data_for_tft(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare input data for TFT prediction.

        Converts simulation/dashboard format to TFT training format.
        """
        prediction_df = df.copy()

        # Ensure required columns exist
        required_cols = ['server_name', 'timestamp', 'cpu_percent', 'memory_percent',
                        'disk_percent', 'load_average']

        missing = [col for col in required_cols if col not in prediction_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Convert server_name to server_id using hash-based encoder
        if 'server_name' in prediction_df.columns and self.server_encoder:
            # Encode using the same mapping as training
            prediction_df['server_id'] = prediction_df['server_name'].apply(
                self.server_encoder.encode
            )
        elif 'server_name' in prediction_df.columns:
            # Fallback if no encoder available
            print("[WARNING] No server encoder loaded - using fallback encoding")
            prediction_df['server_id'] = prediction_df['server_name']

        # Create time_idx (sequential index)
        prediction_df = prediction_df.sort_values(['server_id', 'timestamp'])
        prediction_df['time_idx'] = prediction_df.groupby('server_id').cumcount()

        # Ensure time features exist
        if 'hour' not in prediction_df.columns:
            prediction_df['timestamp'] = pd.to_datetime(prediction_df['timestamp'])
            prediction_df['hour'] = prediction_df['timestamp'].dt.hour
            prediction_df['day_of_week'] = prediction_df['timestamp'].dt.dayofweek
            prediction_df['month'] = prediction_df['timestamp'].dt.month
            prediction_df['is_weekend'] = (prediction_df['day_of_week'] >= 5).astype(int)

        # Ensure status column exists
        if 'status' not in prediction_df.columns:
            prediction_df['status'] = 'healthy'

        # CRITICAL: Infer profile from server_name (for transfer learning)
        def get_profile(server_name):
            """Infer server profile from naming convention."""
            if server_name.startswith('ppml'): return 'ml_compute'
            if server_name.startswith('ppdb'): return 'database'
            if server_name.startswith('ppweb'): return 'web_api'
            if server_name.startswith('ppcon'): return 'conductor_mgmt'
            if server_name.startswith('ppetl'): return 'data_ingest'
            if server_name.startswith('pprisk'): return 'risk_analytics'
            if server_name.startswith('ppgen'): return 'generic'
            return 'generic'  # Fallback for unknown servers

        if 'profile' not in prediction_df.columns:
            prediction_df['profile'] = prediction_df['server_name'].apply(get_profile)

        return prediction_df

    def _format_tft_predictions(self, raw_predictions, input_df: pd.DataFrame,
                                horizon: int) -> Dict:
        """
        Format TFT raw predictions into our standard format.

        TFT returns quantile predictions (p10, p50, p90) for each timestep.
        """
        predictions = {}

        # Extract prediction tensor - raw_predictions is an Output object, not a tensor
        # Access the prediction attribute which contains the actual tensor
        if hasattr(raw_predictions, 'prediction'):
            pred_tensor = raw_predictions.prediction  # Shape: (batch_size, horizon, num_quantiles)
        elif hasattr(raw_predictions, 'output'):
            pred_tensor = raw_predictions.output
        else:
            # Fallback: raw_predictions might be the tensor directly
            pred_tensor = raw_predictions

        # Get unique servers
        servers = input_df['server_id'].unique()

        # Extract predictions per server
        for idx, server_id in enumerate(servers):
            if idx >= len(pred_tensor):
                break

            # Decode server_id back to server_name
            if self.server_encoder:
                server_name = self.server_encoder.decode(server_id)
            else:
                server_name = server_id  # Fallback

            server_preds = {}

            # TFT predicts cpu_percent (primary target)
            # Extract quantiles: [0.1, 0.5, 0.9] typically
            if hasattr(pred_tensor, 'dim') and pred_tensor.dim() >= 2:
                # pred_tensor[idx] shape: (horizon, 1, 3) for single target
                pred_values = pred_tensor[idx].cpu().numpy()

                # Assuming quantiles are [0.1, 0.5, 0.9]
                if pred_values.shape[-1] >= 3:
                    p10_values = pred_values[:horizon, 0, 0].tolist()
                    p50_values = pred_values[:horizon, 0, 1].tolist()
                    p90_values = pred_values[:horizon, 0, 2].tolist()
                else:
                    # Single quantile (median)
                    p50_values = pred_values[:horizon, 0, 0].tolist()
                    # Estimate uncertainty
                    std_dev = np.std(p50_values) if len(p50_values) > 1 else 5.0
                    p10_values = [max(0, v - std_dev) for v in p50_values]
                    p90_values = [min(100, v + std_dev) for v in p50_values]

                # Get current value from input
                server_data = input_df[input_df['server_id'] == server_id]
                current_cpu = server_data['cpu_percent'].iloc[-1] if len(server_data) > 0 else 50.0

                # Calculate trend
                if len(p50_values) > 0:
                    trend = (p50_values[-1] - current_cpu) / len(p50_values)
                else:
                    trend = 0.0

                server_preds['cpu_percent'] = {
                    'p50': p50_values,
                    'p10': p10_values,
                    'p90': p90_values,
                    'current': float(current_cpu),
                    'trend': float(trend)
                }

            # For other metrics, use heuristic predictions based on CPU
            # (since model was trained primarily on CPU)
            server_data = input_df[input_df['server_id'] == server_id]

            for metric in ['memory_percent', 'disk_percent', 'load_average']:
                if metric in server_data.columns:
                    values = server_data[metric].values[-24:]
                    if len(values) > 0:
                        current = values[-1]
                        if len(values) > 1:
                            trend = np.polyfit(np.arange(len(values)), values, 1)[0]
                        else:
                            trend = 0

                        # Generate correlated predictions
                        p50_forecast = [current + (trend * i) for i in range(1, horizon + 1)]
                        noise = np.std(values) if len(values) > 2 else 2.0
                        p10_forecast = [max(0, v - noise * np.sqrt(i)) for i, v in enumerate(p50_forecast, 1)]
                        p90_forecast = [min(100 if metric.endswith('_percent') else 16,
                                          v + noise * np.sqrt(i)) for i, v in enumerate(p50_forecast, 1)]

                        server_preds[metric] = {
                            'p50': p50_forecast,
                            'p10': p10_forecast,
                            'p90': p90_forecast,
                            'current': float(current),
                            'trend': float(trend)
                        }

            # Use decoded server_name as key
            predictions[server_name] = server_preds

        return predictions

    def _predict_heuristic(self, df: pd.DataFrame, horizon: int) -> Dict:
        """
        Enhanced heuristic predictions (fallback when TFT not available).
        Uses trend analysis and statistical methods.

        Expects production standard columns from metrics_generator.py:
        cpu_pct, mem_pct, disk_io_mb_s, latency_ms, error_rate, gc_pause_ms
        """
        predictions = {}

        # Map production columns to output names for compatibility
        metrics_mapping = {
            'cpu_pct': 'cpu_percent',
            'mem_pct': 'memory_percent',
            'disk_io_mb_s': 'disk_percent',
            'latency_ms': 'load_average',
            'error_rate': 'network_errors',
            'gc_pause_ms': 'java_heap_usage'
        }

        # Group by server for per-server predictions
        servers = df['server_name'].unique() if 'server_name' in df.columns else ['default']

        for server in servers:
            if 'server_name' in df.columns:
                server_data = df[df['server_name'] == server]
            else:
                server_data = df

            server_preds = {}

            # Process each production metric
            for input_col, output_name in metrics_mapping.items():
                if input_col not in server_data.columns:
                    continue

                values = server_data[input_col].values[-24:]  # Last 2 hours

                # Clean data: remove NaN and infinite values
                values = values[np.isfinite(values)]

                if len(values) == 0:
                    continue  # Skip if no valid data

                current = values[-1]

                # Calculate trend with error handling
                trend = 0
                if len(values) > 1:
                    try:
                        x = np.arange(len(values))
                        trend = np.polyfit(x, values, 1)[0]

                        # Detect acceleration
                        if len(values) > 5:
                            recent_trend = np.polyfit(np.arange(5), values[-5:], 1)[0]
                            trend = 0.7 * trend + 0.3 * recent_trend  # Weighted
                    except (np.linalg.LinAlgError, ValueError) as e:
                        # Fallback: simple difference-based trend
                        if len(values) >= 2:
                            trend = (values[-1] - values[0]) / len(values)
                        else:
                            trend = 0

                # Generate forecasts with quantiles
                p50_forecast = []  # Median
                p10_forecast = []  # Lower bound
                p90_forecast = []  # Upper bound

                noise = np.std(values) if len(values) > 2 else 1.0

                for i in range(1, horizon + 1):
                    # Trend-based prediction
                    pred = current + (trend * i)

                    # Apply bounds (check output name for type)
                    if output_name.endswith('_percent') or output_name == 'anomaly_score':
                        pred = max(0, min(100 if output_name.endswith('_percent') else 1.0, pred))
                    else:
                        pred = max(0, pred)

                    # Quantiles (confidence intervals)
                    uncertainty = noise * np.sqrt(i)  # Uncertainty grows with horizon
                    p10 = max(0, pred - 1.28 * uncertainty)
                    p90 = pred + 1.28 * uncertainty

                    if output_name.endswith('_percent'):
                        p10 = min(100, p10)
                        p90 = min(100, p90)

                    p50_forecast.append(float(pred))
                    p10_forecast.append(float(p10))
                    p90_forecast.append(float(p90))

                # Use output name for predictions
                server_preds[output_name] = {
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
        from config import CONFIG
        thresholds = CONFIG.get('alert_thresholds', {})

        alerts = []

        for server, server_preds in predictions.items():
            for metric, forecast in server_preds.items():
                if metric not in thresholds:
                    continue

                p50_values = forecast.get('p50', [])

                for i, value in enumerate(p50_values):
                    minutes_ahead = (i + 1) * 5

                    # Only alert on near-term (next hour)
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
                            'message': f"{server}: {metric} predicted to reach {value:.1f} (critical threshold: {thresholds[metric]['critical']})"
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
                            'message': f"{server}: {metric} predicted to reach {value:.1f} (warning threshold: {thresholds[metric]['warning']})"
                        })

        # Sort by severity and time
        alerts.sort(key=lambda x: (x['severity'] == 'warning', x['minutes_ahead']))

        return alerts

    def _calculate_environment_metrics(self, current_data: pd.DataFrame,
                                      predictions: Dict) -> Dict:
        """Calculate environment-wide incident probabilities."""
        # Extract 30-min and 8-hour predictions
        prob_30m = 0.0
        prob_8h = 0.0

        # Count servers with high risk
        high_risk_count = 0
        total_servers = len(predictions)

        for server, server_preds in predictions.items():
            # Check if any metric exceeds thresholds in next 30 min (6 steps)
            risk_30m = 0.0
            risk_8h = 0.0

            for metric, forecast in server_preds.items():
                p50 = forecast.get('p50', [])
                p90 = forecast.get('p90', [])

                if len(p50) >= 6:
                    # 30-minute risk (next 6 steps)
                    if max(p50[:6]) > 80 or max(p90[:6]) > 90:
                        risk_30m += 0.3

                if len(p50) >= 96:
                    # 8-hour risk (all 96 steps)
                    if max(p50) > 85 or max(p90) > 95:
                        risk_8h += 0.2

            if risk_30m > 0.5:
                high_risk_count += 1

            prob_30m += min(1.0, risk_30m)
            prob_8h += min(1.0, risk_8h)

        # Average across fleet
        if total_servers > 0:
            prob_30m = prob_30m / total_servers
            prob_8h = prob_8h / total_servers

        return {
            'incident_probability_30m': float(min(1.0, prob_30m)),
            'incident_probability_8h': float(min(1.0, prob_8h)),
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
# DAEMON MODE (REST + WebSocket)
# =============================================================================

class InferenceDaemon:
    """
    24/7 Inference Daemon with REST and WebSocket API.

    Features:
    - Continuous inference loop
    - Simulation data source
    - REST API for polling
    - WebSocket for streaming (future-ready)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False
        self.tick_count = 0
        self.start_time = None

        # Defer heavy initialization to start() method
        self.inference = None
        self.generator = None

        # Warmup tracking
        # CRITICAL: Must match min_encoder_length from training (144 timesteps = 12 hours)
        self.warmup_threshold = 150  # Timesteps needed per server for TFT (144 min + buffer)
        self.is_warmed_up = False

        # State
        self.rolling_window = deque(maxlen=config.get('window_size', 8640))
        self.latest_predictions = {}
        self.active_alerts = []

        # WebSocket clients
        self.ws_clients: Set = set()

    def set_scenario(self, mode: str, affected_count: Optional[int] = None):
        """
        Set demo scenario - called from dashboard button.

        Args:
            mode: 'healthy', 'degrading', or 'critical'
            affected_count: Number of servers to affect (default: random 1-5)
        """
        if not self.generator:
            return {"error": "Daemon not fully initialized yet"}
        self.generator.set_scenario(mode, affected_count)
        print(f"[SCENARIO] Switched to: {mode.upper()}")

    def get_scenario_status(self) -> Dict:
        """Get current scenario status."""
        if not self.generator:
            return {"scenario": "initializing", "affected_servers": [], "transition_progress": 0.0}
        return self.generator.get_status()

    def _sync_initialization(self):
        """Heavy initialization that runs in thread pool to avoid blocking event loop."""
        print("[START] TFT Inference Daemon Starting...")

        # Initialize inference engine (heavy - loads model)
        model_path = self.config.get('model_path')
        self.inference = TFTInference(model_path, use_real_model=True)

        # Initialize interactive scenario demo generator (reads training data)
        seed = self.config.get('seed', 42)
        training_data_path = self.config.get('training_data_path', './training/server_metrics.parquet')

        self.generator = ScenarioDemoGenerator(
            training_data_path=training_data_path,
            seed=seed
        )
        print(f"[DEMO] Loaded {self.generator.fleet_size} servers from training data")

        print(f"   Model: {self.config.get('model_path', 'auto-detect')}")
        print(f"   Fleet size: {self.generator.fleet_size} servers (auto-detected from training data)")
        print(f"   Scenario mode: Interactive demo with real-time control")

        # Generate sufficient startup data for immediate TFT predictions
        # Need warmup_threshold (150) timesteps per server for TFT to work
        print(f"[INFO] Generating startup data ({self.warmup_threshold} timesteps for instant warmup)...")
        for i in range(self.warmup_threshold):
            batch = self.generator.generate_tick()
            self.rolling_window.extend(batch)
            if (i + 1) % 30 == 0:  # Progress every 30 ticks (2.5 minutes)
                print(f"  Progress: {i+1}/{self.warmup_threshold} timesteps ({(i+1)/self.warmup_threshold*100:.0f}%)")

        print(f"[OK] Initialized with {len(self.rolling_window)} records ({self.warmup_threshold} timesteps × {self.generator.fleet_size} servers)")
        print(f"[OK] All {self.generator.fleet_size} servers ready for TFT predictions immediately!")

    async def start(self):
        """Start the daemon - runs heavy init in thread pool to avoid blocking event loop."""
        # Run heavy initialization in thread pool so event loop can process HTTP requests
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._sync_initialization)

        self.is_running = True
        self.start_time = datetime.now()

        print("=" * 80)
        print("=" * 80)
        print(f"{'🚀 DAEMON READY! 🚀':^80}")
        print(f"{'Dashboard can connect now - first predictions will generate momentarily':^80}")
        print("=" * 80)
        print("=" * 80)

        # Start inference loop (first iteration will warm up model)
        await self.inference_loop()

    async def inference_loop(self):
        """Main inference loop."""
        print("[LOOP] Inference loop started")

        while self.is_running:
            try:
                # Generate new data
                batch = self.generator.generate_tick()
                self.rolling_window.extend(batch)
                self.tick_count += 1

                # Run predictions in thread pool to avoid blocking event loop
                df = pd.DataFrame(list(self.rolling_window))

                # Check if we have enough data for TFT (need 120+ timesteps per server for stable predictions)
                # TFT needs: max_encoder_length (24) + buffer for dataset filtering (96+)
                if 'server_name' in df.columns:
                    timesteps_per_server = df.groupby('server_name').size().min()
                    if timesteps_per_server < self.warmup_threshold:
                        # Not enough data yet - use heuristic only to avoid TFT errors
                        if self.tick_count % 12 == 0:  # Log every minute
                            warmup_progress = (timesteps_per_server / self.warmup_threshold) * 100
                            print(f"[INFO] Model warming up: {timesteps_per_server}/{self.warmup_threshold} timesteps ({warmup_progress:.0f}%)")
                        # Temporarily disable TFT to avoid error spam
                        self.inference.use_real_model = False
                        # Run prediction in thread pool to not block HTTP server
                        loop = asyncio.get_event_loop()
                        predictions = await loop.run_in_executor(None, self.inference.predict, df, 96)
                        self.inference.use_real_model = True  # Re-enable for next attempt
                    else:
                        # Enough data - try TFT (will auto-fallback to heuristic if still fails)
                        if not self.is_warmed_up:
                            self.is_warmed_up = True
                            print(f"[OK] Model warmed up! Switching to TFT predictions ({timesteps_per_server} timesteps available)")
                        # Run TFT prediction in thread pool to not block HTTP server
                        loop = asyncio.get_event_loop()
                        predictions = await loop.run_in_executor(None, self.inference.predict, df, 96)
                else:
                    loop = asyncio.get_event_loop()
                    predictions = await loop.run_in_executor(None, self.inference.predict, df, 96)

                # Cache results
                self.latest_predictions = predictions
                self.active_alerts = predictions['alerts']

                # Broadcast to WebSocket clients
                await self._broadcast_to_clients(predictions)

                # Log progress
                if self.tick_count % 12 == 0:  # Every minute
                    env = predictions['environment']
                    print(f"[Tick {self.tick_count}] "
                          f"Health: {env['fleet_health']} | "
                          f"P(30m): {env['incident_probability_30m']:.2%} | "
                          f"Alerts: {len(self.active_alerts)} | "
                          f"Clients: {len(self.ws_clients)}")

                # Wait for next tick
                await asyncio.sleep(self.config.get('tick_interval', 5))

            except Exception as e:
                print(f"[ERROR] Error in inference loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(5)

    async def _broadcast_to_clients(self, predictions: Dict):
        """Broadcast to WebSocket clients."""
        if not self.ws_clients:
            return

        message = {
            'type': 'prediction_update',
            'timestamp': datetime.now().isoformat(),
            'tick': self.tick_count,
            'predictions': predictions
        }

        disconnected = set()
        for client in self.ws_clients:
            try:
                await client.send_json(message)
            except Exception:
                disconnected.add(client)

        self.ws_clients -= disconnected

    async def shutdown(self):
        """Clean shutdown."""
        print("🛑 Shutting down daemon...")
        self.is_running = False

        for client in self.ws_clients:
            await client.close()

        print("[OK] Daemon stopped")


# =============================================================================
# REST API (FastAPI)
# =============================================================================

daemon: Optional[InferenceDaemon] = None

def create_api_app(daemon_config: Optional[Dict] = None):
    """Create FastAPI application with optional daemon configuration."""
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse
    except ImportError:
        print("[ERROR] FastAPI not installed. Install with: pip install fastapi uvicorn")
        return None

    app = FastAPI(
        title="TFT Inference Daemon",
        description="24/7 TFT prediction service with REST and WebSocket APIs",
        version="2.0.0"
    )

    # Initialize daemon BEFORE startup event so it doesn't block
    global daemon
    config = daemon_config or {
        'model_path': None,  # Auto-detect
        'fleet_size': 25,
        'seed': 42,
        'simulation_mode': 'business_hours',
        'tick_interval': 5,
        'window_size': 8640
    }
    daemon = InferenceDaemon(config)

    @app.on_event("startup")
    async def startup():
        # CRITICAL: Don't start daemon here - it blocks the event loop
        # We'll start it AFTER Uvicorn is ready using a delayed task
        async def delayed_start():
            # Wait for Uvicorn to complete startup
            await asyncio.sleep(0.1)
            print("[OK] HTTP server ready, starting daemon initialization...")
            await daemon.start()

        # Start daemon after a tiny delay to let Uvicorn finish
        asyncio.create_task(delayed_start())

    @app.on_event("shutdown")
    async def shutdown():
        if daemon:
            await daemon.shutdown()

    @app.get("/health")
    async def health():
        if daemon.inference is None:
            # Still initializing
            return {
                "status": "initializing",
                "message": "Daemon is loading model and generating initial data",
                "uptime_seconds": 0
            }
        return {
            "status": "healthy" if daemon.is_running else "stopped",
            "uptime_seconds": (datetime.now() - daemon.start_time).total_seconds() if daemon.start_time else 0
        }

    @app.get("/status")
    async def status():
        # Calculate warmup progress
        warmup_progress = 0
        if daemon.generator and len(daemon.rolling_window) > 0:
            df = pd.DataFrame(list(daemon.rolling_window))
            if 'server_name' in df.columns:
                timesteps_per_server = df.groupby('server_name').size().min()
                warmup_progress = min(100, (timesteps_per_server / daemon.warmup_threshold) * 100)

        return {
            "running": daemon.is_running,
            "tick_count": daemon.tick_count,
            "window_size": len(daemon.rolling_window),
            "active_alerts": len(daemon.active_alerts),
            "ws_clients": len(daemon.ws_clients),
            "warmup": {
                "is_warmed_up": daemon.is_warmed_up,
                "progress_percent": warmup_progress,
                "threshold": daemon.warmup_threshold,
                "message": "Model warming up - using heuristic predictions" if not daemon.is_warmed_up else "Model ready - using TFT predictions"
            }
        }

    @app.get("/predictions/current")
    async def get_predictions():
        if not daemon.latest_predictions:
            return JSONResponse(status_code=503, content={"error": "No predictions yet"})
        return daemon.latest_predictions

    @app.get("/alerts/active")
    async def get_alerts():
        return {"count": len(daemon.active_alerts), "alerts": daemon.active_alerts}

    @app.post("/feed/data")
    async def feed_data(data: Dict):
        """
        Feed external data to daemon for prediction.
        Used for demo mode to inject scenario data in real-time.

        Expected format:
        {
            "records": [
                {
                    "timestamp": "2025-10-10T12:00:00",
                    "server_name": "server01",
                    "cpu_percent": 45.2,
                    "memory_percent": 62.1,
                    ...
                },
                ...
            ]
        }
        """
        try:
            records = data.get('records', [])
            if not records:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No records provided"}
                )

            # Convert to DataFrame
            import pandas as pd
            df = pd.DataFrame(records)

            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Add each record to rolling window (it's a deque)
            for _, row in df.iterrows():
                daemon.rolling_window.append(row.to_dict())

            # Trigger immediate prediction
            daemon.tick_count += 1

            print(f"[FEED] Received {len(records)} records, window size: {len(daemon.rolling_window)}")

            return {
                "status": "success",
                "records_added": len(records),
                "window_size": len(daemon.rolling_window)
            }

        except Exception as e:
            print(f"[ERROR] Feed data failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )

    @app.post("/scenario/set")
    async def set_scenario(request: Dict):
        """
        Interactive demo scenario control.

        Expected: {"mode": "healthy|degrading|critical", "affected_count": 3}
        """
        try:
            mode = request.get('mode', 'healthy')
            affected_count = request.get('affected_count')

            daemon.set_scenario(mode, affected_count)

            return {
                "status": "success",
                "scenario": mode,
                "affected_servers": len(daemon.generator.affected_servers)
            }
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": str(e)}
            )

    @app.get("/scenario/status")
    async def get_scenario_status():
        """Get current scenario status."""
        return daemon.get_scenario_status()

    @app.websocket("/ws/predictions")
    async def ws_predictions(websocket: WebSocket):
        await websocket.accept()
        daemon.ws_clients.add(websocket)
        print(f"[OK] WebSocket connected (total: {len(daemon.ws_clients)})")

        try:
            await websocket.send_json({
                'type': 'connected',
                'message': 'Connected to TFT Inference Daemon'
            })

            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            daemon.ws_clients.remove(websocket)
            print(f"[ERROR] WebSocket disconnected (remaining: {len(daemon.ws_clients)})")

    return app


# =============================================================================
# CLI MODE (Original functionality)
# =============================================================================

def predict(data: Union[Dict, List[Dict]] = None,
           model_path: Optional[str] = None,
           horizon: int = 96) -> Dict[str, Any]:
    """Module interface for making predictions."""

    # Generate sample data if none provided
    if data is None:
        print("[INFO] Generating sample data...")
        generator = SimulationGenerator(fleet_size=5, seed=42)
        data = []
        for _ in range(24):  # 2 hours of data
            batch = generator.generate_tick()
            data.extend(batch)

    inference = TFTInference(model_path, use_real_model=True)
    return inference.predict(data, horizon)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Command line interface with daemon support."""
    parser = argparse.ArgumentParser(
        description="TFT Inference - CLI and Daemon modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CLI mode (one-shot prediction)
  python tft_inference.py --input data.json --output results.json

  # Daemon mode (24/7 server)
  python tft_inference.py --daemon --port 8000

  # Daemon with custom config
  python tft_inference.py --daemon --fleet-size 50 --simulation-mode gradual_spike
        """
    )

    # Mode selection
    parser.add_argument("--daemon", action="store_true", help="Run as daemon with REST/WebSocket API")

    # CLI mode options
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--input", type=str, help="Input JSON file")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--horizon", type=int, default=96, help="Prediction horizon (default: 96 steps = 8 hours)")

    # Daemon mode options
    parser.add_argument("--host", default="0.0.0.0", help="Daemon bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Daemon port (default: 8000)")
    parser.add_argument("--fleet-size", type=int, default=25, help="Simulation fleet size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation")
    parser.add_argument("--simulation-mode",
                       choices=['stable', 'business_hours', 'gradual_spike', 'sudden_spike', 'cyclic'],
                       default='business_hours',
                       help="Simulation scenario")

    args = parser.parse_args()

    # DAEMON MODE
    if args.daemon:
        try:
            import uvicorn
        except ImportError:
            print("[ERROR] Daemon mode requires: pip install fastapi uvicorn[standard] websockets")
            return 1

        print("="*60)
        print("[START] TFT INFERENCE DAEMON MODE")
        print("="*60)
        print(f"REST API:   http://{args.host}:{args.port}")
        print(f"WebSocket:  ws://{args.host}:{args.port}/ws/predictions")
        print(f"API Docs:   http://{args.host}:{args.port}/docs")
        print(f"Simulation: {args.simulation_mode} ({args.fleet_size} servers)")
        print("="*60)

        # Pass command-line args to daemon config
        daemon_config = {
            'model_path': args.model,
            'fleet_size': args.fleet_size,
            'seed': args.seed,
            'simulation_mode': args.simulation_mode,
            'tick_interval': 5,
            'window_size': 8640
        }

        app = create_api_app(daemon_config)
        if app:
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        return 0

    # CLI MODE
    else:
        print("="*60)
        print("[PREDICT] TFT INFERENCE - CLI MODE")
        print("="*60)

        # Load input data
        data = None
        if args.input:
            with open(args.input, 'r') as f:
                data = json.load(f)
            print(f"📂 Loaded input: {args.input}")

        # Make predictions
        results = predict(data, args.model, args.horizon)

        # Save or display results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"[SAVE] Results saved to: {args.output}")
        else:
            # Display summary
            env = results['environment']
            print(f"\n[INFO] Environment Status: {env['fleet_health'].upper()}")
            print(f"   30-min incident probability: {env['incident_probability_30m']:.1%}")
            print(f"   8-hour incident probability: {env['incident_probability_8h']:.1%}")
            print(f"   High-risk servers: {env['high_risk_servers']}/{env['total_servers']}")

            if results['alerts']:
                print(f"\n[WARNING]  {len(results['alerts'])} alerts (showing first 5):")
                for alert in results['alerts'][:5]:
                    icon = "[CRIT]" if alert['severity'] == 'critical' else "[WARN]"
                    print(f"   {icon} {alert['server']}: {alert['metric']} → {alert['predicted_value']:.1f} "
                          f"in {alert['minutes_ahead']} min")
            else:
                print("\n[OK] No alerts - fleet healthy")

            print(f"\n[INFO]  Model type: {results['metadata']['model_type']}")

        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
