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
No dependencies on the messy old tft_inference.py file.
"""

import asyncio
import json
import warnings
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np
import torch
from safetensors.torch import load_file
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our helper modules (these are clean)
from server_encoder import ServerEncoder
from data_validator import DataValidator, CONTRACT_VERSION, VALID_STATES
from gpu_profiles import setup_gpu

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='lightning.pytorch.utilities.parsing')
warnings.filterwarnings('ignore', message='.*is an instance of.*nn.Module.*')
warnings.filterwarnings('ignore', message='.*dataloader.*does not have many workers.*')
warnings.filterwarnings('ignore', message='.*Tensor Cores.*')

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

                self.training_data = TimeSeriesDataSet(
                    dummy_df,
                    time_idx='time_idx',
                    target='cpu_percent',
                    group_ids=['server_id'],
                    max_encoder_length=24,
                    max_prediction_length=96,
                    min_encoder_length=12,
                    min_prediction_length=1,
                    time_varying_unknown_reals=['cpu_percent', 'memory_percent', 'disk_percent', 'load_average'],
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
            prediction_df = self._prepare_data_for_tft(df)

            from pytorch_forecasting import TimeSeriesDataSet

            prediction_dataset = TimeSeriesDataSet.from_dataset(
                self.training_data,
                prediction_df,
                predict=True,
                stop_randomization=True
            )

            batch_size = self.gpu.get_batch_size('inference') if self.gpu else 64
            num_workers = min(self.gpu.get_num_workers(), 4) if self.gpu else 0

            prediction_dataloader = prediction_dataset.to_dataloader(
                train=False,
                batch_size=batch_size,
                num_workers=num_workers
            )

            with torch.no_grad():
                raw_predictions = self.model.predict(
                    prediction_dataloader,
                    mode="raw",
                    return_x=True
                )

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
        """Prepare input data for TFT prediction."""
        prediction_df = df.copy()

        required_cols = ['server_name', 'timestamp', 'cpu_percent', 'memory_percent',
                        'disk_percent', 'load_average']

        missing = [col for col in required_cols if col not in prediction_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

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

        if hasattr(raw_predictions, 'prediction'):
            pred_tensor = raw_predictions.prediction
        elif hasattr(raw_predictions, 'output'):
            pred_tensor = raw_predictions.output
        else:
            pred_tensor = raw_predictions

        servers = input_df['server_id'].unique()

        for idx, server_id in enumerate(servers):
            if idx >= len(pred_tensor):
                break

            if self.server_encoder:
                server_name = self.server_encoder.decode(server_id)
            else:
                server_name = server_id

            server_preds = {}

            if hasattr(pred_tensor, 'dim') and pred_tensor.dim() >= 2:
                pred_values = pred_tensor[idx].cpu().numpy()

                if pred_values.shape[-1] >= 3:
                    p10_values = pred_values[:horizon, 0, 0].tolist()
                    p50_values = pred_values[:horizon, 0, 1].tolist()
                    p90_values = pred_values[:horizon, 0, 2].tolist()
                else:
                    p50_values = pred_values[:horizon, 0, 0].tolist()
                    std_dev = np.std(p50_values) if len(p50_values) > 1 else 5.0
                    p10_values = [max(0, v - std_dev) for v in p50_values]
                    p90_values = [min(100, v + std_dev) for v in p50_values]

                server_data = input_df[input_df['server_id'] == server_id]
                current_cpu = server_data['cpu_percent'].iloc[-1] if len(server_data) > 0 else 50.0

                trend = (p50_values[-1] - current_cpu) / len(p50_values) if len(p50_values) > 0 else 0.0

                server_preds['cpu_percent'] = {
                    'p50': p50_values,
                    'p10': p10_values,
                    'p90': p90_values,
                    'current': float(current_cpu),
                    'trend': float(trend)
                }

            # Heuristic for other metrics
            server_data = input_df[input_df['server_id'] == server_id]

            for metric in ['memory_percent', 'disk_percent', 'load_average']:
                if metric in server_data.columns:
                    values = server_data[metric].values[-24:]
                    if len(values) > 0:
                        current = values[-1]
                        trend = np.polyfit(np.arange(len(values)), values, 1)[0] if len(values) > 1 else 0

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

            predictions[server_name] = server_preds

        return predictions

    def _predict_heuristic(self, df: pd.DataFrame, horizon: int) -> Dict:
        """Enhanced heuristic predictions (fallback)."""
        predictions = {}

        metrics_mapping = {
            'cpu_pct': 'cpu_percent',
            'mem_pct': 'memory_percent',
            'disk_io_mb_s': 'disk_percent',
            'latency_ms': 'load_average',
            'error_rate': 'network_errors',
            'gc_pause_ms': 'java_heap_usage'
        }

        servers = df['server_name'].unique() if 'server_name' in df.columns else ['default']

        for server in servers:
            if 'server_name' in df.columns:
                server_data = df[df['server_name'] == server]
            else:
                server_data = df

            server_preds = {}

            for input_col, output_name in metrics_mapping.items():
                if input_col not in server_data.columns:
                    continue

                values = server_data[input_col].values[-24:]
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

                    if output_name.endswith('_percent'):
                        pred = max(0, min(100, pred))
                    else:
                        pred = max(0, pred)

                    uncertainty = noise * np.sqrt(i)
                    p10 = max(0, pred - 1.28 * uncertainty)
                    p90 = pred + 1.28 * uncertainty

                    if output_name.endswith('_percent'):
                        p10 = min(100, p10)
                        p90 = min(100, p90)

                    p50_forecast.append(float(pred))
                    p10_forecast.append(float(p10))
                    p90_forecast.append(float(p90))

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

                if len(p50) >= 6:
                    if max(p50[:6]) > 80 or max(p90[:6]) > 90:
                        risk_30m += 0.3

                if len(p50) >= 96:
                    if max(p50) > 85 or max(p90) > 95:
                        risk_8h += 0.2

            if risk_30m > 0.5:
                high_risk_count += 1

            prob_30m += min(1.0, risk_30m)
            prob_8h += min(1.0, risk_8h)

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
# CONFIGURATION
# =============================================================================

WARMUP_THRESHOLD = 150  # Timesteps needed per server before TFT predictions work
WINDOW_SIZE = 6000      # Keep last 6000 records (~8 hours at 5-second intervals for 25 servers)
PORT = 8000

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
    """

    def __init__(self, model_path: str = None, port: int = PORT):
        self.port = port
        self.rolling_window = deque(maxlen=WINDOW_SIZE)
        self.tick_count = 0
        self.start_time = datetime.now()

        # Load TFT model
        print(f"[INIT] Loading TFT model...")
        self.inference = TFTInference(model_path, use_real_model=True)
        print(f"[OK] Model loaded")

        # Track per-server data counts for warmup
        self.server_timesteps = {}

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

        # Track per-server counts for warmup status
        for record in records:
            server = record.get('server_name')
            if server:
                self.server_timesteps[server] = self.server_timesteps.get(server, 0) + 1

        # Check warmup status
        servers_ready = sum(1 for count in self.server_timesteps.values() if count >= WARMUP_THRESHOLD)
        total_servers = len(self.server_timesteps)
        is_warmed_up = servers_ready == total_servers and total_servers > 0

        return {
            "status": "accepted",
            "tick": self.tick_count,
            "window_size": len(self.rolling_window),
            "servers_tracked": total_servers,
            "servers_ready": servers_ready,
            "warmup_complete": is_warmed_up
        }

    def get_predictions(self) -> Dict[str, Any]:
        """
        Run TFT predictions on current rolling window.

        Returns:
            Predictions dict with per-server forecasts
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
            predictions = self.inference.predict(df, horizon=96)
            # Return predictions directly in the format the dashboard expects
            return predictions

        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return self.inference._empty_response()

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status for health checks."""
        servers_ready = sum(1 for count in self.server_timesteps.values() if count >= WARMUP_THRESHOLD)
        total_servers = len(self.server_timesteps)
        is_warmed_up = servers_ready == total_servers and total_servers > 0

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
                "message": warmup_msg
            }
        }

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

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
async def feed_data(request: FeedDataRequest):
    """
    Accept data from external source (metrics_generator.py --stream).

    Example:
        POST /feed/data
        {
            "records": [
                {"timestamp": "2025-10-12T20:00:00", "server_name": "ppdb001", "cpu_pct": 45.2, ...},
                {"timestamp": "2025-10-12T20:00:00", "server_name": "ppdb002", "cpu_pct": 52.1, ...}
            ]
        }
    """
    result = daemon.feed_data(request.records)
    return result

@app.get("/predictions/current")
async def get_predictions():
    """
    Get current predictions for all servers.

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
async def get_active_alerts():
    """
    Get active alerts from latest predictions.

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

@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "service": "TFT Inference Daemon",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "feed": "POST /feed/data",
            "predictions": "GET /predictions/current",
            "alerts": "GET /alerts/active",
            "status": "GET /status"
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
