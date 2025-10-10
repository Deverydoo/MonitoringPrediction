#!/usr/bin/env python3
"""
tft_inference.py - TFT Model Inference with Daemon Mode
Supports: CLI mode, Daemon mode with REST/WebSocket API, Real TFT predictions
"""

import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
from collections import deque
from enum import Enum

import pandas as pd
import torch
import numpy as np
from safetensors.torch import load_file


# =============================================================================
# SIMULATION DATA GENERATOR
# =============================================================================

class SimulationMode(Enum):
    """Simulation scenarios."""
    STABLE = "stable"                  # Healthy baseline
    BUSINESS_HOURS = "business_hours"  # Peak during day, quiet at night
    GRADUAL_SPIKE = "gradual_spike"    # Slow resource exhaustion
    SUDDEN_SPIKE = "sudden_spike"      # Acute incident
    CYCLIC = "cyclic"                  # Wave patterns


class SimulationGenerator:
    """
    Generate realistic server metrics with temporal patterns.
    Seed-based for reproducibility.
    """

    def __init__(self, fleet_size: int = 25, seed: int = 42,
                 mode: SimulationMode = SimulationMode.BUSINESS_HOURS):
        np.random.seed(seed)
        self.fleet_size = fleet_size
        self.seed = seed
        self.mode = mode
        self.tick_count = 0
        self.start_time = datetime.now()

        # Create server fleet with profiles
        self.servers = self._create_fleet()

        # Incident tracking
        self.incident_active = False
        self.incident_start_tick = None
        self.incident_servers = set()

    def _create_fleet(self) -> List[Dict]:
        """Create diverse server fleet."""
        servers = []
        profiles = ['production', 'compute', 'service', 'container']
        profile_weights = [0.4, 0.3, 0.2, 0.1]

        for i in range(self.fleet_size):
            profile = np.random.choice(profiles, p=profile_weights)
            servers.append({
                'server_name': f'{profile[:4]}-{i:03d}',
                'profile': profile,
                'base_cpu': np.random.uniform(20, 50),
                'base_memory': np.random.uniform(40, 65),
                'base_latency': np.random.uniform(10, 80),
                'noise_factor': np.random.uniform(0.85, 1.15),
                'is_problem_child': np.random.random() < 0.1  # 10% are problematic
            })

        return servers

    def generate_tick(self) -> List[Dict]:
        """Generate one tick of data for entire fleet."""
        self.tick_count += 1
        current_time = self.start_time + timedelta(seconds=self.tick_count * 5)

        # Time-based factors
        hour = current_time.hour
        minute = current_time.minute

        # Diurnal pattern (business hours effect)
        if self.mode == SimulationMode.BUSINESS_HOURS:
            if 9 <= hour <= 17:  # Business hours
                time_factor = 1.4 + 0.2 * np.sin(2 * np.pi * (hour - 9) / 8)
            else:  # Off hours
                time_factor = 0.6 + 0.1 * np.sin(2 * np.pi * hour / 24)
        elif self.mode == SimulationMode.STABLE:
            time_factor = 1.0
        elif self.mode == SimulationMode.CYCLIC:
            time_factor = 1.0 + 0.3 * np.sin(2 * np.pi * self.tick_count / 72)  # 6-minute cycle
        else:
            time_factor = 1.0

        # Incident injection (gradual or sudden)
        incident_factor = 1.0
        if self.mode == SimulationMode.GRADUAL_SPIKE and self.tick_count > 60:
            # Gradual increase starting after 5 minutes
            ramp = min(1.0, (self.tick_count - 60) / 120)  # Ramp over 10 minutes
            incident_factor = 1.0 + ramp * 1.5
        elif self.mode == SimulationMode.SUDDEN_SPIKE and 100 < self.tick_count < 200:
            # Sudden spike for specific window
            incident_factor = 2.5

        # Generate metrics for each server
        batch = []
        for server in self.servers:
            # Decide if this server is affected by incident
            is_affected = server['is_problem_child'] or (self.tick_count % 100 == 0)

            # Base metrics with variations
            cpu_pct = server['base_cpu'] * time_factor * server['noise_factor']
            if is_affected:
                cpu_pct *= incident_factor
            cpu_pct += np.random.normal(0, 5)
            cpu_pct = max(5, min(100, cpu_pct))

            memory_pct = server['base_memory'] * (1 + 0.1 * time_factor)
            if is_affected:
                memory_pct *= (incident_factor * 0.8)
            memory_pct += np.random.normal(0, 3)
            memory_pct = max(10, min(98, memory_pct))

            # Disk grows slowly over time (simulates logs/cache)
            disk_pct = 30 + (self.tick_count * 0.01) + np.random.normal(0, 2)
            disk_pct = max(20, min(95, disk_pct))

            # Load average correlates with CPU
            load_average = (cpu_pct / 25) * server['noise_factor']
            load_average += np.random.exponential(0.3)
            load_average = max(0.1, min(16, load_average))

            # Java heap (for production servers)
            if server['profile'] == 'production':
                java_heap_usage = memory_pct * 0.9 + np.random.normal(0, 5)
            else:
                java_heap_usage = memory_pct * 0.5 + np.random.normal(0, 3)
            java_heap_usage = max(10, min(100, java_heap_usage))

            # Network errors (spike during incidents)
            network_errors = np.random.poisson(2 if not is_affected else 15)

            # Anomaly score (composite indicator)
            anomaly_score = 0.0
            if cpu_pct > 80:
                anomaly_score += 0.3
            if memory_pct > 85:
                anomaly_score += 0.3
            if load_average > 8:
                anomaly_score += 0.2
            if network_errors > 10:
                anomaly_score += 0.2
            anomaly_score = min(1.0, anomaly_score + np.random.uniform(0, 0.1))

            # State determination
            if anomaly_score > 0.7:
                state = 'critical_issue'
            elif anomaly_score > 0.4:
                state = 'warning'
            else:
                state = 'healthy'

            # Time features
            batch.append({
                'timestamp': current_time.isoformat(),
                'server_name': server['server_name'],
                'cpu_percent': float(cpu_pct),
                'memory_percent': float(memory_pct),
                'disk_percent': float(disk_pct),
                'load_average': float(load_average),
                'java_heap_usage': float(java_heap_usage),
                'network_errors': int(network_errors),
                'anomaly_score': float(anomaly_score),
                'hour': hour,
                'day_of_week': current_time.weekday(),
                'day_of_month': current_time.day,
                'month': current_time.month,
                'quarter': (current_time.month - 1) // 3 + 1,
                'is_weekend': current_time.weekday() >= 5,
                'is_business_hours': 9 <= hour <= 17,
                'status': state,
                'timeframe': 'realtime',
                'service_type': server['profile'],
                'datacenter': 'dc1',
                'environment': 'production'
            })

        return batch


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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_real_model = use_real_model
        self.model_dir = self._find_model(model_path)
        self.config = self._load_config()
        self.model = None
        self.training_data = None

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

            # Step 1: Create a dummy dataset for model architecture
            # This is needed because pytorch_forecasting requires TimeSeriesDataSet
            dummy_df = self._create_dummy_dataset()

            # Step 2: Create TimeSeriesDataSet matching training config
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

        IMPORTANT: Must match the schema used in training!
        - 8 unique status values (from metrics_generator.py state column)
        - Same features as training data
        """
        # All possible status values from training data
        all_status_values = [
            'critical_issue', 'healthy', 'heavy_load', 'idle',
            'maintenance', 'morning_spike', 'offline', 'recovery'
        ]

        data = []
        for server_idx, server_id in enumerate([f'{i}' for i in range(3)]):
            for time_idx in range(50):  # Enough for encoder + prediction
                # Cycle through status values to ensure all are present
                status = all_status_values[time_idx % len(all_status_values)]

                data.append({
                    'time_idx': time_idx,
                    'server_id': server_id,  # Numeric string (matches training encoding)
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
        Use actual TFT model for predictions.

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

            # Step 3: Create dataloader
            prediction_dataloader = prediction_dataset.to_dataloader(
                train=False,
                batch_size=64,
                num_workers=0  # No multiprocessing for inference
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

        # Rename server_name to server_id for TFT
        if 'server_name' in prediction_df.columns:
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

        return prediction_df

    def _format_tft_predictions(self, raw_predictions, input_df: pd.DataFrame,
                                horizon: int) -> Dict:
        """
        Format TFT raw predictions into our standard format.

        TFT returns quantile predictions (p10, p50, p90) for each timestep.
        """
        predictions = {}

        # Extract prediction tensor
        # Shape: (batch_size, horizon, output_size, num_quantiles)
        pred_tensor = raw_predictions.output

        # Get unique servers
        servers = input_df['server_id'].unique()

        # Extract predictions per server
        for idx, server in enumerate(servers):
            if idx >= len(pred_tensor):
                break

            server_preds = {}

            # TFT predicts cpu_percent (primary target)
            # Extract quantiles: [0.1, 0.5, 0.9] typically
            if pred_tensor.dim() >= 3:
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
                server_data = input_df[input_df['server_id'] == server]
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
            server_data = input_df[input_df['server_id'] == server]

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

            predictions[server] = server_preds

        return predictions

    def _predict_heuristic(self, df: pd.DataFrame, horizon: int) -> Dict:
        """
        Enhanced heuristic predictions (fallback when TFT not available).
        Uses trend analysis and statistical methods.
        """
        predictions = {}
        metrics = ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average',
                   'java_heap_usage', 'network_errors', 'anomaly_score']

        # Group by server for per-server predictions
        servers = df['server_name'].unique() if 'server_name' in df.columns else ['default']

        for server in servers:
            if 'server_name' in df.columns:
                server_data = df[df['server_name'] == server]
            else:
                server_data = df

            server_preds = {}

            for metric in metrics:
                if metric not in server_data.columns:
                    continue

                values = server_data[metric].values[-24:]  # Last 2 hours

                if len(values) > 0:
                    current = values[-1]

                    # Calculate trend
                    if len(values) > 1:
                        x = np.arange(len(values))
                        trend = np.polyfit(x, values, 1)[0]

                        # Detect acceleration
                        if len(values) > 5:
                            recent_trend = np.polyfit(np.arange(5), values[-5:], 1)[0]
                            trend = 0.7 * trend + 0.3 * recent_trend  # Weighted
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

                        # Apply bounds
                        if metric.endswith('_percent') or metric == 'anomaly_score':
                            pred = max(0, min(100 if metric.endswith('_percent') else 1.0, pred))
                        else:
                            pred = max(0, pred)

                        # Quantiles (confidence intervals)
                        uncertainty = noise * np.sqrt(i)  # Uncertainty grows with horizon
                        p10 = max(0, pred - 1.28 * uncertainty)
                        p90 = pred + 1.28 * uncertainty

                        if metric.endswith('_percent'):
                            p10 = min(100, p10)
                            p90 = min(100, p90)

                        p50_forecast.append(float(pred))
                        p10_forecast.append(float(p10))
                        p90_forecast.append(float(p90))

                    server_preds[metric] = {
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

        # Initialize inference engine
        model_path = config.get('model_path')
        self.inference = TFTInference(model_path, use_real_model=True)

        # Initialize data source (simulation mode)
        fleet_size = config.get('fleet_size', 25)
        seed = config.get('seed', 42)
        mode_str = config.get('simulation_mode', 'business_hours')
        mode = SimulationMode(mode_str)

        self.generator = SimulationGenerator(fleet_size, seed, mode)

        # State
        self.rolling_window = deque(maxlen=config.get('window_size', 8640))
        self.latest_predictions = {}
        self.active_alerts = []

        # WebSocket clients
        self.ws_clients: Set = set()

    async def start(self):
        """Start the daemon."""
        print("[START] TFT Inference Daemon Starting...")
        print(f"   Model: {self.config.get('model_path', 'auto-detect')}")
        print(f"   Fleet size: {self.config.get('fleet_size', 25)} servers")
        print(f"   Simulation mode: {self.config.get('simulation_mode', 'business_hours')}")

        self.is_running = True
        self.start_time = datetime.now()

        # Load initial historical data
        print("[INFO] Generating initial 24-hour window...")
        for _ in range(288 * 5):  # 24 hours worth
            batch = self.generator.generate_tick()
            self.rolling_window.extend(batch)

        print(f"[OK] Initialized with {len(self.rolling_window)} historical records")

        # Start inference loop
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

                # Run predictions
                df = pd.DataFrame(list(self.rolling_window))
                predictions = self.inference.predict(df, horizon=96)

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

def create_api_app():
    """Create FastAPI application."""
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

    @app.on_event("startup")
    async def startup():
        global daemon
        config = {
            'model_path': None,  # Auto-detect
            'fleet_size': 25,
            'seed': 42,
            'simulation_mode': 'business_hours',
            'tick_interval': 5,
            'window_size': 8640
        }
        daemon = InferenceDaemon(config)
        asyncio.create_task(daemon.start())

    @app.on_event("shutdown")
    async def shutdown():
        if daemon:
            await daemon.shutdown()

    @app.get("/health")
    async def health():
        return {
            "status": "healthy" if daemon.is_running else "stopped",
            "uptime_seconds": (datetime.now() - daemon.start_time).total_seconds() if daemon.start_time else 0
        }

    @app.get("/status")
    async def status():
        return {
            "running": daemon.is_running,
            "tick_count": daemon.tick_count,
            "window_size": len(daemon.rolling_window),
            "active_alerts": len(daemon.active_alerts),
            "ws_clients": len(daemon.ws_clients)
        }

    @app.get("/predictions/current")
    async def get_predictions():
        if not daemon.latest_predictions:
            return JSONResponse(status_code=503, content={"error": "No predictions yet"})
        return daemon.latest_predictions

    @app.get("/alerts/active")
    async def get_alerts():
        return {"count": len(daemon.active_alerts), "alerts": daemon.active_alerts}

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

        app = create_api_app()
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
