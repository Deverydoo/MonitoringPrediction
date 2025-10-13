#!/usr/bin/env python3
"""
tft_dashboard_refactored.py - File-Based TFT Monitoring Dashboard
Production-level dashboard that reads from data sources (demo or production)
No more random data generation - reproducible and explainable results
"""

import time
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import warnings
warnings.filterwarnings('ignore')

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  'requests' not installed. Daemon mode unavailable.")
    print("   Install with: pip install requests")

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'TICK_INTERVAL_SECONDS': 5,  # How often to ingest new data (simulates real-time arrival)
    'REFRESH_SECONDS': 30,  # How often to refresh dashboard visualizations
    'ROLLING_WINDOW_MIN': 5,  # Show full demo window
    'MAX_ENV_POINTS': 500,
    'SAVE_PLOTS': False,
    'PLOT_DIR': './plots/dashboard/'
}

# =============================================================================
# DATA SOURCE
# =============================================================================

class DataSource:
    """Read metrics from file-based data sources (demo or production)."""

    def __init__(self, data_path: str, data_format: str = 'auto'):
        """
        Initialize data source.

        Args:
            data_path: Path to data file (CSV, Parquet, or directory)
            data_format: 'csv', 'parquet', or 'auto' (detect from extension)
        """
        self.data_path = Path(data_path)
        self.data_format = data_format
        self.df = None
        self.current_index = 0
        self.servers = []

        self._load_data()

    def _load_data(self):
        """Load data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data source not found: {self.data_path}")

        # Auto-detect format
        if self.data_format == 'auto':
            if self.data_path.suffix == '.csv':
                self.data_format = 'csv'
            elif self.data_path.suffix == '.parquet':
                self.data_format = 'parquet'
            elif self.data_path.is_dir():
                # Look for files in directory
                csv_files = list(self.data_path.glob('*.csv'))
                parquet_files = list(self.data_path.glob('*.parquet'))

                if parquet_files:
                    self.data_path = parquet_files[0]
                    self.data_format = 'parquet'
                elif csv_files:
                    self.data_path = csv_files[0]
                    self.data_format = 'csv'
                else:
                    raise ValueError(f"No CSV or Parquet files found in {self.data_path}")

        # Load data
        print(f"📂 Loading data from: {self.data_path}")

        if self.data_format == 'csv':
            self.df = pd.read_csv(self.data_path)
        elif self.data_format == 'parquet':
            self.df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(f"Unsupported format: {self.data_format}")

        # Parse timestamps
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        # Get server list
        self.servers = sorted(self.df['server_name'].unique().tolist())

        print(f"✅ Loaded {len(self.df)} records")
        print(f"   Servers: {len(self.servers)}")
        print(f"   Time range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"   Duration: {(self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds() / 60:.1f} minutes")

    def get_next_batch(self) -> Optional[pd.DataFrame]:
        """
        Get next batch of data (one tick for all servers).

        Returns:
            DataFrame with next tick's data, or None if exhausted
        """
        if self.current_index >= len(self.df):
            return None

        # Get current timestamp
        current_time = self.df.iloc[self.current_index]['timestamp']

        # Get all records with this timestamp
        batch = self.df[self.df['timestamp'] == current_time].copy()

        # Move index forward
        self.current_index += len(batch)

        return batch

    def reset(self):
        """Reset to beginning of data."""
        self.current_index = 0

    def get_current_progress(self) -> float:
        """Get current progress through data (0.0 to 1.0)."""
        if len(self.df) == 0:
            return 1.0
        return self.current_index / len(self.df)


# =============================================================================
# MODEL ADAPTERS
# =============================================================================

class TFTDaemonClient:
    """
    Client for TFT Inference Daemon.

    Connects to daemon via REST API to get REAL TFT predictions.
    This is the production adapter that uses the actual trained model.
    """

    def __init__(self, daemon_url: str = "http://localhost:8000"):
        self.daemon_url = daemon_url.rstrip('/')
        self.connected = False
        self.last_predictions = None
        self.last_fetch_time = None

        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for daemon mode")

        self._check_connection()

    def _check_connection(self):
        """Verify daemon is running and accessible."""
        try:
            response = requests.get(f"{self.daemon_url}/health", timeout=2)
            if response.ok:
                health = response.json()
                print(f"✅ Connected to TFT Daemon: {self.daemon_url}")
                print(f"   Status: {health.get('status', 'unknown')}")
                print(f"   Using REAL TFT MODEL predictions! 🤖")
                self.connected = True
                return True
        except requests.exceptions.ConnectionError:
            print(f"❌ Cannot connect to daemon at {self.daemon_url}")
            print(f"   Start daemon with: python tft_inference.py --daemon")
        except Exception as e:
            print(f"⚠️  Daemon connection error: {e}")

        self.connected = False
        return False

    def predict_per_server(self, df_window: pd.DataFrame) -> pd.DataFrame:
        """
        Get per-server risk predictions from TFT daemon.

        Returns DataFrame with server_name and risk_30m columns.
        """
        if not self.connected:
            if not self._check_connection():
                return pd.DataFrame(columns=['server_name', 'risk_30m'])

        try:
            # Fetch predictions from daemon
            response = requests.get(
                f"{self.daemon_url}/predictions/current",
                timeout=5
            )

            if not response.ok:
                print(f"⚠️  Daemon returned {response.status_code}: {response.text[:100]}")
                return pd.DataFrame(columns=['server_name', 'risk_30m'])

            data = response.json()
            self.last_predictions = data
            self.last_fetch_time = datetime.now()

            # Extract predictions
            predictions = data.get('predictions', {})

            if not predictions:
                print("⚠️  No predictions in daemon response")
                return pd.DataFrame(columns=['server_name', 'risk_30m'])

            # Convert TFT predictions to risk scores
            server_risks = []
            for server, server_preds in predictions.items():
                risk_30m = self._calculate_risk_from_tft(server_preds)
                server_risks.append({
                    'server_name': server,
                    'risk_30m': risk_30m
                })

            return pd.DataFrame(server_risks)

        except requests.exceptions.Timeout:
            print("⚠️  Daemon request timeout")
            return pd.DataFrame(columns=['server_name', 'risk_30m'])
        except Exception as e:
            print(f"❌ Error fetching predictions: {e}")
            return pd.DataFrame(columns=['server_name', 'risk_30m'])

    def predict_environment(self, df_window: pd.DataFrame) -> Dict[str, float]:
        """
        Get environment-wide incident probabilities from TFT daemon.

        Returns dict with prob_30m and prob_8h.
        """
        if not self.connected:
            if not self._check_connection():
                return {'prob_30m': 0.0, 'prob_8h': 0.0}

        try:
            # Use cached predictions if recent (< 5 seconds old)
            if (self.last_predictions and self.last_fetch_time and
                (datetime.now() - self.last_fetch_time).total_seconds() < 5):
                data = self.last_predictions
            else:
                # Fetch fresh predictions
                response = requests.get(
                    f"{self.daemon_url}/predictions/current",
                    timeout=5
                )

                if not response.ok:
                    return {'prob_30m': 0.0, 'prob_8h': 0.0}

                data = response.json()
                self.last_predictions = data
                self.last_fetch_time = datetime.now()

            # Extract environment metrics
            env = data.get('environment', {})

            return {
                'prob_30m': env.get('incident_probability_30m', 0.0),
                'prob_8h': env.get('incident_probability_8h', 0.0)
            }

        except Exception as e:
            print(f"❌ Error fetching environment predictions: {e}")
            return {'prob_30m': 0.0, 'prob_8h': 0.0}

    def _calculate_risk_from_tft(self, server_preds: Dict) -> float:
        """
        Calculate 30-minute risk score from TFT predictions.

        Uses p50 (median) and p90 (upper bound) forecasts for next 6 steps (30 min).
        """
        risk = 0.0

        # CPU risk (most important)
        if 'cpu_percent' in server_preds:
            cpu = server_preds['cpu_percent']
            p50 = cpu.get('p50', [])
            p90 = cpu.get('p90', [])

            if len(p50) >= 6:
                max_p50 = max(p50[:6])
                max_p90 = max(p90[:6]) if len(p90) >= 6 else max_p50

                # Critical thresholds
                if max_p90 > 95 or max_p50 > 90:
                    risk += 0.5
                elif max_p90 > 85 or max_p50 > 80:
                    risk += 0.3
                elif max_p50 > 75:
                    risk += 0.15

        # Memory risk
        if 'memory_percent' in server_preds:
            mem = server_preds['memory_percent']
            p50 = mem.get('p50', [])
            p90 = mem.get('p90', [])

            if len(p50) >= 6:
                max_p50 = max(p50[:6])
                max_p90 = max(p90[:6]) if len(p90) >= 6 else max_p50

                if max_p90 > 95 or max_p50 > 90:
                    risk += 0.3
                elif max_p50 > 85:
                    risk += 0.15

        # Load average risk
        if 'load_average' in server_preds:
            load = server_preds['load_average']
            p50 = load.get('p50', [])

            if len(p50) >= 6:
                max_load = max(p50[:6])

                if max_load > 12:
                    risk += 0.2
                elif max_load > 8:
                    risk += 0.1

        return min(1.0, risk)

    def get_model_info(self) -> Dict:
        """Get information about the TFT model from daemon."""
        try:
            response = requests.get(f"{self.daemon_url}/status", timeout=2)
            if response.ok:
                return response.json()
        except Exception:
            pass
        return {}


class ModelAdapter:
    """
    Heuristic model adapter (fallback when daemon not available).

    This is the backup prediction method using simple statistics.
    """

    def predict_per_server(self, df_window: pd.DataFrame) -> pd.DataFrame:
        """Predict 30-minute risk per server."""
        if df_window.empty:
            return pd.DataFrame(columns=['server_name', 'risk_30m'])

        # Calculate risk based on recent metrics
        recent_window = df_window.tail(20)  # Last 20 ticks
        server_risks = []

        for server in df_window['server_name'].unique():
            server_data = recent_window[recent_window['server_name'] == server]

            if server_data.empty:
                risk = 0.1
            else:
                # Risk factors
                cpu_risk = (server_data['cpu_pct'].mean() - 40) / 60
                latency_risk = (server_data['latency_ms'].mean() - 50) / 200
                error_risk = server_data['error_rate'].mean() * 2
                state_risk = 0.8 if (server_data['state'] == 'critical_issue').any() else 0
                state_risk += 0.3 if (server_data['state'] == 'warning').any() else 0

                # Check for incident phase if available
                incident_risk = 0
                if 'incident_phase' in server_data.columns:
                    if (server_data['incident_phase'] == 'peak').any():
                        incident_risk = 0.7
                    elif (server_data['incident_phase'] == 'escalation').any():
                        incident_risk = 0.4

                risk = max(0, min(1, cpu_risk + latency_risk + error_risk + state_risk + incident_risk))

            server_risks.append({'server_name': server, 'risk_30m': risk})

        return pd.DataFrame(server_risks)

    def predict_environment(self, df_window: pd.DataFrame) -> Dict[str, float]:
        """Predict environment incident probabilities."""
        if df_window.empty:
            return {'prob_30m': 0.1, 'prob_8h': 0.2}

        recent = df_window.tail(50)  # Last 50 ticks

        # Environment risk indicators
        p95_latency = recent['latency_ms'].quantile(0.95)
        mean_error_rate = recent['error_rate'].mean()
        critical_fraction = (recent['state'] == 'critical_issue').mean()
        warning_fraction = (recent['state'] == 'warning').mean()

        # Check for incident phase
        incident_active = False
        if 'incident_phase' in recent.columns:
            active_phases = recent['incident_phase'].unique()
            incident_active = any(phase in ['escalation', 'peak'] for phase in active_phases)

        # Calculate probabilities
        latency_factor = min(0.4, p95_latency / 500)
        error_factor = min(0.3, mean_error_rate * 10)
        critical_factor = critical_fraction * 0.6
        warning_factor = warning_fraction * 0.3
        incident_factor = 0.7 if incident_active else 0

        prob_30m = min(0.95, latency_factor + error_factor + critical_factor + warning_factor + incident_factor)
        prob_8h = min(0.90, prob_30m * 0.8 + 0.2)  # 8h is generally higher baseline

        return {'prob_30m': prob_30m, 'prob_8h': prob_8h}


# =============================================================================
# DASHBOARD
# =============================================================================

class LiveDashboard:
    """
    File-based real-time monitoring dashboard.

    Supports two modes:
    1. Daemon mode: Connects to TFT inference daemon for REAL predictions
    2. Fallback mode: Uses heuristic predictions when daemon unavailable
    """

    def __init__(self, data_source: DataSource,
                 daemon_url: Optional[str] = None,
                 use_daemon: bool = True,
                 model_adapter=None,
                 config=None):
        self.config = config or CONFIG
        self.data_source = data_source

        # Choose model adapter
        if model_adapter:
            # Explicitly provided adapter
            self.model_adapter = model_adapter
            self.using_tft = isinstance(model_adapter, TFTDaemonClient)
        elif use_daemon and daemon_url and REQUESTS_AVAILABLE:
            # Try to connect to daemon
            try:
                self.model_adapter = TFTDaemonClient(daemon_url)
                self.using_tft = True
                print("🤖 Dashboard Mode: REAL TFT PREDICTIONS")
            except Exception as e:
                print(f"⚠️  Failed to connect to daemon: {e}")
                print("📊 Falling back to heuristic predictions")
                self.model_adapter = ModelAdapter()
                self.using_tft = False
        else:
            # Fallback to heuristics
            self.model_adapter = ModelAdapter()
            self.using_tft = False
            if use_daemon:
                print("📊 Dashboard Mode: Heuristic Predictions (No daemon specified)")

        self.df_window = pd.DataFrame()
        self.env_prob_history = []
        self.server_order = sorted(data_source.servers)  # Stable ordering for plots

        # Dashboard state
        self.start_time = datetime.now()
        self.last_refresh = self.start_time
        self.tick_count = 0

        # Setup matplotlib
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10

        if self.config['SAVE_PLOTS']:
            Path(self.config['PLOT_DIR']).mkdir(parents=True, exist_ok=True)

    def run(self):
        """Main dashboard loop."""
        print(f"🚀 Starting TFT Monitoring Dashboard")
        print(f"   Data source: {self.data_source.data_path}")
        print(f"   Fleet size: {len(self.data_source.servers)} servers")
        print(f"   Tick interval: every {self.config['TICK_INTERVAL_SECONDS']} seconds (data ingestion)")
        print(f"   Refresh interval: every {self.config['REFRESH_SECONDS']} seconds (visualization)")
        print("=" * 60)

        try:
            last_tick_time = datetime.now()

            while True:
                # Check if it's time to ingest next data tick
                elapsed_since_tick = (datetime.now() - last_tick_time).total_seconds()

                if elapsed_since_tick >= self.config['TICK_INTERVAL_SECONDS']:
                    # Get next batch from data source
                    tick_batch = self.data_source.get_next_batch()

                    if tick_batch is None:
                        print("\n✅ Reached end of data")
                        break

                    # Update window
                    self._update_window(tick_batch)
                    self.tick_count += 1
                    last_tick_time = datetime.now()

                    # Log current phase if available
                    if 'incident_phase' in tick_batch.columns:
                        phase = tick_batch['incident_phase'].iloc[0]
                        progress = self.data_source.get_current_progress()
                        current_time = tick_batch['timestamp'].iloc[0]
                        print(f"[Tick {self.tick_count}] {current_time} | Phase: {phase} | Progress: {progress*100:.1f}%")

                # Refresh dashboard if interval elapsed
                elapsed_since_refresh = (datetime.now() - self.last_refresh).total_seconds()
                if elapsed_since_refresh >= self.config['REFRESH_SECONDS'] and not self.df_window.empty:
                    self._refresh_dashboard()
                    self.last_refresh = datetime.now()

                # Small sleep to prevent CPU spinning
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n⏹️  Dashboard stopped by user")

        self._print_summary()

    def _update_window(self, tick_batch: pd.DataFrame):
        """Update rolling window."""
        self.df_window = pd.concat([self.df_window, tick_batch], ignore_index=True)

        # Keep only recent data (rolling window)
        if not self.df_window.empty:
            cutoff_time = self.df_window['timestamp'].max() - timedelta(minutes=self.config['ROLLING_WINDOW_MIN'])
            self.df_window = self.df_window[self.df_window['timestamp'] > cutoff_time]

    def _refresh_dashboard(self):
        """Refresh all dashboard figures."""
        if self.df_window.empty:
            return

        # Get predictions
        server_risks = self.model_adapter.predict_per_server(self.df_window)
        env_probs = self.model_adapter.predict_environment(self.df_window)

        # Store environment probability history
        self.env_prob_history.append({
            'timestamp': self.df_window['timestamp'].max(),
            'prob_30m': env_probs['prob_30m'],
            'prob_8h': env_probs['prob_8h']
        })

        # Limit history size
        if len(self.env_prob_history) > self.config['MAX_ENV_POINTS']:
            self.env_prob_history = self.env_prob_history[-self.config['MAX_ENV_POINTS']:]

        # Clear and refresh display
        clear_output(wait=True)

        # Create all figures
        self._create_kpi_figure(env_probs)
        self._create_problem_servers_figure(server_risks)
        self._create_env_probability_figure()
        self._create_fleet_risk_strip(server_risks)
        self._create_rolling_metrics_figure()

    def _create_kpi_figure(self, env_probs: Dict[str, float]):
        """Figure 1: Header KPIs."""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')

        # Environment health status
        prob_30m = env_probs['prob_30m']
        if prob_30m < 0.2:
            health_status = "GOOD"
            health_color = 'green'
        elif prob_30m < 0.5:
            health_status = "WARNING"
            health_color = 'orange'
        else:
            health_status = "CRITICAL"
            health_color = 'red'

        # Fleet stats
        latest_batch = self.df_window[self.df_window['timestamp'] == self.df_window['timestamp'].max()]
        active_servers = (latest_batch['state'] != 'offline').sum()
        fleet_size = len(self.df_window['server_name'].unique())

        # Current phase
        current_phase = 'unknown'
        if 'incident_phase' in latest_batch.columns:
            current_phase = latest_batch['incident_phase'].iloc[0]

        # Layout KPIs
        kpi_text = f"""
ENVIRONMENT HEALTH: {health_status}                    INCIDENT PROBABILITY
Current Phase: {current_phase}                         30 min: {prob_30m:.1%}
Fleet: {active_servers}/{fleet_size} servers active                    8 hour: {env_probs['prob_8h']:.1%}
Current Time: {self.df_window['timestamp'].max().strftime('%H:%M:%S')}
Progress: {self.data_source.get_current_progress()*100:.1f}%                    Ticks: {self.tick_count}
"""

        ax.text(0.05, 0.5, kpi_text, fontsize=14, fontfamily='monospace',
                verticalalignment='center', transform=ax.transAxes)

        # Health status highlight
        ax.text(0.25, 0.8, health_status, fontsize=24, fontweight='bold',
                color=health_color, transform=ax.transAxes)

        plt.title('TFT Monitoring Dashboard - System Status', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

    def _create_problem_servers_figure(self, server_risks: pd.DataFrame):
        """Figure 2: Top-5 Problem Servers."""
        if server_risks.empty:
            return

        top_5 = server_risks.nlargest(5, 'risk_30m')

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(range(len(top_5)), top_5['risk_30m'],
                      color=['red' if x > 0.7 else 'orange' if x > 0.4 else 'yellow' for x in top_5['risk_30m']])

        # Add server info labels
        labels = []
        for _, row in top_5.iterrows():
            server_info = self.df_window[self.df_window['server_name'] == row['server_name']].iloc[-1]
            labels.append(f"{row['server_name']} ({server_info['profile']} | {server_info['state']})")

        ax.set_yticks(range(len(top_5)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('30-Minute Risk Score')
        ax.set_title('Top 5 Problem Servers', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)

        # Add value annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.2f}', ha='left', va='center', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def _create_env_probability_figure(self):
        """Figure 3: Environment Incident Probability Over Time."""
        if len(self.env_prob_history) < 2:
            return

        df_probs = pd.DataFrame(self.env_prob_history)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df_probs['timestamp'], df_probs['prob_30m'],
               label='30-min probability', linewidth=2, color='red', marker='o')
        ax.plot(df_probs['timestamp'], df_probs['prob_8h'],
               label='8-hour probability', linewidth=2, color='blue', marker='s')

        # Add phase boundaries if available
        if 'incident_phase' in self.df_window.columns:
            phase_changes = self.df_window[['timestamp', 'incident_phase']].drop_duplicates('incident_phase')
            for _, row in phase_changes.iterrows():
                ax.axvline(row['timestamp'], color='gray', linestyle='--', alpha=0.5)
                ax.text(row['timestamp'], 0.95, row['incident_phase'],
                       rotation=90, verticalalignment='top', fontsize=8)

        ax.set_ylabel('Incident Probability')
        ax.set_title('Environment Incident Probability Trend', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Format x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def _create_fleet_risk_strip(self, server_risks: pd.DataFrame):
        """Figure 4: Fleet Risk Strip."""
        if server_risks.empty:
            return

        # Create risk array in stable server order
        risk_array = []
        for server in self.server_order:
            risk_row = server_risks[server_risks['server_name'] == server]
            risk_value = risk_row['risk_30m'].iloc[0] if not risk_row.empty else 0.0
            risk_array.append(risk_value)

        risk_array = np.array(risk_array).reshape(1, -1)

        fig, ax = plt.subplots(figsize=(15, 3))

        im = ax.imshow(risk_array, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

        # Server labels
        ax.set_xticks(range(len(self.server_order)))
        ax.set_xticklabels([s.split('-')[0] for s in self.server_order], rotation=45, fontsize=8)
        ax.set_yticks([])
        ax.set_title('Fleet Risk Heat Map (0=Low Risk, 1=High Risk)', fontsize=14, fontweight='bold')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.1)
        cbar.set_label('Risk Level')

        plt.tight_layout()
        plt.show()

    def _create_rolling_metrics_figure(self):
        """Figure 5: Rolling Fleet Metrics."""
        if len(self.df_window) < 2:
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # Group by timestamp and calculate fleet-wide metrics
        metrics_by_time = self.df_window.groupby('timestamp').agg({
            'cpu_pct': 'median',
            'latency_ms': lambda x: x.quantile(0.95),
            'error_rate': 'mean'
        }).reset_index()

        # CPU Usage
        axes[0].plot(metrics_by_time['timestamp'], metrics_by_time['cpu_pct'],
                    color='blue', linewidth=2, marker='o', markersize=3)
        axes[0].set_ylabel('CPU % (Median)')
        axes[0].set_title('Fleet Rolling Metrics', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        axes[0].axhline(75, color='orange', linestyle='--', alpha=0.5, label='Warning')
        axes[0].axhline(90, color='red', linestyle='--', alpha=0.5, label='Critical')
        axes[0].legend(fontsize=8)

        # Latency P95
        axes[1].plot(metrics_by_time['timestamp'], metrics_by_time['latency_ms'],
                    color='orange', linewidth=2, marker='s', markersize=3)
        axes[1].set_ylabel('Latency P95 (ms)')
        axes[1].grid(True, alpha=0.3)

        # Error Rate
        axes[2].plot(metrics_by_time['timestamp'], metrics_by_time['error_rate'],
                    color='red', linewidth=2, marker='^', markersize=3)
        axes[2].set_ylabel('Error Rate (Mean)')
        axes[2].set_xlabel('Time')
        axes[2].grid(True, alpha=0.3)

        # Format x-axis
        import matplotlib.dates as mdates
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def _print_summary(self):
        """Print final dashboard summary."""
        runtime = datetime.now() - self.start_time

        print("\n" + "="*60)
        print("📊 DASHBOARD SUMMARY")
        print("="*60)
        print(f"Runtime: {runtime}")
        print(f"Ticks processed: {self.tick_count}")
        print(f"Final window size: {len(self.df_window)} records")

        if not self.df_window.empty:
            final_env_probs = self.model_adapter.predict_environment(self.df_window)
            server_risks = self.model_adapter.predict_per_server(self.df_window)

            print(f"Final environment probabilities:")
            print(f"  30-min: {final_env_probs['prob_30m']:.1%}")
            print(f"  8-hour: {final_env_probs['prob_8h']:.1%}")

            if not server_risks.empty:
                top_3 = server_risks.nlargest(3, 'risk_30m')
                print(f"Top 3 risk servers:")
                for _, row in top_3.iterrows():
                    print(f"  {row['server_name']}: {row['risk_30m']:.2f}")


# =============================================================================
# MODULE INTERFACES
# =============================================================================

def run_dashboard(data_path: str,
                 daemon_url: str = "http://localhost:8000",
                 use_daemon: bool = True,
                 data_format: str = 'auto',
                 tick_interval_sec: int = 5,
                 refresh_sec: int = 30,
                 save_plots: bool = False):
    """
    Run dashboard with file-based data source.

    Args:
        data_path: Path to data file or directory
        daemon_url: TFT inference daemon URL
        use_daemon: Whether to use TFT daemon (True) or heuristic fallback (False)
        data_format: 'csv', 'parquet', or 'auto'
        tick_interval_sec: How often to ingest new data (simulates real-time arrival)
        refresh_sec: How often to refresh dashboard visualizations
        save_plots: Whether to save plots to files
    """
    config = CONFIG.copy()
    config.update({
        'TICK_INTERVAL_SECONDS': tick_interval_sec,
        'REFRESH_SECONDS': refresh_sec,
        'SAVE_PLOTS': save_plots
    })

    data_source = DataSource(data_path, data_format)
    dashboard = LiveDashboard(
        data_source,
        daemon_url=daemon_url if use_daemon else None,
        use_daemon=use_daemon,
        config=config
    )
    dashboard.run()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="TFT Real-Time Monitoring Dashboard")
    parser.add_argument("data_path", help="Path to data file or directory")
    parser.add_argument("--daemon-url", default="http://localhost:8000",
                       help="TFT inference daemon URL (default: http://localhost:8000)")
    parser.add_argument("--no-daemon", action="store_true",
                       help="Use heuristic predictions instead of TFT daemon")
    parser.add_argument("--format", choices=['csv', 'parquet', 'auto'], default='auto',
                       help="Data format")
    parser.add_argument("--tick-interval", type=int, default=5,
                       help="Data ingestion interval in seconds (simulates real-time arrival)")
    parser.add_argument("--refresh", type=int, default=30,
                       help="Dashboard refresh interval in seconds")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save plots to files")

    args = parser.parse_args()

    try:
        run_dashboard(
            data_path=args.data_path,
            daemon_url=args.daemon_url,
            use_daemon=not args.no_daemon,
            data_format=args.format,
            tick_interval_sec=args.tick_interval,
            refresh_sec=args.refresh,
            save_plots=args.save_plots
        )
        return 0
    except Exception as e:
        print(f"❌ Dashboard failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
