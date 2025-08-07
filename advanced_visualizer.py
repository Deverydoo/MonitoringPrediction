#!/usr/bin/env python3
"""
Advanced TFT Visualizer - Stunning visualizations for 8-hour predictions
Supports both module import and CLI usage
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Enhanced styling
plt.style.use('dark_background')
sns.set_palette("viridis")

class AdvancedTFTVisualizer:
    """Advanced visualizer for TFT predictions with stunning graphics."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._load_config()
        self.setup_styling()
        
    def _load_config(self) -> Dict:
        """Load configuration with visualization settings."""
        try:
            with open('tft_config_adjusted.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        
        # Enhanced default config
        default_config = {
            "prediction_horizon": 96,  # 8 hours
            "poll_interval_minutes": 5,
            "visualization": {
                "enabled": True,
                "save_plots": True,
                "plot_dir": "./plots/",
                "dpi": 150,
                "figsize": [20, 14],
                "style": "dark_background",
                "show_confidence_intervals": True,
                "show_attention_weights": True
            },
            "alert_thresholds": {
                "cpu_percent": {"warning": 75.0, "critical": 90.0},
                "memory_percent": {"warning": 80.0, "critical": 93.0},
                "disk_percent": {"warning": 85.0, "critical": 95.0},
                "load_average": {"warning": 4.0, "critical": 8.0}
            }
        }
        
        # Merge with defaults
        for key, value in default_config.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def setup_styling(self):
        """Setup enhanced matplotlib styling."""
        plt.style.use('dark_background')
        
        # Enhanced color palette
        self.colors = {
            'primary': '#00D2FF',      # Cyan
            'secondary': '#FF0080',    # Magenta  
            'accent': '#00FF94',       # Green
            'warning': '#FFB000',      # Orange
            'critical': '#FF4757',     # Red
            'prediction': '#74B9FF',   # Light Blue
            'uncertainty': '#A29BFE',  # Purple
            'background': '#2F3542',   # Dark Gray
            'grid': '#57606F'          # Medium Gray
        }
        
        # Set default params
        plt.rcParams.update({
            'figure.facecolor': '#1E1E1E',
            'axes.facecolor': '#2F3542',
            'axes.edgecolor': '#57606F',
            'axes.linewidth': 0.8,
            'grid.color': '#57606F',
            'grid.alpha': 0.3,
            'text.color': '#FFFFFF',
            'axes.labelcolor': '#FFFFFF',
            'xtick.color': '#FFFFFF',
            'ytick.color': '#FFFFFF',
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 13,
            'legend.fontsize': 11,
            'figure.titlesize': 18
        })
    
    def create_prediction_dashboard(self, 
                                  historical_data: List[Dict],
                                  predictions: Dict[str, List[float]],
                                  alerts: List[Dict] = None,
                                  confidence_intervals: Dict = None,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Create stunning prediction dashboard."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 14), facecolor='#1E1E1E')
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25, 
                             left=0.06, right=0.94, top=0.93, bottom=0.07)
        
        # Main metrics (2x2 grid)
        metrics = ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        # Create time arrays
        hist_times = [datetime.fromisoformat(d['timestamp']) for d in historical_data[-48:]]
        pred_times = [hist_times[-1] + timedelta(minutes=5*i) for i in range(1, len(predictions[metrics[0]])+1)]
        
        for i, (metric, pos) in enumerate(zip(metrics, positions)):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            self._plot_metric_prediction(ax, metric, hist_times, pred_times, 
                                       historical_data, predictions, confidence_intervals)
        
        # Alert heatmap
        ax_heatmap = fig.add_subplot(gs[0, 2])
        self._plot_alert_heatmap(ax_heatmap, predictions, alerts)
        
        # Attention weights visualization
        ax_attention = fig.add_subplot(gs[1, 2])
        self._plot_attention_weights(ax_attention, len(hist_times), len(pred_times))
        
        # System health timeline
        ax_timeline = fig.add_subplot(gs[2, :])
        self._plot_system_health_timeline(ax_timeline, hist_times, pred_times, 
                                        historical_data, predictions, alerts)
        
        # Add title and metadata
        fig.suptitle('🔮 TFT Server Monitoring - 8 Hour Forecast', 
                    fontsize=24, fontweight='bold', color='#00D2FF', y=0.97)
        
        # Add timestamp
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        fig.text(0.02, 0.02, timestamp_text, fontsize=10, color='#A0A0A0', alpha=0.7)
        
        if save_path:
            self._save_plot(fig, save_path)
        
        return fig
    
    def _plot_metric_prediction(self, ax, metric: str, hist_times: List[datetime], 
                              pred_times: List[datetime], historical_data: List[Dict],
                              predictions: Dict, confidence_intervals: Dict = None):
        """Plot individual metric with predictions and confidence intervals."""
        
        # Extract historical values
        hist_values = []
        for d in historical_data[-len(hist_times):]:
            if metric in d:
                hist_values.append(d[metric])
            else:
                hist_values.append(np.nan)
        
        # Get predictions
        pred_values = predictions.get(metric, [])
        
        # Plot historical data
        ax.plot(hist_times, hist_values, color=self.colors['primary'], 
               linewidth=2.5, label='Historical', alpha=0.9, marker='o', markersize=3)
        
        # Plot predictions
        if pred_values:
            ax.plot(pred_times, pred_values, color=self.colors['prediction'], 
                   linewidth=3, label='Prediction', alpha=0.9, linestyle='--', marker='s', markersize=4)
        
        # Add confidence intervals
        if confidence_intervals and metric in confidence_intervals:
            lower = confidence_intervals[metric]['lower']
            upper = confidence_intervals[metric]['upper']
            ax.fill_between(pred_times, lower, upper, 
                           color=self.colors['uncertainty'], alpha=0.2, label='Uncertainty')
        
        # Add thresholds
        if metric in self.config['alert_thresholds']:
            thresholds = self.config['alert_thresholds'][metric]
            
            # Warning threshold
            ax.axhline(y=thresholds['warning'], color=self.colors['warning'], 
                      linestyle=':', alpha=0.7, linewidth=2, label='Warning')
            
            # Critical threshold  
            ax.axhline(y=thresholds['critical'], color=self.colors['critical'], 
                      linestyle=':', alpha=0.7, linewidth=2, label='Critical')
            
            # Highlight danger zones
            y_max = ax.get_ylim()[1]
            ax.fill_between(ax.get_xlim(), thresholds['warning'], thresholds['critical'],
                           color=self.colors['warning'], alpha=0.1)
            ax.fill_between(ax.get_xlim(), thresholds['critical'], y_max,
                           color=self.colors['critical'], alpha=0.1)
        
        # Styling
        ax.set_title(f'{metric.replace("_", " ").title()}', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.8)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add units
        if metric.endswith('_percent'):
            ax.set_ylabel('Percentage (%)')
        elif metric == 'load_average':
            ax.set_ylabel('Load Average')
        else:
            ax.set_ylabel('Value')
    
    def _plot_alert_heatmap(self, ax, predictions: Dict, alerts: List[Dict] = None):
        """Create alert risk heatmap."""
        metrics = list(predictions.keys())[:4]  # Top 4 metrics
        horizon = len(predictions[metrics[0]])
        
        # Create risk matrix
        risk_matrix = np.zeros((len(metrics), min(horizon, 48)))  # Max 4 hours for heatmap
        
        for i, metric in enumerate(metrics):
            values = predictions[metric][:48]  # First 4 hours
            thresholds = self.config['alert_thresholds'].get(metric, {})
            
            for j, value in enumerate(values):
                if 'critical' in thresholds and value >= thresholds['critical']:
                    risk_matrix[i, j] = 3  # Critical
                elif 'warning' in thresholds and value >= thresholds['warning']:
                    risk_matrix[i, j] = 2  # Warning
                elif value >= np.mean(values) * 1.2:
                    risk_matrix[i, j] = 1  # Elevated
                else:
                    risk_matrix[i, j] = 0  # Normal
        
        # Create heatmap
        im = ax.imshow(risk_matrix, cmap='RdYlGn_r', aspect='auto', alpha=0.8)
        
        # Labels
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels([m.replace('_', ' ').title() for m in metrics])
        
        # Time labels (every 30 minutes)
        time_ticks = range(0, min(horizon, 48), 6)
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([f'+{t//12:.1f}h' for t in time_ticks])
        
        ax.set_title('Risk Heatmap (Next 4 Hours)', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(['Normal', 'Elevated', 'Warning', 'Critical'])
    
    def _plot_attention_weights(self, ax, hist_len: int, pred_len: int):
        """Visualize attention weights (simulated)."""
        # Simulate attention weights
        attention_matrix = np.random.random((pred_len//4, hist_len//4))  # Downsample for visibility
        
        # Add realistic patterns
        for i in range(attention_matrix.shape[0]):
            for j in range(attention_matrix.shape[1]):
                # Higher attention for recent history
                recency_weight = np.exp(-(hist_len//4 - j) / (hist_len//8))
                attention_matrix[i, j] *= recency_weight
        
        # Normalize
        max_val = attention_matrix.max() if attention_matrix.size > 0 else 1.0
        if max_val > 0:
            attention_matrix = attention_matrix / max_val
        
        # Plot
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='auto', alpha=0.8)
        
        ax.set_xlabel('Historical Time Steps')
        ax.set_ylabel('Prediction Steps')
        ax.set_title('TFT Attention Weights', fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8, label='Attention Weight')
    
    def _plot_system_health_timeline(self, ax, hist_times: List[datetime], 
                                   pred_times: List[datetime], historical_data: List[Dict],
                                   predictions: Dict, alerts: List[Dict] = None):
        """Create system health timeline."""
        
        # Calculate system health score
        all_times = hist_times + pred_times
        health_scores = []
        
        # Historical health scores
        for d in historical_data[-len(hist_times):]:
            score = 100
            for metric in ['cpu_percent', 'memory_percent', 'disk_percent']:
                if metric in d and metric in self.config['alert_thresholds']:
                    value = d[metric]
                    thresholds = self.config['alert_thresholds'][metric]
                    if value >= thresholds['critical']:
                        score -= 30
                    elif value >= thresholds['warning']:
                        score -= 15
            health_scores.append(max(0, score))
        
        # Predicted health scores
        for i in range(len(pred_times)):
            score = 100
            for metric in ['cpu_percent', 'memory_percent', 'disk_percent']:
                if metric in predictions and i < len(predictions[metric]):
                    value = predictions[metric][i]
                    if metric in self.config['alert_thresholds']:
                        thresholds = self.config['alert_thresholds'][metric]
                        if value >= thresholds['critical']:
                            score -= 30
                        elif value >= thresholds['warning']:
                            score -= 15
            health_scores.append(max(0, score))
        
        # Plot health score
        colors = []
        for score in health_scores:
            if score >= 80:
                colors.append(self.colors['accent'])     # Green
            elif score >= 60:
                colors.append(self.colors['warning'])    # Orange
            else:
                colors.append(self.colors['critical'])   # Red
        
        # Split historical and predicted
        hist_scores = health_scores[:len(hist_times)]
        pred_scores = health_scores[len(hist_times):]
        
        ax.plot(hist_times, hist_scores, color=self.colors['primary'], 
               linewidth=3, label='Historical Health', marker='o', markersize=4)
        ax.plot(pred_times, pred_scores, color=self.colors['prediction'], 
               linewidth=3, label='Predicted Health', linestyle='--', marker='s', markersize=4)
        
        # Fill areas
        ax.fill_between(hist_times, hist_scores, alpha=0.3, color=self.colors['primary'])
        ax.fill_between(pred_times, pred_scores, alpha=0.3, color=self.colors['prediction'])
        
        # Add alert markers
        if alerts:
            for alert in alerts:
                alert_time = hist_times[-1] + timedelta(minutes=alert['time_ahead_minutes'])
                if alert['severity'] == 'critical':
                    ax.scatter([alert_time], [20], color=self.colors['critical'], 
                             s=200, marker='X', zorder=10, alpha=0.8)
                else:
                    ax.scatter([alert_time], [40], color=self.colors['warning'], 
                             s=150, marker='!', zorder=10, alpha=0.8)
        
        # Styling
        ax.set_ylim(0, 100)
        ax.set_ylabel('System Health Score')
        ax.set_xlabel('Time')
        ax.set_title('System Health Timeline - 8 Hour Forecast', fontweight='bold', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', framealpha=0.8)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add vertical line for current time
        ax.axvline(x=hist_times[-1], color='white', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(hist_times[-1], 90, 'Now', rotation=90, va='bottom', ha='right', 
               color='white', fontweight='bold')
    
    def _save_plot(self, fig: plt.Figure, save_path: str):
        """Save plot with high quality."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        base_path = Path(save_path).with_suffix('')
        
        fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight', 
                   facecolor='#1E1E1E', edgecolor='none')
        fig.savefig(f"{base_path}.svg", bbox_inches='tight', 
                   facecolor='#1E1E1E', edgecolor='none')
        
        print(f"✅ Plots saved to: {base_path}.[png|svg]")


# Enhanced prediction generator for 8 hours
def generate_enhanced_predictions(base_data: List[Dict], horizon: int = 96) -> Dict:
    """Generate enhanced predictions with confidence intervals."""
    
    if not base_data:
        # Generate sample data
        base_data = []
        for i in range(48):  # 4 hours of history
            timestamp = datetime.now() - timedelta(minutes=5*i)
            base_data.insert(0, {
                'timestamp': timestamp.isoformat(),
                'cpu_percent': 45 + np.random.normal(0, 8) + 10*np.sin(i/12),
                'memory_percent': 60 + np.random.normal(0, 5) + 5*np.cos(i/24),
                'disk_percent': 40 + np.random.normal(0, 3) + i*0.1,
                'load_average': 1.5 + np.random.normal(0, 0.5) + 0.5*np.sin(i/6)
            })
    
    predictions = {}
    confidence_intervals = {}
    
    metrics = ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average']
    
    for metric in metrics:
        # Get recent values
        recent_values = [d.get(metric, 50) for d in base_data[-12:]]
        current_value = recent_values[-1] if recent_values else 50
        
        # Calculate trend
        if len(recent_values) > 1:
            x = np.arange(len(recent_values))
            trend = np.polyfit(x, recent_values, 1)[0]
        else:
            trend = 0
        
        # Generate predictions with realistic patterns
        pred_values = []
        lower_bounds = []
        upper_bounds = []
        
        for i in range(1, horizon + 1):
            # Base prediction with trend
            base_pred = current_value + (trend * i)
            
            # Add cyclical patterns
            if metric == 'cpu_percent':
                # CPU has daily patterns
                daily_cycle = 10 * np.sin(2 * np.pi * i / 288)  # 24-hour cycle
                hourly_cycle = 5 * np.sin(2 * np.pi * i / 12)   # Hourly variation
                base_pred += daily_cycle + hourly_cycle
            elif metric == 'memory_percent':
                # Memory tends to grow slowly over time
                base_pred += i * 0.05
                base_pred += 3 * np.cos(2 * np.pi * i / 144)    # 12-hour cycle
            elif metric == 'load_average':
                # Load average has business hour patterns
                business_cycle = 0.8 * np.sin(2 * np.pi * (i + 180) / 288)
                base_pred += business_cycle
            
            # Add some realistic noise
            noise = np.random.normal(0, max(1, abs(trend) * 0.5))
            base_pred += noise
            
            # Bound values appropriately
            if metric.endswith('_percent'):
                base_pred = max(0, min(100, base_pred))
            elif metric == 'load_average':
                base_pred = max(0, min(20, base_pred))
            
            pred_values.append(base_pred)
            
            # Calculate confidence intervals (wider for longer horizons)
            uncertainty = 2 + (i / horizon) * 8  # Increasing uncertainty
            lower_bounds.append(max(0, base_pred - uncertainty))
            upper_bounds.append(min(100 if metric.endswith('_percent') else base_pred + uncertainty, 
                                  base_pred + uncertainty))
        
        predictions[metric] = pred_values
        confidence_intervals[metric] = {
            'lower': lower_bounds,
            'upper': upper_bounds
        }
    
    # Generate alerts
    alerts = []
    thresholds = {
        'cpu_percent': {'warning': 75.0, 'critical': 90.0},
        'memory_percent': {'warning': 80.0, 'critical': 93.0},
        'disk_percent': {'warning': 85.0, 'critical': 95.0},
        'load_average': {'warning': 4.0, 'critical': 8.0}
    }
    
    for metric, values in predictions.items():
        if metric in thresholds:
            for i, value in enumerate(values):
                if value >= thresholds[metric]['critical']:
                    alerts.append({
                        'metric': metric,
                        'severity': 'critical',
                        'value': value,
                        'time_ahead_minutes': (i + 1) * 5,
                        'threshold': thresholds[metric]['critical']
                    })
                elif value >= thresholds[metric]['warning']:
                    alerts.append({
                        'metric': metric,
                        'severity': 'warning', 
                        'value': value,
                        'time_ahead_minutes': (i + 1) * 5,
                        'threshold': thresholds[metric]['warning']
                    })
    
    return {
        'predictions': predictions,
        'confidence_intervals': confidence_intervals,
        'alerts': alerts,
        'metadata': {
            'horizon_steps': horizon,
            'horizon_hours': horizon * 5 / 60,
            'prediction_time': datetime.now().isoformat(),
            'method': 'enhanced_tft_simulation'
        }
    }


# Module interface
def visualize_predictions(historical_data: List[Dict] = None, 
                         predictions: Dict = None,
                         save_path: str = "./plots/tft_8hour_forecast") -> plt.Figure:
    """Main visualization function for module usage."""
    
    # Generate data if not provided
    if historical_data is None or predictions is None:
        result = generate_enhanced_predictions(historical_data or [])
        if predictions is None:
            predictions = result['predictions']
        confidence_intervals = result.get('confidence_intervals')
        alerts = result.get('alerts', [])
    else:
        confidence_intervals = None
        alerts = []
    
    # Create visualizer and dashboard
    visualizer = AdvancedTFTVisualizer()
    fig = visualizer.create_prediction_dashboard(
        historical_data=historical_data or [],
        predictions=predictions,
        alerts=alerts,
        confidence_intervals=confidence_intervals,
        save_path=save_path
    )
    
    return fig


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description="Advanced TFT Visualization")
    parser.add_argument("--data", type=str, help="JSON file with historical data")
    parser.add_argument("--predictions", type=str, help="JSON file with predictions")
    parser.add_argument("--output", type=str, default="./plots/tft_8hour_forecast",
                       help="Output path for plots")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    
    args = parser.parse_args()
    
    try:
        # Load data
        historical_data = None
        if args.data:
            with open(args.data, 'r') as f:
                historical_data = json.load(f)
        
        predictions = None
        if args.predictions:
            with open(args.predictions, 'r') as f:
                pred_data = json.load(f)
                predictions = pred_data.get('predictions', pred_data)
        
        # Create visualization
        fig = visualize_predictions(historical_data, predictions, args.output)
        
        if args.show:
            plt.show()
        
        print("🎨 Advanced visualization completed!")
        return 0
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())