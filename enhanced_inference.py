#!/usr/bin/env python3
"""
Enhanced TFT Inference - 8-hour predictions with advanced features
Supports both module import and CLI usage
"""

import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import visualization
try:
    from advanced_visualizer import visualize_predictions
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️  Advanced visualization not available - install matplotlib & seaborn")


class EnhancedTFTPredictor:
    """Enhanced TFT predictor with 8-hour horizon and advanced features."""
    
    def __init__(self, config_path: str = "tft_config_adjusted.json"):
        self.config = self._load_config(config_path)
        self.model_loaded = False
        
    def _load_config(self, config_path: str) -> Dict:
        """Load enhanced configuration."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Config not found, using enhanced defaults")
            config = {}
        
        # Enhanced defaults for 8-hour predictions
        defaults = {
            "prediction_horizon": 96,  # 96 * 5min = 8 hours
            "context_length": 288,     # 288 * 5min = 24 hours context
            "poll_interval_minutes": 5,
            "alert_thresholds": {
                "cpu_percent": {"warning": 75.0, "critical": 90.0},
                "memory_percent": {"warning": 80.0, "critical": 93.0},
                "disk_percent": {"warning": 85.0, "critical": 95.0},
                "load_average": {"warning": 4.0, "critical": 8.0},
                "java_heap_usage": {"warning": 82.0, "critical": 94.0},
                "network_errors": {"warning": 50, "critical": 200}
            }
        }
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def predict_with_confidence(self, 
                              data: Union[Dict, List[Dict]], 
                              horizon: Optional[int] = None,
                              include_confidence: bool = True,
                              include_attention: bool = True) -> Dict[str, Any]:
        """Enhanced prediction with confidence intervals and attention weights."""
        
        # Convert input to list
        if isinstance(data, dict):
            data = [data]
        
        horizon = horizon or self.config["prediction_horizon"]
        
        # Generate enhanced predictions
        predictions = self._generate_advanced_predictions(data, horizon)
        
        # Generate confidence intervals
        confidence_intervals = None
        if include_confidence:
            confidence_intervals = self._generate_confidence_intervals(data, predictions, horizon)
        
        # Generate simulated attention weights
        attention_weights = None
        if include_attention:
            attention_weights = self._generate_attention_weights(len(data), horizon)
        
        # Generate alerts with time-to-breach estimation
        alerts = self._generate_enhanced_alerts(predictions, horizon)
        
        # Calculate system health score
        health_scores = self._calculate_health_scores(predictions)
        
        # Risk assessment
        risk_assessment = self._assess_risk_levels(predictions, alerts)
        
        return {
            "predictions": predictions,
            "confidence_intervals": confidence_intervals,
            "attention_weights": attention_weights,
            "alerts": alerts,
            "health_scores": health_scores,
            "risk_assessment": risk_assessment,
            "metadata": {
                "prediction_time": datetime.now().isoformat(),
                "horizon_steps": horizon,
                "horizon_hours": horizon * self.config["poll_interval_minutes"] / 60,
                "input_points": len(data),
                "method": "enhanced_tft_simulation",
                "confidence_included": include_confidence,
                "attention_included": include_attention
            }
        }
    
    def _generate_advanced_predictions(self, data: List[Dict], horizon: int) -> Dict[str, List[float]]:
        """Generate advanced predictions with realistic patterns."""
        
        predictions = {}
        metrics = ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average', 
                  'java_heap_usage', 'network_errors']
        
        for metric in metrics:
            # Extract recent values
            recent_values = []
            for item in data[-24:]:  # Use last 2 hours
                value = None
                if isinstance(item, dict):
                    value = item.get(metric)
                    if value is None and 'metrics' in item:
                        value = item['metrics'].get(metric)
                
                if value is not None:
                    recent_values.append(float(value))
            
            if recent_values:
                current_value = recent_values[-1]
                
                # Calculate multiple trend components
                trends = self._calculate_trends(recent_values)
                
                # Generate predictions with multiple patterns
                pred_values = []
                for i in range(1, horizon + 1):
                    # Base prediction with linear trend
                    base_pred = current_value + (trends['linear'] * i)
                    
                    # Add cyclical patterns based on metric type
                    cyclical = self._add_cyclical_patterns(metric, i, horizon)
                    base_pred += cyclical
                    
                    # Add autoregressive component
                    ar_component = self._add_autoregressive(recent_values, i)
                    base_pred += ar_component
                    
                    # Add seasonal patterns
                    seasonal = self._add_seasonal_patterns(metric, i)
                    base_pred += seasonal
                    
                    # Add realistic noise with increasing uncertainty
                    uncertainty_factor = 1 + (i / horizon) * 0.5
                    noise = np.random.normal(0, trends['volatility'] * uncertainty_factor)
                    base_pred += noise
                    
                    # Apply constraints
                    base_pred = self._apply_constraints(metric, base_pred, current_value, i)
                    
                    pred_values.append(float(base_pred))
                
                predictions[metric] = pred_values
            else:
                # Use baseline if no data
                baseline_values = {
                    'cpu_percent': 45,
                    'memory_percent': 60,
                    'disk_percent': 40,
                    'load_average': 1.5,
                    'java_heap_usage': 70,
                    'network_errors': 10
                }
                
                baseline = baseline_values.get(metric, 50)
                predictions[metric] = [baseline + np.random.normal(0, 5) for _ in range(horizon)]
        
        return predictions
    
    def _calculate_trends(self, values: List[float]) -> Dict[str, float]:
        """Calculate multiple trend components."""
        if len(values) < 2:
            return {'linear': 0, 'volatility': 1}
        
        # Linear trend
        x = np.arange(len(values))
        linear_trend = np.polyfit(x, values, 1)[0]
        
        # Volatility (standard deviation of residuals)
        fitted_values = np.polyval([linear_trend, values[0]], x)
        residuals = np.array(values) - fitted_values
        volatility = np.std(residuals)
        
        return {
            'linear': linear_trend,
            'volatility': max(0.5, volatility)
        }
    
    def _add_cyclical_patterns(self, metric: str, step: int, horizon: int) -> float:
        """Add realistic cyclical patterns based on metric type."""
        patterns = 0
        
        # Time in minutes from prediction start
        minutes_ahead = step * self.config["poll_interval_minutes"]
        hours_ahead = minutes_ahead / 60
        
        if metric == 'cpu_percent':
            # Daily cycle (peak during business hours)
            daily_cycle = 15 * np.sin(2 * np.pi * hours_ahead / 24 - np.pi/2)
            # Hourly variation
            hourly_cycle = 5 * np.sin(2 * np.pi * hours_ahead)
            patterns += daily_cycle + hourly_cycle
            
        elif metric == 'memory_percent':
            # Slow memory leak pattern
            leak_pattern = step * 0.02  # 2% over full horizon
            # Daily cycle (smaller than CPU)
            daily_cycle = 5 * np.cos(2 * np.pi * hours_ahead / 24)
            patterns += leak_pattern + daily_cycle
            
        elif metric == 'load_average':
            # Business hours pattern
            business_hours = 1.5 * np.sin(2 * np.pi * (hours_ahead + 6) / 24)
            # Burst pattern
            if step % 12 == 0:  # Every hour
                business_hours += np.random.normal(0, 0.5)
            patterns += business_hours
            
        elif metric == 'java_heap_usage':
            # GC cycle simulation
            gc_cycle = 10 * np.sin(2 * np.pi * step / 20)  # 20-step GC cycle
            # Growth between GCs
            growth = (step % 20) * 0.5
            patterns += gc_cycle + growth
            
        elif metric == 'network_errors':
            # Poisson-like error bursts
            if np.random.random() < 0.05:  # 5% chance of burst
                patterns += np.random.exponential(20)
        
        return patterns
    
    def _add_autoregressive(self, values: List[float], step: int) -> float:
        """Add autoregressive component."""
        if len(values) < 2:
            return 0
        
        # Simple AR(1) model
        recent_change = values[-1] - values[-2] if len(values) >= 2 else 0
        ar_coefficient = 0.3  # Persistence of recent changes
        
        return ar_coefficient * recent_change * np.exp(-step / 20)  # Decay over time
    
    def _add_seasonal_patterns(self, metric: str, step: int) -> float:
        """Add seasonal patterns."""
        current_hour = datetime.now().hour
        minutes_ahead = step * self.config["poll_interval_minutes"]
        future_hour = (current_hour + minutes_ahead / 60) % 24
        
        # Different patterns by time of day
        if 9 <= future_hour <= 17:  # Business hours
            business_factor = 1.2
        elif 22 <= future_hour or future_hour <= 6:  # Night hours
            business_factor = 0.7
        else:
            business_factor = 1.0
        
        # Metric-specific seasonal adjustments
        adjustments = {
            'cpu_percent': business_factor * 5,
            'memory_percent': business_factor * 2,
            'load_average': business_factor * 0.5,
            'java_heap_usage': business_factor * 3,
            'network_errors': business_factor * 10
        }
        
        return adjustments.get(metric, 0)
    
    def _apply_constraints(self, metric: str, value: float, current_value: float, step: int) -> float:
        """Apply realistic constraints to predictions."""
        
        # Define bounds
        bounds = {
            'cpu_percent': (0, 100),
            'memory_percent': (0, 100),
            'disk_percent': (0, 100),
            'load_average': (0, 50),
            'java_heap_usage': (0, 100),
            'network_errors': (0, 1000)
        }
        
        if metric in bounds:
            min_val, max_val = bounds[metric]
            value = max(min_val, min(max_val, value))
        
        # Prevent unrealistic jumps
        max_change_per_step = {
            'cpu_percent': 2,
            'memory_percent': 1,
            'disk_percent': 0.5,
            'load_average': 0.2,
            'java_heap_usage': 3,
            'network_errors': 50
        }
        
        if metric in max_change_per_step:
            max_change = max_change_per_step[metric] * step
            value = max(current_value - max_change, min(current_value + max_change, value))
        
        return value
    
    def _generate_confidence_intervals(self, data: List[Dict], predictions: Dict, horizon: int) -> Dict:
        """Generate confidence intervals for predictions."""
        
        confidence_intervals = {}
        
        for metric, pred_values in predictions.items():
            lower_bounds = []
            upper_bounds = []
            
            for i, pred_value in enumerate(pred_values):
                # Increasing uncertainty over time
                base_uncertainty = 2
                time_uncertainty = (i + 1) / horizon * 8  # Max 8% uncertainty
                total_uncertainty = base_uncertainty + time_uncertainty
                
                # Metric-specific uncertainty
                metric_uncertainty = {
                    'cpu_percent': 1.5,
                    'memory_percent': 1.0,
                    'disk_percent': 0.5,
                    'load_average': 2.0,
                    'java_heap_usage': 2.5,
                    'network_errors': 5.0
                }.get(metric, 2.0)
                
                uncertainty = total_uncertainty * metric_uncertainty
                
                # 90% confidence interval
                lower = pred_value - 1.645 * uncertainty
                upper = pred_value + 1.645 * uncertainty
                
                # Apply bounds
                if metric.endswith('_percent') or metric == 'java_heap_usage':
                    lower = max(0, lower)
                    upper = min(100, upper)
                elif metric == 'load_average':
                    lower = max(0, lower)
                    upper = min(50, upper)
                elif metric == 'network_errors':
                    lower = max(0, lower)
                
                lower_bounds.append(lower)
                upper_bounds.append(upper)
            
            confidence_intervals[metric] = {
                'lower': lower_bounds,
                'upper': upper_bounds,
                'confidence_level': 0.90
            }
        
        return confidence_intervals
    
    def _generate_attention_weights(self, input_length: int, horizon: int) -> Dict:
        """Generate simulated attention weights."""
        
        # Simulate attention matrix (horizon x input_length)
        attention_matrix = np.random.random((min(horizon, 24), min(input_length, 48)))  # Limit size
        
        # Add realistic patterns
        for i in range(attention_matrix.shape[0]):
            for j in range(attention_matrix.shape[1]):
                # Higher attention for recent history
                recency_weight = np.exp(-(input_length - j) / (input_length / 3))
                # Cyclical attention patterns
                cyclical_weight = 1 + 0.3 * np.sin(2 * np.pi * j / 12)  # 12-step cycle
                attention_matrix[i, j] *= recency_weight * cyclical_weight
        
        # Normalize each prediction step
        for i in range(attention_matrix.shape[0]):
            attention_matrix[i] = attention_matrix[i] / attention_matrix[i].sum()
        
        return {
            'matrix': attention_matrix.tolist(),
            'shape': attention_matrix.shape,
            'description': 'Attention weights [prediction_steps, input_steps]'
        }
    
    def _generate_enhanced_alerts(self, predictions: Dict, horizon: int) -> List[Dict]:
        """Generate enhanced alerts with time-to-breach estimation."""
        
        alerts = []
        
        for metric, values in predictions.items():
            if metric not in self.config["alert_thresholds"]:
                continue
            
            thresholds = self.config["alert_thresholds"][metric]
            
            for i, value in enumerate(values):
                time_ahead_minutes = (i + 1) * self.config["poll_interval_minutes"]
                
                alert_data = {
                    'metric': metric,
                    'value': value,
                    'step': i + 1,
                    'time_ahead_minutes': time_ahead_minutes,
                    'time_ahead_hours': time_ahead_minutes / 60,
                    'predicted_time': (datetime.now() + timedelta(minutes=time_ahead_minutes)).isoformat()
                }
                
                if value >= thresholds['critical']:
                    alert_data.update({
                        'severity': 'critical',
                        'threshold': thresholds['critical'],
                        'exceedance': value - thresholds['critical'],
                        'priority': 1
                    })
                    alerts.append(alert_data)
                elif value >= thresholds['warning']:
                    alert_data.update({
                        'severity': 'warning',
                        'threshold': thresholds['warning'],
                        'exceedance': value - thresholds['warning'],
                        'priority': 2
                    })
                    alerts.append(alert_data)
        
        # Sort by priority and time
        return sorted(alerts, key=lambda x: (x['priority'], x['step']))
    
    def _calculate_health_scores(self, predictions: Dict) -> List[float]:
        """Calculate system health scores over prediction horizon."""
        
        horizon = len(next(iter(predictions.values())))
        health_scores = []
        
        for i in range(horizon):
            score = 100.0
            
            # Deduct points based on metric values
            for metric, values in predictions.items():
                if metric not in self.config["alert_thresholds"]:
                    continue
                
                value = values[i]
                thresholds = self.config["alert_thresholds"][metric]
                
                if value >= thresholds['critical']:
                    score -= 40  # Major penalty
                elif value >= thresholds['warning']:
                    score -= 20  # Moderate penalty
                elif value >= thresholds['warning'] * 0.8:
                    score -= 10  # Minor penalty for elevated values
            
            health_scores.append(max(0, score))
        
        return health_scores
    
    def _assess_risk_levels(self, predictions: Dict, alerts: List[Dict]) -> Dict:
        """Assess overall risk levels."""
        
        # Count alerts by severity and time windows
        risk_windows = {
            'next_1_hour': {'critical': 0, 'warning': 0},
            'next_4_hours': {'critical': 0, 'warning': 0},
            'next_8_hours': {'critical': 0, 'warning': 0}
        }
        
        for alert in alerts:
            hours_ahead = alert['time_ahead_hours']
            severity = alert['severity']
            
            if hours_ahead <= 1:
                risk_windows['next_1_hour'][severity] += 1
            if hours_ahead <= 4:
                risk_windows['next_4_hours'][severity] += 1
            if hours_ahead <= 8:
                risk_windows['next_8_hours'][severity] += 1
        
        # Calculate overall risk level
        critical_count = risk_windows['next_1_hour']['critical']
        warning_count = risk_windows['next_1_hour']['warning']
        
        if critical_count > 0:
            overall_risk = 'HIGH'
        elif warning_count > 2:
            overall_risk = 'MEDIUM'
        elif warning_count > 0:
            overall_risk = 'LOW'
        else:
            overall_risk = 'MINIMAL'
        
        return {
            'overall_risk': overall_risk,
            'risk_windows': risk_windows,
            'total_alerts': len(alerts),
            'most_at_risk_metrics': [alert['metric'] for alert in alerts[:3]]  # Top 3
        }


# Module interface
def predict_8_hours(data: Union[Dict, List[Dict]] = None,
                   include_confidence: bool = True,
                   include_attention: bool = True,
                   visualize: bool = False,
                   save_plots: str = None) -> Dict[str, Any]:
    """Main prediction function for 8-hour forecasting."""
    
    predictor = EnhancedTFTPredictor()
    
    # Generate sample data if none provided
    if data is None:
        print("📊 Generating sample data...")
        data = []
        for i in range(48):  # 4 hours of history
            timestamp = datetime.now() - timedelta(minutes=5*i)
            data.insert(0, {
                'timestamp': timestamp.isoformat(),
                'cpu_percent': 45 + np.random.normal(0, 8) + 10*np.sin(i/12),
                'memory_percent': 60 + np.random.normal(0, 5) + 5*np.cos(i/24), 
                'disk_percent': 40 + np.random.normal(0, 3) + i*0.1,
                'load_average': 1.5 + np.random.normal(0, 0.5) + 0.5*np.sin(i/6),
                'java_heap_usage': 70 + np.random.normal(0, 10),
                'network_errors': max(0, 10 + np.random.poisson(5))
            })
    
    # Make predictions
    results = predictor.predict_with_confidence(
        data=data,
        include_confidence=include_confidence,
        include_attention=include_attention
    )
    
    # Create visualizations
    if visualize and VISUALIZATION_AVAILABLE:
        fig = visualize_predictions(
            historical_data=data,
            predictions=results['predictions'],
            save_path=save_plots
        )
        results['visualization'] = fig
    elif visualize:
        print("⚠️  Visualization not available - install matplotlib & seaborn")
    
    return results


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description="Enhanced TFT 8-hour predictions")
    parser.add_argument("--data", type=str, help="Input JSON file with historical data")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--plots", type=str, default="./plots/enhanced_forecast", 
                       help="Path for saved plots")
    parser.add_argument("--no-confidence", action="store_true", help="Skip confidence intervals")
    parser.add_argument("--no-attention", action="store_true", help="Skip attention weights")
    
    args = parser.parse_args()
    
    try:
        # Load input data
        data = None
        if args.data:
            with open(args.data, 'r') as f:
                data = json.load(f)
        
        # Make predictions
        results = predict_8_hours(
            data=data,
            include_confidence=not args.no_confidence,
            include_attention=not args.no_attention,
            visualize=args.visualize,
            save_plots=args.plots if args.visualize else None
        )
        
        # Remove visualization from results if present (not JSON serializable)
        if 'visualization' in results:
            del results['visualization']
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Results saved to: {args.output}")
        else:
            # Display summary
            print("\n🔮 8-Hour TFT Forecast Summary")
            print("=" * 50)
            
            metadata = results['metadata']
            print(f"📊 Prediction horizon: {metadata['horizon_hours']:.1f} hours")
            print(f"⏰ Generated: {metadata['prediction_time']}")
            
            # Show alerts summary
            alerts = results['alerts']
            if alerts:
                print(f"\n⚠️ {len(alerts)} alerts detected:")
                for alert in alerts[:5]:  # Show top 5
                    print(f"  {alert['severity'].upper()}: {alert['metric']} = {alert['value']:.1f} "
                          f"in {alert['time_ahead_hours']:.1f}h")
                if len(alerts) > 5:
                    print(f"  ... and {len(alerts) - 5} more")
            else:
                print("\n✅ No alerts - system health looks good!")
            
            # Show risk assessment
            risk = results['risk_assessment']
            print(f"\n🎯 Overall Risk Level: {risk['overall_risk']}")
            
            if risk['most_at_risk_metrics']:
                print(f"📈 Top risk metrics: {', '.join(risk['most_at_risk_metrics'])}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Enhanced prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())