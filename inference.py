#!/usr/bin/env python3
"""
inference.py - TFT Model Inference
Run predictions using trained TFT model
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer
import numpy as np

from config import CONFIG


class TFTPredictor:
    """TFT model predictor for inference."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        if model_path:
            self.load_model(model_path)
        else:
            self._find_latest_model()
    
    def _find_latest_model(self):
        """Find and load the latest trained model."""
        models_dir = Path(CONFIG["models_dir"])
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        model_dirs = sorted(models_dir.glob("tft_model_*"), reverse=True)
        if not model_dirs:
            raise FileNotFoundError("No trained models found")
        
        latest_model = model_dirs[0] / "model.ckpt"
        print(f"📥 Loading latest model: {latest_model}")
        self.load_model(str(latest_model))
    
    def load_model(self, model_path: str):
        """Load trained TFT model."""
        try:
            self.model = TemporalFusionTransformer.load_from_checkpoint(model_path)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(self, data: Union[Dict, List[Dict]], horizon: Optional[int] = None) -> Dict[str, Any]:
        """
        Make predictions on input data.
        
        Args:
            data: Input metrics data
            horizon: Prediction horizon (uses config default if None)
            
        Returns:
            Dictionary with predictions and alerts
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        
        # Convert input to list of dicts
        if isinstance(data, dict):
            data = [data]
        
        # Simple prediction simulation (replace with actual model inference)
        predictions = self._simulate_predictions(data, horizon)
        alerts = self._generate_alerts(predictions)
        
        return {
            "predictions": predictions,
            "alerts": alerts,
            "metadata": {
                "prediction_time": datetime.now().isoformat(),
                "input_points": len(data),
                "horizon": horizon or CONFIG["prediction_horizon"]
            }
        }
    
    def _simulate_predictions(self, data: List[Dict], horizon: Optional[int]) -> Dict[str, List[float]]:
        """Simulate predictions (replace with actual model inference)."""
        horizon = horizon or CONFIG["prediction_horizon"]
        
        # Get recent values
        predictions = {}
        for metric in ["cpu_percent", "memory_percent", "disk_percent", "load_average"]:
            if data and metric in data[-1]:
                current_value = data[-1][metric]
                
                # Simple trend-based prediction
                pred_values = []
                for i in range(1, horizon + 1):
                    # Add some trend and noise
                    trend = np.random.normal(0, 2)
                    value = current_value + trend * i
                    
                    # Bound values
                    if metric.endswith("_percent"):
                        value = max(0, min(100, value))
                    else:
                        value = max(0, value)
                    
                    pred_values.append(float(value))
                
                predictions[metric] = pred_values
        
        return predictions
    
    def _generate_alerts(self, predictions: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Generate alerts based on predictions."""
        alerts = []
        thresholds = CONFIG["alert_thresholds"]
        
        for metric, values in predictions.items():
            if metric in thresholds:
                for i, value in enumerate(values):
                    if value >= thresholds[metric]["critical"]:
                        alerts.append({
                            "metric": metric,
                            "severity": "critical",
                            "value": value,
                            "threshold": thresholds[metric]["critical"],
                            "steps_ahead": i + 1,
                            "time_ahead_minutes": (i + 1) * 5
                        })
                    elif value >= thresholds[metric]["warning"]:
                        alerts.append({
                            "metric": metric,
                            "severity": "warning",
                            "value": value,
                            "threshold": thresholds[metric]["warning"],
                            "steps_ahead": i + 1,
                            "time_ahead_minutes": (i + 1) * 5
                        })
        
        return sorted(alerts, key=lambda x: (x["severity"] == "warning", x["steps_ahead"]))


# Module interface
def predict(data: Union[Dict, List[Dict]], 
           model_path: Optional[str] = None,
           horizon: Optional[int] = None) -> Dict[str, Any]:
    """Make predictions - module interface."""
    predictor = TFTPredictor(model_path)
    return predictor.predict(data, horizon)


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description="Run TFT model inference")
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--input", type=str, help="Input JSON file with metrics")
    parser.add_argument("--output", type=str, help="Output JSON file for predictions")
    parser.add_argument("--horizon", type=int, help="Prediction horizon")
    
    args = parser.parse_args()
    
    # Load input data
    if args.input:
        with open(args.input, 'r') as f:
            data = json.load(f)
    else:
        # Generate sample data
        print("📊 Using sample data...")
        data = []
        for i in range(30):
            data.append({
                "timestamp": (datetime.now() - timedelta(minutes=5*i)).isoformat(),
                "cpu_percent": 40 + np.random.normal(0, 10),
                "memory_percent": 50 + np.random.normal(0, 15),
                "disk_percent": 35 + np.random.normal(0, 5),
                "load_average": 1.5 + np.random.normal(0, 0.3)
            })
    
    # Make predictions
    try:
        results = predict(data, args.model, args.horizon)
        
        # Save or display results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Predictions saved to: {args.output}")
        else:
            print("\n🔮 Predictions:")
            for metric, values in results["predictions"].items():
                print(f"  {metric}: {values[0]:.1f} → {values[-1]:.1f}")
            
            if results["alerts"]:
                print(f"\n⚠️ {len(results['alerts'])} alerts:")
                for alert in results["alerts"][:3]:
                    print(f"  {alert['severity'].upper()}: {alert['metric']} = {alert['value']:.1f} "
                          f"in {alert['time_ahead_minutes']} minutes")
            else:
                print("\n✅ No alerts - system healthy")
        
        return 0
        
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())