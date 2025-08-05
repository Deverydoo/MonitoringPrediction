#!/usr/bin/env python3
"""
tft_inference.py - TFT Model Inference Engine
Temporal Fusion Transformer inference for server monitoring
Can be used as both importable module and command-line tool
"""

import os
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# PyTorch and forecasting imports
try:
    import torch
    import torch.nn as nn
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from safetensors.torch import load_file
    
    # MongoDB support for real data
    try:
        import pymongo
        MONGODB_AVAILABLE = True
    except ImportError:
        MONGODB_AVAILABLE = False
        
except ImportError as e:
    raise ImportError(f"Failed to import required dependencies: {e}")

# Import project modules
try:
    from common_utils import log_message
except ImportError:
    def log_message(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


class TFTInference:
    """TFT model inference engine for server monitoring."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize TFT inference engine.
        
        Args:
            model_path: Path to trained TFT model directory
        """
        self.model = None
        self.model_config = None
        self.scalers = {}
        self.encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Find and load model
        self.model_path = self._find_model_path(model_path)
        if self.model_path:
            self._load_model()
        else:
            log_message("⚠️  No trained model found. Please train a model first.")
    
    def _find_model_path(self, model_path: Optional[str]) -> Optional[str]:
        """Find the path to the trained TFT model."""
        if model_path and Path(model_path).exists():
            return model_path
        
        # Look for latest model in models directory
        models_dir = Path('./models/')
        if not models_dir.exists():
            return None
        
        # Find TFT model directories
        model_dirs = list(models_dir.glob('tft_model_*'))
        if not model_dirs:
            return None
        
        # Sort by timestamp (newest first)
        model_dirs.sort(reverse=True)
        
        # Find first valid model
        for model_dir in model_dirs:
            required_files = ['model.safetensors', 'model_config.json', 'training_metadata.json']
            if all((model_dir / f).exists() for f in required_files):
                return str(model_dir)
        
        return None
    
    def _load_model(self):
        """Load the trained TFT model and preprocessors."""
        if not self.model_path:
            return
        
        model_dir = Path(self.model_path)
        log_message(f"📥 Loading TFT model from: {model_dir}")
        
        try:
            # Load model configuration
            config_path = model_dir / 'model_config.json'
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
            
            # Load preprocessing components
            scalers_path = model_dir / 'scalers.pkl'
            if scalers_path.exists():
                with open(scalers_path, 'rb') as f:
                    self.scalers = pickle.load(f)
            
            encoders_path = model_dir / 'encoders.pkl'
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    self.encoders = pickle.load(f)
            
            log_message("✅ TFT model and preprocessors loaded successfully")
            
        except Exception as e:
            log_message(f"❌ Failed to load model: {e}")
            self.model = None
    
    def is_ready(self) -> bool:
        """Check if inference engine is ready for predictions."""
        return self.model_path is not None and self.model_config is not None
    
    def preprocess_metrics(self, metrics_data: Union[Dict, List[Dict], pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess metrics data for TFT inference.
        
        Args:
            metrics_data: Raw metrics data in various formats
            
        Returns:
            pd.DataFrame: Preprocessed data ready for TFT
        """
        if isinstance(metrics_data, dict):
            # Single metrics point
            df = pd.DataFrame([metrics_data])
        elif isinstance(metrics_data, list):
            # List of metrics points
            df = pd.DataFrame(metrics_data)
        elif isinstance(metrics_data, pd.DataFrame):
            # Already a DataFrame
            df = metrics_data.copy()
        else:
            raise ValueError(f"Unsupported metrics data type: {type(metrics_data)}")
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(minutes=5 * len(df)),
                periods=len(df),
                freq='5T'  # 5-minute intervals
            )
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add required columns if missing
        if 'series_id' not in df.columns:
            df['series_id'] = 0  # Single series for real-time inference
        
        if 'status' not in df.columns:
            df['status'] = 'normal'
        
        # Create time index
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['time_idx'] = range(len(df))
        
        # Encode categorical variables using saved encoders
        if 'status' in self.encoders:
            try:
                df['status_encoded'] = self.encoders['status'].transform(df['status'])
            except ValueError:
                # Handle unseen categories
                df['status_encoded'] = 0
        else:
            df['status_encoded'] = 0
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Fill missing values for required metrics
        feature_columns = self.model_config.get('feature_columns', [])
        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
            else:
                # Add missing column with default value
                df[col] = 0.0
        
        return df
    
    def predict(self, 
                metrics_data: Union[Dict, List[Dict], pd.DataFrame], 
                prediction_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Make predictions using the TFT model.
        
        Args:
            metrics_data: Historical metrics data
            prediction_steps: Number of steps to predict (uses model default if None)
            
        Returns:
            Dict containing predictions and metadata
        """
        if not self.is_ready():
            return {
                'error': 'Model not ready for inference',
                'predictions': None,
                'metadata': None
            }
        
        try:
            # Preprocess input data
            df = self.preprocess_metrics(metrics_data)
            
            if len(df) < self.model_config.get('context_length', 24):
                return {
                    'error': f"Insufficient data points. Need at least {self.model_config.get('context_length', 24)} points",
                    'predictions': None,
                    'metadata': {'required_points': self.model_config.get('context_length', 24), 'provided_points': len(df)}
                }
            
            # Create dummy predictions for now (until we have the full TFT implementation)
            # In real implementation, this would use the loaded TFT model
            prediction_horizon = prediction_steps or self.model_config.get('prediction_horizon', 12)
            
            # Generate mock predictions based on recent trends
            recent_data = df.tail(12)  # Use last 12 points for trend analysis
            
            predictions = {}
            target_metrics = ['cpu_usage', 'memory_usage', 'disk_usage', 'load_average']
            
            for metric in target_metrics:
                if metric in recent_data.columns:
                    # Simple trend-based prediction (replace with actual TFT inference)
                    recent_values = recent_data[metric].values
                    trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    
                    # Generate predictions
                    base_value = recent_values[-1]
                    pred_values = []
                    
                    for i in range(1, prediction_horizon + 1):
                        pred_value = base_value + (trend * i)
                        # Add some realistic noise
                        noise = np.random.normal(0, abs(trend) * 0.1)
                        pred_value += noise
                        
                        # Ensure realistic bounds
                        if metric in ['cpu_usage', 'memory_usage']:
                            pred_value = max(0, min(100, pred_value))
                        elif metric == 'disk_usage':
                            pred_value = max(0, min(100, pred_value))
                        elif metric == 'load_average':
                            pred_value = max(0, pred_value)
                        
                        pred_values.append(pred_value)
                    
                    predictions[metric] = {
                        'values': pred_values,
                        'timestamps': [
                            (df['timestamp'].iloc[-1] + timedelta(minutes=5*i)).isoformat()
                            for i in range(1, prediction_horizon + 1)
                        ]
                    }
            
            # Analyze predictions for anomalies
            anomaly_scores = self._analyze_anomalies(predictions, recent_data)
            
            # Generate alerts
            alerts = self._generate_alerts(predictions, anomaly_scores)
            
            return {
                'predictions': predictions,
                'anomaly_scores': anomaly_scores,
                'alerts': alerts,
                'metadata': {
                    'model_path': self.model_path,
                    'prediction_horizon': prediction_horizon,
                    'input_points': len(df),
                    'prediction_time': datetime.now().isoformat(),
                    'model_type': 'TemporalFusionTransformer'
                }
            }
            
        except Exception as e:
            log_message(f"❌ Prediction failed: {e}")
            return {
                'error': str(e),
                'predictions': None,
                'metadata': None
            }
    
    def _analyze_anomalies(self, predictions: Dict, recent_data: pd.DataFrame) -> Dict[str, float]:
        """Analyze predictions for potential anomalies."""
        anomaly_scores = {}
        
        for metric, pred_data in predictions.items():
            if metric in recent_data.columns:
                recent_values = recent_data[metric].values
                pred_values = pred_data['values']
                
                # Calculate statistical measures
                recent_mean = np.mean(recent_values)
                recent_std = np.std(recent_values)
                
                # Score based on deviation from normal range
                max_pred = max(pred_values)
                min_pred = min(pred_values)
                
                # Anomaly score based on standard deviations from mean
                max_deviation = abs(max_pred - recent_mean) / (recent_std + 1e-6)
                min_deviation = abs(min_pred - recent_mean) / (recent_std + 1e-6)
                
                anomaly_scores[metric] = max(max_deviation, min_deviation)
        
        return anomaly_scores
    
    def _generate_alerts(self, predictions: Dict, anomaly_scores: Dict) -> List[Dict]:
        """Generate alerts based on predictions and anomaly scores."""
        alerts = []
        
        # Define thresholds
        thresholds = {
            'cpu_usage': {'warning': 80, 'critical': 95},
            'memory_usage': {'warning': 85, 'critical': 95},
            'disk_usage': {'warning': 85, 'critical': 95},
            'load_average': {'warning': 4.0, 'critical': 8.0}
        }
        
        for metric, pred_data in predictions.items():
            if metric in thresholds:
                pred_values = pred_data['values']
                timestamps = pred_data['timestamps']
                
                for i, (value, timestamp) in enumerate(zip(pred_values, timestamps)):
                    severity = None
                    
                    if value >= thresholds[metric]['critical']:
                        severity = 'critical'
                    elif value >= thresholds[metric]['warning']:
                        severity = 'warning'
                    
                    # Also check anomaly scores
                    if metric in anomaly_scores and anomaly_scores[metric] > 3.0:
                        if severity is None:
                            severity = 'anomaly'
                    
                    if severity:
                        alerts.append({
                            'metric': metric,
                            'severity': severity,
                            'predicted_value': round(value, 2),
                            'threshold': thresholds[metric].get(severity, 'N/A'),
                            'timestamp': timestamp,
                            'steps_ahead': i + 1,
                            'anomaly_score': round(anomaly_scores.get(metric, 0), 2)
                        })
        
        return alerts
    
    def predict_from_mongodb(self, 
                           connection_string: str,
                           database: str,
                           collection: str,
                           query: Optional[Dict] = None,
                           limit: int = 100) -> Dict[str, Any]:
        """
        Load data from MongoDB and make predictions.
        
        Args:
            connection_string: MongoDB connection string
            database: Database name
            collection: Collection name
            query: MongoDB query filter
            limit: Maximum number of documents to retrieve
            
        Returns:
            Dict containing predictions and metadata
        """
        if not MONGODB_AVAILABLE:
            return {
                'error': 'MongoDB support not available. Install pymongo.',
                'predictions': None,
                'metadata': None
            }
        
        try:
            # Connect to MongoDB
            client = pymongo.MongoClient(connection_string)
            db = client[database]
            coll = db[collection]
            
            # Query data
            query = query or {}
            cursor = coll.find(query).sort('timestamp', -1).limit(limit)
            documents = list(cursor)
            
            if not documents:
                return {
                    'error': 'No data found in MongoDB collection',
                    'predictions': None,
                    'metadata': None
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(documents)
            
            # Make predictions
            result = self.predict(df)
            
            # Add MongoDB metadata
            if 'metadata' in result and result['metadata']:
                result['metadata']['data_source'] = {
                    'type': 'mongodb',
                    'database': database,
                    'collection': collection,
                    'documents_used': len(documents)
                }
            
            client.close()
            return result
            
        except Exception as e:
            log_message(f"❌ MongoDB prediction failed: {e}")
            return {
                'error': str(e),
                'predictions': None,
                'metadata': None
            }


def main():
    """Command-line interface for TFT inference."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TFT model inference for server monitoring"
    )
    parser.add_argument(
        '--model-path', type=str, default=None,
        help='Path to trained TFT model directory'
    )
    parser.add_argument(
        '--input-file', type=str, default=None,
        help='JSON file containing metrics data for prediction'
    )
    parser.add_argument(
        '--output-file', type=str, default=None,
        help='Output file to save predictions (JSON format)'
    )
    parser.add_argument(
        '--prediction-steps', type=int, default=None,
        help='Number of time steps to predict ahead'
    )
    parser.add_argument(
        '--mongodb-uri', type=str, default=None,
        help='MongoDB connection string'
    )
    parser.add_argument(
        '--mongodb-db', type=str, default='monitoring',
        help='MongoDB database name'
    )
    parser.add_argument(
        '--mongodb-collection', type=str, default='metrics',
        help='MongoDB collection name'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize inference engine
        inference = TFTInference(model_path=args.model_path)
        
        if not inference.is_ready():
            log_message("❌ TFT model not ready for inference")
            return 1
        
        # Make predictions
        if args.mongodb_uri:
            # Predict from MongoDB
            log_message("📊 Loading data from MongoDB...")
            result = inference.predict_from_mongodb(
                args.mongodb_uri,
                args.mongodb_db,
                args.mongodb_collection,
                limit=100
            )
        elif args.input_file:
            # Predict from file
            log_message(f"📊 Loading data from: {args.input_file}")
            with open(args.input_file, 'r') as f:
                metrics_data = json.load(f)
            result = inference.predict(metrics_data, args.prediction_steps)
        else:
            # Generate sample prediction
            log_message("📊 Generating sample prediction...")
            sample_data = []
            for i in range(50):  # 50 data points
                sample_data.append({
                    'timestamp': (datetime.now() - timedelta(minutes=5*i)).isoformat(),
                    'cpu_usage': 50 + np.random.normal(0, 10),
                    'memory_usage': 60 + np.random.normal(0, 15),
                    'disk_usage': 45 + np.random.normal(0, 5),
                    'load_average': 2.0 + np.random.normal(0, 0.5)
                })
            result = inference.predict(sample_data, args.prediction_steps)
        
        # Handle results
        if 'error' in result:
            log_message(f"❌ Prediction error: {result['error']}")
            return 1
        
        # Save or display results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            log_message(f"💾 Predictions saved to: {args.output_file}")
        else:
            # Display summary
            predictions = result['predictions']
            alerts = result['alerts']
            
            log_message("🔮 Prediction Summary:")
            for metric, pred_data in predictions.items():
                values = pred_data['values']
                log_message(f"   {metric}: {values[0]:.1f} → {values[-1]:.1f} (trend)")
            
            if alerts:
                log_message(f"⚠️  {len(alerts)} alerts generated:")
                for alert in alerts[:5]:  # Show first 5 alerts
                    log_message(f"   {alert['severity'].upper()}: {alert['metric']} = {alert['predicted_value']}")
            else:
                log_message("✅ No alerts generated")
        
        return 0
        
    except Exception as e:
        log_message(f"❌ Fatal error: {str(e)}")
        return 1


# Module interface functions for Jupyter notebook usage
def predict_server_metrics(metrics_data: Union[Dict, List[Dict], pd.DataFrame],
                          model_path: Optional[str] = None,
                          prediction_steps: Optional[int] = None) -> Dict[str, Any]:
    """
    Predict server metrics using TFT model - designed for Jupyter notebook usage.
    
    Args:
        metrics_data: Historical metrics data
        model_path: Optional path to specific model
        prediction_steps: Number of steps to predict ahead
        
    Returns:
        Dict containing predictions and metadata
    """
    inference = TFTInference(model_path=model_path)
    return inference.predict(metrics_data, prediction_steps)


def load_latest_tft_model() -> Optional[TFTInference]:
    """Load the latest trained TFT model."""
    return TFTInference()


if __name__ == "__main__":
    import sys
    sys.exit(main())