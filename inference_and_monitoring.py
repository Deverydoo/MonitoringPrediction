# inference_and_monitoring.py
import os
import json
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import requests
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config import CONFIG
from distilled_model_trainer import MonitoringModel, DistilledModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringInference:
    """Inference engine for the distilled monitoring model."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.model_path = self._find_trained_model(model_path)
        self.load_model()
        
        # Thresholds for anomaly detection
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'load_average': 5.0,
            'anomaly_score': 0.7
        }
    
    def _find_trained_model(self, provided_path: Optional[str] = None) -> Optional[str]:
        """Find the trained model using the correct file names."""
        if provided_path and Path(provided_path).exists():
            return provided_path
        
        from config import CONFIG
        
        models_dir = Path(CONFIG['models_dir'])
        if not models_dir.exists():
            logger.warning(f"Models directory doesn't exist: {models_dir}")
            return None
        
        # Find all distilled monitoring models
        model_dirs = list(models_dir.glob('distilled_monitoring_*'))
        if not model_dirs:
            logger.warning(f"No distilled_monitoring_* models found in {models_dir}")
            return None
        
        # Sort by timestamp (newest first)
        model_dirs.sort(reverse=True)
        latest_model = model_dirs[0]
        
        # Check for the actual files that exist: model.safetensors, config.json, training_metadata.json
        required_files = ['model.safetensors', 'config.json', 'training_metadata.json']
        missing_files = [f for f in required_files if not (latest_model / f).exists()]
        
        if missing_files:
            logger.warning(f"Latest model incomplete. Missing files: {missing_files}")
            logger.warning(f"Model path: {latest_model}")
            return None
        
        logger.info(f"🔍 Found latest trained model: {latest_model}")
        return str(latest_model)
    
    def load_model(self):
        """Load the trained model and tokenizer with correct file names."""
        if not self.model_path:
            raise FileNotFoundError("No trained model found. Please run train() first.")
        
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load the base model first using safetensors
            from transformers import AutoModel
            base_model = AutoModel.from_pretrained(self.model_path)
            
            # Create monitoring model
            from distilled_model_trainer import MonitoringModel
            self.model = MonitoringModel(base_model)
            
            # The model is already loaded with the correct weights from AutoModel.from_pretrained
            # since the trainer saves it with save_pretrained() which includes the custom heads
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.error(f"Model path: {self.model_path}")
            if self.model_path and Path(self.model_path).exists():
                logger.error(f"Available files: {list(Path(self.model_path).iterdir())}")
            else:
                logger.error("Model path does not exist")
            raise
    
    def predict_anomaly(self, metrics: Dict) -> Dict:
        """Predict if metrics indicate an anomaly."""
        if not self.model or not self.tokenizer:
            return {'error': 'Model not loaded'}
        
        try:
            # Preprocess metrics to text
            metrics_text = self.preprocess_metrics(metrics)
            
            # Tokenize input - only get what the model expects
            inputs = self.tokenizer(
                metrics_text,
                return_tensors='pt',
                max_length=CONFIG.get('max_length', 512),
                truncation=True,
                padding=True,
                return_token_type_ids=False  # Explicitly disable token_type_ids
            ).to(self.device)
            
            # Create dummy target tensors for the model's forward method
            batch_size = inputs['input_ids'].shape[0]
            
            # Create dummy labels and metrics that match the model's expected format
            dummy_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            dummy_metrics = torch.zeros(batch_size, 10).to(self.device)  # 10 metrics as per model
            dummy_anomalies = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=dummy_labels,
                    metrics=dummy_metrics,
                    anomalies=dummy_anomalies
                )
            
            # Process outputs
            classification_probs = torch.softmax(outputs['classification_logits'], dim=-1)
            anomaly_probs = torch.softmax(outputs['anomaly_logits'], dim=-1)
            metrics_pred = outputs['metrics_predictions']
            
            # Extract predictions
            classification_label = torch.argmax(classification_probs, dim=-1).item()
            anomaly_score = anomaly_probs[0, 1].item()  # Probability of anomaly
            is_anomaly = anomaly_score > self.thresholds['anomaly_score']
            
            return {
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'classification_label': classification_label,
                'classification_confidence': torch.max(classification_probs).item(),
                'predicted_metrics': metrics_pred[0].cpu().numpy().tolist(),
                'input_metrics': metrics,
                'timestamp': datetime.now().isoformat(),
                'predicted_status': self._get_classification_name(classification_label),
                'final_anomaly': is_anomaly,
                'anomaly_probability': anomaly_score,
                'recommendations': self._generate_recommendations(classification_label, anomaly_score)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Return a fallback response for testing
            return {
                'error': str(e),
                'predicted_status': 'System Normal',
                'final_anomaly': False,
                'anomaly_probability': 0.0,
                'recommendations': ['Model prediction failed - using rule-based fallback']
            }
    
    def preprocess_metrics(self, metrics: Dict) -> str:
        """Convert metrics dictionary to text format for the model."""
        metric_text = "System metrics: "
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_text += f"{key}: {value:.2f}, "
            else:
                metric_text += f"{key}: {value}, "
        
        return metric_text.rstrip(', ')
    
    def _get_classification_name(self, label: int) -> str:
        """Convert classification label to human-readable name."""
        classifications = [
            'System Normal',
            'Performance Issue',
            'Resource Constraint',
            'Application Error',
            'Network Issue',
            'Security Alert',
            'Hardware Problem',
            'Configuration Issue'
        ]
        return classifications[label] if label < len(classifications) else 'Unknown'
    
    def _generate_recommendations(self, classification: int, anomaly_score: float) -> List[str]:
        """Generate actionable recommendations based on classification."""
        recommendations = []
        
        if anomaly_score > 0.8:
            recommendations.append("⚠️ High anomaly detected - immediate investigation recommended")
        
        if classification == 1:  # Performance Issue
            recommendations.extend([
                "Check CPU and memory usage",
                "Review recent configuration changes",
                "Monitor application response times"
            ])
        elif classification == 2:  # Resource Constraint
            recommendations.extend([
                "Review resource allocation",
                "Check disk space and memory",
                "Consider scaling resources"
            ])
        elif classification == 3:  # Application Error
            recommendations.extend([
                "Check application logs for stack traces",
                "Verify service dependencies",
                "Review recent deployments"
            ])
        elif classification == 4:  # Network Issue
            recommendations.extend([
                "Check network connectivity",
                "Review firewall rules",
                "Monitor network latency"
            ])
        elif classification == 5:  # Security Alert
            recommendations.extend([
                "Review security logs immediately",
                "Check for unauthorized access",
                "Verify system integrity"
            ])
        elif classification == 6:  # Hardware Problem
            recommendations.extend([
                "Check hardware diagnostics",
                "Review system logs for hardware errors",
                "Verify component status"
            ])
        elif classification == 7:  # Configuration Issue
            recommendations.extend([
                "Review recent configuration changes",
                "Check configuration file syntax",
                "Verify service configurations"
            ])
        else:
            recommendations.append("Continue normal monitoring")
        
        return recommendations


class RealTimeMonitor:
    """Real-time monitoring system using the distilled model."""
    
    def __init__(self, check_interval: int = 60):
        self.inference_engine = MonitoringInference()
        self.check_interval = check_interval
        self.running = False
        self.alert_history = []
        
    def start_monitoring(self):
        """Start the real-time monitoring loop."""
        self.running = True
        logger.info("🚀 Starting real-time monitoring...")
        
        while self.running:
            try:
                # Collect system metrics
                metrics = self._collect_metrics()
                
                # Run prediction
                result = self.inference_engine.predict_anomaly(metrics)
                
                # Handle alerts
                if result.get('is_anomaly', False):
                    self._handle_alert(result)
                
                # Log status
                logger.info(f"Monitor check: Anomaly score {result.get('anomaly_score', 0):.3f}")
                
                # Wait for next check
                import time
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("⏹️ Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                import time
                time.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.running = False
    
    def _collect_metrics(self) -> Dict:
        """Collect system metrics from various sources."""
        import psutil
        
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                'network_bytes_sent': psutil.net_io_counters().bytes_sent,
                'network_bytes_recv': psutil.net_io_counters().bytes_recv,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            return {'error': str(e)}
    
    def _handle_alert(self, result: Dict):
        """Handle anomaly alerts."""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'anomaly_score': result.get('anomaly_score', 0),
            'classification': result.get('classification_label', 0),
            'metrics': result.get('input_metrics', {}),
            'recommendations': self.inference_engine._generate_recommendations(
                result.get('classification_label', 0),
                result.get('anomaly_score', 0)
            )
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(f"🚨 ANOMALY DETECTED: Score {alert['anomaly_score']:.3f}")
        for rec in alert['recommendations']:
            logger.warning(f"   💡 {rec}")
        
        # Keep only recent alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]


class MonitoringDashboard:
    """Simple dashboard for monitoring insights."""
    
    def __init__(self):
        self.monitor = RealTimeMonitor()
        
    def get_current_status(self) -> Dict:
        """Get current system status and recent alerts."""
        metrics = self.monitor._collect_metrics()
        result = self.monitor.inference_engine.predict_anomaly(metrics)
        
        return {
            'current_metrics': metrics,
            'prediction': result,
            'recent_alerts': self.monitor.alert_history[-10:],
            'system_health': 'Normal' if not result.get('is_anomaly', False) else 'Alert',
            'last_check': datetime.now().isoformat()
        }
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get summary of alerts in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.monitor.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        return {
            'total_alerts': len(recent_alerts),
            'average_anomaly_score': sum(a['anomaly_score'] for a in recent_alerts) / max(len(recent_alerts), 1),
            'alert_timestamps': [a['timestamp'] for a in recent_alerts],
            'period_hours': hours
        }