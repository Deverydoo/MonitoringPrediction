# inference_and_monitoring.py
import os
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Setup basic logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Framework detection
try:
    from config import CONFIG, FRAMEWORK_BACKEND
except ImportError:
    FRAMEWORK_BACKEND = os.environ.get('ML_FRAMEWORK', 'pytorch').lower()

# Framework-specific imports
if FRAMEWORK_BACKEND == 'tensorflow':
    try:
        import tensorflow as tf
        from tensorflow import keras
        from transformers import TFAutoModel, AutoTokenizer
        FRAMEWORK_AVAILABLE = True
        logger.info("🔥 TensorFlow inference backend loaded")
    except ImportError as e:
        logger.warning(f"TensorFlow import failed: {e}, falling back to PyTorch")
        FRAMEWORK_AVAILABLE = False
        FRAMEWORK_BACKEND = 'pytorch'
else:
    FRAMEWORK_BACKEND = 'pytorch'

if FRAMEWORK_BACKEND == 'pytorch':
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        FRAMEWORK_AVAILABLE = True
        logger.info("🔥 PyTorch inference backend loaded")
    except ImportError as e:
        logger.error(f"Both TensorFlow and PyTorch import failed: {e}")
        FRAMEWORK_AVAILABLE = False

class MonitoringInference:
    """Enhanced inference engine supporting both PyTorch and TensorFlow"""
    
    def __init__(self, model_path: Optional[str] = None, framework: Optional[str] = None):
        self.framework = framework or FRAMEWORK_BACKEND
        self.model = None
        self.tokenizer = None
        self.model_path = self._find_trained_model(model_path)
        
        # Framework-specific device setup
        if self.framework == 'tensorflow':
            # Check for GPU availability
            gpus = tf.config.experimental.list_physical_devices('GPU')
            self.device = "/GPU:0" if gpus else "/CPU:0"
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model if available
        if self.model_path:
            self.load_model()
        else:
            logger.warning("No trained model found - using rule-based fallback")
        
        # Thresholds for anomaly detection (will be dynamically adjusted)
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'load_average': 5.0,
            'java_heap_usage': 85.0,
            'java_gc_time': 15.0,
            'network_io_rate': 80.0,
            'disk_io_rate': 75.0,
            'anomaly_score': 0.7
        }
        
        # Continual learning components
        self.feedback_history = []
        self.performance_metrics = {
            'predictions': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy': 0.0
        }
    
    def _find_trained_model(self, provided_path: Optional[str] = None) -> Optional[str]:
        """Find the trained model using framework-aware detection."""
        if provided_path and Path(provided_path).exists():
            return provided_path
        
        try:
            from config import CONFIG
            models_dir = Path(CONFIG['models_dir'])
        except ImportError:
            models_dir = Path('./models/')
        
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
        
        # Check for framework-specific files
        if self.framework == 'tensorflow':
            required_files = ['tf_model', 'config.json', 'training_metadata.json']
        else:
            required_files = ['config.json', 'training_metadata.json']
        
        missing_files = [f for f in required_files if not (latest_model / f).exists()]
        
        if missing_files:
            logger.warning(f"Latest model incomplete. Missing files: {missing_files}")
            logger.warning(f"Model path: {latest_model}")
            return None
        
        logger.info(f"🔍 Found latest trained model: {latest_model}")
        return str(latest_model)
    
    def load_model(self):
        """Load the trained model with framework detection."""
        if not self.model_path:
            raise FileNotFoundError("No trained model found. Please run train() first.")
        
        try:
            logger.info(f"Loading {self.framework} model from: {self.model_path}")
            
            # Load tokenizer (common for both frameworks)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            if self.framework == 'tensorflow':
                self._load_tensorflow_model()
            else:
                self._load_pytorch_model()
            
            logger.info(f"✅ {self.framework.title()} model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {self.framework} model: {e}")
            logger.error(f"Model path: {self.model_path}")
            if self.model_path and Path(self.model_path).exists():
                logger.error(f"Available files: {list(Path(self.model_path).iterdir())}")
            else:
                logger.error("Model path does not exist")
            raise
    
    def _load_tensorflow_model(self):
        """Load TensorFlow model."""
        try:
            # Load the saved TensorFlow model
            tf_model_path = Path(self.model_path) / 'tf_model'
            self.model = keras.models.load_model(str(tf_model_path))
            
            # Set device context
            with tf.device(self.device):
                # Warm up the model
                dummy_input = {
                    'input_ids': tf.random.uniform((1, 512), maxval=1000, dtype=tf.int32),
                    'attention_mask': tf.ones((1, 512), dtype=tf.int32)
                }
                _ = self.model(dummy_input)
            
            logger.info(f"🔥 TensorFlow model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"TensorFlow model loading failed: {e}")
            raise
    
    def _load_pytorch_model(self):
        """Load PyTorch model."""
        try:
            # Load the base model
            base_model = AutoModel.from_pretrained(self.model_path)
            
            # Recreate the monitoring model
            from distilled_model_trainer import MonitoringModel
            self.model = MonitoringModel(base_model)
            
            # Load custom heads if they exist
            custom_heads_path = Path(self.model_path) / 'custom_heads.pt'
            if custom_heads_path.exists():
                heads_data = torch.load(custom_heads_path, map_location=self.device)
                self.model.classifier.load_state_dict(heads_data['classifier_state_dict'])
                self.model.anomaly_detector.load_state_dict(heads_data['anomaly_detector_state_dict'])
                self.model.metrics_regressor.load_state_dict(heads_data['metrics_regressor_state_dict'])
                logger.info("📦 Custom task heads loaded")
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"🔥 PyTorch model loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"PyTorch model loading failed: {e}")
            raise
    
    def predict_anomaly(self, metrics: Dict) -> Dict:
        """Predict if metrics indicate an anomaly with framework-specific inference."""
        if not self.model or not self.tokenizer:
            return self._rule_based_prediction(metrics)
        
        try:
            # Preprocess metrics to text
            metrics_text = self.preprocess_metrics(metrics)
            
            # Framework-specific prediction
            if self.framework == 'tensorflow':
                prediction = self._predict_tensorflow(metrics_text, metrics)
            else:
                prediction = self._predict_pytorch(metrics_text, metrics)
            
            # Update performance tracking
            self.performance_metrics['predictions'] += 1
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Fallback to rule-based prediction
            return self._rule_based_prediction(metrics)
    
    def _predict_tensorflow(self, metrics_text: str, original_metrics: Dict) -> Dict:
        """TensorFlow-specific prediction logic."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                metrics_text,
                return_tensors='tf',
                max_length=CONFIG.get('max_length', 512),
                truncation=True,
                padding=True
            )
            
            # Run inference
            with tf.device(self.device):
                outputs = self.model(inputs, training=False)
            
            # Process outputs
            classification_probs = tf.nn.softmax(outputs['classification_logits'], axis=-1)
            anomaly_probs = tf.nn.softmax(outputs['anomaly_logits'], axis=-1)
            metrics_pred = outputs['metrics_predictions']
            
            # Extract predictions
            classification_label = tf.argmax(classification_probs, axis=-1).numpy()[0]
            anomaly_score = anomaly_probs[0, 1].numpy()  # Probability of anomaly
            is_anomaly = anomaly_score > self.thresholds['anomaly_score']
            
            return {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(anomaly_score),
                'classification_label': int(classification_label),
                'classification_confidence': float(tf.reduce_max(classification_probs).numpy()),
                'predicted_metrics': metrics_pred[0].numpy().tolist(),
                'input_metrics': original_metrics,
                'timestamp': datetime.now().isoformat(),
                'predicted_status': self._get_classification_name(classification_label),
                'final_anomaly': bool(is_anomaly),
                'anomaly_probability': float(anomaly_score),
                'recommendations': self._generate_recommendations(classification_label, anomaly_score),
                'framework': 'tensorflow'
            }
            
        except Exception as e:
            logger.error(f"TensorFlow prediction error: {e}")
            return self._rule_based_prediction(original_metrics)
    
    def _predict_pytorch(self, metrics_text: str, original_metrics: Dict) -> Dict:
        """PyTorch-specific prediction logic."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                metrics_text,
                return_tensors='pt',
                max_length=CONFIG.get('max_length', 512),
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Create dummy target tensors for the model's forward method
            batch_size = inputs['input_ids'].shape[0]
            dummy_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            dummy_metrics = torch.zeros(batch_size, 10).to(self.device)
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
                'input_metrics': original_metrics,
                'timestamp': datetime.now().isoformat(),
                'predicted_status': self._get_classification_name(classification_label),
                'final_anomaly': is_anomaly,
                'anomaly_probability': anomaly_score,
                'recommendations': self._generate_recommendations(classification_label, anomaly_score),
                'framework': 'pytorch'
            }
            
        except Exception as e:
            logger.error(f"PyTorch prediction error: {e}")
            return self._rule_based_prediction(original_metrics)
    
    def _rule_based_prediction(self, metrics: Dict) -> Dict:
        """Enhanced rule-based fallback prediction."""
        cpu = metrics.get('cpu_usage', 0)
        memory = metrics.get('memory_usage', 0)
        disk = metrics.get('disk_usage', 0)
        load = metrics.get('load_average', 0)
        java_heap = metrics.get('java_heap_usage', 0)
        
        # Enhanced rule-based logic with multiple severity levels
        anomaly_score = 0.0
        status = "System Normal"
        recommendations = ["Continue normal monitoring"]
        anomaly_reasons = []
        
        # CPU analysis
        if cpu > 95:
            anomaly_score = max(anomaly_score, 0.95)
            status = "Critical Performance Issue"
            recommendations.extend(["Immediate CPU investigation required", "Check for runaway processes"])
            anomaly_reasons.append(f"Critical CPU usage: {cpu}%")
        elif cpu > 90:
            anomaly_score = max(anomaly_score, 0.9)
            status = "Performance Issue"
            recommendations.extend(["Check CPU-intensive processes", "Review recent deployments"])
            anomaly_reasons.append(f"High CPU usage: {cpu}%")
        elif cpu > 80:
            anomaly_score = max(anomaly_score, 0.7)
            status = "Performance Warning"
            recommendations.append("Monitor CPU usage closely")
            anomaly_reasons.append(f"Elevated CPU usage: {cpu}%")
        
        # Memory analysis
        if memory > 95:
            anomaly_score = max(anomaly_score, 0.95)
            status = "Critical Resource Constraint"
            recommendations.extend(["Immediate memory investigation", "Check for memory leaks"])
            anomaly_reasons.append(f"Critical memory usage: {memory}%")
        elif memory > 90:
            anomaly_score = max(anomaly_score, 0.85)
            status = "Resource Constraint"
            recommendations.extend(["Check memory usage", "Look for memory leaks"])
            anomaly_reasons.append(f"High memory usage: {memory}%")
        elif memory > 80:
            anomaly_score = max(anomaly_score, 0.6)
            status = "Resource Warning"
            recommendations.append("Monitor memory usage")
            anomaly_reasons.append(f"Elevated memory usage: {memory}%")
        
        # Disk analysis
        if disk > 95:
            anomaly_score = max(anomaly_score, 0.9)
            status = "Critical Storage Issue"
            recommendations.extend(["Immediate disk cleanup required", "Check log rotation"])
            anomaly_reasons.append(f"Critical disk usage: {disk}%")
        elif disk > 90:
            anomaly_score = max(anomaly_score, 0.8)
            status = "Resource Constraint"
            recommendations.extend(["Clean up disk space", "Review log rotation"])
            anomaly_reasons.append(f"High disk usage: {disk}%")
        
        # Load average analysis
        if load > 10:
            anomaly_score = max(anomaly_score, 0.8)
            status = "Performance Issue"
            recommendations.extend(["Check system load", "Review running processes"])
            anomaly_reasons.append(f"Very high load average: {load}")
        elif load > 8:
            anomaly_score = max(anomaly_score, 0.75)
            status = "Performance Issue"
            recommendations.append("Monitor system load")
            anomaly_reasons.append(f"High load average: {load}")
        
        # Java heap analysis
        if java_heap > 95:
            anomaly_score = max(anomaly_score, 0.9)
            status = "Application Issue"
            recommendations.extend(["Check Java heap settings", "Analyze heap dumps"])
            anomaly_reasons.append(f"Critical Java heap usage: {java_heap}%")
        elif java_heap > 85:
            anomaly_score = max(anomaly_score, 0.7)
            status = "Application Warning"
            recommendations.append("Monitor Java heap usage")
            anomaly_reasons.append(f"High Java heap usage: {java_heap}%")
        
        # Combined analysis for multiple issues
        if len(anomaly_reasons) > 1:
            anomaly_score = min(anomaly_score + 0.1, 1.0)  # Boost score for multiple issues
            status = "Multiple System Issues"
            recommendations.insert(0, "Multiple performance issues detected - prioritize investigation")
        
        return {
            'predicted_status': status,
            'final_anomaly': anomaly_score > self.thresholds['anomaly_score'],
            'anomaly_probability': anomaly_score,
            'recommendations': recommendations,
            'anomaly_reasons': anomaly_reasons,
            'model_type': 'rule_based_fallback',
            'is_anomaly': anomaly_score > self.thresholds['anomaly_score'],
            'anomaly_score': anomaly_score,
            'classification_label': 1 if anomaly_score > 0.5 else 0,
            'classification_confidence': min(anomaly_score + 0.2, 1.0),
            'input_metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'framework': 'rule_based'
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
                "Check CPU and memory usage trends",
                "Review recent configuration changes",
                "Monitor application response times",
                "Consider scaling resources if needed"
            ])
        elif classification == 2:  # Resource Constraint
            recommendations.extend([
                "Review resource allocation and limits",
                "Check disk space and memory utilization",
                "Consider horizontal or vertical scaling",
                "Implement resource cleanup policies"
            ])
        elif classification == 3:  # Application Error
            recommendations.extend([
                "Check application logs for stack traces",
                "Verify service dependencies and health",
                "Review recent deployments and changes",
                "Test application functionality"
            ])
        elif classification == 4:  # Network Issue
            recommendations.extend([
                "Check network connectivity and latency",
                "Review firewall rules and security groups",
                "Monitor network traffic patterns",
                "Verify DNS resolution"
            ])
        elif classification == 5:  # Security Alert
            recommendations.extend([
                "Review security logs immediately",
                "Check for unauthorized access attempts",
                "Verify system integrity and compliance",
                "Update security policies if needed"
            ])
        elif classification == 6:  # Hardware Problem
            recommendations.extend([
                "Check hardware diagnostics and health",
                "Review system logs for hardware errors",
                "Verify component status and temperatures",
                "Consider hardware replacement if needed"
            ])
        elif classification == 7:  # Configuration Issue
            recommendations.extend([
                "Review recent configuration changes",
                "Check configuration file syntax and validity",
                "Verify service configurations and settings",
                "Test configuration in staging environment"
            ])
        else:
            recommendations.append("Continue normal monitoring")
        
        return recommendations
    
    def update_thresholds(self, feedback: Dict):
        """Update thresholds based on feedback for continual learning."""
        if not CONFIG.get('continual_learning_enabled', True):
            return
        
        # Store feedback for learning
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'metrics': feedback.get('metrics', {}),
            'predicted_anomaly': feedback.get('predicted_anomaly', False),
            'actual_anomaly': feedback.get('actual_anomaly', False),
            'user_feedback': feedback.get('user_feedback', 'unknown')
        }
        
        self.feedback_history.append(feedback_entry)
        
        # Limit feedback history
        max_history = CONFIG.get('feedback_retention_days', 30) * 24  # Assuming hourly feedback
        if len(self.feedback_history) > max_history:
            self.feedback_history = self.feedback_history[-max_history:]
        
        # Update performance metrics
        predicted = feedback.get('predicted_anomaly', False)
        actual = feedback.get('actual_anomaly', False)
        
        if predicted == actual:
            self.performance_metrics['correct_predictions'] += 1
        elif predicted and not actual:
            self.performance_metrics['false_positives'] += 1
        elif not predicted and actual:
            self.performance_metrics['false_negatives'] += 1
        
        # Calculate accuracy
        total_predictions = self.performance_metrics['predictions']
        if total_predictions > 0:
            self.performance_metrics['accuracy'] = (
                self.performance_metrics['correct_predictions'] / total_predictions
            )
        
        # Adjust thresholds if auto-adjustment is enabled
        if CONFIG.get('auto_threshold_adjustment', True):
            self._adjust_thresholds()
    
    def _adjust_thresholds(self):
        """Automatically adjust thresholds based on performance metrics."""
        if len(self.feedback_history) < 10:  # Need minimum feedback
            return
        
        recent_feedback = self.feedback_history[-50:]  # Last 50 entries
        false_positive_rate = len([f for f in recent_feedback 
                                 if f['predicted_anomaly'] and not f['actual_anomaly']]) / len(recent_feedback)
        false_negative_rate = len([f for f in recent_feedback 
                                 if not f['predicted_anomaly'] and f['actual_anomaly']]) / len(recent_feedback)
        
        adjustment_rate = CONFIG.get('threshold_adjustment_rate', 0.05)
        
        # Adjust anomaly score threshold
        if false_positive_rate > 0.1:  # Too many false positives
            self.thresholds['anomaly_score'] = min(0.9, self.thresholds['anomaly_score'] + adjustment_rate)
            logger.info(f"Increased anomaly threshold to {self.thresholds['anomaly_score']:.3f} (reducing false positives)")
        elif false_negative_rate > 0.1:  # Too many false negatives
            self.thresholds['anomaly_score'] = max(0.3, self.thresholds['anomaly_score'] - adjustment_rate)
            logger.info(f"Decreased anomaly threshold to {self.thresholds['anomaly_score']:.3f} (reducing false negatives)")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        return {
            **self.performance_metrics,
            'thresholds': self.thresholds.copy(),
            'feedback_count': len(self.feedback_history),
            'framework': self.framework,
            'model_loaded': self.model is not None
        }


class RealTimeMonitor:
    """Enhanced real-time monitoring system with continual learning."""
    
    def __init__(self, check_interval: int = 60, framework: Optional[str] = None):
        self.inference_engine = MonitoringInference(framework=framework)
        self.check_interval = check_interval
        self.running = False
        self.alert_history = []
        self.metrics_history = []
        
        # Enhanced monitoring capabilities
        self.performance_baseline = {}
        self.trend_analysis = {}
        self.alert_suppression = {}
        
    def start_monitoring(self):
        """Start the enhanced real-time monitoring loop."""
        self.running = True
        logger.info("🚀 Starting enhanced real-time monitoring...")
        logger.info(f"🔧 Using {self.inference_engine.framework} framework")
        
        while self.running:
            try:
                # Collect system metrics
                metrics = self._collect_enhanced_metrics()
                
                # Run prediction with trend analysis
                result = self._analyze_with_trends(metrics)
                
                # Handle alerts with intelligent suppression
                if result.get('is_anomaly', False):
                    self._handle_intelligent_alert(result)
                
                # Update baselines and trends
                self._update_performance_baseline(metrics)
                self._update_trend_analysis(metrics, result)
                
                # Log status with more detail
                logger.info(f"Monitor check: Anomaly score {result.get('anomaly_score', 0):.3f}, "
                          f"Status: {result.get('predicted_status', 'Unknown')}")
                
                # Store metrics history for trend analysis
                self.metrics_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': metrics,
                    'result': result
                })
                
                # Limit history size
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
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
    
    def _collect_enhanced_metrics(self) -> Dict:
        """Collect enhanced system metrics from various sources."""
        try:
            import psutil
            
            # Basic system metrics
            metrics = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                'network_bytes_sent': psutil.net_io_counters().bytes_sent,
                'network_bytes_recv': psutil.net_io_counters().bytes_recv,
                'timestamp': datetime.now().isoformat()
            }
            
            # Enhanced metrics
            try:
                # CPU per-core usage
                cpu_per_core = psutil.cpu_percent(percpu=True)
                metrics['cpu_core_max'] = max(cpu_per_core)
                metrics['cpu_core_avg'] = sum(cpu_per_core) / len(cpu_per_core)
                
                # Memory details
                memory = psutil.virtual_memory()
                metrics['memory_available_gb'] = memory.available / (1024**3)
                metrics['memory_used_gb'] = memory.used / (1024**3)
                
                # Disk I/O
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics['disk_read_bytes'] = disk_io.read_bytes
                    metrics['disk_write_bytes'] = disk_io.write_bytes
                
                # Network I/O rates (calculated from previous measurement)
                if hasattr(self, '_prev_network_metrics'):
                    time_diff = 1  # Assuming 1-second interval
                    prev = self._prev_network_metrics
                    metrics['network_in_rate'] = (metrics['network_bytes_recv'] - prev['network_bytes_recv']) / time_diff
                    metrics['network_out_rate'] = (metrics['network_bytes_sent'] - prev['network_bytes_sent']) / time_diff
                
                self._prev_network_metrics = {
                    'network_bytes_recv': metrics['network_bytes_recv'],
                    'network_bytes_sent': metrics['network_bytes_sent']
                }
                
            except Exception as e:
                logger.debug(f"Enhanced metrics collection failed: {e}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _analyze_with_trends(self, metrics: Dict) -> Dict:
        """Analyze metrics with trend consideration."""
        # Run basic prediction
        result = self.inference_engine.predict_anomaly(metrics)
        
        # Add trend analysis
        if len(self.metrics_history) > 5:
            trend_info = self._calculate_trends(metrics)
            result.update(trend_info)
            
            # Adjust anomaly score based on trends
            if trend_info.get('trend_severity', 0) > 0.5:
                result['anomaly_score'] = min(1.0, result.get('anomaly_score', 0) + 0.1)
                result['recommendations'].append("Negative trend detected - monitor closely")
        
        return result
    
    def _calculate_trends(self, current_metrics: Dict) -> Dict:
        """Calculate trend information for key metrics."""
        if len(self.metrics_history) < 5:
            return {'trends': {}, 'trend_severity': 0}
        
        trends = {}
        trend_severity = 0
        
        # Get recent history for trend calculation
        recent_history = self.metrics_history[-10:]
        key_metrics = ['cpu_percent', 'memory_percent', 'disk_percent', 'load_average']
        
        for metric in key_metrics:
            values = [h['metrics'].get(metric, 0) for h in recent_history if metric in h['metrics']]
            if len(values) >= 3:
                # Simple trend calculation (positive = increasing)
                trend = (values[-1] - values[0]) / len(values)
                trends[metric] = {
                    'trend': trend,
                    'direction': 'increasing' if trend > 0 else 'decreasing',
                    'magnitude': abs(trend)
                }
                
                # Calculate severity for concerning trends
                if metric in ['cpu_percent', 'memory_percent', 'disk_percent'] and trend > 5:
                    trend_severity = max(trend_severity, min(trend / 20, 1.0))
        
        return {
            'trends': trends,
            'trend_severity': trend_severity
        }
    
    def _handle_intelligent_alert(self, result: Dict):
        """Handle alerts with intelligent suppression and escalation."""
        alert_key = result.get('predicted_status', 'unknown')
        current_time = datetime.now()
        
        # Check for alert suppression
        if alert_key in self.alert_suppression:
            last_alert_time = self.alert_suppression[alert_key]['last_alert']
            suppression_window = timedelta(minutes=30)  # 30-minute suppression window
            
            if current_time - last_alert_time < suppression_window:
                # Increment suppressed count but don't create new alert
                self.alert_suppression[alert_key]['suppressed_count'] += 1
                return
        
        # Create alert
        alert = {
            'timestamp': current_time.isoformat(),
            'anomaly_score': result.get('anomaly_score', 0),
            'classification': result.get('predicted_status', 'Unknown'),
            'metrics': result.get('input_metrics', {}),
            'recommendations': result.get('recommendations', []),
            'framework': result.get('framework', 'unknown'),
            'trends': result.get('trends', {}),
            'severity': self._calculate_alert_severity(result)
        }
        
        self.alert_history.append(alert)
        
        # Update suppression tracking
        self.alert_suppression[alert_key] = {
            'last_alert': current_time,
            'suppressed_count': 0
        }
        
        # Log alert with severity-based formatting
        severity = alert['severity']
        if severity >= 0.8:
            logger.error(f"🚨 CRITICAL ANOMALY: {alert['classification']} (Score: {alert['anomaly_score']:.3f})")
        elif severity >= 0.6:
            logger.warning(f"⚠️ HIGH ANOMALY: {alert['classification']} (Score: {alert['anomaly_score']:.3f})")
        else:
            logger.info(f"📊 ANOMALY DETECTED: {alert['classification']} (Score: {alert['anomaly_score']:.3f})")
        
        # Log recommendations
        for rec in alert['recommendations'][:3]:  # Top 3 recommendations
            logger.info(f"   💡 {rec}")
        
        # Keep only recent alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    def _calculate_alert_severity(self, result: Dict) -> float:
        """Calculate alert severity based on multiple factors."""
        base_score = result.get('anomaly_score', 0)
        
        # Factors that increase severity
        severity_multiplier = 1.0
        
        # Classification-based adjustment
        classification = result.get('classification_label', 0)
        if classification in [5, 6]:  # Security or Hardware issues
            severity_multiplier += 0.2
        elif classification in [1, 2]:  # Performance or Resource issues
            severity_multiplier += 0.1
        
        # Trend-based adjustment
        trend_severity = result.get('trend_severity', 0)
        severity_multiplier += trend_severity * 0.3
        
        # Multiple metrics affected
        affected_metrics = len([m for m, v in result.get('input_metrics', {}).items() 
                              if isinstance(v, (int, float)) and v > 80])
        if affected_metrics > 2:
            severity_multiplier += 0.1
        
        return min(1.0, base_score * severity_multiplier)
    
    def _update_performance_baseline(self, metrics: Dict):
        """Update performance baseline for trend analysis."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key != 'timestamp':
                if key not in self.performance_baseline:
                    self.performance_baseline[key] = {'values': [], 'avg': 0, 'std': 0}
                
                baseline = self.performance_baseline[key]
                baseline['values'].append(value)
                
                # Keep only last 100 values for baseline
                if len(baseline['values']) > 100:
                    baseline['values'] = baseline['values'][-100:]
                
                # Update statistics
                baseline['avg'] = np.mean(baseline['values'])
                baseline['std'] = np.std(baseline['values'])
    
    def _update_trend_analysis(self, metrics: Dict, result: Dict):
        """Update trend analysis data."""
        # This could be expanded for more sophisticated trend analysis
        pass
    
    def provide_feedback(self, metrics: Dict, predicted_anomaly: bool, actual_anomaly: bool, user_feedback: str = ""):
        """Provide feedback to the inference engine for continual learning."""
        feedback = {
            'metrics': metrics,
            'predicted_anomaly': predicted_anomaly,
            'actual_anomaly': actual_anomaly,
            'user_feedback': user_feedback
        }
        
        self.inference_engine.update_thresholds(feedback)
        logger.info(f"📚 Feedback provided: Predicted={predicted_anomaly}, Actual={actual_anomaly}")


class MonitoringDashboard:
    """Enhanced dashboard for monitoring insights with framework awareness."""
    
    def __init__(self, framework: Optional[str] = None):
        self.monitor = RealTimeMonitor(framework=framework)
        
    def get_current_status(self) -> Dict:
        """Get current system status and recent alerts."""
        metrics = self.monitor._collect_enhanced_metrics()
        result = self.monitor.inference_engine.predict_anomaly(metrics)
        
        return {
            'current_metrics': metrics,
            'prediction': result,
            'recent_alerts': self.monitor.alert_history[-10:],
            'system_health': 'Normal' if not result.get('is_anomaly', False) else 'Alert',
            'last_check': datetime.now().isoformat(),
            'framework': self.monitor.inference_engine.framework,
            'performance_metrics': self.monitor.inference_engine.get_performance_metrics(),
            'baseline_metrics': self.monitor.performance_baseline
        }
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get summary of alerts in the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.monitor.alert_history
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]
        
        # Alert severity distribution
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        for alert in recent_alerts:
            severity = alert.get('severity', 0)
            if severity >= 0.8:
                severity_counts['critical'] += 1
            elif severity >= 0.6:
                severity_counts['high'] += 1
            elif severity >= 0.4:
                severity_counts['medium'] += 1
            else:
                severity_counts['low'] += 1
        
        return {
            'total_alerts': len(recent_alerts),
            'severity_distribution': severity_counts,
            'average_anomaly_score': sum(a['anomaly_score'] for a in recent_alerts) / max(len(recent_alerts), 1),
            'alert_timestamps': [a['timestamp'] for a in recent_alerts],
            'period_hours': hours,
            'most_common_issues': self._get_most_common_issues(recent_alerts)
        }
    
    def _get_most_common_issues(self, alerts: List[Dict]) -> Dict:
        """Get most common issue types from alerts."""
        issue_counts = {}
        for alert in alerts:
            classification = alert.get('classification', 'Unknown')
            issue_counts[classification] = issue_counts.get(classification, 0) + 1
        
        return dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))
    
    def export_metrics_history(self, filename: Optional[str] = None) -> str:
        """Export metrics history to JSON file."""
        if filename is None:
            filename = f"metrics_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'framework': self.monitor.inference_engine.framework,
            'metrics_history': self.monitor.metrics_history,
            'alert_history': self.monitor.alert_history,
            'performance_metrics': self.monitor.inference_engine.get_performance_metrics(),
            'baseline_metrics': self.monitor.performance_baseline
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"📁 Metrics history exported to: {filename}")
        return filename