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
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config import CONFIG
from distilled_model_trainer import DistilledMonitoringModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringInference:
    """Inference engine for the distilled monitoring model."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(CONFIG['models_dir'], 'distilled_monitoring_model')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.load_model()
        
        # Thresholds for anomaly detection
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'load_average': 5.0,
            'anomaly_score': 0.7
        }
    
    def load_model(self):
        """Load the trained model and tokenizer."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = DistilledMonitoringModel(CONFIG['model_name'])
            model_file = os.path.join(self.model_path, 'pytorch_model.bin')
            self.model.load_state_dict(torch.load(model_file, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def preprocess_metrics(self, metrics: Dict) -> str:
        """Convert metrics dictionary to text format for the model."""
        metric_text = "System metrics: "
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_text += f"{key}: {value:.2f}, "
            else:
                metric_text += f"{key}: {value}, "
        
        return metric_text.rstrip(', ')
    
    def predict_anomaly(self, metrics: Dict) -> Dict:
        """Predict if metrics indicate an anomaly."""
        # Convert metrics to text
        text_input = self.preprocess_metrics(metrics)
        
        # Tokenize
        inputs = self.tokenizer(
            text_input,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=CONFIG['max_length']
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Get predictions
            classification_probs = torch.softmax(outputs['classification_logits'], dim=-1)
            anomaly_prob = torch.sigmoid(outputs['anomaly_logits']).item()
            predicted_metrics = outputs['metrics_predictions'].squeeze().cpu().numpy()
            
            # Determine classification
            class_pred = torch.argmax(classification_probs, dim=-1).item()
            class_confidence = classification_probs.max().item()
            
            # Status mapping: 0=normal, 1=error, 2=anomaly
            status_map = {0: "normal", 1: "error", 2: "anomaly"}
            predicted_status = status_map.get(class_pred, "unknown")
            
            # Rule-based anomaly detection as backup
            rule_based_anomaly = self.rule_based_anomaly_detection(metrics)
            
            # Combine model and rule-based predictions
            final_anomaly = anomaly_prob > self.thresholds['anomaly_score'] or rule_based_anomaly
            
            return {
                'timestamp': datetime.now().isoformat(),
                'predicted_status': predicted_status,
                'classification_confidence': class_confidence,
                'anomaly_probability': anomaly_prob,
                'rule_based_anomaly': rule_based_anomaly,
                'final_anomaly': final_anomaly,
                'predicted_metrics': predicted_metrics.tolist(),
                'input_metrics': metrics,
                'recommendations': self.generate_recommendations(metrics, final_anomaly, predicted_status)
            }
    
    def rule_based_anomaly_detection(self, metrics: Dict) -> bool:
        """Rule-based anomaly detection as fallback."""
        anomaly_indicators = []
        
        # Check individual thresholds
        if metrics.get('cpu_usage', 0) > self.thresholds['cpu_usage']:
            anomaly_indicators.append('high_cpu')
        
        if metrics.get('memory_usage', 0) > self.thresholds['memory_usage']:
            anomaly_indicators.append('high_memory')
        
        if metrics.get('disk_usage', 0) > self.thresholds['disk_usage']:
            anomaly_indicators.append('high_disk')
        
        if metrics.get('load_average', 0) > self.thresholds['load_average']:
            anomaly_indicators.append('high_load')
        
        # Check for combined indicators
        if len(anomaly_indicators) >= 2:
            return True
        
        # Check for specific patterns
        if (metrics.get('cpu_usage', 0) > 90 and 
            metrics.get('load_average', 0) > 8):
            return True
        
        if (metrics.get('memory_usage', 0) > 95 and 
            metrics.get('java_gc_time', 0) > 20):
            return True
        
        return False
    
    def generate_recommendations(self, metrics: Dict, is_anomaly: bool, status: str) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        if not is_anomaly and status == "normal":
            recommendations.append("System is operating normally")
            recommendations.append("Continue regular monitoring")
            return recommendations
        
        # CPU-related recommendations
        if metrics.get('cpu_usage', 0) > 80:
            recommendations.extend([
                "High CPU usage detected",
                "Run 'top' or 'htop' to identify CPU-intensive processes",
                "Consider process optimization or resource scaling"
            ])
        
        # Memory-related recommendations
        if metrics.get('memory_usage', 0) > 85:
            recommendations.extend([
                "High memory usage detected",
                "Check for memory leaks using 'jmap' for Java applications",
                "Consider increasing available memory or optimizing applications"
            ])
        
        # Disk-related recommendations
        if metrics.get('disk_usage', 0) > 90:
            recommendations.extend([
                "Disk space critical",
                "Clean up temporary files and old logs",
                "Use 'du -sh /*' to identify large directories"
            ])
        
        # Java-specific recommendations
        if metrics.get('java_gc_time', 0) > 15:
            recommendations.extend([
                "High GC time detected in Java application",
                "Analyze heap dump to identify memory issues",
                "Consider JVM tuning parameters"
            ])
        
        # Spark-specific recommendations
        if metrics.get('spark_stage_duration', 0) > 600:
            recommendations.extend([
                "Long-running Spark stages detected",
                "Check for data skew and optimize partitioning",
                "Review Spark configuration and resource allocation"
            ])
        
        return recommendations

class RealTimeMonitor:
    """Real-time monitoring system using the distilled model."""
    
    def __init__(self, inference_engine: MonitoringInference):
        self.inference_engine = inference_engine
        self.alert_history = []
        self.metrics_history = []
        self.max_history = 1000
    
    def collect_system_metrics(self) -> Dict:
        """Collect current system metrics."""
        try:
            # This is a placeholder - in real implementation, you would
            # integrate with your actual monitoring systems
            
            # Simulate collecting metrics from various sources
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': np.random.uniform(10, 30),  # Replace with actual collection
                'memory_usage': np.random.uniform(40, 70),
                'disk_usage': np.random.uniform(20, 80),
                'load_average': np.random.uniform(1, 3),
                'network_io': np.random.uniform(1000, 10000),
                'disk_io': np.random.uniform(100, 1000),
                'processes': np.random.randint(100, 300),
                'java_heap_usage': np.random.uniform(30, 70),
                'java_gc_time': np.random.uniform(1, 5),
                'spark_executor_count': np.random.randint(5, 15),
                'spark_stage_duration': np.random.uniform(30, 120)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {}
    
    def collect_from_splunk(self, query: str, time_range: str = "-15m") -> Dict:
        """Collect metrics from Splunk (placeholder)."""
        # Placeholder for Splunk integration
        # In real implementation, use Splunk SDK or REST API
        try:
            # Example Splunk REST API call
            splunk_url = CONFIG.get('splunk_url')
            if splunk_url:
                headers = {
                    'Authorization': f"Bearer {CONFIG.get('splunk_token')}",
                    'Content-Type': 'application/json'
                }
                
                params = {
                    'search': query,
                    'earliest_time': time_range,
                    'output_mode': 'json'
                }
                
                # This would be actual API call in production
                # response = requests.get(f"{splunk_url}/services/search/jobs/export", 
                #                        headers=headers, params=params)
                
                # For now, return simulated data
                return self.collect_system_metrics()
            
        except Exception as e:
            logger.error(f"Failed to collect from Splunk: {e}")
        
        return {}
    
    def collect_from_spectrum(self) -> Dict:
        """Collect metrics from IBM Spectrum Conductor."""
        try:
            spectrum_url = CONFIG.get('spectrum_rest_url')
            if spectrum_url:
                auth = CONFIG.get('spectrum_auth', {})
                
                # Example API calls to Spectrum REST interface
                endpoints = [
                    '/platform/rest/conductor/v1/clusters',
                    '/platform/rest/conductor/v1/consumers',
                    '/platform/rest/conductor/v1/resourcegroups'
                ]
                
                spectrum_metrics = {}
                
                for endpoint in endpoints:
                    try:
                        # This would be actual API call in production
                        # response = requests.get(f"{spectrum_url}{endpoint}", 
                        #                        auth=(auth.get('username'), auth.get('password')))
                        
                        # For now, return simulated data
                        spectrum_metrics.update({
                            'spectrum_active_jobs': np.random.randint(10, 50),
                            'spectrum_cluster_utilization': np.random.uniform(30, 80),
                            'spectrum_available_slots': np.random.randint(50, 200)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to collect from Spectrum endpoint {endpoint}: {e}")
                
                return spectrum_metrics
                
        except Exception as e:
            logger.error(f"Failed to collect from Spectrum: {e}")
        
        return {}
    
    def process_metrics(self, metrics: Dict) -> Dict:
        """Process collected metrics through the model."""
        if not metrics:
            return {}
        
        # Run inference
        prediction = self.inference_engine.predict_anomaly(metrics)
        
        # Store in history
        self.metrics_history.append(prediction)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        # Generate alerts if needed
        if prediction.get('final_anomaly', False):
            alert = {
                'timestamp': prediction['timestamp'],
                'severity': self.determine_severity(prediction),
                'message': self.generate_alert_message(prediction),
                'recommendations': prediction['recommendations'],
                'metrics': metrics
            }
            
            self.alert_history.append(alert)
            if len(self.alert_history) > self.max_history:
                self.alert_history.pop(0)
            
            # Log alert
            logger.warning(f"ALERT: {alert['message']}")
        
        return prediction
    
    def determine_severity(self, prediction: Dict) -> str:
        """Determine alert severity based on prediction."""
        anomaly_prob = prediction.get('anomaly_probability', 0)
        status = prediction.get('predicted_status', 'normal')
        metrics = prediction.get('input_metrics', {})
        
        # Critical conditions
        if (metrics.get('cpu_usage', 0) > 95 or 
            metrics.get('memory_usage', 0) > 95 or 
            metrics.get('disk_usage', 0) > 95):
            return 'critical'
        
        # High severity
        if anomaly_prob > 0.9 or status == 'error':
            return 'high'
        
        # Medium severity
        if anomaly_prob > 0.7:
            return 'medium'
        
        return 'low'
    
    def generate_alert_message(self, prediction: Dict) -> str:
        """Generate human-readable alert message."""
        status = prediction.get('predicted_status', 'unknown')
        prob = prediction.get('anomaly_probability', 0)
        metrics = prediction.get('input_metrics', {})
        
        if status == 'error':
            return f"System error detected (confidence: {prob:.2f})"
        elif status == 'anomaly':
            return f"System anomaly detected (confidence: {prob:.2f})"
        else:
            # Identify specific issues
            issues = []
            if metrics.get('cpu_usage', 0) > 80:
                issues.append(f"High CPU usage ({metrics['cpu_usage']:.1f}%)")
            if metrics.get('memory_usage', 0) > 85:
                issues.append(f"High memory usage ({metrics['memory_usage']:.1f}%)")
            if metrics.get('disk_usage', 0) > 90:
                issues.append(f"High disk usage ({metrics['disk_usage']:.1f}%)")
            
            if issues:
                return f"Performance issues detected: {', '.join(issues)}"
            else:
                return f"Anomaly detected (confidence: {prob:.2f})"
    
    def get_system_health_summary(self) -> Dict:
        """Get overall system health summary."""
        if not self.metrics_history:
            return {'status': 'no_data', 'message': 'No monitoring data available'}
        
        recent_predictions = self.metrics_history[-10:]  # Last 10 predictions
        anomaly_count = sum(1 for p in recent_predictions if p.get('final_anomaly', False))
        
        # Calculate average metrics
        avg_cpu = np.mean([p.get('input_metrics', {}).get('cpu_usage', 0) for p in recent_predictions])
        avg_memory = np.mean([p.get('input_metrics', {}).get('memory_usage', 0) for p in recent_predictions])
        avg_anomaly_prob = np.mean([p.get('anomaly_probability', 0) for p in recent_predictions])
        
        # Determine overall health
        if anomaly_count >= 5:
            health_status = 'critical'
        elif anomaly_count >= 3:
            health_status = 'warning'
        elif anomaly_count >= 1:
            health_status = 'caution'
        else:
            health_status = 'healthy'
        
        return {
            'status': health_status,
            'recent_anomalies': anomaly_count,
            'avg_cpu_usage': avg_cpu,
            'avg_memory_usage': avg_memory,
            'avg_anomaly_probability': avg_anomaly_prob,
            'total_alerts': len(self.alert_history),
            'last_update': datetime.now().isoformat()
        }
    
    def run_continuous_monitoring(self, interval_seconds: int = 60):
        """Run continuous monitoring loop."""
        logger.info(f"Starting continuous monitoring (interval: {interval_seconds}s)")
        
        try:
            import time
            while True:
                # Collect metrics from all sources
                system_metrics = self.collect_system_metrics()
                splunk_metrics = self.collect_from_splunk("index=system_logs")
                spectrum_metrics = self.collect_from_spectrum()
                
                # Combine all metrics
                combined_metrics = {**system_metrics, **splunk_metrics, **spectrum_metrics}
                
                if combined_metrics:
                    # Process through model
                    prediction = self.process_metrics(combined_metrics)
                    
                    # Log status
                    if prediction.get('final_anomaly', False):
                        logger.warning(f"Anomaly detected: {prediction.get('predicted_status')}")
                    else:
                        logger.info("System operating normally")
                
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")

class MonitoringDashboard:
    """Simple dashboard for monitoring results."""
    
    def __init__(self, monitor: RealTimeMonitor):
        self.monitor = monitor
    
    def display_current_status(self):
        """Display current system status."""
        health_summary = self.monitor.get_system_health_summary()
        
        print("\n" + "="*50)
        print("SYSTEM HEALTH DASHBOARD")
        print("="*50)
        print(f"Overall Status: {health_summary['status'].upper()}")
        print(f"Recent Anomalies: {health_summary['recent_anomalies']}/10")
        print(f"Average CPU Usage: {health_summary['avg_cpu_usage']:.1f}%")
        print(f"Average Memory Usage: {health_summary['avg_memory_usage']:.1f}%")
        print(f"Total Alerts: {health_summary['total_alerts']}")
        print(f"Last Update: {health_summary['last_update']}")
        
        # Show recent alerts
        if self.monitor.alert_history:
            print("\nRECENT ALERTS:")
            print("-" * 30)
            for alert in self.monitor.alert_history[-5:]:  # Last 5 alerts
                print(f"[{alert['severity'].upper()}] {alert['timestamp'][:19]}")
                print(f"  {alert['message']}")
                print(f"  Recommendations: {alert['recommendations'][0] if alert['recommendations'] else 'None'}")
                print()
        
        print("="*50)
    
    def export_metrics_history(self, filename: str = None):
        """Export metrics history to JSON file."""
        if filename is None:
            filename = f"metrics_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'metrics_history': self.monitor.metrics_history,
            'alert_history': self.monitor.alert_history,
            'exported_at': datetime.now().isoformat(),
            'summary': self.monitor.get_system_health_summary()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Metrics history exported to {filename}")

# Usage example functions
def run_single_prediction():
    """Run a single prediction example."""
    print("Running single prediction example...")
    
    # Initialize inference engine
    inference_engine = MonitoringInference()
    
    # Example metrics
    test_metrics = {
        'cpu_usage': 85.5,
        'memory_usage': 92.3,
        'disk_usage': 45.2,
        'load_average': 6.8,
        'network_io': 15000,
        'disk_io': 2500,
        'java_heap_usage': 87.5,
        'java_gc_time': 18.2,
        'spark_executor_count': 8,
        'spark_stage_duration': 450
    }
    
    # Get prediction
    prediction = inference_engine.predict_anomaly(test_metrics)
    
    print("\nPrediction Results:")
    print(f"Status: {prediction['predicted_status']}")
    print(f"Anomaly Probability: {prediction['anomaly_probability']:.3f}")
    print(f"Final Anomaly: {prediction['final_anomaly']}")
    print(f"Confidence: {prediction['classification_confidence']:.3f}")
    
    print("\nRecommendations:")
    for rec in prediction['recommendations']:
        print(f"  - {rec}")

def run_monitoring_demo(duration_minutes: int = 5):
    """Run monitoring demo for specified duration."""
    print(f"Running monitoring demo for {duration_minutes} minutes...")
    
    # Initialize components
    inference_engine = MonitoringInference()
    monitor = RealTimeMonitor(inference_engine)
    dashboard = MonitoringDashboard(monitor)
    
    import time
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    try:
        while time.time() < end_time:
            # Collect and process metrics
            metrics = monitor.collect_system_metrics()
            
            # Occasionally inject anomalies for demo
            if np.random.random() < 0.3:  # 30% chance of anomaly
                metrics['cpu_usage'] = np.random.uniform(85, 99)
                metrics['memory_usage'] = np.random.uniform(90, 99)
            
            prediction = monitor.process_metrics(metrics)
            
            # Display status every 30 seconds
            if int(time.time() - start_time) % 30 == 0:
                dashboard.display_current_status()
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\nDemo stopped by user")
    
    # Final summary
    dashboard.display_current_status()
    dashboard.export_metrics_history()

if __name__ == "__main__":
    # Example usage
    print("Monitoring System Demo")
    print("1. Single prediction")
    print("2. Monitoring demo")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        run_single_prediction()
    elif choice == "2":
        duration = int(input("Duration in minutes (default 5): ") or "5")
        run_monitoring_demo(duration)
    else:
        print("Invalid choice")