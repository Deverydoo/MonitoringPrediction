import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from config import CONFIG, setup_directories, detect_training_environment, setup_fallback_system, test_fallback_system
from dataset_generator import EnhancedDatasetGenerator
from distilled_model_trainer import DistilledModelTrainer
from inference_and_monitoring import MonitoringInference, RealTimeMonitor, MonitoringDashboard
from common_utils import (
    analyze_existing_datasets, check_models_like_trainer, 
    ensure_directory_structure, get_dataset_paths, DATASET_FORMATS
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistilledMonitoringSystem:
    """Clean interface synchronized with working trainer and generator."""
    
    def __init__(self):
        self.setup_complete = False
        self.fallback_ready = False
        self.generator = None
        self.trainer = None
        
        # Use common utilities for consistent checking
        self.datasets_exist = self._check_datasets_using_common_utils()
        self.model_trained = check_models_like_trainer(Path(CONFIG['models_dir']))
        
    def setup(self):
        """Setup system environment and fallbacks."""
        print("🚀 Setting up distilled monitoring system...")
        
        # Create directories using common utilities
        ensure_directory_structure(CONFIG)
        
        # Setup fallback system
        print(f"\n{'='*50}")
        print("FALLBACK SYSTEM SETUP")
        print(f"{'='*50}")
        
        self.fallback_ready = setup_fallback_system()
        if self.fallback_ready:
            test_fallback_system()
            print("✅ Fallback system ready")
        else:
            print("❌ Fallback system failed")
            return False
        
        # Initialize generator using enhanced class
        self.generator = EnhancedDatasetGenerator()
        
        self.setup_complete = True
        print("\n✅ Setup complete!")
        return True
    
    def generate_datasets(self, language_count: int = None, metrics_count: int = None):
        """Generate training datasets using enhanced generator."""
        if not self.setup_complete:
            print("❌ Run setup() first")
            return None, None
        
        print("\n📊 DATASET GENERATION")
        print("="*50)
        
        try:
            # Use the correct method name from EnhancedDatasetGenerator
            language_data, metrics_data = self.generator.generate_complete_dataset(
                language_count, metrics_count
            )
            
            # Update status using common utilities
            self.datasets_exist = self._check_datasets_using_common_utils()
            print("✅ Dataset generation completed!")
            return language_data, metrics_data
            
        except KeyboardInterrupt:
            print("\n⏸️  Generation interrupted - progress saved")
            return None, None
        except Exception as e:
            print(f"\n❌ Generation failed: {e}")
            logger.error(f"Dataset generation error: {e}")
            return None, None
    
    def train(self):
        """Train using exact trainer logic."""
        if not self.datasets_exist:
            print("❌ No datasets found. Run generate_datasets() first")
            return False
        
        print("\n🏋️ TRAINING DISTILLED MODEL")
        print("="*40)
        print(f"Environment: {detect_training_environment()}")
        
        try:
            # Use exact same trainer class and logic
            trainer = DistilledModelTrainer(CONFIG, resume_training=True)
            success = trainer.train()
            
            if success:
                # Update status using common utilities
                self.model_trained = check_models_like_trainer(Path(CONFIG['models_dir']))
                print("✅ Training completed!")
                return True
            else:
                print("❌ Training failed")
                return False
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            logger.error(f"Training error: {e}")
            return False
    
    def test(self):
        """Test model inference with rule-based fallback."""
        print("\n🧪 TESTING MODEL INFERENCE")
        print("="*40)
        
        try:
            # Try ML model first
            inference = MonitoringInference()
            
            test_cases = [
                {
                    'name': 'Normal Operation',
                    'metrics': {
                        'cpu_usage': 25.5, 'memory_usage': 45.2, 'disk_usage': 35.8,
                        'load_average': 1.2, 'java_heap_usage': 55.0
                    }
                },
                {
                    'name': 'High CPU',
                    'metrics': {
                        'cpu_usage': 92.3, 'memory_usage': 65.1, 'disk_usage': 45.2,
                        'load_average': 8.7, 'java_heap_usage': 75.0
                    }
                },
                {
                    'name': 'Memory Pressure',
                    'metrics': {
                        'cpu_usage': 35.2, 'memory_usage': 96.8, 'disk_usage': 55.1,
                        'load_average': 2.1, 'java_heap_usage': 97.5
                    }
                }
            ]
            
            ml_success = False
            for test_case in test_cases:
                print(f"\n--- {test_case['name']} ---")
                
                # Try ML prediction
                prediction = inference.predict_anomaly(test_case['metrics'])
                
                if 'error' in prediction:
                    print(f"ML Model Error: {prediction['error']}")
                    # Fall back to rule-based prediction
                    prediction = self._rule_based_prediction(test_case['metrics'])
                    print("Using rule-based fallback")
                else:
                    ml_success = True
                
                print(f"Status: {prediction.get('predicted_status', 'Unknown')}")
                print(f"Anomaly: {prediction.get('final_anomaly', False)} ({prediction.get('anomaly_probability', 0.0):.3f})")
                print(f"Recommendation: {prediction.get('recommendations', ['None'])[0]}")
            
            if ml_success:
                print("\n✅ ML Model testing successful!")
            else:
                print("\n⚠️  ML Model needs debugging, but rule-based fallback works")
            
            # Update status
            self.model_trained = check_models_like_trainer(Path(CONFIG['models_dir']))
            return True
            
        except FileNotFoundError as e:
            print(f"❌ No trained model found. Run train() first")
            print(f"Details: {e}")
            self.model_trained = False
            return False
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            logger.error(f"Testing error: {e}")
            return False

    def _rule_based_prediction(self, metrics: Dict) -> Dict:
        """Rule-based fallback prediction for testing."""
        cpu = metrics.get('cpu_usage', 0)
        memory = metrics.get('memory_usage', 0)
        disk = metrics.get('disk_usage', 0)
        load = metrics.get('load_average', 0)
        
        # Simple rule-based logic
        anomaly_score = 0.0
        status = "System Normal"
        recommendations = ["Continue normal monitoring"]
        
        if cpu > 90:
            anomaly_score = 0.9
            status = "Performance Issue"
            recommendations = ["Check CPU-intensive processes", "Review recent deployments"]
        elif memory > 90:
            anomaly_score = 0.85
            status = "Resource Constraint"
            recommendations = ["Check memory usage", "Look for memory leaks"]
        elif disk > 90:
            anomaly_score = 0.8
            status = "Resource Constraint"
            recommendations = ["Clean up disk space", "Review log rotation"]
        elif load > 8:
            anomaly_score = 0.75
            status = "Performance Issue"
            recommendations = ["Check system load", "Review running processes"]
        elif cpu > 80 or memory > 80:
            anomaly_score = 0.6
            status = "Performance Issue"
            recommendations = ["Monitor resource usage closely"]
        
        return {
            'predicted_status': status,
            'final_anomaly': anomaly_score > 0.7,
            'anomaly_probability': anomaly_score,
            'recommendations': recommendations,
            'model_type': 'rule_based_fallback'
        }
    
    def demo(self, minutes: int = 5):
        """Run monitoring demo."""
        print(f"\n🎭 MONITORING DEMO ({minutes} minutes)")
        print("="*40)
        
        # Check model using common utilities
        if not check_models_like_trainer(Path(CONFIG['models_dir'])):
            print("❌ No trained model found. Run train() first")
            return
        
        try:
            import time
            import numpy as np
            
            inference = MonitoringInference()
            monitor = RealTimeMonitor(check_interval=12)
            dashboard = MonitoringDashboard()
            
            start_time = time.time()
            end_time = start_time + (minutes * 60)
            iteration = 0
            
            while time.time() < end_time:
                iteration += 1
                
                # Collect basic metrics
                metrics = {
                    'cpu_usage': np.random.uniform(15, 45),
                    'memory_usage': np.random.uniform(30, 60),
                    'disk_usage': np.random.uniform(20, 50),
                    'load_average': np.random.uniform(0.5, 2.0),
                    'java_heap_usage': np.random.uniform(40, 70)
                }
                
                # Inject anomalies for demo
                if iteration % 8 == 0:
                    print("\n🔥 Simulating CPU spike...")
                    metrics.update({
                        'cpu_usage': np.random.uniform(85, 99),
                        'load_average': np.random.uniform(8, 15)
                    })
                elif iteration % 12 == 0:
                    print("\n💾 Simulating memory pressure...")
                    metrics.update({
                        'memory_usage': np.random.uniform(88, 97),
                        'java_heap_usage': np.random.uniform(92, 99)
                    })
                
                # Process and display
                prediction = inference.predict_anomaly(metrics)
                
                status = "🔴 ANOMALY" if prediction.get('final_anomaly') else "🟢 Normal"
                print(f"{status} - Iteration {iteration} (confidence: {prediction['anomaly_probability']:.3f})")
                
                time.sleep(12)
            
            print("\n✅ Demo completed!")
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.error(f"Demo error: {e}")
    
    def status(self):
        """Show system status using common utilities and enhanced generator methods."""
        print(f"\n{'='*50}")
        print("SYSTEM STATUS")
        print(f"{'='*50}")
        
        print(f"Setup: {'✅' if self.setup_complete else '❌'}")
        print(f"Datasets: {'✅' if self._check_datasets_using_common_utils() else '❌'}")
        print(f"Model: {'✅' if check_models_like_trainer(Path(CONFIG['models_dir'])) else '❌'}")
        print(f"Fallbacks: {'✅' if self.fallback_ready else '❌'}")
        
        # Show progress using generator's method
        if self.generator:
            self.generator.show_progress_with_multiplier()
        
        # File status using common utilities
        dataset_analysis = analyze_existing_datasets(Path(CONFIG['training_dir']))
        
        print(f"\nFiles:")
        print(f"  Language Dataset: {'✅' if dataset_analysis['language_dataset']['exists'] else '❌'}")
        if dataset_analysis['language_dataset']['exists']:
            print(f"    Samples: {dataset_analysis['language_dataset']['samples']}")
        
        print(f"  Metrics Dataset: {'✅' if dataset_analysis['metrics_dataset']['exists'] else '❌'}")
        if dataset_analysis['metrics_dataset']['exists']:
            print(f"    Samples: {dataset_analysis['metrics_dataset']['samples']}")
        
        # Model check using common utilities
        model_found = check_models_like_trainer(Path(CONFIG['models_dir']))
        print(f"  Trained Model: {'✅' if model_found else '❌'}")
        
        print(f"\nEnvironment: {detect_training_environment()}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
    
    def show_progress(self):
        """Show dataset generation progress using generator's method."""
        if self.generator:
            self.generator.show_progress_with_multiplier()
        else:
            print("❌ Generator not initialized. Run setup() first.")
    
    def reset_progress(self):
        """Reset dataset generation progress using generator's method."""
        if self.generator:
            self.generator.reset_progress()
        else:
            print("❌ Generator not initialized. Run setup() first.")
    
    def retry_failed(self):
        """Retry failed generation items using generator's method."""
        if self.generator:
            self.generator.retry_failed()
        else:
            print("❌ Generator not initialized. Run setup() first.")
    
    def _check_datasets_using_common_utils(self):
        """Check if datasets exist using common utilities."""
        analysis = analyze_existing_datasets(Path(CONFIG['training_dir']))
        return (analysis['language_dataset']['exists'] and 
                analysis['metrics_dataset']['exists'])

# Global system instance
system = DistilledMonitoringSystem()

# Simple interface functions - all use exact same logic as components
def setup():
    """Setup environment and fallback system."""
    return system.setup()

def generate_datasets(language_count: int = None, metrics_count: int = None):
    """Generate training datasets."""
    return system.generate_datasets(language_count, metrics_count)

def train():
    """Train the distilled model."""
    return system.train()

def test():
    """Test model inference."""
    return system.test()

def demo(minutes: int = 5):
    """Run monitoring demo."""
    return system.demo(minutes)

def status():
    """Show system status."""
    return system.status()

def show_progress():
    """Show dataset generation progress."""
    return system.show_progress()

def reset_progress():
    """Reset dataset generation progress."""
    return system.reset_progress()

def retry_failed():
    """Retry failed generation items."""
    return system.retry_failed()

def quick_start_guide():
    """Display quick start guide."""
    print("""
🚀 DISTILLED MONITORING SYSTEM - QUICK START
============================================

WORKFLOW:
1. setup()                    # Setup environment and fallback system
2. generate_datasets()        # Generate training datasets
3. train()                   # Train the distilled model
4. test()                    # Test model inference
5. demo(minutes=5)           # Run monitoring demo

PROGRESS MANAGEMENT:
- show_progress()            # Check dataset generation progress
- reset_progress()           # Start dataset generation fresh
- retry_failed()             # Retry failed generation items

CONFIGURATION:
- Edit YAML files in ./data_config/ for customization
- System discovers models and YAML files automatically
- Fallback order: Remote API → Ollama → Local Model → Static

The system is designed for dynamic environments with automatic discovery.
""")

# Display guide on import
print("🚀 Distilled Monitoring System")
print("📊 Predictive monitoring with dynamic discovery")
print()
print("Type quick_start_guide() for usage instructions")
print("Type status() to check system status")