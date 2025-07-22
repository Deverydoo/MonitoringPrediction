import os
import logging
from pathlib import Path
from datetime import datetime

from config import CONFIG, setup_directories, detect_training_environment, setup_fallback_system, test_fallback_system
from dataset_generator import DatasetGenerator
from distilled_model_trainer import DistilledModelTrainer
from inference_and_monitoring import MonitoringInference, RealTimeMonitor, MonitoringDashboard

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DistilledMonitoringSystem:
    """Clean, simple interface for the distilled monitoring system."""
    
    def __init__(self):
        self.setup_complete = False
        self.fallback_ready = False
        self.generator = None
        
        # Check what exists
        self.datasets_exist = self._check_datasets()
        self.model_trained = self._check_model()
        
    def setup(self):
        """Setup system environment and fallbacks."""
        print("🚀 Setting up distilled monitoring system...")
        
        # Create directories
        setup_directories()
        
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
        
        # Initialize generator
        self.generator = DatasetGenerator()
        
        self.setup_complete = True
        print("\n✅ Setup complete!")
        return True
    
    def generate_datasets(self, language_count: int = None, metrics_count: int = None):
        """Generate training datasets."""
        if not self.setup_complete:
            print("❌ Run setup() first")
            return None, None
        
        print("\n📊 DATASET GENERATION")
        print("="*50)
        
        try:
            language_data, metrics_data = self.generator.generate_complete_dataset(
                language_count, metrics_count
            )
            
            self.datasets_exist = True
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
        """Train the distilled model."""
        if not self.datasets_exist:
            print("❌ No datasets found. Run generate_datasets() first")
            return False
        
        print("\n🏋️ TRAINING DISTILLED MODEL")
        print("="*40)
        print(f"Environment: {detect_training_environment()}")
        
        try:
            trainer = DistilledModelTrainer()
            trainer.train_model()
            self.model_trained = True
            print("✅ Training completed!")
            return True
        except Exception as e:
            print(f"❌ Training failed: {e}")
            logger.error(f"Training error: {e}")
            return False
    
    def test(self):
        """Test model inference."""
        if not self.model_trained:
            print("❌ No trained model found. Run train() first")
            return False
        
        print("\n🧪 TESTING MODEL INFERENCE")
        print("="*40)
        
        try:
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
            
            for test_case in test_cases:
                print(f"\n--- {test_case['name']} ---")
                prediction = inference.predict_anomaly(test_case['metrics'])
                
                print(f"Status: {prediction['predicted_status']}")
                print(f"Anomaly: {prediction['final_anomaly']} ({prediction['anomaly_probability']:.3f})")
                print(f"Recommendation: {prediction['recommendations'][0] if prediction['recommendations'] else 'None'}")
            
            print("\n✅ Testing completed!")
            return True
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            logger.error(f"Testing error: {e}")
            return False
    
    def demo(self, minutes: int = 5):
        """Run monitoring demo."""
        if not self.model_trained:
            print("❌ No trained model found. Run train() first")
            return
        
        print(f"\n🎭 MONITORING DEMO ({minutes} minutes)")
        print("="*40)
        
        try:
            import time
            import numpy as np
            
            inference = MonitoringInference()
            monitor = RealTimeMonitor(inference)
            dashboard = MonitoringDashboard(monitor)
            
            start_time = time.time()
            end_time = start_time + (minutes * 60)
            iteration = 0
            
            while time.time() < end_time:
                iteration += 1
                
                # Collect metrics with occasional anomalies
                metrics = monitor.collect_system_metrics()
                
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
                prediction = monitor.process_metrics(metrics)
                
                status = "🔴 ANOMALY" if prediction.get('final_anomaly') else "🟢 Normal"
                print(f"{status} - Iteration {iteration} (confidence: {prediction['anomaly_probability']:.3f})")
                
                # Show dashboard occasionally
                if prediction.get('final_anomaly') or iteration % 5 == 0:
                    dashboard.display_current_status()
                
                time.sleep(12)
            
            print("\n✅ Demo completed!")
            dashboard.export_metrics_history()
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            logger.error(f"Demo error: {e}")
    
    def status(self):
        """Show system status."""
        print(f"\n{'='*50}")
        print("SYSTEM STATUS")
        print(f"{'='*50}")
        
        print(f"Setup: {'✅' if self.setup_complete else '❌'}")
        print(f"Datasets: {'✅' if self.datasets_exist else '❌'}")
        print(f"Model: {'✅' if self.model_trained else '❌'}")
        print(f"Fallbacks: {'✅' if self.fallback_ready else '❌'}")
        
        # Show progress if generator exists
        if self.generator:
            self.generator.show_progress()
        
        # File status - Fixed: files is a list of tuples, not a dict
        print(f"\nFiles:")
        files = [
            (f"{CONFIG['training_dir']}/language_dataset.json", "Language Dataset"),
            (f"{CONFIG['training_dir']}/metrics_dataset.json", "Metrics Dataset"),
            (f"{CONFIG['models_dir']}/distilled_monitoring_model", "Trained Model")
        ]
        
        for file_path, description in files:  # Changed from files.items()
            exists = Path(file_path).exists()
            print(f"  {description}: {'✅' if exists else '❌'}")
        
        print(f"\nEnvironment: {detect_training_environment()}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")
    
    def show_progress(self):
        """Show dataset generation progress."""
        if self.generator:
            self.generator.show_progress()
        else:
            print("❌ Generator not initialized. Run setup() first.")
    
    def reset_progress(self):
        """Reset dataset generation progress."""
        if self.generator:
            self.generator.reset_progress()
        else:
            print("❌ Generator not initialized. Run setup() first.")
    
    def retry_failed(self):
        """Retry failed generation items."""
        if self.generator:
            self.generator.retry_failed()
        else:
            print("❌ Generator not initialized. Run setup() first.")
    
    def _check_datasets(self):
        """Check if datasets exist."""
        lang_file = Path(CONFIG['training_dir']) / 'language_dataset.json'
        metrics_file = Path(CONFIG['training_dir']) / 'metrics_dataset.json'
        return lang_file.exists() and metrics_file.exists()
    
    def _check_model(self):
        """Check if trained model exists."""
        model_path = Path(CONFIG['models_dir']) / 'distilled_monitoring_model'
        return model_path.exists()

# Global system instance
system = DistilledMonitoringSystem()

# Simple interface functions
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