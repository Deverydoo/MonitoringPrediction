#!/usr/bin/env python3
"""
main_notebook.py - Streamlined TFT Interface
Clean interface focused only on TFT time-series prediction
Removes all legacy BERT/language model code
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Import streamlined TFT components
from config import CONFIG, validate_tft_environment, detect_training_environment, get_system_info
from metrics_generator import MetricsDatasetGenerator
from tft_model_trainer import DistilledModelTrainer
from tft_inference import TFTInference
from common_utils import (
    analyze_metrics_dataset, check_tft_model_exists, ensure_tft_directories,
    validate_metrics_dataset, get_dataset_stats, cleanup_tft_artifacts, log_message
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TFTMonitoringSystem:
    """Streamlined TFT-only monitoring system."""
    
    def __init__(self):
        self.setup_complete = False
        self.generator = None
        self.trainer = None
        self.inference = None
        
        # Check current state
        analysis = analyze_metrics_dataset(Path(CONFIG['training_dir']))
        self.dataset_exists = analysis['exists']
        self.model_trained = check_tft_model_exists(Path(CONFIG['models_dir']))
        
        log_message("🎯 TFT Monitoring System initialized")
        log_message(f"📊 Dataset: {'✅' if self.dataset_exists else '❌'}")
        log_message(f"🤖 Model: {'✅' if self.model_trained else '❌'}")
    
    def setup(self) -> bool:
        """Setup TFT environment and directories."""
        log_message("🚀 Setting up TFT monitoring system...")
        
        # Validate TFT environment
        if not validate_tft_environment():
            log_message("❌ TFT environment validation failed")
            return False
        
        # Create directories
        if not ensure_tft_directories(CONFIG):
            log_message("❌ Failed to create directories")
            return False
        
        # Initialize components
        self.generator = MetricsDatasetGenerator(CONFIG)
        
        self.setup_complete = True
        log_message("✅ TFT setup complete!")
        
        # Show system info
        info = get_system_info()
        log_message(f"🎮 Environment: {info['environment']}")
        if info.get('gpu_name'):
            log_message(f"🔥 GPU: {info['gpu_name']} ({info['gpu_memory_gb']}GB)")
        
        return True
    
    def generate_dataset(self, hours: int = 168, force_regenerate: bool = False) -> bool:
        """Generate metrics dataset for TFT training."""
        if not self.setup_complete:
            log_message("❌ Run setup() first")
            return False
        
        dataset_path = Path(CONFIG['training_dir']) / 'metrics_dataset.json'
        
        # Check if dataset already exists
        if dataset_path.exists() and not force_regenerate:
            stats = get_dataset_stats(dataset_path)
            existing_hours = stats.get('time_span_hours', 0)
            
            if existing_hours >= hours * 0.9:  # Within 90% of requested hours
                log_message(f"✅ Dataset already exists with {existing_hours:.1f} hours")
                log_message("💡 Use force_regenerate=True to regenerate")
                self.dataset_exists = True
                return True
        
        log_message(f"📊 Generating {hours} hours of TFT training data...")
        log_message(f"🎯 Target: ~{hours * 12 * CONFIG.get('servers_count', 57):,} samples")
        
        try:
            # Generate dataset
            dataset = self.generator.generate_dataset(
                total_hours=hours,
                output_file=str(dataset_path)
            )
            
            if dataset:
                # Validate generated dataset
                if validate_metrics_dataset(dataset_path):
                    self.dataset_exists = True
                    
                    # Show stats
                    stats = get_dataset_stats(dataset_path)
                    log_message("🎉 Dataset generation completed!")
                    log_message(f"   📊 Total samples: {stats['total_samples']:,}")
                    log_message(f"   ⏱️  Time span: {stats['time_span_hours']:.1f} hours")
                    log_message(f"   🖥️  Servers: {stats['servers_count']}")
                    log_message(f"   ⚠️  Anomaly ratio: {stats['anomaly_ratio']*100:.1f}%")
                    
                    return True
                else:
                    log_message("❌ Generated dataset failed validation")
                    return False
            else:
                log_message("❌ Dataset generation failed")
                return False
                
        except KeyboardInterrupt:
            log_message("⏸️  Generation interrupted")
            return False
        except Exception as e:
            log_message(f"❌ Generation failed: {e}")
            return False
    
    def train(self, resume: bool = False) -> bool:
        """Train TFT model."""
        if not self.dataset_exists:
            log_message("❌ No dataset found. Run generate_dataset() first")
            return False
        
        log_message("🏋️ Training TFT model...")
        log_message(f"🎯 Model: Temporal Fusion Transformer")
        log_message(f"⚡ Environment: {detect_training_environment()}")
        log_message(f"📊 Config: {CONFIG['epochs']} epochs, batch size {CONFIG['batch_size']}")
        
        try:
            # Create trainer
            self.trainer = DistilledModelTrainer(CONFIG, resume_training=resume)
            
            # Train model
            success = self.trainer.train()
            
            if success:
                self.model_trained = check_tft_model_exists(Path(CONFIG['models_dir']))
                log_message("🎉 TFT training completed!")
                log_message("💡 Model capabilities:")
                log_message("   - Multi-horizon time-series forecasting")
                log_message("   - Attention-based feature importance")
                log_message("   - Uncertainty quantification")
                return True
            else:
                log_message("❌ Training failed")
                return False
                
        except KeyboardInterrupt:
            log_message("⏸️ Training interrupted")
            return False
        except Exception as e:
            log_message(f"❌ Training failed: {e}")
            return False
    
    def test(self) -> bool:
        """Test TFT model inference."""
        if not self.model_trained:
            log_message("❌ No trained model found. Run train() first")
            return False
        
        log_message("🧪 Testing TFT model inference...")
        
        try:
            # Initialize inference engine
            self.inference = TFTInference()
            
            if not self.inference.is_ready():
                log_message("❌ TFT inference engine not ready")
                return False
            
            # Generate test scenarios
            test_scenarios = self._create_test_scenarios()
            
            success_count = 0
            for i, scenario in enumerate(test_scenarios, 1):
                log_message(f"\n--- Test {i}: {scenario['name']} ---")
                
                # Run prediction
                result = self.inference.predict(scenario['data'])
                
                if 'error' in result:
                    log_message(f"❌ Error: {result['error']}")
                    continue
                
                # Display results
                predictions = result.get('predictions', {})
                alerts = result.get('alerts', [])
                
                log_message(f"✅ Prediction successful")
                
                # Show key predictions
                for metric, pred_data in list(predictions.items())[:3]:  # Show first 3 metrics
                    values = pred_data['values']
                    log_message(f"   {metric}: {values[0]:.1f} → {values[-1]:.1f}")
                
                # Show alerts
                if alerts:
                    critical_alerts = [a for a in alerts if a['severity'] == 'critical']
                    warning_alerts = [a for a in alerts if a['severity'] == 'warning']
                    
                    if critical_alerts:
                        log_message(f"   🚨 {len(critical_alerts)} critical alerts")
                    if warning_alerts:
                        log_message(f"   ⚠️  {len(warning_alerts)} warning alerts")
                else:
                    log_message("   ✅ No alerts generated")
                
                success_count += 1
            
            log_message(f"\n🎉 Testing completed: {success_count}/{len(test_scenarios)} successful")
            return success_count > 0
            
        except Exception as e:
            log_message(f"❌ Testing failed: {e}")
            return False
    
    def demo(self, minutes: int = 5) -> bool:
        """Run TFT monitoring demo."""
        if not self.model_trained:
            log_message("❌ No trained model found. Run train() first")
            return False
        
        log_message(f"🎭 Running TFT monitoring demo ({minutes} minutes)...")
        log_message("📈 Features: Multi-horizon forecasting, attention weights, uncertainty quantification")
        
        try:
            import time
            import random
            
            if not self.inference:
                self.inference = TFTInference()
            
            start_time = time.time()
            end_time = start_time + (minutes * 60)
            iteration = 0
            
            log_message("🚀 Starting live TFT prediction demo...")
            
            while time.time() < end_time:
                iteration += 1
                
                # Generate realistic server metrics sequence
                demo_data = self._generate_demo_sequence()
                
                # Run TFT prediction
                result = self.inference.predict(demo_data)
                
                if 'error' not in result:
                    predictions = result.get('predictions', {})
                    alerts = result.get('alerts', [])
                    
                    # Show prediction summary
                    status = "🔴 ALERTS" if alerts else "🟢 Normal"
                    log_message(f"Iteration {iteration}: {status}")
                    
                    # Show interesting predictions
                    if iteration % 3 == 0:  # Every 3rd iteration
                        for metric, pred_data in list(predictions.items())[:2]:
                            values = pred_data['values']
                            trend = "↗️" if values[-1] > values[0] else "↘️" if values[-1] < values[0] else "→"
                            log_message(f"   {metric}: {values[0]:.1f} {trend} {values[-1]:.1f}")
                    
                    # Show critical alerts
                    critical_alerts = [a for a in alerts if a['severity'] == 'critical']
                    if critical_alerts:
                        alert = critical_alerts[0]
                        log_message(f"   🚨 CRITICAL: {alert['metric']} predicted at {alert['predicted_value']}")
                
                time.sleep(10)  # 10 second intervals
            
            log_message("🎉 Demo completed successfully!")
            log_message("💡 Demo showcased:")
            log_message("   - Real-time multi-horizon forecasting")
            log_message("   - Automated alert generation")
            log_message("   - Temporal pattern recognition")
            
            return True
            
        except KeyboardInterrupt:
            log_message("⏹️  Demo stopped by user")
            return False
        except Exception as e:
            log_message(f"❌ Demo failed: {e}")
            return False
    
    def status(self):
        """Show comprehensive system status."""
        log_message(f"\n{'='*50}")
        log_message("TFT MONITORING SYSTEM STATUS")
        log_message(f"{'='*50}")
        
        # System status
        log_message(f"Setup: {'✅' if self.setup_complete else '❌'}")
        log_message(f"Dataset: {'✅' if self.dataset_exists else '❌'}")
        log_message(f"Model: {'✅' if self.model_trained else '❌'}")
        
        # Dataset info
        if self.dataset_exists:
            dataset_path = Path(CONFIG['training_dir']) / 'metrics_dataset.json'
            stats = get_dataset_stats(dataset_path)
            
            log_message(f"\n📊 Dataset Information:")
            log_message(f"   Samples: {stats.get('total_samples', 0):,}")
            log_message(f"   Time span: {stats.get('time_span_hours', 0):.1f} hours")
            log_message(f"   Servers: {stats.get('servers_count', 0)}")
            log_message(f"   Anomaly ratio: {stats.get('anomaly_ratio', 0)*100:.1f}%")
            log_message(f"   Format: {'Enhanced' if stats.get('enhanced') else 'Standard'}")
        
        # Model info
        if self.model_trained:
            from common_utils import get_latest_tft_model_path
            model_path = get_latest_tft_model_path(Path(CONFIG['models_dir']))
            log_message(f"\n🤖 Model Information:")
            log_message(f"   Type: Temporal Fusion Transformer")
            log_message(f"   Path: {model_path}")
            log_message(f"   Framework: PyTorch Forecasting")
        
        # System info
        info = get_system_info()
        log_message(f"\n🖥️  System Information:")
        log_message(f"   Environment: {info['environment']}")
        log_message(f"   Framework: {info['framework']}")
        if info.get('gpu_name'):
            log_message(f"   GPU: {info['gpu_name']} ({info['gpu_memory_gb']}GB)")
        
        log_message(f"\n⚙️  Configuration:")
        log_message(f"   Epochs: {CONFIG['epochs']}")
        log_message(f"   Batch size: {CONFIG['batch_size']}")
        log_message(f"   Context length: {CONFIG['context_length']}")
        log_message(f"   Prediction horizon: {CONFIG['prediction_horizon']}")
        
        log_message(f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_message(f"{'='*50}")
    
    def cleanup(self):
        """Run system cleanup."""
        log_message("🧹 Running TFT system cleanup...")
        try:
            cleanup_tft_artifacts(CONFIG)
            log_message("✅ Cleanup completed")
        except Exception as e:
            log_message(f"❌ Cleanup failed: {e}")
    
    def _create_test_scenarios(self) -> list:
        """Create test scenarios for TFT inference."""
        scenarios = []
        
        # Scenario 1: Normal operation trend
        normal_data = []
        for i in range(30):  # 30 time points (2.5 hours at 5min intervals)
            normal_data.append({
                'timestamp': (datetime.now() - timedelta(minutes=5*i)).isoformat(),
                'cpu_usage': 25 + random.gauss(0, 5),
                'memory_usage': 45 + random.gauss(0, 8),
                'disk_usage': 35 + random.gauss(0, 3),
                'load_average': 1.2 + random.gauss(0, 0.3)
            })
        
        scenarios.append({
            'name': 'Normal Operation Trend',
            'data': normal_data
        })
        
        # Scenario 2: Gradual degradation
        degradation_data = []
        for i in range(30):
            base_cpu = 30 + (i * 2)  # Gradually increasing
            degradation_data.append({
                'timestamp': (datetime.now() - timedelta(minutes=5*i)).isoformat(),
                'cpu_usage': base_cpu + random.gauss(0, 3),
                'memory_usage': 50 + (i * 1.5) + random.gauss(0, 5),
                'disk_usage': 40 + random.gauss(0, 2),
                'load_average': 1.5 + (i * 0.1) + random.gauss(0, 0.2)
            })
        
        scenarios.append({
            'name': 'Gradual Performance Degradation',
            'data': degradation_data
        })
        
        # Scenario 3: Spike pattern
        spike_data = []
        for i in range(30):
            if 10 <= i <= 15:  # Spike in middle
                cpu_val = 85 + random.gauss(0, 5)
                mem_val = 80 + random.gauss(0, 8)
            else:
                cpu_val = 25 + random.gauss(0, 5)
                mem_val = 45 + random.gauss(0, 8)
            
            spike_data.append({
                'timestamp': (datetime.now() - timedelta(minutes=5*i)).isoformat(),
                'cpu_usage': cpu_val,
                'memory_usage': mem_val,
                'disk_usage': 35 + random.gauss(0, 3),
                'load_average': 1.2 + random.gauss(0, 0.3)
            })
        
        scenarios.append({
            'name': 'Spike Pattern Detection',
            'data': spike_data
        })
        
        return scenarios
    
    def _generate_demo_sequence(self) -> list:
        """Generate demo data sequence."""
        import random
        from datetime import timedelta
        
        # Generate 24 time points (2 hours of history)
        demo_data = []
        base_time = datetime.now()
        
        for i in range(24):
            # Add some variability and trends
            phase = i / 24.0  # 0 to 1
            cpu_trend = 30 + 20 * phase + random.gauss(0, 5)  # Gradual increase
            mem_trend = 50 + 15 * phase + random.gauss(0, 8)
            
            demo_data.append({
                'timestamp': (base_time - timedelta(minutes=5*i)).isoformat(),
                'cpu_usage': max(0, min(100, cpu_trend)),
                'memory_usage': max(0, min(100, mem_trend)),
                'disk_usage': 35 + random.gauss(0, 3),
                'load_average': 1.2 + phase * 2 + random.gauss(0, 0.3),
                'java_heap_usage': 55 + phase * 20 + random.gauss(0, 5),
                'network_errors': random.poisson(2)
            })
        
        return demo_data


# Global system instance
system = TFTMonitoringSystem()

# Simplified interface functions
def setup() -> bool:
    """Setup TFT environment."""
    return system.setup()

def generate_dataset(hours: int = 168, force_regenerate: bool = False) -> bool:
    """Generate metrics dataset for TFT training."""
    return system.generate_dataset(hours, force_regenerate)

def train(resume: bool = False) -> bool:
    """Train TFT model."""
    return system.train(resume)

def test() -> bool:
    """Test TFT model inference."""
    return system.test()

def demo(minutes: int = 5) -> bool:
    """Run TFT monitoring demo."""
    return system.demo(minutes)

def status():
    """Show system status."""
    return system.status()

def cleanup():
    """Run system cleanup."""
    return system.cleanup()

def quick_start_guide():
    """Display TFT quick start guide."""
    print("""
🚀 TFT MONITORING SYSTEM - QUICK START
======================================

STREAMLINED TFT WORKFLOW:
✨ Pure PyTorch Forecasting implementation
✨ Temporal Fusion Transformer architecture
✨ Multi-horizon time-series prediction
✨ Attention-based feature importance
✨ Safetensors secure model storage

SIMPLE 4-STEP WORKFLOW:
1. setup()                    # Setup TFT environment
2. generate_dataset(168)      # Generate 1 week of data
3. train()                   # Train TFT model
4. test()                    # Test predictions

DEMO & MONITORING:
- demo(minutes=5)            # Live prediction demo
- status()                   # System status
- cleanup()                  # Clean old files

ADVANCED OPTIONS:
- generate_dataset(hours=720, force_regenerate=True)  # 30 days, force regen
- train(resume=True)         # Resume training from checkpoint

CONFIGURATION:
- Edit CONFIG in config.py for model parameters
- TFT optimizes automatically for your GPU
- Uses mixed precision training on CUDA

WHAT'S INCLUDED:
📊 Enhanced metrics generator (57 servers, realistic patterns)
🤖 TFT model (6-step ahead prediction, 24-step context)
⚡ GPU-optimized training (automatic batch size tuning)
🔒 Secure model storage (Safetensors format)
📈 Real-time inference with uncertainty quantification
🚨 Automated alert generation with thresholds

The system generates realistic server behavior patterns perfect for
training predictive models that can forecast server failures and
performance issues before they occur.
""")

# Display guide on import
print("🎯 TFT Monitoring System - Temporal Fusion Transformer")
print("📈 Multi-horizon time-series prediction for server monitoring")
print("⚡ PyTorch Forecasting + GPU acceleration")
print()
print("Type quick_start_guide() for usage instructions")
print("Type status() to check system status")

# Missing import for test scenarios
from datetime import timedelta
import random