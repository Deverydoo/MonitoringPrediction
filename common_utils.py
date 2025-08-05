#!/usr/bin/env python3
"""
common_utils.py - Complete TFT-focused utilities
Streamlined utilities for TFT training, Safetensors model storage, and metrics datasets
Includes all missing functions for TFT system
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

# TFT-specific dataset format
METRICS_DATASET_FORMAT = {
    "filename": "metrics_dataset.json",
    "structure": {
        "training_samples": [
            {
                "id": "string",
                "timestamp": "ISO timestamp",
                "server_name": "string",
                "metrics": "dict",
                "status": "normal|anomaly",
                "timeframe": "string",
                "explanation": "string",
                "severity": "string"
            }
        ],
        "metadata": {
            "generated_at": "ISO timestamp",
            "total_samples": "int",
            "time_span_hours": "int", 
            "servers_count": "int",
            "poll_interval_seconds": "int",
            "anomaly_ratio": "float",
            "format_version": "string"
        }
    }
}

def load_metrics_dataset(file_path: Path) -> Dict[str, Any]:
    """Load metrics dataset for TFT training."""
    try:
        if not file_path.exists():
            logger.warning(f"Metrics dataset not found: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ Loaded metrics dataset: {file_path.name}")
        
        # Validate structure
        if "training_samples" not in data:
            logger.warning("Metrics dataset missing 'training_samples'")
            return {}
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to load metrics dataset {file_path}: {e}")
        return {}

def load_dataset_file(file_path: Path) -> Dict[str, Any]:
    """Load dataset file - alias for load_metrics_dataset."""
    return load_metrics_dataset(file_path)

def get_dataset_paths(training_dir: Path) -> Dict[str, Path]:
    """Get paths to dataset files."""
    return {
        'metrics_dataset': training_dir / 'metrics_dataset.json',
        'training_dir': training_dir,
        'models_dir': training_dir.parent / 'models',
        'checkpoints_dir': training_dir.parent / 'checkpoints',
        'logs_dir': training_dir.parent / 'logs'
    }

def save_metrics_dataset(data: Dict[str, Any], file_path: Path) -> bool:
    """Save metrics dataset with metadata."""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata if missing
        if "metadata" not in data:
            data["metadata"] = create_metrics_metadata(data.get("training_samples", []))
        
        # Atomic write
        temp_file = file_path.with_suffix('.json.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        temp_file.replace(file_path)
        logger.info(f"💾 Saved metrics dataset: {file_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save metrics dataset {file_path}: {e}")
        if temp_file.exists():
            temp_file.unlink()
        return False

def create_metrics_metadata(samples: List[Dict]) -> Dict:
    """Create metadata for metrics dataset."""
    anomaly_count = sum(1 for s in samples if s.get('status') == 'anomaly')
    anomaly_ratio = anomaly_count / len(samples) if samples else 0
    
    # Extract time span from samples
    if samples:
        timestamps = [datetime.fromisoformat(s['timestamp'].replace('Z', '+00:00')) 
                     for s in samples if 'timestamp' in s]
        if timestamps:
            time_span_hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600
        else:
            time_span_hours = 0
    else:
        time_span_hours = 0
    
    # Count unique servers
    servers = set(s.get('server_name', 'unknown') for s in samples)
    
    return {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(samples),
        "anomaly_samples": anomaly_count,
        "normal_samples": len(samples) - anomaly_count,
        "anomaly_ratio": anomaly_ratio,
        "time_span_hours": time_span_hours,
        "servers_count": len(servers),
        "format_version": "2.0",
        "enhanced": True
    }

def analyze_metrics_dataset(training_dir: Path) -> Dict[str, Any]:
    """Analyze existing metrics dataset."""
    analysis = {
        "exists": False,
        "samples": 0,
        "anomaly_ratio": 0,
        "metadata": {}
    }
    
    metrics_file = training_dir / METRICS_DATASET_FORMAT["filename"]
    if metrics_file.exists():
        data = load_metrics_dataset(metrics_file)
        if data:
            samples = data.get("training_samples", [])
            anomaly_count = sum(1 for s in samples if s.get('status') == 'anomaly')
            
            analysis = {
                "exists": True,
                "samples": len(samples),
                "anomaly_ratio": anomaly_count / len(samples) if samples else 0,
                "metadata": data.get("metadata", {}),
                "enhanced_format": data.get("metadata", {}).get("enhanced", False)
            }
    
    return analysis

def check_tft_model_exists(models_dir: Path) -> bool:
    """Check if trained TFT model exists."""
    if not models_dir.exists():
        return False
    
    # Look for TFT model directories
    model_dirs = list(models_dir.glob('tft_monitoring_*'))
    if not model_dirs:
        return False
    
    # Sort by timestamp (newest first)
    model_dirs.sort(reverse=True)
    latest_model = model_dirs[0]
    
    # Verify TFT model files exist
    required_files = ['model.safetensors', 'config.json', 'training_metadata.json']
    return all((latest_model / f).exists() for f in required_files)

def check_models_like_trainer(models_dir: Path) -> bool:
    """Check if models exist (alias for backward compatibility)."""
    return check_tft_model_exists(models_dir)

def get_latest_tft_model_path(models_dir: Path) -> Optional[str]:
    """Get path to latest trained TFT model."""
    if not check_tft_model_exists(models_dir):
        return None
    
    model_dirs = list(models_dir.glob('tft_monitoring_*'))
    model_dirs.sort(reverse=True)
    return str(model_dirs[0])

def ensure_tft_directories(config: Dict) -> bool:
    """Ensure all required TFT directories exist."""
    directories = [
        config.get('training_dir', './training/'),
        config.get('models_dir', './models/'),
        config.get('checkpoints_dir', './checkpoints/'),
        config.get('logs_dir', './logs/'),
        config.get('data_config_dir', './data_config/')
    ]
    
    try:
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False

def cleanup_tft_artifacts(config: Dict):
    """Clean up old TFT training artifacts."""
    if not config.get("cleanup_enabled", True):
        return
    
    try:
        # Clean up old checkpoints
        checkpoints_dir = Path(config.get('checkpoints_dir', './checkpoints/'))
        if checkpoints_dir.exists():
            checkpoint_files = list(checkpoints_dir.glob('*.ckpt'))
            keep_recent = config.get('keep_recent_checkpoints', 3)
            
            if len(checkpoint_files) > keep_recent:
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                to_remove = checkpoint_files[keep_recent:]
                
                for checkpoint in to_remove:
                    checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint.name}")
                
                logger.info(f"🧹 Cleaned up {len(to_remove)} old checkpoints")
        
        # Clean up old logs
        logs_dir = Path(config.get('logs_dir', './logs/'))
        if logs_dir.exists():
            from datetime import timedelta
            retention_days = config.get('log_retention_days', 30)
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            old_logs = []
            for log_file in logs_dir.rglob('*.log'):
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    old_logs.append(log_file)
            
            for log_file in old_logs:
                log_file.unlink()
            
            if old_logs:
                logger.info(f"🧹 Removed {len(old_logs)} old log files")
        
        # Clean up temporary files
        temp_files = list(Path('.').glob('*.tmp')) + list(Path('.').glob('*.pkl.tmp'))
        for temp_file in temp_files:
            temp_file.unlink()
            logger.debug(f"Removed temp file: {temp_file}")
        
        if temp_files:
            logger.info(f"🧹 Removed {len(temp_files)} temporary files")
                
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def get_optimal_workers() -> int:
    """Get optimal number of DataLoader workers for TFT training."""
    try:
        import torch
        if torch.cuda.is_available():
            return min(4, os.cpu_count() // 4)  # Less workers for GPU training
        return min(2, os.cpu_count() // 8)  # Minimal workers for CPU
    except ImportError:
        return 2

def log_message(message: str, level: str = "INFO"):
    """Simple logging function for TFT training."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"[{timestamp}] {message}")
    
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)

def save_tft_model_safely(model_state: Dict, model_dir: Path, metadata: Dict):
    """Save TFT model using Safetensors format."""
    try:
        from safetensors.torch import save_file
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model weights with Safetensors
        model_path = model_dir / 'model.safetensors'
        save_file(model_state, str(model_path))
        logger.info(f"💾 TFT model weights saved: {model_path}")
        
        # Save metadata
        metadata_path = model_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save model config
        config_path = model_dir / 'config.json'
        model_config = {
            'model_type': 'TemporalFusionTransformer',
            'framework': 'pytorch_forecasting',
            'saved_at': datetime.now().isoformat(),
            'safetensors_format': True
        }
        
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        logger.info(f"✅ TFT model saved successfully: {model_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save TFT model: {e}")
        return False

def load_tft_model_safely(model_dir: Path):
    """Load TFT model from Safetensors format."""
    try:
        from safetensors.torch import load_file
        
        model_path = model_dir / 'model.safetensors'
        if not model_path.exists():
            raise FileNotFoundError(f"TFT model not found: {model_path}")
        
        # Load model state
        model_state = load_file(str(model_path))
        logger.info(f"📥 TFT model weights loaded: {model_path}")
        
        # Load metadata
        metadata_path = model_dir / 'training_metadata.json'
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        return model_state, metadata
        
    except Exception as e:
        logger.error(f"❌ Failed to load TFT model: {e}")
        return None, None

def validate_metrics_dataset(dataset_path: Path) -> bool:
    """Validate metrics dataset format for TFT training."""
    try:
        data = load_metrics_dataset(dataset_path)
        if not data:
            return False
        
        # Check required structure
        if "training_samples" not in data:
            logger.error("Missing 'training_samples' in dataset")
            return False
        
        samples = data["training_samples"]
        if not samples:
            logger.error("No samples found in dataset")
            return False
        
        # Validate sample structure
        sample = samples[0]
        required_fields = ['id', 'timestamp', 'server_name', 'metrics', 'status']
        missing_fields = [field for field in required_fields if field not in sample]
        
        if missing_fields:
            logger.error(f"Sample missing required fields: {missing_fields}")
            return False
        
        # Validate metrics structure
        metrics = sample['metrics']
        if not isinstance(metrics, dict):
            logger.error("Metrics field must be a dictionary")
            return False
        
        # Check for numeric metrics
        numeric_metrics = [k for k, v in metrics.items() if isinstance(v, (int, float))]
        if len(numeric_metrics) < 3:
            logger.error("Need at least 3 numeric metrics for TFT training")
            return False
        
        logger.info(f"✅ Dataset validation passed: {len(samples)} samples")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset validation failed: {e}")
        return False

def get_dataset_stats(dataset_path: Path) -> Dict[str, Any]:
    """Get comprehensive statistics about the metrics dataset."""
    try:
        data = load_metrics_dataset(dataset_path)
        if not data:
            return {}
        
        samples = data.get("training_samples", [])
        metadata = data.get("metadata", {})
        
        if not samples:
            return {"error": "No samples found"}
        
        # Basic stats
        stats = {
            "total_samples": len(samples),
            "time_span_hours": metadata.get("time_span_hours", 0),
            "servers_count": metadata.get("servers_count", 0),
            "anomaly_ratio": metadata.get("anomaly_ratio", 0),
            "format_version": metadata.get("format_version", "unknown"),
            "enhanced": metadata.get("enhanced", False)
        }
        
        # Status distribution
        status_counts = {}
        for sample in samples:
            status = sample.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        stats["status_distribution"] = status_counts
        
        # Server distribution
        server_counts = {}
        for sample in samples:
            server = sample.get("server_name", "unknown")
            server_counts[server] = server_counts.get(server, 0) + 1
        
        stats["servers_with_most_samples"] = sorted(
            server_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # Metrics analysis
        if samples:
            sample_metrics = samples[0].get("metrics", {})
            numeric_metrics = [k for k, v in sample_metrics.items() if isinstance(v, (int, float))]
            stats["numeric_metrics_count"] = len(numeric_metrics)
            stats["sample_metrics"] = list(sample_metrics.keys())
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get dataset stats: {e}")
        return {"error": str(e)}

def create_tft_training_summary(model_dir: Path, training_stats: Dict) -> Dict[str, Any]:
    """Create training summary for TFT model."""
    summary = {
        "model_type": "TemporalFusionTransformer",
        "framework": "pytorch_forecasting",
        "model_path": str(model_dir),
        "training_completed": True,
        "created_at": datetime.now().isoformat(),
        **training_stats
    }
    
    return summary

def analyze_existing_datasets(training_dir: Path) -> Dict[str, Any]:
    """
    Analyze existing datasets in the training directory.
    
    Args:
        training_dir: Path to training directory
        
    Returns:
        Dict containing analysis of available datasets
    """
    analysis = {
        'datasets_found': [],
        'metrics_dataset': {
            'exists': False,
            'path': None,
            'samples': 0,
            'size_mb': 0,
            'anomaly_ratio': 0,
            'time_span_hours': 0,
            'servers_count': 0,
            'format_version': 'unknown',
            'enhanced': False
        },
        'total_datasets': 0,
        'ready_for_training': False,
        'recommendations': []
    }
    
    try:
        if not training_dir.exists():
            analysis['recommendations'].append("Training directory does not exist - run setup() first")
            return analysis
        
        # Check for metrics dataset (primary dataset)
        metrics_path = training_dir / 'metrics_dataset.json'
        if metrics_path.exists():
            analysis['datasets_found'].append('metrics_dataset')
            analysis['metrics_dataset']['exists'] = True
            analysis['metrics_dataset']['path'] = str(metrics_path)
            
            # Get file size
            file_size = metrics_path.stat().st_size
            analysis['metrics_dataset']['size_mb'] = file_size / (1024 * 1024)
            
            # Load and analyze dataset content
            try:
                dataset_stats = get_dataset_stats(metrics_path)
                if dataset_stats and 'error' not in dataset_stats:
                    analysis['metrics_dataset'].update({
                        'samples': dataset_stats.get('total_samples', 0),
                        'anomaly_ratio': dataset_stats.get('anomaly_ratio', 0),
                        'time_span_hours': dataset_stats.get('time_span_hours', 0),
                        'servers_count': dataset_stats.get('servers_count', 0),
                        'format_version': dataset_stats.get('format_version', 'unknown'),
                        'enhanced': dataset_stats.get('enhanced', False)
                    })
                    
                    logger.info(f"✅ Metrics dataset: {analysis['metrics_dataset']['samples']:,} samples")
                else:
                    analysis['recommendations'].append("Metrics dataset exists but appears corrupted")
                    
            except Exception as e:
                logger.warning(f"Failed to analyze metrics dataset: {e}")
                analysis['recommendations'].append("Failed to read metrics dataset - may be corrupted")
        
        # Check for other potential dataset files
        dataset_patterns = [
            ('language_dataset.json', 'language_dataset'),
            ('training_data.json', 'training_data'),
            ('server_logs.json', 'server_logs'),
            ('enhanced_metrics.json', 'enhanced_metrics')
        ]
        
        for filename, dataset_type in dataset_patterns:
            file_path = training_dir / filename
            if file_path.exists():
                analysis['datasets_found'].append(dataset_type)
                try:
                    file_size = file_path.stat().st_size / (1024 * 1024)
                    logger.info(f"✅ Found {dataset_type}: {file_size:.1f} MB")
                except Exception:
                    pass
        
        # Count total datasets
        analysis['total_datasets'] = len(analysis['datasets_found'])
        
        # Determine if ready for training
        metrics_ready = (
            analysis['metrics_dataset']['exists'] and 
            analysis['metrics_dataset']['samples'] > 1000  # Need reasonable amount of data
        )
        
        analysis['ready_for_training'] = metrics_ready
        
        # Generate recommendations
        if not analysis['metrics_dataset']['exists']:
            analysis['recommendations'].append("No metrics dataset found - run generate_dataset() first")
        elif analysis['metrics_dataset']['samples'] < 1000:
            analysis['recommendations'].append("Metrics dataset too small - generate more data")
        elif analysis['metrics_dataset']['samples'] < 10000:
            analysis['recommendations'].append("Consider generating more training data for better model performance")
        
        if analysis['metrics_dataset']['time_span_hours'] < 24:
            analysis['recommendations'].append("Dataset covers less than 24 hours - consider longer time span")
        
        if analysis['metrics_dataset']['anomaly_ratio'] < 0.05:
            analysis['recommendations'].append("Low anomaly ratio - model may not learn to detect issues well")
        elif analysis['metrics_dataset']['anomaly_ratio'] > 0.3:
            analysis['recommendations'].append("High anomaly ratio - may need to balance dataset")
        
        if not analysis['metrics_dataset']['enhanced']:
            analysis['recommendations'].append("Consider regenerating with enhanced format for better features")
        
        # Log summary
        if analysis['ready_for_training']:
            logger.info("✅ Datasets ready for TFT training")
        else:
            logger.warning("⚠️  Datasets not ready for training")
            
        if analysis['recommendations']:
            logger.info("💡 Recommendations:")
            for rec in analysis['recommendations']:
                logger.info(f"   - {rec}")
                
    except Exception as e:
        logger.error(f"Failed to analyze datasets: {e}")
        analysis['recommendations'].append(f"Analysis failed: {str(e)}")
    
    return analysis

# Export key functions for TFT workflow
__all__ = [
    'load_metrics_dataset',
    'load_dataset_file',
    'get_dataset_paths',
    'save_metrics_dataset', 
    'analyze_metrics_dataset',
    'check_tft_model_exists',
    'check_models_like_trainer',
    'get_latest_tft_model_path',
    'ensure_tft_directories',
    'cleanup_tft_artifacts',
    'validate_metrics_dataset',
    'get_dataset_stats',
    'save_tft_model_safely',
    'load_tft_model_safely',
    'log_message',
    'get_optimal_workers'
]