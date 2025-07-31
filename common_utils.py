#!/usr/bin/env python3
"""
common_utils.py - Shared utilities for dataset generation, training, and inference
Centralizes data format handling and common operations to minimize script modifications
"""
import os, sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import torch as tf

logger = logging.getLogger(__name__)

# Standardized dataset formats
DATASET_FORMATS = {
    "language_dataset": {
        "filename": "language_dataset.json",
        "structure": {
            "metadata": {
                "generated_at": "ISO timestamp",
                "total_samples": "int",
                "session_id": "string",
                "models_per_question": "int",
                "generation_stats": "dict",
                "sample_distribution": "dict"
            },
            "samples": [
                {
                    "type": "string",
                    "prompt": "string", 
                    "response": "string",
                    "model": "string",
                    "session_id": "string",
                    "timestamp": "ISO timestamp"
                }
            ]
        }
    },
    "metrics_dataset": {
        "filename": "metrics_dataset.json",
        "structure": {
            "training_samples": [
                {
                    "id": "string",
                    "timestamp": "ISO timestamp",
                    "server_name": "string",
                    "metrics": "dict",
                    "status": "normal|anomaly",
                    "explanation": "string"
                }
            ],
            "metadata": {
                "generated_at": "ISO timestamp",
                "total_samples": "int",
                "anomaly_ratio": "float",
                "session_id": "string"
            }
        }
    }
}

def load_dataset_file(file_path: Path, expected_format: str = None) -> Dict[str, Any]:
    """Load dataset file with format validation."""
    try:
        if not file_path.exists():
            logger.warning(f"Dataset file not found: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✅ Loaded {file_path.name}")
        
        if expected_format and expected_format in DATASET_FORMATS:
            _validate_dataset_format(data, expected_format)
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return {}

def save_dataset_file(data: Dict[str, Any], file_path: Path, format_type: str = None) -> bool:
    """Save dataset file with format consistency."""
    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata if missing
        if format_type == "language_dataset" and "metadata" not in data:
            data["metadata"] = _create_language_metadata(data.get("samples", []))
        elif format_type == "metrics_dataset" and "metadata" not in data:
            data["metadata"] = _create_metrics_metadata(data.get("training_samples", []))
        
        # Atomic write
        temp_file = file_path.with_suffix('.json.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        temp_file.replace(file_path)
        logger.info(f"💾 Saved {file_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")
        if temp_file.exists():
            temp_file.unlink()
        return False

def _validate_dataset_format(data: Dict, format_type: str) -> bool:
    """Validate dataset follows expected format."""
    expected = DATASET_FORMATS.get(format_type, {})
    structure = expected.get("structure", {})
    
    if format_type == "language_dataset":
        if "samples" not in data:
            logger.warning("Language dataset missing 'samples' key")
            return False
        if not isinstance(data["samples"], list):
            logger.warning("Language dataset 'samples' is not a list")
            return False
            
    elif format_type == "metrics_dataset":
        if "training_samples" not in data:
            logger.warning("Metrics dataset missing 'training_samples' key")
            return False
        if not isinstance(data["training_samples"], list):
            logger.warning("Metrics dataset 'training_samples' is not a list")
            return False
    
    return True

def _create_language_metadata(samples: List[Dict]) -> Dict:
    """Create metadata for language dataset."""
    sample_distribution = {}
    for sample in samples:
        sample_type = sample.get('type', 'unknown')
        sample_distribution[sample_type] = sample_distribution.get(sample_type, 0) + 1
    
    return {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(samples),
        "sample_distribution": sample_distribution,
        "format_version": "1.0"
    }

def _create_metrics_metadata(samples: List[Dict]) -> Dict:
    """Create metadata for metrics dataset."""
    anomaly_count = sum(1 for s in samples if s.get('status') == 'anomaly')
    anomaly_ratio = anomaly_count / len(samples) if samples else 0
    
    return {
        "generated_at": datetime.now().isoformat(),
        "total_samples": len(samples),
        "anomaly_samples": anomaly_count,
        "normal_samples": len(samples) - anomaly_count,
        "anomaly_ratio": anomaly_ratio,
        "format_version": "1.0"
    }

def analyze_existing_datasets(training_dir: Path) -> Dict[str, Any]:
    """Analyze existing datasets using standardized format."""
    analysis = {
        "language_dataset": {"exists": False, "samples": 0, "distribution": {}},
        "metrics_dataset": {"exists": False, "samples": 0, "anomaly_ratio": 0}
    }
    
    # Language dataset
    lang_file = training_dir / DATASET_FORMATS["language_dataset"]["filename"]
    if lang_file.exists():
        lang_data = load_dataset_file(lang_file, "language_dataset")
        if lang_data:
            samples = lang_data.get("samples", [])
            analysis["language_dataset"] = {
                "exists": True,
                "samples": len(samples),
                "distribution": _get_sample_distribution(samples),
                "metadata": lang_data.get("metadata", {})
            }
    
    # Metrics dataset
    metrics_file = training_dir / DATASET_FORMATS["metrics_dataset"]["filename"]
    if metrics_file.exists():
        metrics_data = load_dataset_file(metrics_file, "metrics_dataset")
        if metrics_data:
            samples = metrics_data.get("training_samples", [])
            anomaly_count = sum(1 for s in samples if s.get('status') == 'anomaly')
            analysis["metrics_dataset"] = {
                "exists": True,
                "samples": len(samples),
                "anomaly_ratio": anomaly_count / len(samples) if samples else 0,
                "metadata": metrics_data.get("metadata", {})
            }
    
    return analysis

def _get_sample_distribution(samples: List[Dict]) -> Dict[str, int]:
    """Get distribution of sample types."""
    distribution = {}
    for sample in samples:
        sample_type = sample.get('type', 'unknown')
        distribution[sample_type] = distribution.get(sample_type, 0) + 1
    return distribution

def check_models_like_trainer(models_dir: Path) -> bool:
    """Check if trained model exists using exact same logic as trainer."""
    if not models_dir.exists():
        return False
    
    # Use EXACT same pattern as distilled_model_trainer.py
    model_dirs = list(models_dir.glob('distilled_monitoring_*'))
    if not model_dirs:
        return False
    
    # Sort by timestamp (newest first) - EXACT same as trainer
    model_dirs.sort(reverse=True)
    latest_model = model_dirs[0]
    
    # Verify using EXACT same files as trainer saves
    required_files = ['model.safetensors', 'config.json', 'training_metadata.json']
    return all((latest_model / f).exists() for f in required_files)

def get_latest_model_path(models_dir: Path) -> Optional[str]:
    """Get path to latest trained model."""
    if not check_models_like_trainer(models_dir):
        return None
    
    model_dirs = list(models_dir.glob('distilled_monitoring_*'))
    model_dirs.sort(reverse=True)
    return str(model_dirs[0])

def ensure_directory_structure(config: Dict) -> bool:
    """Ensure all required directories exist."""
    directories = [
        config.get('training_dir', './training/'),
        config.get('models_dir', './models/'),
        config.get('checkpoints_dir', './checkpoints/'),
        config.get('logs_dir', './logs/'),
        config.get('hf_cache_dir', './hf_cache/'),
        config.get('local_model_path', './local_models/'),
        config.get('pretrained_dir', './pretrained/'),
        config.get('static_fallback_path', './static_responses/'),
        config.get('data_config_dir', './data_config/')
    ]
    
    try:
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return False

def get_cache_status(config: Dict) -> Dict[str, Any]:
    """Get comprehensive cache status."""
    status = {
        "cache_type": "shared" if config.get("shared_cache_enabled") else "local",
        "directories": {},
        "total_size_gb": 0,
        "model_counts": {}
    }
    
    cache_dirs = {
        "hf_cache": config["hf_cache_dir"],
        "models": config["models_dir"],
        "local_models": config["local_model_path"],
        "pretrained": config["pretrained_dir"]
    }
    
    for name, directory in cache_dirs.items():
        dir_path = Path(directory)
        if dir_path.exists():
            try:
                # Calculate size
                total_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                size_gb = total_size / (1024**3)
                
                # Count items
                item_count = len([d for d in dir_path.iterdir() if d.is_dir()])
                
                status["directories"][name] = {
                    "path": str(directory),
                    "exists": True,
                    "size_gb": round(size_gb, 2),
                    "item_count": item_count
                }
                status["total_size_gb"] += size_gb
                status["model_counts"][name] = item_count
            except Exception as e:
                status["directories"][name] = {
                    "path": str(directory),
                    "exists": False,
                    "error": str(e),
                    "size_gb": 0,
                    "item_count": 0
                }
        else:
            status["directories"][name] = {
                "path": str(directory),
                "exists": False,
                "size_gb": 0,
                "item_count": 0
            }
    
    status["total_size_gb"] = round(status["total_size_gb"], 2)
    return status

# Export standard dataset file paths
def get_dataset_paths(training_dir: Path) -> Dict[str, Path]:
    """Get standardized dataset file paths."""
    return {
        "language_dataset": training_dir / DATASET_FORMATS["language_dataset"]["filename"],
        "metrics_dataset": training_dir / DATASET_FORMATS["metrics_dataset"]["filename"]
    }

# Progress tracking utilities
def save_generation_progress(progress_data: Dict, progress_file: Path) -> bool:
    """Save generation progress with error handling."""
    try:
        progress_data['last_updated'] = datetime.now().isoformat()
        
        temp_file = progress_file.with_suffix('.pkl.tmp')
        import pickle
        with open(temp_file, 'wb') as f:
            pickle.dump(progress_data, f)
        
        temp_file.replace(progress_file)
        return True
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")
        return False

def load_generation_progress(progress_file: Path) -> Optional[Dict]:
    """Load generation progress with error handling."""
    if not progress_file.exists():
        return None
    
    try:
        import pickle
        with open(progress_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load progress: {e}")
        return None

def load_generation_progress(progress_file: Path) -> Optional[Dict]:
    """Load generation progress with error handling."""
    if not progress_file.exists():
        return None
    
    try:
        import pickle
        with open(progress_file, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load progress: {e}")
        return None

def save_generation_progress(progress_data: Dict, progress_file: Path) -> bool:
    """Save generation progress with error handling."""
    try:
        progress_data['last_updated'] = datetime.now().isoformat()
        
        temp_file = progress_file.with_suffix('.pkl.tmp')
        import pickle
        with open(temp_file, 'wb') as f:
            pickle.dump(progress_data, f)
        
        temp_file.replace(progress_file)
        return True
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")
        return False

def log_message(message: str, level: str = "INFO"):
    """Log to both console and file for Spark compatibility."""
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)
    
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            log_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{log_time}] {level}: {message}\n")
            f.flush()
    except Exception:
        pass

def clear_model_discovery_cache():
    """Clear model discovery cache to force fresh discovery."""
    cache_file = Path("./model_discovery_cache.pkl")
    if cache_file.exists():
        cache_file.unlink()
        logger.info("🗑️ Model discovery cache cleared")

def get_optimal_workers():
    """Get optimal number of DataLoader workers."""
    if tf.cuda.is_available():
        return min(4, os.cpu_count() // 4)  # Less workers for GPU training
    return min(2, os.cpu_count() // 8)  # Minimal workers for CPU

def cleanup_generation_artifacts():
    """Clean up temporary generation artifacts periodically."""
    import shutil
    
    try:
        # Clean up pickle files after data is transferred
        progress_file = Path("./generation_progress.pkl")
        if progress_file.exists():
            # Check if data has been successfully transferred to datasets
            from config import CONFIG
            training_dir = Path(CONFIG.get('training_dir', './training/'))
            dataset_paths = get_dataset_paths(training_dir)
            
            # If both datasets exist and have data, we can clean up pickle
            if (dataset_paths['language_dataset'].exists() and 
                dataset_paths['metrics_dataset'].exists()):
                
                lang_data = load_dataset_file(dataset_paths['language_dataset'])
                metrics_data = load_dataset_file(dataset_paths['metrics_dataset'])
                
                if (lang_data.get('samples') and metrics_data.get('training_samples')):
                    progress_file.unlink()
                    logger.info("🗑️ Cleaned up generation progress pickle")
        
        # Clean up temporary files
        temp_files = list(Path('.').glob('*.tmp')) + list(Path('.').glob('*.pkl.tmp'))
        for temp_file in temp_files:
            temp_file.unlink()
            logger.info(f"🗑️ Cleaned up temp file: {temp_file}")
        
        # Clean up old discovery cache
        cache_file = Path("./model_discovery_cache.pkl")
        if cache_file.exists():
            from datetime import datetime, timedelta
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age > timedelta(hours=1):  # Cache older than 1 hour
                cache_file.unlink()
                logger.info("🗑️ Cleaned up old model discovery cache")
        
    except Exception as e:
        logger.error(f"Cleanup error: {e}")

def manage_training_checkpoints(checkpoints_dir: Path, models_dir: Path, keep_recent: int = 1, keep_best: int = 1):
    """Manage training checkpoints - handle both safetensors and pytorch formats."""
    try:
        if not checkpoints_dir.exists():
            return
        
        # Get all checkpoint files (both formats)
        checkpoint_files = (list(checkpoints_dir.glob('checkpoint_*.pt')) + 
                          list(checkpoints_dir.glob('checkpoint_*_metadata.json')))
        
        # Group by checkpoint name (safetensors creates multiple files)
        checkpoint_groups = {}
        for f in checkpoint_files:
            if f.name.endswith('_metadata.json'):
                base_name = f.name.replace('_metadata.json', '')
            elif f.name.endswith('.pt'):
                base_name = f.name.replace('.pt', '')
            else:
                continue
                
            if base_name not in checkpoint_groups:
                checkpoint_groups[base_name] = []
            checkpoint_groups[base_name].append(f)
        
        if len(checkpoint_groups) <= (keep_recent + keep_best):
            return  # Not enough checkpoints to clean up
        
        # Sort groups by modification time (newest first)
        sorted_groups = sorted(checkpoint_groups.items(), 
                             key=lambda x: max(f.stat().st_mtime for f in x[1]), 
                             reverse=True)
        
        # Keep the most recent ones
        groups_to_keep = set([name for name, _ in sorted_groups[:keep_recent]])
        
        # Find best performing checkpoints by loss
        best_checkpoints = []
        for group_name, files in checkpoint_groups.items():
            try:
                # Look for metadata file first (safetensors format)
                metadata_file = next((f for f in files if f.name.endswith('_metadata.json')), None)
                
                if metadata_file:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    loss = metadata.get('training_stats', {}).get('best_loss', float('inf'))
                else:
                    # Try PyTorch format
                    pt_file = next((f for f in files if f.name.endswith('.pt')), None)
                    if pt_file:
                        import torch
                        checkpoint = torch.load(pt_file, map_location='cpu')
                        loss = checkpoint.get('training_stats', {}).get('best_loss', float('inf'))
                    else:
                        loss = float('inf')
                
                best_checkpoints.append((group_name, loss))
            except Exception:
                continue
        
        # Sort by loss (best first) and keep the best ones
        best_checkpoints.sort(key=lambda x: x[1])
        groups_to_keep.update([name for name, _ in best_checkpoints[:keep_best]])
        
        # Remove checkpoint groups not in keep list
        removed_count = 0
        for group_name, files in checkpoint_groups.items():
            if group_name not in groups_to_keep:
                for file in files:
                    file.unlink()
                    removed_count += 1
                logger.info(f"🗑️ Removed checkpoint group: {group_name}")
        
        if removed_count > 0:
            logger.info(f"🧹 Checkpoint cleanup: removed {removed_count} files, kept {len(groups_to_keep)} checkpoint groups")
        
    except Exception as e:
        logger.error(f"Checkpoint cleanup error: {e}")

def periodic_cleanup(config: Dict):
    """Comprehensive periodic cleanup function."""
    logger.info("🧹 Starting periodic cleanup...")
    
    # Clean generation artifacts
    cleanup_generation_artifacts()
    
    # Manage checkpoints
    checkpoints_dir = Path(config.get('checkpoints_dir', './checkpoints/'))
    models_dir = Path(config.get('models_dir', './models/'))
    manage_training_checkpoints(checkpoints_dir, models_dir)
    
    # Clean up old log files (keep last 30 days)
    logs_dir = Path(config.get('logs_dir', './logs/'))
    if logs_dir.exists():
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=30)
        
        old_logs = []
        for log_file in logs_dir.glob('*.log'):
            if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                old_logs.append(log_file)
        
        for log_file in old_logs:
            log_file.unlink()
            logger.info(f"🗑️ Cleaned up old log: {log_file.name}")
        
        if old_logs:
            logger.info(f"🧹 Log cleanup: removed {len(old_logs)} old log files")
    
    logger.info("✅ Periodic cleanup completed")