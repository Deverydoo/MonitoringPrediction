# config.py - Enhanced configuration with PyTorch/TensorFlow selection and dynamic discovery
import os
import json
import logging
import time
import random
import subprocess
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

# Framework selection - set before imports
FRAMEWORK = os.environ.get('ML_FRAMEWORK', 'pytorch').lower()  # 'pytorch' or 'tensorflow'

if FRAMEWORK == 'tensorflow':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF logging
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable optimizations
    try:
        import tensorflow as tf
        # Configure GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        FRAMEWORK_AVAILABLE = True
        DEVICE_TYPE = 'GPU' if gpus else 'CPU'
    except ImportError:
        FRAMEWORK_AVAILABLE = False
        DEVICE_TYPE = 'CPU'
else:  # PyTorch default
    os.environ['USE_TF'] = 'NO'
    os.environ['USE_TORCH'] = 'YES'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    try:
        import torch
        FRAMEWORK_AVAILABLE = True
        DEVICE_TYPE = 'CUDA' if torch.cuda.is_available() else 'CPU'
    except ImportError:
        FRAMEWORK_AVAILABLE = False
        DEVICE_TYPE = 'CPU'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced core configuration with framework selection
CONFIG = {
    # Framework selection
    "framework": FRAMEWORK,
    "framework_available": FRAMEWORK_AVAILABLE,
    "device_type": DEVICE_TYPE,

    # Project directories
    "training_dir": "./training/",
    "checkpoints_dir": "./checkpoints/",
    "logs_dir": "./logs/",
    "data_config_dir": "./data_config/",

    # Cleanup settings.
    "cleanup_enabled": True,
    "cleanup_interval_hours": 6,  # Run cleanup every 6 hours
    "keep_recent_checkpoints": 1,
    "keep_best_checkpoints": 1,
    "log_retention_days": 30,
    "auto_cleanup_on_epoch": True,
    
    # Model and training settings
    "model_name": "bert-base-uncased",
    "max_length": 512,
    "batch_size": 24,
    "learning_rate": 2e-5,
    "epochs": 13,
    "warmup_steps": 100,
    "weight_decay": 0.01,

    

    # Dask GPU acceleration (for production Spark environments)
    "use_dask_gpu": True,  # Enable Dask GPU acceleration when available
    "dask_scheduler_address": os.environ.get('DASK_SCHEDULER_ADDRESS', ''),  # Connect to existing cluster
    "dask_gpu_memory_limit": "auto",  # GPU memory limit for Dask workers
    "dask_workers": "auto",  # Number of Dask workers (auto-detect)

    # Framework-specific optimizations
    "mixed_precision": True if DEVICE_TYPE in ['GPU', 'CUDA'] else False,
    "gradient_accumulation_steps": 2,
    "dataloader_num_workers": 4,
    "pin_memory": True if DEVICE_TYPE in ['GPU', 'CUDA'] else False,
    "persistent_workers": True,
    
    # TensorFlow specific settings
    "tf_mixed_precision_policy": "mixed_float16" if DEVICE_TYPE == 'GPU' else "float32",
    "tf_xla_compile": True,
    "tf_memory_growth": True,
    
    # PyTorch specific settings
    "torch_compile": False,  # PyTorch 2.0+ compilation
    "torch_dtype": "float32",
    "use_cuda": DEVICE_TYPE == 'CUDA',
    "force_cpu": False,
    
    # Dynamic discovery settings
    "auto_discover_models": True,
    "auto_discover_yaml": True,
    "yaml_config_dir": os.environ.get('DISTILLED_YAML_DIR', "./data_config/"),
    "model_discovery_timeout": 15,
    "yaml_discovery_interval": 300,  # Re-scan every 5 minutes
    
    # Flexible cache directories
    "hf_cache_dir": os.environ.get('DISTILLED_HF_CACHE', "./hf_cache/"),
    "models_dir": os.environ.get('DISTILLED_MODELS', "./models/"),
    "local_model_path": os.environ.get('DISTILLED_LOCAL_MODELS', "./local_models/"),
    "pretrained_dir": os.environ.get('DISTILLED_PRETRAINED', "./pretrained/"),
    "shared_cache_enabled": os.environ.get('DISTILLED_SHARED_CACHE', 'false').lower() == 'true',

    # Dataset generation with dynamic calculation
    "calculate_samples_dynamically": True,
    "language_samples": 9000,
    "metrics_samples": 200000,
    "base_samples_per_yaml": 50,
    "variety_multiplier": 1.5,
    "quality_over_quantity": True,
    "anomaly_ratio": 0.2,
    
    # Performance settings
    "response_quality_threshold": 15,
    "api_rate_limit": 0.1,
    "max_retries": 3,
    "retry_backoff": 2.0,
    
    # Continual learning framework
    "continual_learning_enabled": True,
    "learning_batch_size": 50,
    "threshold_adjustment_rate": 0.05,
    "feedback_retention_days": 30,
    "auto_threshold_adjustment": True,
    "learning_rate_decay": 0.95,
    "performance_tracking_window": 100,

    # Remote LLM (primary)
    "llm_url": os.environ.get('REMOTE_LLM_URL', ""),
    "llm_key": os.environ.get('REMOTE_LLM_KEY', ""),
    "llm_timeout": 30,
    "llm_max_tokens": 2000,
    
    # Ollama configuration with enhanced discovery
    "ollama_enabled": True,
    "ollama_url": os.environ.get('OLLAMA_URL', "http://localhost:11434"),
    "ollama_timeout": 90,
    "ollama_max_tokens": 20000,
    "ollama_temperature": 0.7,
    "ollama_auto_discover": True,
    "ollama_discovery_interval": 240,
    
    # Ollama performance optimizations
    "ollama_parallel_requests": 6,
    "ollama_max_loaded_models": 3,
    "ollama_max_queue": 1024,
    "ollama_gpu_layers": -1,
    "ollama_context_size": 4096,
    "ollama_batch_size": 512,
    "ollama_flash_attention": True,
    "ollama_keep_alive": "10m",
    
    # Model rotation and variety settings
    "model_rotation_enabled": True,
    "model_swap_interval": 25,
    "models_per_question": 1,
    "max_concurrent_models": 4,
    "model_pool_size": 20,
    "preload_next_model": True,
    "generation_batch_size": 4,
    "model_priority_weights": {
        "remote": 1.0,
        "ollama": 0.9,
        "local": 0.5,
        "static": 0.1
    },
    
    # Local model fallback with enhanced discovery
    "local_model_enabled": True,
    "local_model_max_tokens": 4000,
    "local_model_temperature": 0.7,
    "local_model_auto_scan": True,
    "local_model_scan_dirs": [
        "./local_models/",
        "./hf_cache/",
        "./pretrained/",
        os.environ.get('HF_HOME', ''),
        os.path.expanduser("~/.cache/huggingface/")
    ],
    
    # Static fallback
    "enable_static_fallback": True,
    "static_fallback_path": "./static_responses/",
    "static_response_categories": ["technical", "errors", "troubleshooting", "best_practices"],

    # Data source integration
    "integrate_real_data": True,
    "real_data_samples_per_round": 50,
    "real_data_quality_threshold": 0.8,
    
    # Enhanced discovery
    "discovery_optimization": True,
    "cache_discovery_results": True,
    "discovery_cache_ttl": 300,
    
    # Performance optimizations
    "parallel_model_discovery": True,
    "batch_ollama_discovery": True,
    "efficient_yaml_scanning": True,
    
    # Frequent saves
    "save_frequency": "rotation_round",
    "rotation_round_size": 50,
    "time_based_save_interval": 300,
    
    # Alert thresholds (will be dynamically adjusted)
    "alert_thresholds": {
        "cpu_usage": 80.0,
        "memory_usage": 85.0,
        "disk_usage": 90.0,
        "load_average": 5.0,
        "java_heap_usage": 85.0,
        "java_gc_time": 15.0,
        "network_io_rate": 80.0,
        "disk_io_rate": 75.0,
        "anomaly_score": 0.7
    },
    
    # Multi-source data integration
    "splunk_integration": {
        "enabled": os.environ.get('SPLUNK_ENABLED', 'false').lower() == 'true',
        "url": os.environ.get('SPLUNK_URL', ''),
        "token": os.environ.get('SPLUNK_TOKEN', ''),
        "timeout": 30,
        "max_results": 1000,
        "default_queries": {
            "vemkd_logs": 'index=linux sourcetype="vemkd" | head 1000',
            "error_logs": 'index=linux ("error" OR "exception" OR "critical") | head 500',
            "performance": 'index=system | stats avg(cpu_usage), avg(memory_usage) by host',
            "spectrum_logs": 'index=spectrum sourcetype="conductor" | head 1000',
            "security_events": 'index=security | head 100'
        }
    },
    
    "jira_integration": {
        "enabled": os.environ.get('JIRA_ENABLED', 'false').lower() == 'true',
        "url": os.environ.get('JIRA_URL', ''),
        "username": os.environ.get('JIRA_USER', ''),
        "token": os.environ.get('JIRA_TOKEN', ''),
        "timeout": 20,
        "project_keys": os.environ.get('JIRA_PROJECTS', 'IT,OPS,INFRA').split(','),
        "issue_types": ["Bug", "Incident", "Task", "Story"],
        "max_issues_per_query": 100
    },
    
    "confluence_integration": {
        "enabled": os.environ.get('CONFLUENCE_ENABLED', 'false').lower() == 'true',
        "url": os.environ.get('CONFLUENCE_URL', ''),
        "username": os.environ.get('CONFLUENCE_USER', ''),
        "token": os.environ.get('CONFLUENCE_TOKEN', ''),
        "spaces": os.environ.get('CONFLUENCE_SPACES', 'IT,OPS').split(','),
        "content_types": ["page", "blogpost"]
    },
    
    "spectrum_integration": {
        "enabled": os.environ.get('SPECTRUM_ENABLED', 'false').lower() == 'true',
        "url": os.environ.get('SPECTRUM_URL', ''),
        "username": os.environ.get('SPECTRUM_USER', ''),
        "password": os.environ.get('SPECTRUM_PASS', ''),
        "endpoints": [
            "/platform/rest/conductor/v1/clusters",
            "/platform/rest/conductor/v1/consumers",
            "/platform/rest/conductor/v1/resourcegroups",
            "/platform/rest/conductor/v1/workloads"
        ],
        "polling_interval": 60
    },
    
    # Enhanced model configurations
    "local_pretrained_models": {
        "bert-base-uncased": "./pretrained/bert-base-uncased/",
        "distilbert-base-uncased": "./pretrained/distilbert-base-uncased/",
        "microsoft/DialoGPT-medium": "./local_models/microsoft_DialoGPT-medium/",
        "microsoft/DialoGPT-small": "./local_models/microsoft_DialoGPT-small/",
    },
    
    # Model performance tracking
    "track_model_performance": True,
    "model_performance_metrics": ["response_time", "quality_score", "success_rate"],
    "model_rotation_based_on_performance": True,

    # Dynamic cache configuration
    "dynamic_cache_setup": True,
    
    # Discovery optimization
    "discovery_cache_ttl": 300,
    "rediscovery_interval": 1800,
    "max_ollama_models": 20,
    "max_local_models": 10,
    
    # Generation optimization
    "rotation_round_size": 50,
    "yaml_discovery_interval": 300,
    
    # Performance tracking
    "track_generation_performance": True,
    "log_performance_every": 100,
    
    # Advanced caching
    "cache_responses": True,
    "cache_ttl": 3600,
    "cache_max_size": 10000,
    "cache_compression": True,

    # Environment detection
    "auto_detect_shared_storage": True,
    "shared_storage_indicators": ["/shared/", "/nfs/", "/mnt/shared/", "//"],
    
    # Shared storage alternatives (if detected)
    "shared_hf_cache": os.environ.get('SHARED_HF_CACHE', "/shared/ml_models/huggingface/"),
    "shared_models": os.environ.get('SHARED_MODELS', "/shared/ml_models/distilled/"),
    "shared_pretrained": os.environ.get('SHARED_PRETRAINED', "/shared/ml_models/pretrained/"),
    
    # Cache optimization settings
    "enable_cache_compression": True,
    "cache_cleanup_threshold_gb": 50,
    "cache_retention_days": 30,
    "verify_cache_integrity": True,
    
    # Performance settings
    "parallel_downloads": 3,
    "chunk_size_mb": 10,
    "enable_symlinks": True,
    "cache_lock_timeout": 300,
}

def detect_training_environment():
    """Enhanced training environment detection with framework optimization."""
    if CONFIG["force_cpu"]:
        return "cpu"
    
    if CONFIG["framework"] == "tensorflow":
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_name = tf.config.experimental.get_device_details(gpus[0])['device_name']
                memory_info = tf.config.experimental.get_memory_info(f'GPU:0')
                logger.info(f"🎮 TensorFlow GPU: {gpu_name}")
                
                # Set TensorFlow optimizations
                if CONFIG['tf_mixed_precision_policy'] == 'mixed_float16':
                    tf.keras.mixed_precision.set_global_policy('mixed_float16')
                
                return "tensorflow_gpu"
            else:
                logger.info("💻 TensorFlow CPU")
                return "tensorflow_cpu"
        except ImportError:
            logger.warning("TensorFlow not available")
    
    else:  # PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                
                logger.info(f"🎮 PyTorch CUDA GPU: {gpu_name} ({gpu_memory}GB)")
                
                # Set optimizations based on GPU
                if gpu_memory >= 8:
                    CONFIG['batch_size'] = min(CONFIG['batch_size'], 16)
                elif gpu_memory >= 4:
                    CONFIG['batch_size'] = min(CONFIG['batch_size'], 8)
                else:
                    CONFIG['batch_size'] = min(CONFIG['batch_size'], 4)
                
                return "pytorch_cuda"
            
            # Check Apple Silicon (MPS)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                logger.info("🍎 PyTorch Apple Silicon (MPS)")
                CONFIG['batch_size'] = min(CONFIG['batch_size'], 8)
                return "pytorch_mps"
        except ImportError:
            logger.warning("PyTorch not available")
    
    # Check for Spark
    try:
        from pyspark.sql import SparkSession
        logger.info("⚡ Spark environment detected")
        return "spark"
    except ImportError:
        pass
    
    # CPU fallback
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    logger.info(f"💻 CPU training: {cpu_count} cores")
    CONFIG['batch_size'] = min(CONFIG['batch_size'], max(2, cpu_count // 2))
    
    return "cpu"

def setup_dynamic_cache_environment():
    """Setup cache directories with automatic shared storage detection."""
    
    # Detect shared storage environment
    shared_detected = False
    if CONFIG["auto_detect_shared_storage"]:
        current_dir = os.getcwd()
        for indicator in CONFIG["shared_storage_indicators"]:
            if indicator in current_dir:
                shared_detected = True
                logger.info(f"🔍 Shared storage detected: {indicator} in {current_dir}")
                break
    
    # Configure cache paths based on environment
    if shared_detected:
        CONFIG.update({
            "hf_cache_dir": CONFIG["shared_hf_cache"],
            "models_dir": CONFIG["shared_models"],
            "local_model_path": CONFIG["shared_pretrained"],
            "pretrained_dir": CONFIG["shared_pretrained"],
            "shared_cache_enabled": True
        })
        logger.info("📁 Using shared storage cache configuration")
    else:
        logger.info("📁 Using local cache configuration")
    
    # Create and verify cache directories
    cache_dirs = [
        CONFIG["hf_cache_dir"],
        CONFIG["models_dir"],
        CONFIG["local_model_path"],
        CONFIG["pretrained_dir"]
    ]
    
    for cache_dir in cache_dirs:
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Test write access
            test_file = Path(cache_dir) / ".cache_test"
            test_file.write_text("test")
            test_file.unlink()
            
            logger.info(f"✅ Cache ready: {cache_dir}")
            
        except Exception as e:
            logger.error(f"❌ Cache setup failed for {cache_dir}: {e}")
            # Fallback to local
            fallback_dir = f"./{Path(cache_dir).name}/"
            CONFIG[cache_dir] = fallback_dir
            Path(fallback_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"🔄 Fallback to: {fallback_dir}")
    
    return True

def get_cache_status():
    """Get comprehensive cache status."""
    status = {
        "framework": CONFIG["framework"],
        "framework_available": CONFIG["framework_available"],
        "device_type": CONFIG["device_type"],
        "cache_type": "shared" if CONFIG.get("shared_cache_enabled") else "local",
        "directories": {},
        "total_size_gb": 0,
        "model_counts": {}
    }
    
    cache_dirs = {
        "hf_cache": CONFIG["hf_cache_dir"],
        "models": CONFIG["models_dir"],
        "local_models": CONFIG["local_model_path"],
        "pretrained": CONFIG["pretrained_dir"]
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

class EnhancedModelChain:
    """Optimized model chain with consolidated discovery and intelligent rotation."""
    
    _discovery_cache_file = Path("./model_discovery_cache.pkl")
    _discovery_cache = {}
    _last_discovery_time = 0
    
    def __init__(self):
        # Single discovery method
        self.available_models = self._discover_all_models()
        self.model_rotation_pool = self._build_rotation_pool()
        
        self.current_model_index = 0
        self.models_per_question = CONFIG.get('models_per_question', 1)
        self.rotation_counter = 0
        self.rotation_interval = CONFIG.get('model_swap_interval', 25)
        self.performance_tracker = {}
        
        logger.info(f"✅ Model chain initialized: {len(self.available_models)} models")
    
    def _discover_all_models(self) -> Dict[str, Dict]:
        """SINGLE consolidated model discovery method."""
        current_time = time.time()
        
        # Use cached results if recent
        if (current_time - self._last_discovery_time) < CONFIG.get('discovery_cache_ttl', 300):
            if self._discovery_cache:
                return self._discovery_cache.copy()
        
        models = {}
        
        # 1. Remote API models
        if CONFIG.get('llm_url') and CONFIG.get('llm_key'):
            if self._test_remote_api():
                models['remote_primary'] = {
                    'type': 'remote',
                    'available': True,
                    'priority': 1,
                    'performance_score': 1.0
                }
        
        # 2. Ollama models (batch discovery)
        if CONFIG.get('ollama_enabled'):
            ollama_models = self._discover_ollama_batch()
            for i, model_name in enumerate(ollama_models):
                key = f'ollama_{model_name.replace(":", "_").replace("/", "_")}'
                models[key] = {
                    'type': 'ollama',
                    'model_name': model_name,
                    'available': True,
                    'priority': 2,
                    'performance_score': 0.9 - (i * 0.01)
                }
        
        # 3. Local models (efficient scanning)
        if CONFIG.get('local_model_enabled'):
            local_models = self._discover_local_models()
            for i, (model_name, model_path) in enumerate(local_models):
                key = f'local_{model_name.replace("/", "_")}'
                models[key] = {
                    'type': 'local',
                    'model_name': model_name,
                    'model_path': model_path,
                    'available': True,
                    'priority': 3,
                    'performance_score': 0.7 - (i * 0.05)
                }
        
        # 4. Static fallback
        models['static'] = {
            'type': 'static',
            'available': True,
            'priority': 4,
            'performance_score': 0.1
        }
        
        # Cache results
        self._discovery_cache = models.copy()
        self._last_discovery_time = current_time
        EnhancedModelChain._last_discovery_time = current_time
        EnhancedModelChain._discovery_cache = models.copy()
        
        logger.info(f"📋 Discovered {len(models)} total models")
        return models
    
    def _discover_ollama_batch(self) -> List[str]:
        """Single batch Ollama discovery."""
        models = []
        
        try:
            import requests
            response = requests.get(
                f"{CONFIG['ollama_url']}/api/tags", 
                timeout=CONFIG['model_discovery_timeout']
            )
            
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                logger.info(f"📋 Batch discovered {len(models)} Ollama models")
            else:
                # CLI fallback
                result = subprocess.run(
                    ["ollama", "list"], 
                    capture_output=True, 
                    text=True, 
                    timeout=CONFIG['model_discovery_timeout']
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[1:]
                    models = [
                        line.split()[0] for line in lines 
                        if line.strip() and not line.startswith('NAME')
                    ]
                    logger.info(f"📋 CLI batch discovered {len(models)} Ollama models")
                    
        except Exception as e:
            logger.warning(f"Ollama batch discovery failed: {e}")
            models = ["qwen2.5-coder:latest", "phi4:latest", "deepseek-r1:latest"]
        
        return models[:CONFIG.get('max_ollama_models', 20)]
    
    def _discover_local_models(self) -> List[tuple]:
        """Single efficient local model discovery."""
        local_models = []
        scan_dirs = [dir_path for dir_path in CONFIG['local_model_scan_dirs'] if dir_path]
        
        for scan_dir in scan_dirs:
            try:
                scan_path = Path(scan_dir).expanduser()
                if not scan_path.exists():
                    continue
                
                # Efficient glob scanning
                for config_file in scan_path.glob("*/config.json"):
                    model_dir = config_file.parent
                    
                    # Quick model file check
                    model_files = [
                        "pytorch_model.bin", "model.safetensors", 
                        "tf_model.h5", "model.onnx"
                    ]
                    
                    if any((model_dir / model_file).exists() for model_file in model_files):
                        model_name = model_dir.name.replace('_', '/')
                        local_models.append((model_name, str(model_dir)))
                        
                        if len(local_models) >= CONFIG.get('max_local_models', 10):
                            break
                            
            except Exception as e:
                logger.debug(f"Error scanning {scan_dir}: {e}")
                continue
        
        # Add configured models
        for model_name, model_path in CONFIG['local_pretrained_models'].items():
            if Path(model_path).exists():
                local_models.append((model_name, model_path))
        
        logger.info(f"📁 Efficiently discovered {len(local_models)} local models")
        return local_models

    def generate_responses(self, prompt: str, max_tokens: int = 300) -> List[Dict]:
        """Generate responses using the model chain fallback system."""
        responses = []
        
        # Try each model in the rotation pool
        for model_key in self.model_rotation_pool[:self.models_per_question]:
            try:
                model_info = self.available_models.get(model_key, {})
                model_type = model_info.get('type', 'unknown')
                
                if model_type == 'remote' and CONFIG.get('llm_url') and CONFIG.get('llm_key'):
                    response = self._query_remote_api(prompt, max_tokens)
                elif model_type == 'ollama':
                    response = self._query_ollama(model_info.get('model_name'), prompt, max_tokens)
                elif model_type == 'local':
                    response = self._query_local_model(model_info, prompt, max_tokens)
                elif model_type == 'static':
                    response = self._query_static_fallback(prompt)
                else:
                    continue
                
                if response and len(response.strip()) >= CONFIG.get('response_quality_threshold', 15):
                    responses.append({
                        'response': response,
                        'model': model_key,
                        'type': model_type,
                        'quality_score': len(response) / 100.0
                    })
                    
            except Exception as e:
                logger.debug(f"Model {model_key} failed: {e}")
                continue
        
        return responses
    
    def _query_remote_api(self, prompt: str, max_tokens: int) -> str:
        """Query remote API."""
        try:
            import requests
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(
                CONFIG['llm_url'],
                json=payload,
                headers={"x-api-key": CONFIG['llm_key']},
                timeout=CONFIG['llm_timeout']
            )
            
            if response.status_code == 200:
                return response.json().get('content', [{}])[0].get('text', '')
        except Exception:
            pass
        return ""
    
    def _query_ollama(self, model_name: str, prompt: str, max_tokens: int) -> str:
        """Query Ollama model."""
        try:
            import requests
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": CONFIG.get('ollama_temperature', 0.7),
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{CONFIG['ollama_url']}/api/generate",
                json=payload,
                timeout=CONFIG['ollama_timeout']
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
        except Exception:
            pass
        return ""
    
    def _query_local_model(self, model_info: Dict, prompt: str, max_tokens: int) -> str:
        """Query local HuggingFace model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            model_path = model_info.get('model_path')
            if not model_path or not Path(model_path).exists():
                return ""
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=min(max_tokens, 100),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
            
        except Exception:
            pass
        return ""
    
    def _query_static_fallback(self, prompt: str) -> str:
        """Query static fallback responses."""
        try:
            static_file = Path(CONFIG["static_fallback_path"]) / "static_responses.json"
            if static_file.exists():
                with open(static_file, 'r') as f:
                    static_data = json.load(f)
                
                prompt_lower = prompt.lower()
                
                # Simple keyword matching
                for category, responses in static_data.items():
                    if isinstance(responses, dict):
                        for key, response in responses.items():
                            if key.lower() in prompt_lower or any(word in prompt_lower for word in key.lower().split()):
                                return response
                
                # Default response
                return "This appears to be a system monitoring question. Check system resources, logs, and recent changes for troubleshooting."
        except Exception:
            pass
        return "System monitoring response not available."
        
    def _build_rotation_pool(self) -> List[str]:
        """Build optimized rotation pool."""
        available_models = [
            (name, info) for name, info in self.available_models.items()
            if info.get('available', False)
        ]
        
        # Sort by performance score * priority weight
        def model_score(item):
            name, info = item
            priority_weight = CONFIG['model_priority_weights'].get(info['type'], 0.5)
            performance_score = info.get('performance_score', 0.5)
            return priority_weight * performance_score
        
        available_models.sort(key=model_score, reverse=True)
        
        # Build diverse pool
        pool = []
        type_counts = {}
        max_per_type = {
            'remote': 2,
            'ollama': CONFIG['model_pool_size'] - 5,
            'local': 3,
            'static': 1
        }
        
        for name, info in available_models:
            model_type = info['type']
            current_count = type_counts.get(model_type, 0)
            
            if current_count < max_per_type.get(model_type, 1):
                pool.append(name)
                type_counts[model_type] = current_count + 1
                
            if len(pool) >= CONFIG['model_pool_size']:
                break
        
        logger.info(f"🎯 Built rotation pool: {len(pool)} models")
        for model_type, count in type_counts.items():
            logger.info(f"   {model_type}: {count} models")
        
        return pool
    
    def _test_remote_api(self) -> bool:
        """Test remote API availability."""
        try:
            import requests
            test_payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "test"}]
            }
            
            response = requests.post(
                CONFIG['llm_url'],
                json=test_payload,
                headers={"x-api-key": CONFIG['llm_key']},
                timeout=5
            )
            
            return response.status_code in [200, 400]
            
        except Exception:
            return False

def setup_directories():
    """Create necessary directories with enhanced structure."""
    dirs = [
        CONFIG["training_dir"], 
        CONFIG["checkpoints_dir"], 
        CONFIG["logs_dir"],
        CONFIG["models_dir"], 
        CONFIG["hf_cache_dir"], 
        CONFIG["local_model_path"],
        CONFIG["static_fallback_path"],
        CONFIG["data_config_dir"],
        CONFIG["pretrained_dir"]
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_config(config_path="./config.json"):
    """Load configuration from JSON file with validation."""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            # Validate critical settings
            for key, value in loaded_config.items():
                if key in CONFIG:
                    # Type validation
                    if isinstance(CONFIG[key], type(value)) or CONFIG[key] is None:
                        CONFIG[key] = value
                    else:
                        logger.warning(f"Config type mismatch for {key}: expected {type(CONFIG[key])}, got {type(value)}")
                else:
                    logger.info(f"New config key: {key} = {value}")
                    CONFIG[key] = value
            
            logger.info(f"📖 Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")

def save_config(config_path="./config.json"):
    """Save configuration to JSON file."""
    try:
        config_copy = CONFIG.copy()
        
        # Remove non-serializable items
        non_serializable = ["framework_available", "torch_dtype"]
        for key in non_serializable:
            config_copy.pop(key, None)
        
        # Convert Path objects to strings
        for key, value in config_copy.items():
            if isinstance(value, Path):
                config_copy[key] = str(value)
        
        with open(config_path, 'w') as f:
            json.dump(config_copy, f, indent=2)
        
        logger.info(f"💾 Configuration saved to {config_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save config: {e}")

def setup_fallback_system():
    """Setup comprehensive fallback system with validation."""
    print("Setting up comprehensive fallback system...")
    
    available_methods = []
    
    # Check Remote LLM
    print("\n1. Checking Remote LLM...")
    if CONFIG.get('llm_url') and CONFIG.get('llm_key'):
        print("   ✅ Remote LLM configured")
        available_methods.append("Remote LLM")
    else:
        print("   ❌ Remote LLM not configured")
    
    # Check Ollama with enhanced discovery
    print("\n2. Checking Ollama...")
    try:
        import requests
        response = requests.get(f"{CONFIG['ollama_url']}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print(f"   ✅ Ollama available with {len(models)} models")
                available_methods.append("Ollama")
                
                # Log discovered models
                for model in models[:5]:  # Show first 5
                    print(f"      • {model['name']}")
                if len(models) > 5:
                    print(f"      • ... and {len(models) - 5} more")
            else:
                print("   ❌ Ollama running but no models found")
        else:
            print("   ❌ Ollama server not responding")
    except Exception as e:
        print(f"   ❌ Ollama not available: {e}")
    
    # Check Local Models with enhanced scanning
    print("\n3. Checking Local Models...")
    try:
        if CONFIG["framework"] == "tensorflow":
            from transformers import TFAutoTokenizer
            tokenizer_class = TFAutoTokenizer
        else:
            from transformers import AutoTokenizer
            tokenizer_class = AutoTokenizer
            
        local_count = 0
        
        for scan_dir in CONFIG['local_model_scan_dirs']:
            if not scan_dir:
                continue
                
            scan_path = Path(scan_dir).expanduser()
            if scan_path.exists():
                for item in scan_path.iterdir():
                    if item.is_dir() and (item / "config.json").exists():
                        local_count += 1
                        if local_count <= 3:  # Show first 3
                            print(f"      • {item.name}")
        
        if local_count > 0:
            print(f"   ✅ {local_count} local models available")
            available_methods.append("Local Models")
        else:
            print("   ❌ No local models found")
            
    except ImportError:
        print(f"   ❌ {CONFIG['framework'].title()} not available for local models")
    except Exception as e:
        print(f"   ❌ Local model check failed: {e}")
    
    # Setup Static Fallback
    print("\n4. Setting up Static Fallback...")
    if CONFIG.get('enable_static_fallback', True):
        success = create_static_fallback_responses()
        if success:
            print("   ✅ Static fallback responses created")
            available_methods.append("Static Fallback")
        else:
            print("   ⚠️  Static fallback setup had issues")
    
    print(f"\n✅ Fallback system ready with {len(available_methods)} methods: {', '.join(available_methods)}")
    return len(available_methods) > 0

def create_static_fallback_responses():
    """Create comprehensive static fallback responses."""
    try:
        static_responses = {
            "technical_explanations": {
                "cpu_usage": "CPU usage represents the percentage of processing power being used. High CPU usage (>80%) may indicate heavy processes, inefficient code, or system stress. Monitor with tools like top, htop, or sar to identify resource-intensive processes.",
                
                "memory_usage": "Memory usage shows RAM consumption by system processes. High memory usage (>85%) can cause swapping and performance degradation. Use free, ps, or /proc/meminfo to monitor. Consider memory leaks if usage grows continuously.",
                
                "disk_usage": "Disk usage indicates storage space consumption on filesystems. High disk usage (>90%) can cause application failures and system instability. Monitor with df, du, or lsblk. Implement log rotation and cleanup policies.",
                
                "load_average": "Load average represents system load over 1, 5, and 15-minute periods. Values above CPU core count indicate system stress. Use uptime or top to monitor. High load may indicate CPU bottlenecks or I/O waits.",
                
                "java_heap_usage": "Java heap usage shows memory allocated to Java applications. High heap usage (>85%) may indicate memory leaks or undersized heap. Use jstat, jmap, or heap dumps for analysis. Tune with -Xmx and -Xms flags.",
                
                "network_io": "Network I/O measures data transfer rates over network interfaces. High network I/O may indicate heavy traffic, inefficient protocols, or network congestion. Monitor with netstat, ss, iftop, or nload.",
                
                "systemd": "systemd is a system and service manager for Linux. It manages system initialization, service lifecycle, and dependencies. Use systemctl for service management and journalctl for log viewing."
            }
        }
        
        # Save static responses
        static_dir = Path(CONFIG["static_fallback_path"])
        static_dir.mkdir(parents=True, exist_ok=True)
        
        with open(static_dir / "static_responses.json", "w") as f:
            json.dump(static_responses, f, indent=2)
        
        logger.info(f"📝 Static responses created: {sum(len(cat) for cat in static_responses.values())} responses")
        return True
        
    except Exception as e:
        logger.error(f"❌ Static response creation failed: {e}")
        return False

def test_fallback_system():
    """Test the fallback system with realistic prompts."""
    print("Testing fallback system...")
    
    test_prompts = [
        "Explain what high CPU usage indicates in system monitoring.",
        "What causes java.lang.OutOfMemoryError in applications?",
        "How do you troubleshoot network connectivity issues on Linux?"
    ]
    
    print("\nRunning test queries...")
    successful_tests = 0
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt[:50]}...")
        try:
            responses = model_chain.generate_responses(prompt, max_tokens=150)
            
            if responses and len(responses) > 0:
                first_response = responses[0].get('response', '')
                if len(first_response.strip()) >= CONFIG.get('response_quality_threshold', 15):
                    print(f"✅ Success: {len(first_response)} chars, {len(responses)} model(s)")
                    print(f"   Sample: {first_response[:80]}...")
                    successful_tests += 1
                else:
                    print(f"❌ Response too short: {len(first_response)} chars")
            else:
                print(f"❌ No response generated")
                
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
    
    print(f"\n📊 TEST RESULTS: {successful_tests}/{len(test_prompts)} successful")
    return successful_tests > 0

# Initialize enhanced model chain
model_chain = EnhancedModelChain()

# Initialize directories and shared cache on import
setup_directories()
if CONFIG.get('shared_cache_enabled'):
    setup_dynamic_cache_environment()

# Load user configuration if exists
load_config()

if __name__ == "__main__":
    print("🚀 Enhanced Distilled Monitoring System Configuration")
    print("=" * 60)
    print(f"Framework: {CONFIG['framework'].title()}")
    print(f"Device: {CONFIG['device_type']}")
    print(f"Framework Available: {CONFIG['framework_available']}")
    
    # Setup fallback system
    setup_success = setup_fallback_system()
    
    if setup_success:
        # Test the system
        test_success = test_fallback_system()
        
        if test_success:
            print("\n✅ Configuration and fallback system ready!")
            print(f"   Models available: {len(model_chain.available_models)}")
            print(f"   Rotation pool: {len(model_chain.model_rotation_pool)}")
        else:
            print("\n⚠️  System configured but some tests failed")
    else:
        print("\n❌ Fallback system setup failed")
    
    # Show configuration summary
    print(f"\nCache directories:")
    print(f"  HF Cache: {CONFIG['hf_cache_dir']}")
    print(f"  Models: {CONFIG['models_dir']}")
    print(f"  Local Models: {CONFIG['local_model_path']}")
    print(f"  Shared Cache: {'Enabled' if CONFIG['shared_cache_enabled'] else 'Disabled'}")
    
    print(f"\nTraining environment: {detect_training_environment()}")
    print(f"Continual learning: {'Enabled' if CONFIG['continual_learning_enabled'] else 'Disabled'}")
    print("=" * 60)