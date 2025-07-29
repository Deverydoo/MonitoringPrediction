#!/usr/bin/env python3
"""
distilled_model_trainer.py
Technology-specific predictive LLM for monitoring and troubleshooting
Enhanced with PyTorch/TensorFlow framework selection and Keras interface
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import traceback
from collections import defaultdict
import numpy as np

# Configure encoding for Windows/Linux compatibility
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Setup basic logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import framework based on config
try:
    from config import CONFIG, FRAMEWORK, FRAMEWORK_AVAILABLE, DEVICE_TYPE
except ImportError:
    FRAMEWORK = os.environ.get('ML_FRAMEWORK', 'pytorch').lower()
    FRAMEWORK_AVAILABLE = True
    DEVICE_TYPE = 'CPU'

# Framework-specific imports
if FRAMEWORK == 'tensorflow' and FRAMEWORK_AVAILABLE:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, Model, optimizers, losses, metrics
        from transformers import TFAutoModel, TFAutoModelForCausalLM, AutoTokenizer
        
        # Configure GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        FRAMEWORK_BACKEND = 'tensorflow'
        logger.info(f"🔥 TensorFlow backend loaded: {tf.__version__}")
        
    except ImportError as e:
        logger.warning(f"TensorFlow import failed: {e}, falling back to PyTorch")
        FRAMEWORK_BACKEND = 'pytorch'
else:
    FRAMEWORK_BACKEND = 'pytorch'

if FRAMEWORK_BACKEND == 'pytorch':
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset, DataLoader, TensorDataset
        from transformers import (
            AutoTokenizer, AutoModel, AutoModelForCausalLM, 
            Trainer, TrainingArguments, get_linear_schedule_with_warmup
        )
        
        # Disable dynamo/compilation issues
        try:
            torch._dynamo.config.disable = True
            torch._dynamo.config.suppress_errors = True
        except:
            pass
            
        logger.info(f"🔥 PyTorch backend loaded: {torch.__version__}")
        
    except ImportError as e:
        logger.error(f"Both TensorFlow and PyTorch import failed: {e}")
        raise ImportError("No suitable ML framework available")

def setup_enhanced_logging():
    """Setup enhanced logging with local file output for Spark environments."""
    
    # Ensure logs directory exists
    logs_dir = Path('./logs/')
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'training_{timestamp}.log'
    
    # Configure logging with both console and file handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # Console handler (for local/interactive use)
            logging.StreamHandler(),
            # File handler (for Spark environments)
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🗂️ Enhanced logging initialized - local log: {log_file}")
    logger.info(f"🔧 Framework: {FRAMEWORK_BACKEND}")
    
    return logger, log_file
    
logger, training_log_file = setup_enhanced_logging()

class TrainingEnvironment:
    """Manages training environment with framework-specific optimizations"""
    
    def __init__(self):
        self.framework = FRAMEWORK_BACKEND
        self.device = self._detect_best_device()
        self.env_type = self._get_environment_type()
        self._setup_environment()
    
    def _detect_best_device(self) -> str:
        """Detect best available device with framework-specific logic"""
        if self.framework == 'tensorflow':
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    gpu_name = tf.config.experimental.get_device_details(gpus[0])['device_name']
                    logger.info(f"🎮 TensorFlow GPU: {gpu_name}")
                    return f"/GPU:0"
                else:
                    logger.info("💻 TensorFlow CPU")
                    return "/CPU:0"
            except Exception as e:
                logger.warning(f"TensorFlow device detection failed: {e}")
                return "/CPU:0"
        
        else:  # PyTorch
            try:
                if torch.cuda.is_available():
                    device = f"cuda:{torch.cuda.current_device()}"
                    gpu_name = torch.cuda.get_device_name()
                    logger.info(f"🎮 PyTorch CUDA GPU: {gpu_name}")
                    return device
                
                # Check Apple Silicon (MPS)
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    logger.info("🍎 PyTorch Apple Silicon (MPS)")
                    return "mps"
                
                logger.info("💻 PyTorch CPU")
                return "cpu"
            except Exception as e:
                logger.warning(f"PyTorch device detection failed: {e}")
                return "cpu"
    
    def _get_environment_type(self) -> str:
        """Get environment type for optimization"""
        if self.framework == 'tensorflow':
            return "tensorflow_gpu" if "GPU" in self.device else "tensorflow_cpu"
        else:
            if "cuda" in self.device:
                return "pytorch_cuda"
            elif "mps" in self.device:
                return "pytorch_mps"
            return "pytorch_cpu"
    
    def _setup_environment(self):
        """Setup framework-specific optimizations"""
        if self.framework == 'tensorflow':
            self._setup_tensorflow()
        else:
            self._setup_pytorch()
    
    def _setup_tensorflow(self):
        """Setup TensorFlow optimizations"""
        try:
            # Mixed precision
            if CONFIG.get('tf_mixed_precision_policy') == 'mixed_float16' and "GPU" in self.device:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logger.info("🚀 TensorFlow mixed precision enabled")
            
            # XLA compilation
            if CONFIG.get('tf_xla_compile', True):
                tf.config.optimizer.set_jit(True)
                logger.info("🚀 TensorFlow XLA compilation enabled")
            
            # Memory growth
            if CONFIG.get('tf_memory_growth', True):
                gpus = tf.config.experimental.list_physical_devices('GPU')
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("🚀 TensorFlow memory growth enabled")
                
        except Exception as e:
            logger.warning(f"TensorFlow optimization setup failed: {e}")
    
    def _setup_pytorch(self):
        """Setup PyTorch optimizations"""
        try:
            if "cuda" in self.device:
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("🚀 PyTorch CUDA optimizations enabled")
            
            elif self.device == "cpu":
                torch.set_num_threads(os.cpu_count())
                logger.info(f"🔧 PyTorch CPU threads: {os.cpu_count()}")
                
        except Exception as e:
            logger.warning(f"PyTorch optimization setup failed: {e}")

class DatasetLoader:
    """Efficient dataset loader with dynamic discovery and caching"""
    
    def __init__(self, training_dir: str, tokenizer=None):
        self.training_dir = Path(training_dir)
        self.tokenizer = tokenizer
        self.cache = {}
        self.supported_formats = ['.json', '.jsonl']
        
    def discover_datasets(self) -> Dict[str, List[Path]]:
        """Dynamically discover all training datasets"""
        datasets = defaultdict(list)
        
        if not self.training_dir.exists():
            logger.error(f"Training directory not found: {self.training_dir}")
            return dict(datasets)
        
        for file_path in self.training_dir.rglob("*"):
            if file_path.suffix.lower() in self.supported_formats:
                dataset_type = self._classify_dataset(file_path.name)
                datasets[dataset_type].append(file_path)
        
        # Log discovery results
        total_files = sum(len(files) for files in datasets.values())
        logger.info(f"📁 Discovered {total_files} dataset files")
        for dtype, files in datasets.items():
            if files:
                logger.info(f"  {dtype}: {len(files)} files")
                for file in files:
                    size_mb = file.stat().st_size / (1024 * 1024)
                    logger.info(f"    - {file.name} ({size_mb:.1f}MB)")
        
        return dict(datasets)
    
    def _classify_dataset(self, filename: str) -> str:
        """Classify dataset type from filename"""
        filename_lower = filename.lower()
        
        if 'language' in filename_lower:
            return 'language'
        elif 'metric' in filename_lower:
            return 'metrics'
        elif 'splunk' in filename_lower:
            return 'splunk_logs'
        elif 'jira' in filename_lower:
            return 'jira_tickets'
        elif 'confluence' in filename_lower:
            return 'confluence_docs'
        elif 'spectrum' in filename_lower:
            return 'spectrum_logs'
        elif 'vemkd' in filename_lower:
            return 'vemkd_logs'
        else:
            return 'general'
    
    def load_dataset_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single dataset file with proper encoding"""
        try:
            # Try UTF-8 first
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to Latin-1 for problematic files
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                logger.warning(f"Used latin-1 encoding for {file_path}")
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
                return {}
        
        try:
            data = json.loads(content)
            logger.info(f"✅ Loaded {file_path.name}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return {}
    
    def extract_training_samples(self, datasets: Dict[str, List[Path]]) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Extract and consolidate training samples from all datasets"""
        texts = []
        labels = []
        metrics = []
        anomalies = []
        
        for dataset_type, files in datasets.items():
            for file_path in files:
                data = self.load_dataset_file(file_path)
                if not data:
                    continue
                
                # Process based on dataset structure
                samples = self._extract_samples_from_data(data, dataset_type)
                if samples:
                    t, l, m, a = samples
                    texts.extend(t)
                    labels.extend(l)
                    metrics.extend(m)
                    anomalies.extend(a)
                    logger.info(f"✅ Extracted {len(t)} samples from {file_path.name}")
        
        # Clean and validate data
        texts, labels, metrics, anomalies = self._clean_training_data(
            texts, labels, metrics, anomalies
        )
        
        logger.info(f"🎯 Final dataset: {len(texts)} samples")
        logger.info(f"   Labels: {len(set(labels))} unique")
        logger.info(f"   Anomalies: {sum(anomalies)} anomaly samples")
        
        return texts, labels, metrics, anomalies
    
    def _extract_samples_from_data(self, data: Dict, dataset_type: str) -> Optional[Tuple[List[str], List[int], List[List[float]], List[int]]]:
        """Extract samples from data based on structure"""
        texts, labels, metrics, anomalies = [], [], [], []
        
        # Handle language dataset
        if dataset_type == 'language' and 'samples' in data:
            for sample in data['samples']:
                if isinstance(sample, dict) and 'explanation' in sample:
                    text = sample['explanation']
                    label = 0  # Default technical label
                    metric = [0.0] * 10  # Default metrics
                    anomaly = 0  # Default normal
                    
                    texts.append(text)
                    labels.append(label)
                    metrics.append(metric)
                    anomalies.append(anomaly)
        
        # Handle metrics dataset
        elif dataset_type == 'metrics' and 'training_samples' in data:
            for sample in data['training_samples']:
                if isinstance(sample, dict):
                    text = sample.get('explanation', 'System metrics sample')
                    label = 1 if sample.get('status') == 'anomaly' else 0
                    metric = list(sample.get('metrics', {}).values())[:10]
                    # Pad or truncate to exactly 10 values
                    while len(metric) < 10:
                        metric.append(0.0)
                    metric = metric[:10]
                    anomaly = 1 if sample.get('status') == 'anomaly' else 0
                    
                    texts.append(text)
                    labels.append(label)
                    metrics.append(metric)
                    anomalies.append(anomaly)
        
        return texts, labels, metrics, anomalies if texts else None
    
    def _clean_training_data(self, texts: List[str], labels: List[int], 
                           metrics: List[List[float]], anomalies: List[int]) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Clean and validate training data"""
        cleaned_texts = []
        cleaned_labels = []
        cleaned_metrics = []
        cleaned_anomalies = []
        
        for i, text in enumerate(texts):
            # Skip very short texts
            if len(text.strip()) < 10:
                continue
            
            # Clean text
            clean_text = text.strip().replace('\x00', '').replace('\ufffd', '')
            if len(clean_text) < 10:
                continue
            
            cleaned_texts.append(clean_text)
            cleaned_labels.append(labels[i])
            cleaned_metrics.append(metrics[i])
            cleaned_anomalies.append(anomalies[i])
        
        logger.info(f"🧹 Cleaned: {len(cleaned_texts)}/{len(texts)} samples retained")
        return cleaned_texts, cleaned_labels, cleaned_metrics, cleaned_anomalies

# Framework-specific model classes
if FRAMEWORK_BACKEND == 'tensorflow':
    class MonitoringModel(keras.Model):
        """Multi-task monitoring model with TensorFlow/Keras"""
        
        def __init__(self, base_model, num_labels: int = 8, num_metrics: int = 10, **kwargs):
            super().__init__(**kwargs)
            self.base_model = base_model
            self.hidden_size = base_model.config.hidden_size
            
            # Multi-task heads
            self.classifier = layers.Dense(num_labels, activation='softmax', name='classifier')
            self.anomaly_detector = layers.Dense(2, activation='softmax', name='anomaly_detector')
            self.metrics_regressor = layers.Dense(num_metrics, name='metrics_regressor')
            
            # Task weights for loss balancing
            self.classification_weight = 0.4
            self.anomaly_weight = 0.3
            self.metrics_weight = 0.3
        
        def call(self, inputs, training=None):
            """Forward pass with multi-task outputs"""
            if isinstance(inputs, dict):
                input_ids = inputs.get('input_ids')
                attention_mask = inputs.get('attention_mask', None)
            else:
                input_ids = inputs
                attention_mask = None
            
            # Get base model outputs
            if attention_mask is not None:
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, training=training)
            else:
                outputs = self.base_model(input_ids=input_ids, training=training)
            
            pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token
            
            # Multi-task predictions
            classification_logits = self.classifier(pooled_output)
            anomaly_logits = self.anomaly_detector(pooled_output)
            metrics_pred = self.metrics_regressor(pooled_output)
            
            return {
                'classification_logits': classification_logits,
                'anomaly_logits': anomaly_logits,
                'metrics_predictions': metrics_pred
            }
        
        def compute_loss(self, labels, metrics_true, anomalies, y_pred):
            """Compute multi-task loss"""
            classification_loss = tf.keras.losses.sparse_categorical_crossentropy(
                labels, y_pred['classification_logits']
            )
            anomaly_loss = tf.keras.losses.sparse_categorical_crossentropy(
                anomalies, y_pred['anomaly_logits']
            )
            # Fix: Use tf.reduce_mean with tf.square for MSE
            metrics_loss = tf.reduce_mean(tf.square(metrics_true - y_pred['metrics_predictions']))
            
            total_loss = (self.classification_weight * tf.reduce_mean(classification_loss) + 
                         self.anomaly_weight * tf.reduce_mean(anomaly_loss) + 
                         self.metrics_weight * metrics_loss)
            
            return total_loss

else:  # PyTorch
    class MonitoringModel(nn.Module):
        """Multi-task monitoring model with PyTorch"""
        
        def __init__(self, base_model, num_labels: int = 8, num_metrics: int = 10):
            super().__init__()
            self.base_model = base_model
            self.hidden_size = base_model.config.hidden_size
            
            # Multi-task heads
            self.classifier = nn.Linear(self.hidden_size, num_labels)
            self.anomaly_detector = nn.Linear(self.hidden_size, 2)
            self.metrics_regressor = nn.Linear(self.hidden_size, num_metrics)
            
            # Task weights for loss balancing
            self.classification_weight = 0.4
            self.anomaly_weight = 0.3
            self.metrics_weight = 0.3
            
            self._init_weights()
        
        def _init_weights(self):
            """Initialize task-specific heads"""
            for module in [self.classifier, self.anomaly_detector, self.metrics_regressor]:
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    if module.bias is not None:
                        module.bias.data.zero_()
        
        def forward(self, input_ids, attention_mask, labels=None, metrics=None, anomalies=None):
            """Forward pass with multi-task outputs"""
            # Get base model outputs
            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token
            
            # Multi-task predictions
            classification_logits = self.classifier(pooled_output)
            anomaly_logits = self.anomaly_detector(pooled_output)
            metrics_pred = self.metrics_regressor(pooled_output)
            
            loss = None
            if labels is not None:
                # Calculate multi-task loss
                ce_loss = nn.CrossEntropyLoss()
                mse_loss = nn.MSELoss()
                
                classification_loss = ce_loss(classification_logits, labels)
                anomaly_loss = ce_loss(anomaly_logits, anomalies) if anomalies is not None else 0
                metrics_loss = mse_loss(metrics_pred, metrics) if metrics is not None else 0
                
                loss = (self.classification_weight * classification_loss + 
                       self.anomaly_weight * anomaly_loss + 
                       self.metrics_weight * metrics_loss)
            
            return {
                'loss': loss,
                'classification_logits': classification_logits,
                'anomaly_logits': anomaly_logits,
                'metrics_predictions': metrics_pred
            }

# Framework-specific dataset classes
if FRAMEWORK_BACKEND == 'tensorflow':
    class MonitoringDataset:
        """TensorFlow dataset wrapper"""
        
        def __init__(self, texts, labels, metrics, anomalies, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.metrics = metrics
            self.anomalies = anomalies
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def create_tf_dataset(self, batch_size):
            """Create TensorFlow dataset"""
            def generator():
                for i in range(len(self.texts)):
                    encoded = self.tokenizer(
                        self.texts[i],
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='tf'
                    )
                    
                    yield (
                        {
                            'input_ids': encoded['input_ids'][0],
                            'attention_mask': encoded['attention_mask'][0]
                        },
                        {
                            'labels': self.labels[i],
                            'metrics': self.metrics[i],
                            'anomalies': self.anomalies[i]
                        }
                    )
            
            output_signature = (
                {
                    'input_ids': tf.TensorSpec(shape=(self.max_length,), dtype=tf.int32),
                    'attention_mask': tf.TensorSpec(shape=(self.max_length,), dtype=tf.int32)
                },
                {
                    'labels': tf.TensorSpec(shape=(), dtype=tf.int32),
                    'metrics': tf.TensorSpec(shape=(10,), dtype=tf.float32),
                    'anomalies': tf.TensorSpec(shape=(), dtype=tf.int32)
                }
            )
            
            dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            return dataset

else:  # PyTorch
    class MonitoringDataset(Dataset):
        """PyTorch dataset for monitoring data"""
        
        def __init__(self, texts, labels, metrics, anomalies, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.metrics = metrics
            self.anomalies = anomalies
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            
            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long),
                'metrics': torch.tensor(self.metrics[idx], dtype=torch.float32),
                'anomalies': torch.tensor(self.anomalies[idx], dtype=torch.long)
            }

class DistilledModelTrainer:
    """Main trainer class with framework selection and optimization"""
    
    def __init__(self, config: Dict[str, Any], resume_training: bool = False):
        """Initialize trainer with framework selection"""
        self.config = config
        self.framework = FRAMEWORK_BACKEND
        self.env = TrainingEnvironment()
        self.loader = DatasetLoader(config['training_dir'])
        
        # Framework-specific device setup
        if self.framework == 'tensorflow':
            self.device = self.env.device
        else:
            self.device = torch.device(self.env.device if 'cuda' in self.env.device else 'cpu')

        # Store log file path for reference
        self.training_log_file = training_log_file
        logger.info(f"📁 Training logs will be written to: {self.training_log_file}")
        logger.info(f"🔧 Using {self.framework.title()} framework")
        
        # Progress tracking
        self.training_progress = {
            'start_time': None,
            'current_epoch': 0,
            'total_epochs': config.get('epochs', 3),
            'current_step': 0,
            'total_steps': 0,
            'best_loss': float('inf'),
            'losses': []
        }
        
        # Check for resume option
        if resume_training:
            latest_model = self.find_latest_model()
            if latest_model and self.load_from_checkpoint(latest_model):
                logger.info("🔄 Resuming from existing model")
                return
            else:
                logger.info("🆕 No valid checkpoint found, starting fresh")

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with framework selection"""
        model_name = self.config.get('model_name', 'bert-base-uncased')
        
        # Try local model first
        local_path = Path(CONFIG['pretrained_dir']) / model_name
        if local_path.exists():
            logger.info(f"📁 Loading from local: {local_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(local_path), local_files_only=True
                )
                
                if self.framework == 'tensorflow':
                    base_model = TFAutoModel.from_pretrained(
                        str(local_path), local_files_only=True
                    )
                else:
                    base_model = AutoModel.from_pretrained(
                        str(local_path), local_files_only=True
                    )
                    
                logger.info(f"✅ Loaded local model: {model_name}")
            except Exception as e:
                logger.error(f"❌ Local model load failed: {e}")
                raise
        else:
            # Download and cache
            logger.info(f"📥 Downloading {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=CONFIG['hf_cache_dir']
            )
            
            if self.framework == 'tensorflow':
                base_model = TFAutoModel.from_pretrained(
                    model_name, cache_dir=CONFIG['hf_cache_dir']
                )
            else:
                base_model = AutoModel.from_pretrained(
                    model_name, cache_dir=CONFIG['hf_cache_dir']
                )
            
            # Save locally for next time
            local_path.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(str(local_path))
            base_model.save_pretrained(str(local_path))
            logger.info(f"💾 Cached model locally: {local_path}")
        
        # Create monitoring model
        self.model = MonitoringModel(base_model)
        
        if self.framework == 'pytorch':
            self.model.to(self.device)
            
            # Enable compilation if supported
            if hasattr(torch, 'compile') and self.env.env_type == "pytorch_cuda":
                try:
                    self.model = torch.compile(self.model)
                    logger.info("🚀 PyTorch model compilation enabled")
                except Exception as e:
                    logger.warning(f"Model compilation failed: {e}")
    
    def prepare_training_data(self):
        """Prepare and tokenize training data"""
        logger.info("📊 Discovering and loading datasets...")
        
        # Discover datasets
        datasets = self.loader.discover_datasets()
        if not any(datasets.values()):
            raise ValueError("No training datasets found!")
        
        # Extract samples
        texts, labels, metrics, anomalies = self.loader.extract_training_samples(datasets)
        if not texts:
            raise ValueError("No training samples extracted!")
        
        # Create dataset
        logger.info("🔤 Creating training dataset...")
        dataset = MonitoringDataset(texts, labels, metrics, anomalies, self.tokenizer, self.config.get('max_length', 512))
        
        if self.framework == 'tensorflow':
            # Create TensorFlow dataset
            tf_dataset = dataset.create_tf_dataset(self.config['batch_size'])
            logger.info(f"✅ TensorFlow dataset ready: {len(texts)} samples")
            return tf_dataset, len(texts)
        else:
            # Create PyTorch DataLoader
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=True,
                num_workers=self.config.get('dataloader_num_workers', 2),
                pin_memory=self.config.get('pin_memory', True),
                persistent_workers=self.config.get('persistent_workers', True)
            )
            logger.info(f"✅ PyTorch dataset ready: {len(texts)} samples")
            return dataloader, len(texts)
    
    def train(self) -> bool:
        """Train the distilled model with framework selection."""
        
        # Enhanced logging setup
        logs_dir = Path('./logs/')
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f'training_{timestamp}.log'
        
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
        
        log_message("🏋️ Starting distilled model training")
        log_message(f"📁 Local log file: {log_file}")
        log_message(f"🖥️ Environment: {self.env.env_type}")
        log_message(f"🔧 Framework: {self.framework.title()}")
        log_message(f"🎮 Device: {self.device}")
        
        try:
            # Initialize model
            log_message("🤖 Setting up model and tokenizer...")
            self.setup_model_and_tokenizer()
            log_message(f"✅ Model loaded: {self.config.get('model_name', 'unknown')}")
            
            # Prepare training data
            log_message("📊 Preparing training data...")
            dataset, sample_count = self.prepare_training_data()
            
            log_message(f"📈 Training samples: {sample_count}")
            log_message(f"📦 Batch size: {self.config['batch_size']}")
            
            # Framework-specific training
            if self.framework == 'tensorflow':
                success = self._train_tensorflow(dataset, sample_count, log_message)
            else:
                success = self._train_pytorch(dataset, sample_count, log_message)
            
            if success:
                # Save final model
                log_message("💾 Saving final model...")
                final_path = self._save_final_model()
                
                # Training summary
                duration = datetime.now() - self.training_progress['start_time']
                log_message(f"🎉 Training completed successfully!")
                log_message(f"⏱️ Duration: {duration}")
                log_message(f"📈 Best loss: {self.training_progress['best_loss']:.4f}")
                log_message(f"💾 Final model saved: {final_path}")
                log_message(f"📋 Training log saved: {log_file}")
                
                return True
            else:
                log_message("❌ Training failed", "ERROR")
                return False
            
        except KeyboardInterrupt:
            log_message("⏸️ Training interrupted by user", "WARNING")
            return False
            
        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            log_message(error_msg, "ERROR")
            log_message(f"🔍 Traceback: {traceback.format_exc()}", "ERROR")
            return False
    
    def _train_tensorflow(self, dataset, sample_count, log_message) -> bool:
        """TensorFlow-specific training logic"""
        log_message("🔥 Starting TensorFlow training...")
        
        # Setup optimizer and metrics
        optimizer = optimizers.AdamW(
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            run_eagerly=False  # Use graph mode for performance
        )
        
        # Setup callbacks
        callbacks = []
        
        # Model checkpoint
        checkpoint_dir = Path(self.config.get('checkpoints_dir', './checkpoints/'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_dir / 'best_model.h5'),
            monitor='loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Progress tracking
        self.training_progress['start_time'] = datetime.now()
        steps_per_epoch = sample_count // self.config['batch_size']
        self.training_progress['total_steps'] = steps_per_epoch * self.config['epochs']
        
        # Custom training loop for multi-task learning
        @tf.function
        def train_step(inputs, targets):
            with tf.GradientTape() as tape:
                predictions = self.model(inputs, training=True)
                loss = self.model.compute_loss(
                    targets['labels'], 
                    targets['metrics'], 
                    targets['anomalies'], 
                    predictions
                )
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            return loss
        
        # Training loop
        try:
            epoch_losses = []
            for epoch in range(self.config['epochs']):
                log_message(f"🔄 Starting epoch {epoch + 1}/{self.config['epochs']}")
                epoch_loss = 0.0
                step_count = 0
                
                for batch in dataset:
                    inputs, targets = batch
                    loss = train_step(inputs, targets)
                    epoch_loss += loss
                    step_count += 1
                    
                    if step_count % 100 == 0:
                        log_message(f"   Step {step_count}: Loss={loss:.4f}")
                
                avg_loss = epoch_loss / step_count
                epoch_losses.append(avg_loss)
                log_message(f"✅ Epoch {epoch + 1} completed: Avg Loss={avg_loss:.4f}")
                
                # Update best loss
                if avg_loss < self.training_progress['best_loss']:
                    self.training_progress['best_loss'] = avg_loss
                    log_message(f"🏆 New best model: Loss={avg_loss:.4f}")
            
            self.training_progress['losses'] = epoch_losses
            return True
            
        except Exception as e:
            log_message(f"TensorFlow training failed: {e}", "ERROR")
            return False
    
    def _train_pytorch(self, dataloader, sample_count, log_message) -> bool:
        """PyTorch-specific training logic"""
        log_message("🔥 Starting PyTorch training...")
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=len(dataloader) * self.config['epochs']
        )
        
        # Mixed precision setup for efficiency
        scaler = None
        if self.config.get('mixed_precision', False) and self.env.env_type == "pytorch_cuda":
            scaler = torch.cuda.amp.GradScaler()
            log_message("🚀 Mixed precision training enabled")
        
        # Training progress tracking
        total_steps = len(dataloader) * self.config['epochs']
        self.training_progress = {
            'start_time': datetime.now(),
            'current_step': 0,
            'total_steps': total_steps,
            'best_loss': float('inf'),
            'losses': []
        }
        
        log_message(f"🎯 Training configuration:")
        log_message(f"   Epochs: {self.config['epochs']}")
        log_message(f"   Learning rate: {self.config['learning_rate']}")
        log_message(f"   Total training steps: {total_steps}")
        log_message(f"   Batches per epoch: {len(dataloader)}")
        
        # Training loop
        self.model.train()
        
        try:
            for epoch in range(self.config['epochs']):
                epoch_loss = 0.0
                log_message(f"🔄 Starting epoch {epoch + 1}/{self.config['epochs']}")
                
                for step, batch in enumerate(dataloader):
                    self.training_progress['current_step'] += 1
                    
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    metrics = batch['metrics'].to(self.device)
                    anomalies = batch['anomalies'].to(self.device)
                    
                    # Forward pass with optional mixed precision
                    optimizer.zero_grad()
                    
                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                metrics=metrics,
                                anomalies=anomalies
                            )
                            loss = outputs['loss']
                        
                        # Backward pass with scaling
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            metrics=metrics,
                            anomalies=anomalies
                        )
                        loss = outputs['loss']
                        
                        # Standard backward pass
                        loss.backward()
                        optimizer.step()
                    
                    if scheduler:
                        scheduler.step()
                    
                    epoch_loss += loss.item()
                    
                    # Progress logging every 100 steps
                    if step % 100 == 0:
                        progress_pct = (self.training_progress['current_step'] / 
                                      self.training_progress['total_steps']) * 100
                        current_lr = optimizer.param_groups[0]['lr']
                        log_message(f"   Step {step}: Loss={loss.item():.4f}, LR={current_lr:.2e} ({progress_pct:.1f}% complete)")
                    
                    # Save checkpoint every 500 steps
                    if step % 500 == 0 and step > 0:
                        checkpoint_name = f"checkpoint_epoch_{epoch+1}_step_{step}"
                        self._save_checkpoint(checkpoint_name)
                        log_message(f"💾 Checkpoint saved: {checkpoint_name}")
                
                # End of epoch processing
                avg_loss = epoch_loss / len(dataloader)
                self.training_progress['losses'].append(avg_loss)
                
                log_message(f"✅ Epoch {epoch + 1} completed: Avg Loss={avg_loss:.4f}")
                
                # Save best model
                if avg_loss < self.training_progress['best_loss']:
                    self.training_progress['best_loss'] = avg_loss
                    self._save_checkpoint('best_model')
                    log_message(f"🏆 New best model saved: Loss={avg_loss:.4f}")
            
            return True
            
        except Exception as e:
            log_message(f"PyTorch training failed: {e}", "ERROR")
            return False
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.get('checkpoints_dir', './checkpoints/'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if self.framework == 'tensorflow':
            checkpoint_path = checkpoint_dir / f"{name}.h5"
            self.model.save_weights(str(checkpoint_path))
        else:
            checkpoint_path = checkpoint_dir / f"{name}.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'training_progress': self.training_progress,
                'config': self.config
            }, checkpoint_path)
    
    def _save_final_model(self) -> str:
        """Save final trained model with framework detection"""
        models_dir = Path(CONFIG['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"distilled_monitoring_{timestamp}"
        model_path.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        if self.framework == 'tensorflow':
            # Save TensorFlow model
            self.model.save(str(model_path / 'tf_model'))
            self.tokenizer.save_pretrained(str(model_path))
        else:
            # Save PyTorch model using transformers format
            self.model.base_model.save_pretrained(str(model_path))
            self.tokenizer.save_pretrained(str(model_path))
            
            # Save custom heads separately
            torch.save({
                'classifier_state_dict': self.model.classifier.state_dict(),
                'anomaly_detector_state_dict': self.model.anomaly_detector.state_dict(),
                'metrics_regressor_state_dict': self.model.metrics_regressor.state_dict(),
                'model_config': {
                    'num_labels': 8,
                    'num_metrics': 10,
                    'hidden_size': self.model.hidden_size
                }
            }, model_path / 'custom_heads.pt')
    
        # Convert config to JSON-serializable format
        def make_json_serializable(obj):
            """Convert numpy/torch types to JSON-serializable types"""
            if obj is None:
                return None
            elif hasattr(obj, 'dtype'):
                if str(type(obj)).startswith("<class 'numpy.dtype"):
                    return str(obj)
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                else:
                    return str(obj)
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, type):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Clean config for JSON serialization
        clean_config = make_json_serializable(self.config)
        
        # Save training metadata
        metadata = {
            'model_type': 'distilled_monitoring',
            'framework': self.framework,
            'base_model': self.config.get('model_name'),
            'training_samples': int(self.training_progress.get('total_steps', 0)),
            'best_loss': float(self.training_progress.get('best_loss', 0.0)),
            'training_time': str(datetime.now() - self.training_progress['start_time']),
            'device_type': self.env.env_type,
            'config': clean_config,
            'timestamp': timestamp,
            'model_path': str(model_path)
        }
        
        with open(model_path / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Model saved to: {model_path}")
        return str(model_path)

    def find_latest_model(self) -> Optional[str]:
        """Find the most recent trained model for resuming"""
        models_dir = Path(CONFIG['models_dir'])
        if not models_dir.exists():
            return None
        
        # Find all distilled monitoring models
        model_dirs = list(models_dir.glob('distilled_monitoring_*'))
        if not model_dirs:
            return None
        
        # Sort by timestamp (newest first)
        model_dirs.sort(reverse=True)
        latest_model = model_dirs[0]
        
        # Verify it's a complete model
        if self.framework == 'tensorflow':
            required_files = ['tf_model', 'config.json', 'training_metadata.json']
        else:
            required_files = ['config.json', 'training_metadata.json']
            
        if all((latest_model / f).exists() for f in required_files):
            logger.info(f"🔍 Found latest model: {latest_model}")
            return str(latest_model)
        
        logger.warning(f"⚠️ Latest model incomplete: {latest_model}")
        return None

    def load_from_checkpoint(self, model_path: str) -> bool:
        """Load model from previous training checkpoint"""
        try:
            model_path = Path(model_path)
            
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            if self.framework == 'tensorflow':
                # Load TensorFlow model
                self.model = keras.models.load_model(str(model_path / 'tf_model'))
            else:
                # Load PyTorch model
                base_model = AutoModel.from_pretrained(str(model_path))
                self.model = MonitoringModel(base_model)
                
                # Load custom heads if they exist
                custom_heads_path = model_path / 'custom_heads.pt'
                if custom_heads_path.exists():
                    heads_data = torch.load(custom_heads_path, map_location=self.device)
                    self.model.classifier.load_state_dict(heads_data['classifier_state_dict'])
                    self.model.anomaly_detector.load_state_dict(heads_data['anomaly_detector_state_dict'])
                    self.model.metrics_regressor.load_state_dict(heads_data['metrics_regressor_state_dict'])
                
                self.model.to(self.device)
            
            # Load training metadata if available
            metadata_path = model_path / 'training_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"📊 Previous training: {metadata.get('training_samples', 0)} samples")
                    logger.info(f"📈 Previous best loss: {metadata.get('best_loss', 'unknown')}")
                    logger.info(f"🔧 Previous framework: {metadata.get('framework', 'unknown')}")
            
            logger.info(f"✅ Loaded model from: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            return False

def main():
    """Main training function"""
    from config import CONFIG, setup_directories
    
    # Setup environment
    setup_directories()
    
    # Initialize trainer
    trainer = DistilledModelTrainer(CONFIG)
    
    # Start training
    success = trainer.train()
    
    if success:
        logger.info("🎉 Training completed successfully!")
        return True
    else:
        logger.error("❌ Training failed!")
        return False

if __name__ == "__main__":
    main()