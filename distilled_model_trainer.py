# distilled_model_trainer.py
import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple, Any

# Force PyTorch-only environment before any imports
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Try to import transformers and other ML libraries with PyTorch-only
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        TrainingArguments, Trainer, 
        DataCollatorWithPadding
    )
    from datasets import Dataset
    import torch.nn as nn
    import torch.nn.functional as F
    
    # Verify PyTorch backend
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print("PyTorch backend loaded successfully")
    
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"Warning: transformers not available. Error: {e}")

try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

from config import CONFIG, detect_training_environment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{CONFIG['logs_dir']}/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def find_local_model_path(model_name: str) -> Optional[str]:
    """Find local model path with multiple fallback options."""
    search_paths = [
        f"./pretrained/{model_name}/",
        f"./pretrained/{model_name.replace('/', '_')}/", 
        f"{CONFIG.get('hf_cache_dir', './hf_cache')}/{model_name}/",
        f"{CONFIG.get('hf_cache_dir', './hf_cache')}/{model_name.replace('/', '_')}/",
        f"{CONFIG.get('local_model_path', './local_models')}/{model_name}/",
        f"{CONFIG.get('local_model_path', './local_models')}/{model_name.replace('/', '_')}/"
    ]
    
    for path in search_paths:
        path_obj = Path(path)
        if path_obj.exists():
            # Check for essential files
            config_exists = (path_obj / "config.json").exists()
            model_exists = any([
                (path_obj / "pytorch_model.bin").exists(),
                (path_obj / "model.safetensors").exists(),
                (path_obj / "pytorch_model.safetensors").exists()
            ])
            
            if config_exists and model_exists:
                logger.info(f"📁 Found local model at: {path}")
                return str(path_obj)
    
    return None

class DistilledMonitoringModel(nn.Module):
    """Custom distilled model for system monitoring - PyTorch only."""
    
    def __init__(self, base_model, num_labels=8, num_metrics=10):
        super().__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.num_metrics = num_metrics
        
        # Get hidden size from base model
        hidden_size = base_model.config.hidden_size
        
        # Classification heads
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.anomaly_detector = nn.Linear(hidden_size, 2)  # binary: normal/anomaly
        
        # Metrics regression head - updated to 10 outputs
        self.metrics_regressor = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_metrics)  # 10 metrics output
        )
        
        # Multi-task loss weights
        self.classification_weight = 1.0
        self.anomaly_weight = 1.0
        self.metrics_weight = 0.5
        
        try:
            # Enhanced local model discovery with multiple path options
            local_paths_to_check = [
                # Direct pretrained directory (your current setup)
                f"./pretrained/{base_model_name}/",
                # HF cache format
                f"{CONFIG['hf_cache_dir']}/{base_model_name}/",
                # Local models directory
                f"{CONFIG['local_model_path']}/{base_model_name}/",
                # Safe filename format (replace / with _)
                f"./pretrained/{base_model_name.replace('/', '_')}/",
                f"{CONFIG['hf_cache_dir']}/{base_model_name.replace('/', '_')}/",
                f"{CONFIG['local_model_path']}/{base_model_name.replace('/', '_')}/"
            ]
            
            model_loaded = False
            
            # Try each local path
            for local_path in local_paths_to_check:
                local_path_obj = Path(local_path)
                if local_path_obj.exists():
                    # Check for essential model files
                    required_files = ['config.json']
                    model_files = ['pytorch_model.bin', 'model.safetensors', 'pytorch_model.safetensors']
                    
                    has_config = any((local_path_obj / f).exists() for f in required_files)
                    has_model = any((local_path_obj / f).exists() for f in model_files)
                    
                    if has_config and has_model:
                        try:
                            logger.info(f"Loading model from local path: {local_path}")
                            
                            # Load from local directory with explicit local_files_only
                            self.config = AutoConfig.from_pretrained(
                                local_path,
                                local_files_only=True,
                                trust_remote_code=False
                            )
                            
                            self.base_model = AutoModel.from_pretrained(
                                local_path,
                                config=self.config,
                                local_files_only=True,
                                torch_dtype=CONFIG.get('torch_dtype', torch.float32),
                                trust_remote_code=False
                            )
                            
                            model_loaded = True
                            logger.info(f"✅ Successfully loaded local model from: {local_path}")
                            break
                            
                        except Exception as e:
                            logger.warning(f"Failed to load from {local_path}: {e}")
                            continue
            
            # Fallback to HuggingFace if no local model found
            if not model_loaded:
                logger.warning("No local model found, attempting HuggingFace download...")
                logger.warning("This will fail in offline environments!")
                
                try:
                    self.config = AutoConfig.from_pretrained(
                        base_model_name,
                        cache_dir=CONFIG['hf_cache_dir'],
                        local_files_only=False  # Allow download as last resort
                    )
                    self.base_model = AutoModel.from_pretrained(
                        base_model_name,
                        config=self.config,
                        cache_dir=CONFIG['hf_cache_dir'],
                        torch_dtype=CONFIG.get('torch_dtype', torch.float32),
                        local_files_only=False
                    )
                    model_loaded = True
                    logger.info(f"✅ Downloaded and cached model: {base_model_name}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to load model {base_model_name}: {e}")
                    logger.error("Ensure model is cached locally or internet connection available")
                    raise RuntimeError(f"Could not load model {base_model_name} - no local copy and no internet")
            
            if not model_loaded:
                raise RuntimeError(f"Could not load model {base_model_name} from any location")
                
            # Initialize custom heads
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(self.config.hidden_size, num_labels)
            self.regressor = nn.Linear(self.config.hidden_size, 20)
            self.anomaly_detector = nn.Linear(self.config.hidden_size, 1)
            self._init_weights()
            
        except Exception as e:
            logger.error(f"Failed to initialize model {base_model_name}: {e}")
            raise
        
        # Optional: Compile model for PyTorch 2.0+ performance boost
        if torch.__version__ >= "2.0.0" and CONFIG.get('torch_compile', False):
            try:
                self.base_model = torch.compile(self.base_model)
                logger.info("Model compiled with torch.compile for better performance")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        
    def _init_weights(self):
        """Initialize the weights of the custom heads."""
        for module in [self.classifier, self.regressor, self.anomaly_detector]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
        
    def forward(self, input_ids, attention_mask, labels=None, metrics=None, anomalies=None):
        # Get base model outputs
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]  # Use [CLS] token
        
        # Multi-task outputs
        classification_logits = self.classifier(pooled_output)
        anomaly_logits = self.anomaly_detector(pooled_output)
        metrics_pred = self.metrics_regressor(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            mse_loss = nn.MSELoss()
            
            # Classification loss
            classification_loss = loss_fn(classification_logits, labels)
            
            # Anomaly detection loss
            anomaly_loss = loss_fn(anomaly_logits, anomalies) if anomalies is not None else 0
            
            # Metrics regression loss
            metrics_loss = mse_loss(metrics_pred, metrics) if metrics is not None else 0
            
            # Combined loss
            loss = (self.classification_weight * classification_loss + 
                   self.anomaly_weight * anomaly_loss + 
                   self.metrics_weight * metrics_loss)
        
        return {
            'loss': loss,
            'classification_logits': classification_logits,
            'anomaly_logits': anomaly_logits,
            'metrics_predictions': metrics_pred
        }

class MonitoringDataset:
    """Dataset class for monitoring data - PyTorch only."""
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_training_data(self) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Load and prepare training data from JSON files."""
        texts = []
        labels = []
        metrics = []
        anomalies = []
        
        # Load language dataset
        lang_file = os.path.join(CONFIG['training_dir'], 'language_dataset.json')
        if os.path.exists(lang_file):
            with open(lang_file, 'r') as f:
                lang_data = json.load(f)
            
            for item in lang_data:
                if item['type'] == 'technical_explanation':
                    text = f"Term: {item['term']}. Explanation: {item['explanation']}"
                    texts.append(text)
                    labels.append(0)  # Normal explanation
                    metrics.append([0.0] * 20)  # Placeholder metrics
                    anomalies.append(0)
                elif item['type'] == 'error_interpretation':
                    text = f"Error: {item['error_message']}. Interpretation: {item['interpretation']}"
                    texts.append(text)
                    labels.append(1)  # Error case
                    metrics.append([0.0] * 20)
                    anomalies.append(1)
        
        # Load metrics dataset
        metrics_file = os.path.join(CONFIG['training_dir'], 'metrics_dataset.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            for sample in metrics_data.get('training_samples', []):
                # Create text representation of metrics
                metric_text = f"System metrics: "
                metric_values = []
                
                for key, value in sample['metrics'].items():
                    metric_text += f"{key}: {value:.2f}, "
                    metric_values.append(float(value))
                
                # Pad or truncate to 20 values
                while len(metric_values) < 20:
                    metric_values.append(0.0)
                metric_values = metric_values[:20]
                
                metric_text += f"Status: {sample['status']}. Explanation: {sample['explanation']}"
                
                texts.append(metric_text)
                labels.append(2 if sample['status'] == 'anomaly' else 0)
                metrics.append(metric_values)
                anomalies.append(1 if sample['status'] == 'anomaly' else 0)
        
        logger.info(f"Loaded {len(texts)} training samples")
        return texts, labels, metrics, anomalies
    
    def create_dataset(self) -> Dataset:
        """Create HuggingFace dataset with PyTorch tensors."""
        texts, labels, metrics, anomalies = self.load_training_data()
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert to PyTorch tensors
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
            'metrics_targets': torch.tensor(metrics, dtype=torch.float32),
            'anomaly_targets': torch.tensor(anomalies, dtype=torch.float32)
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        return dataset

class DistilledModelTrainer:
    """Main trainer class for the distilled monitoring model - PyTorch only."""
    
    def __init__(self):
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        self.training_env = detect_training_environment()
        
        logger.info(f"Training environment: {self.training_env}")
        logger.info(f"Device: {self.device}")
        
    def _setup_device(self):
        """Setup training device based on availability - PyTorch 2.0+ optimized."""
        # Check PyTorch version
        torch_version = torch.__version__
        logger.info(f"PyTorch version: {torch_version}")
        
        if torch_version < "2.0.1":
            logger.warning("PyTorch 2.0.1+ recommended for optimal performance and compatibility")
        
        if torch.cuda.is_available() and not CONFIG['force_cpu']:
            device = torch.device('cuda')
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
            
            # PyTorch 2.0+ CUDA optimizations
            if torch_version >= "2.0.0":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 for better performance on Ampere+ GPUs")
                
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')  # Apple Silicon support
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for training")
            
            # CPU optimizations for PyTorch 2.0+
            if torch_version >= "2.0.0":
                torch.set_num_threads(torch.get_num_threads())
                logger.info(f"Using {torch.get_num_threads()} CPU threads")
                
        return device
    
    def load_or_create_model(self) -> DistilledMonitoringModel:
        """Load existing model or create new one."""
        model_path = os.path.join(CONFIG['models_dir'], 'distilled_monitoring_model')
        
        if os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            try:
                model = DistilledMonitoringModel(CONFIG['model_name'])
                model.load_state_dict(torch.load(
                    os.path.join(model_path, 'pytorch_model.bin'),
                    map_location=self.device
                ))
                return model
            except Exception as e:
                logger.warning(f"Could not load existing model: {e}")
        
        logger.info("Creating new model")
        model = DistilledMonitoringModel(CONFIG['model_name'])
        return model
    
    def load_tokenizer(self):
        """Load tokenizer with PyTorch backend and proper local fallback."""
        try:
            # Check for local pretrained model first (matching your working test)
            local_model_path = CONFIG['local_pretrained_models'].get(CONFIG['model_name'])
            
            if local_model_path and os.path.exists(local_model_path):
                logger.info(f"Loading tokenizer from local pretrained: {local_model_path}")
                tokenizer = AutoTokenizer.from_pretrained(
                    local_model_path,
                    local_files_only=True,
                    trust_remote_code=CONFIG.get('trust_remote_code', False)
                )
                logger.info(f"Successfully loaded local tokenizer: {CONFIG['model_name']}")
            else:
                # Fallback to HuggingFace with cache
                logger.info(f"Loading tokenizer from HuggingFace: {CONFIG['model_name']}")
                tokenizer = AutoTokenizer.from_pretrained(
                    CONFIG['model_name'],
                    cache_dir=CONFIG['hf_cache_dir'],
                    trust_remote_code=CONFIG.get('trust_remote_code', False),
                    use_auth_token=CONFIG.get('use_auth_token', False)
                )
            
            # Add pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Could not load tokenizer: {e}")
            raise RuntimeError(f"No tokenizer available for {CONFIG['model_name']}")
    
    def train_model(self):
        """Main training loop with PyTorch backend."""
        logger.info("Starting model training...")
        
        # Load tokenizer and model
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_or_create_model()
        self.model.to(self.device)
        
        # Prepare dataset
        dataset_handler = MonitoringDataset(self.tokenizer, CONFIG['max_length'])
        train_dataset = dataset_handler.create_dataset()
        
        if len(train_dataset) == 0:
            logger.error("No training data found. Please run data generation scripts first.")
            return
        
        # Split dataset
        train_size = int(0.8 * len(train_dataset))
        eval_size = len(train_dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, eval_size]
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")
        
        # Training arguments with PyTorch 2.0+ optimizations
        training_args = TrainingArguments(
            output_dir=CONFIG['checkpoints_dir'],
            num_train_epochs=CONFIG['epochs'],
            per_device_train_batch_size=CONFIG['batch_size'],
            per_device_eval_batch_size=CONFIG['batch_size'],
            warmup_steps=CONFIG['warmup_steps'],
            weight_decay=CONFIG['weight_decay'],
            learning_rate=CONFIG['learning_rate'],
            logging_dir=CONFIG['logs_dir'],
            logging_steps=10,
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            dataloader_pin_memory=False if self.device.type == 'cpu' else True,
            
            # PyTorch 2.0+ mixed precision settings
            fp16=self.device.type == 'cuda' and torch.__version__ >= "1.6.0",
            bf16=self.device.type == 'cuda' and torch.cuda.is_bf16_supported() and torch.__version__ >= "1.10.0",
            
            # PyTorch 2.0+ optimizations
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,  # Keep all columns for our custom forward
            report_to=None,  # Disable wandb/tensorboard
            use_cpu=self.device.type == 'cpu',
            
            # Additional PyTorch 2.0+ settings
            gradient_checkpointing=False,  # Can enable for memory savings
            optim="adamw_torch",  # Use PyTorch's native AdamW
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=CONFIG['max_length']
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        try:
            trainer.train()
            
            # Save final model
            final_model_path = os.path.join(CONFIG['models_dir'], 'distilled_monitoring_model')
            trainer.save_model(final_model_path)
            self.tokenizer.save_pretrained(final_model_path)
            
            logger.info(f"Training completed. Model saved to {final_model_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

class FlexibleDatasetLoader:
    """Dynamic loader for various training data formats."""
    
    def __init__(self, training_dir: str, tokenizer=None):
        self.training_dir = Path(training_dir)
        self.supported_formats = ['.json', '.jsonl', '.csv', '.txt']
        self.tokenizer = tokenizer
        
        # Initialize tokenizer if not provided
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                from config import CONFIG
                
                # Try to load from local path first
                local_model_path = find_local_model_path(CONFIG['model_name'])
                if local_model_path:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        local_model_path,
                        local_files_only=True
                    )
                    logger.info(f"✅ Loaded tokenizer from local path: {local_model_path}")
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
                    logger.info(f"✅ Loaded tokenizer from HuggingFace: {CONFIG['model_name']}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to initialize tokenizer: {e}")
                self.tokenizer = None
        
    def discover_training_files(self) -> Dict[str, List[Path]]:
        """Discover all training files by type."""
        files_by_type = {
            'language': [],
            'metrics': [], 
            'splunk_logs': [],
            'jira_tickets': [],
            'confluence_docs': [],
            'spectrum_logs': [],
            'vemkd_logs': [],
            'general': []
        }
        
        if not self.training_dir.exists():
            return files_by_type
        
        for file_path in self.training_dir.iterdir():
            if file_path.suffix.lower() in self.supported_formats:
                file_type = self._classify_file(file_path)
                files_by_type[file_type].append(file_path)
        
        # Log discovered files
        total_files = sum(len(files) for files in files_by_type.values())
        logger.info(f"📁 Discovered {total_files} training files")
        for file_type, files in files_by_type.items():
            if files:
                logger.info(f"  {file_type}: {len(files)} files")
        
        return files_by_type

    def train_model_with_debug(self):
        """Train the model with detailed debugging."""
        logger.info("🏋️ Starting model training with debug info...")

        try:
            # Initialize tokenizer first
            logger.info("🔤 Initializing tokenizer...")
            from transformers import AutoTokenizer
            
            local_model_path = find_local_model_path(CONFIG['model_name'])
            if local_model_path:
                tokenizer = AutoTokenizer.from_pretrained(
                    local_model_path,
                    local_files_only=True
                )
                logger.info(f"✅ Loaded tokenizer from: {local_model_path}")
            else:
                tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
                logger.info(f"✅ Loaded tokenizer from HuggingFace")
            
            # Initialize loader with tokenizer
            loader = FlexibleDatasetLoader(CONFIG['training_dir'], tokenizer=tokenizer)
            
            # Load data with debugging
            logger.info("📚 Loading training data...")
            texts, labels, metrics, anomalies = loader.debug_training_data_loading()

            if not texts:
                logger.error("❌ No training data loaded!")
                return False
            
            logger.info(f"✅ Successfully loaded {len(texts)} training samples")
            
            # Continue with tokenization and training...
            logger.info("🔤 Starting tokenization...")
            
            # Test tokenization on a small subset first
            test_texts = texts[:5] if len(texts) >= 5 else texts
            logger.info(f"Testing tokenization on {len(test_texts)} samples...")
            
            encoded = tokenizer(
                test_texts,
                padding=True,
                truncation=True,
                max_length=CONFIG.get('max_length', 512),
                return_tensors="pt"
            )
            
            logger.info(f"✅ Test tokenization successful - shape: {encoded['input_ids'].shape}")
            
            # Now tokenize all data
            logger.info("🔤 Tokenizing all training data...")
            all_encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=CONFIG.get('max_length', 512),
                return_tensors="pt"
            )
            
            logger.info(f"✅ Full tokenization successful - shape: {all_encoded['input_ids'].shape}")
            
            # Create dataset
            from datasets import Dataset
            import torch

            logger.info(f"Checking metrics dimensions...")
            if metrics:
                first_metric = metrics[0]
                logger.info(f"First metric sample length: {len(first_metric)}")
                logger.info(f"First metric sample: {first_metric}")
                
                # Ensure all metrics have exactly 10 values
                normalized_metrics = []
                for metric_sample in metrics:
                    if len(metric_sample) != 10:
                        # Fix the length
                        if len(metric_sample) > 10:
                            metric_sample = metric_sample[:10]
                        else:
                            while len(metric_sample) < 10:
                                metric_sample.append(0.0)
                    normalized_metrics.append(metric_sample)
                
                logger.info(f"Normalized metrics to 10 values each")
                metrics = normalized_metrics
            
            logger.info(f"✅ Dataset created - metrics shape: {torch.tensor(metrics).shape}")
            
            dataset = Dataset.from_dict({
                'input_ids': all_encoded['input_ids'],
                'attention_mask': all_encoded['attention_mask'],
                'labels': torch.tensor(labels, dtype=torch.long),
                'metrics': torch.tensor(metrics, dtype=torch.float32),
                'anomalies': torch.tensor(anomalies, dtype=torch.long)
            })
            
            logger.info(f"✅ Dataset created successfully with {len(dataset)} samples")
            
            # Continue with actual training...
            logger.info("🏋️ Starting model training...")
            return True
            
        except Exception as e:
            logger.error(f"❌ Training with debug failed: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return False

    def debug_training_data_loading(self):
        """Debug the exact point where training data loading fails."""
        logger.info("🔍 Debugging training data loading step by step...")
        
        try:
            # Step 1: Check if files exist
            files_by_type = self.discover_training_files()
            logger.info(f"📁 Files discovered: {sum(len(files) for files in files_by_type.values())}")
            
            for file_type, file_list in files_by_type.items():
                logger.info(f"  {file_type}: {len(file_list)} files")
                for file_path in file_list:
                    file_size = file_path.stat().st_size
                    logger.info(f"    - {file_path} ({file_size} bytes)")
            
            # Step 2: Try loading each file individually
            all_texts = []
            all_labels = []
            all_metrics = []
            all_anomalies = []
            
            for file_type, file_list in files_by_type.items():
                for file_path in file_list:
                    logger.info(f"🔄 Processing {file_path}...")
                    
                    try:
                        # Test basic file reading first
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content_preview = f.read(500)  # Read first 500 chars
                        logger.info(f"✅ Basic read OK - preview: {repr(content_preview[:100])}...")
                        
                        # Now try structured loading
                        file_texts, file_labels, file_metrics, file_anomalies = self._load_single_file(
                            file_path, file_type
                        )
                        
                        logger.info(f"✅ Structured loading OK - loaded {len(file_texts)} samples")
                        
                        all_texts.extend(file_texts)
                        all_labels.extend(file_labels)
                        all_metrics.extend(file_metrics)
                        all_anomalies.extend(file_anomalies)
                        
                    except Exception as e:
                        logger.error(f"❌ Failed loading {file_path}: {type(e).__name__}: {e}")
                        
                        # Try to show exactly where it fails
                        try:
                            with open(file_path, 'rb') as f:
                                raw_content = f.read()
                            logger.info(f"Raw file size: {len(raw_content)} bytes")
                            
                            # Try different encodings
                            for encoding in ['utf-8', 'cp1252', 'latin-1']:
                                try:
                                    decoded = raw_content.decode(encoding)
                                    logger.info(f"{encoding} decode: SUCCESS")
                                except UnicodeDecodeError as ude:
                                    logger.info(f"{encoding} decode: FAILED at position {ude.start}")
                        except Exception as e2:
                            logger.error(f"Raw read also failed: {e2}")
                        
                        raise  # Re-raise to see full stack trace
            
            logger.info(f"🎯 Final totals: {len(all_texts)} texts, {len(all_labels)} labels")
            return all_texts, all_labels, all_metrics, all_anomalies
            
        except Exception as e:
            logger.error(f"❌ Debug loading failed: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise

    def clean_all_training_files(self):
        """Clean all training files to ensure UTF-8 encoding."""
        logger.info("🧹 Cleaning all training files for UTF-8 compatibility...")
        
        files_by_type = self.discover_training_files()
        cleaned_files = 0
        
        for file_type, file_list in files_by_type.items():
            for file_path in file_list:
                try:
                    # Read with multiple encoding attempts
                    content = None
                    original_encoding = None
                    
                    for encoding in ['utf-8', 'cp1252', 'latin-1']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                content = f.read()
                            original_encoding = encoding
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if content is None:
                        logger.error(f"❌ Could not read {file_path} with any encoding")
                        continue
                    
                    # Clean problematic characters
                    cleaned_content = content
                    replacements = {
                        '\x9d': ' ',      # Operating system command
                        '\x9c': '"',      # String terminator  
                        '\x93': '"',      # Set transmit state
                        '\x94': '"',      # Cancel character
                        '\x91': "'",      # Private use 1
                        '\x92': "'",      # Private use 2
                        '\x96': '-',      # Start selected area
                        '\x97': '-',      # End selected area
                        '\x85': '...',    # Next line
                        '\x80': '€',      # Euro sign
                        '\x82': ',',      # Break permitted here
                        '\x83': 'f',      # No break here
                        '\x84': '"',      # Index
                        '\x88': '^',      # Character tabulation set
                        '\x89': '‰',      # Character tabulation with justification
                        '\x8a': 'Š',      # Line tabulation set
                        '\x8b': '<',      # Partial line forward
                        '\x8c': 'Œ',      # Partial line backward
                        '\x8e': 'Ž',      # Single shift two
                        '\x8f': '',       # Single shift three
                    }
                    
                    changed = False
                    for old_char, new_char in replacements.items():
                        if old_char in cleaned_content:
                            cleaned_content = cleaned_content.replace(old_char, new_char)
                            changed = True
                            logger.info(f"Replaced {repr(old_char)} with {repr(new_char)} in {file_path}")
                    
                    # Write back as UTF-8 if changes were made or if original wasn't UTF-8
                    if changed or original_encoding != 'utf-8':
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(cleaned_content)
                        cleaned_files += 1
                        logger.info(f"✅ Cleaned and saved {file_path} as UTF-8")
                    
                except Exception as e:
                    logger.error(f"❌ Error cleaning {file_path}: {e}")
                    continue
        
        logger.info(f"🧹 Cleaned {cleaned_files} training files")
        return cleaned_files > 0
        
    def _classify_file(self, file_path: Path) -> str:
        """Classify file based on name patterns."""
        name_lower = file_path.name.lower()
        
        if 'language' in name_lower or 'conversation' in name_lower:
            return 'language'
        elif 'metric' in name_lower or 'performance' in name_lower:
            return 'metrics'
        elif 'splunk' in name_lower or 'search' in name_lower:
            return 'splunk_logs'
        elif 'jira' in name_lower or 'ticket' in name_lower:
            return 'jira_tickets'
        elif 'confluence' in name_lower or 'wiki' in name_lower:
            return 'confluence_docs'
        elif 'spectrum' in name_lower or 'conductor' in name_lower:
            return 'spectrum_logs'
        elif 'vemkd' in name_lower or 'linux' in name_lower:
            return 'vemkd_logs'
        else:
            return 'general'

    def diagnose_encoding_issue(self) -> Dict[str, Any]:
        """Diagnose which file and position is causing the encoding error."""
        logger.info("🔍 Diagnosing encoding issues in training files...")
        
        files_by_type = self.discover_training_files()
        diagnostic_results = {
            'problematic_files': [],
            'total_files_checked': 0,
            'encoding_issues': []
        }
        
        position_counter = 0
        target_position = 255418
        
        for file_type, file_list in files_by_type.items():
            for file_path in file_list:
                diagnostic_results['total_files_checked'] += 1
                
                try:
                    # Try to read with different methods to pinpoint the issue
                    file_size = file_path.stat().st_size
                    logger.info(f"📄 Checking {file_path} (size: {file_size} bytes)")
                    
                    # Check if this file contains our target position
                    if position_counter <= target_position <= position_counter + file_size:
                        logger.warning(f"🎯 Target position {target_position} found in file: {file_path}")
                        relative_pos = target_position - position_counter
                        
                        # Read the problematic area
                        with open(file_path, 'rb') as f:
                            f.seek(max(0, relative_pos - 50))
                            chunk = f.read(100)
                            
                        logger.info(f"Bytes around position {relative_pos}:")
                        logger.info(f"Hex: {chunk.hex()}")
                        
                        # Try to decode with different encodings
                        for encoding in ['utf-8', 'cp1252', 'latin-1', 'ascii']:
                            try:
                                decoded = chunk.decode(encoding, errors='replace')
                                logger.info(f"{encoding}: {repr(decoded)}")
                            except Exception as e:
                                logger.info(f"{encoding}: Failed - {e}")
                        
                        diagnostic_results['problematic_files'].append({
                            'file': str(file_path),
                            'position': relative_pos,
                            'hex_chunk': chunk.hex(),
                            'file_type': file_type
                        })
                    
                    position_counter += file_size
                    
                    # Test reading the entire file
                    with open(file_path, 'r', encoding='utf-8', errors='strict') as f:
                        content = f.read()
                        logger.info(f"✅ {file_path} - UTF-8 OK")
                        
                except UnicodeDecodeError as e:
                    logger.error(f"❌ {file_path} - Encoding error: {e}")
                    diagnostic_results['encoding_issues'].append({
                        'file': str(file_path),
                        'error': str(e),
                        'file_type': file_type
                    })
                    
                    # Try to fix the file
                    self._fix_encoding_in_file(file_path)
                    
                except Exception as e:
                    logger.error(f"❌ {file_path} - Other error: {e}")
        
        return diagnostic_results
    
    def _fix_encoding_in_file(self, file_path: Path) -> bool:
        """Attempt to fix encoding issues in a specific file."""
        logger.info(f"🔧 Attempting to fix encoding in {file_path}")
        
        backup_path = file_path.with_suffix(file_path.suffix + '.backup')
        
        try:
            # Create backup
            import shutil
            shutil.copy2(file_path, backup_path)
            logger.info(f"📋 Backup created: {backup_path}")
            
            # Try different encodings to read the file
            content = None
            successful_encoding = None
            
            for encoding in ['cp1252', 'latin-1', 'utf-8', 'ascii']:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                        content = f.read()
                    successful_encoding = encoding
                    logger.info(f"✅ Successfully read with {encoding}")
                    break
                except Exception as e:
                    logger.info(f"❌ Failed with {encoding}: {e}")
                    continue
            
            if content is not None:
                # Clean the content - remove or replace problematic characters
                # Byte 0x9d is a Windows-1252 "operating system command" character
                cleaned_content = content.replace('\x9d', ' ')  # Replace with space
                cleaned_content = cleaned_content.replace('\x9c', '"')  # Replace œ with quote
                cleaned_content = cleaned_content.replace('\x93', '"')  # Replace " with quote
                cleaned_content = cleaned_content.replace('\x94', '"')  # Replace " with quote
                cleaned_content = cleaned_content.replace('\x91', "'")  # Replace ' with apostrophe
                cleaned_content = cleaned_content.replace('\x92', "'")  # Replace ' with apostrophe
                
                # Write back as UTF-8
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                logger.info(f"✅ Fixed and saved {file_path} as UTF-8")
                return True
            else:
                logger.error(f"❌ Could not read {file_path} with any encoding")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to fix {file_path}: {e}")
            # Restore backup if it exists
            if backup_path.exists():
                import shutil
                shutil.copy2(backup_path, file_path)
            return False
            
    def load_flexible_training_data(self) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Load training data from all discovered files with encoding diagnosis."""
        logger.info("📚 Loading training data with encoding diagnosis...")
        
        # First, diagnose encoding issues
        diagnostic_results = self.diagnose_encoding_issue()
        
        if diagnostic_results['encoding_issues']:
            logger.warning(f"Found {len(diagnostic_results['encoding_issues'])} files with encoding issues")
            for issue in diagnostic_results['encoding_issues']:
                logger.warning(f"  - {issue['file']}: {issue['error']}")
        
        texts = []
        labels = []
        metrics = []
        anomalies = []
        
        files_by_type = self.discover_training_files()
        
        for file_type, file_list in files_by_type.items():
            for file_path in file_list:
                try:
                    file_texts, file_labels, file_metrics, file_anomalies = self._load_single_file(
                        file_path, file_type
                    )
                    
                    texts.extend(file_texts)
                    labels.extend(file_labels)
                    metrics.extend(file_metrics)
                    anomalies.extend(file_anomalies)
                    
                    logger.info(f"✅ Loaded {len(file_texts)} samples from {file_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Error loading {file_path}: {e}")
                    # Try to fix and reload
                    if self._fix_encoding_in_file(file_path):
                        try:
                            file_texts, file_labels, file_metrics, file_anomalies = self._load_single_file(
                                file_path, file_type
                            )
                            texts.extend(file_texts)
                            labels.extend(file_labels) 
                            metrics.extend(file_metrics)
                            anomalies.extend(file_anomalies)
                            logger.info(f"✅ Fixed and loaded {len(file_texts)} samples from {file_path}")
                        except Exception as e2:
                            logger.error(f"❌ Still failed after fix attempt: {e2}")
                            continue
                    else:
                        continue
        
        logger.info(f"📊 Total loaded: {len(texts)} training samples")
        return texts, labels, metrics, anomalies
    
    def _load_single_file(self, file_path: Path, file_type: str) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Load data from a single file based on its type with UTF-8 encoding."""
        try:
            if file_path.suffix == '.json':
                return self._load_json_file(file_path, file_type)
            elif file_path.suffix == '.jsonl':
                return self._load_jsonl_file(file_path, file_type)
            elif file_path.suffix == '.csv':
                return self._load_csv_file(file_path, file_type)
            elif file_path.suffix == '.txt':
                return self._load_text_file(file_path, file_type)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in {file_path}: {e}")
            # Try with different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    logger.info(f"Retrying {file_path} with {encoding} encoding")
                    return self._load_with_encoding(file_path, file_type, encoding)
                except:
                    continue
            raise ValueError(f"Could not decode {file_path} with any supported encoding")
    
    def _load_json_file(self, file_path: Path, file_type: str) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Load JSON file with correct structure handling."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        metrics = []
        anomalies = []
        
        # Handle language_dataset.json structure
        if file_type == 'language' and 'samples' in data:
            samples = data['samples']
            logger.info(f"Processing {len(samples)} language samples from {file_path}")
            
            for sample in samples:
                # Extract text from different possible fields
                text = ""
                if 'response' in sample:
                    text = sample['response']
                elif 'explanation' in sample:
                    text = sample['explanation']
                elif 'answer' in sample:
                    text = sample['answer']
                else:
                    # Combine available text fields
                    text_parts = []
                    for key in ['term', 'question', 'prompt']:
                        if key in sample and sample[key]:
                            text_parts.append(f"{key}: {sample[key]}")
                    text = " ".join(text_parts)
                
                if text and text.strip():
                    texts.append(text.strip())
                    labels.append(self._generate_label_for_type(file_type))
                    metrics.append(self._generate_default_metrics())
                    anomalies.append(0)  # Default to no anomaly
        
        # Handle metrics_dataset.json structure  
        elif file_type == 'metrics' and 'training_samples' in data:
            samples = data['training_samples']
            logger.info(f"Processing {len(samples)} metrics samples from {file_path}")
            
            for sample in samples:
                # Create text representation
                text_parts = []
                
                # Add metrics info
                if 'metrics' in sample:
                    metric_text = "System metrics: "
                    
                    # Define the expected metric order to match your dataset
                    metric_keys = [
                        'class_count', 'gc_time', 'heap_usage', 'thread_count', 'cpu_usage',
                        'disk_io', 'disk_usage', 'load_average', 'memory_usage', 'network_io'
                    ]
                    
                    metric_values = []
                    for key in metric_keys:
                        if key in sample['metrics']:
                            value = float(sample['metrics'][key])
                            metric_text += f"{key}: {value:.2f}, "
                            metric_values.append(value)
                        else:
                            # Use default value if metric is missing
                            default_val = self._generate_default_metrics()[len(metric_values)]
                            metric_values.append(default_val)
                    
                    # Ensure exactly 10 values
                    while len(metric_values) < 10:
                        metric_values.append(0.0)
                    metric_values = metric_values[:10]
                    
                    text_parts.append(metric_text.rstrip(', '))
                else:
                    metric_values = self._generate_default_metrics()
                
                # Add status and explanation
                if 'status' in sample:
                    text_parts.append(f"Status: {sample['status']}")
                if 'explanation' in sample:
                    text_parts.append(f"Explanation: {sample['explanation']}")
                
                text = " ".join(text_parts)
                
                if text and text.strip():
                    texts.append(text.strip())
                    labels.append(2 if sample.get('status') == 'anomaly' else 0)
                    metrics.append(metric_values)
                    anomalies.append(1 if sample.get('status') == 'anomaly' else 0)
        
        # Handle generic structure
        else:
            logger.warning(f"Unknown structure in {file_path} for type {file_type}")
            # Try to extract any available data
            if isinstance(data, list):
                samples = data
            elif isinstance(data, dict):
                # Try common keys
                samples = data.get('samples', data.get('training_samples', data.get('data', [])))
            else:
                samples = []
            
            logger.info(f"Attempting generic processing of {len(samples)} samples")
            for sample in samples[:10]:  # Limit to prevent issues
                if isinstance(sample, dict):
                    text = str(sample.get('text', sample.get('content', str(sample))))
                    if text and text.strip():
                        texts.append(text.strip())
                        labels.append(self._generate_label_for_type(file_type))
                        metrics.append(self._generate_default_metrics())
                        anomalies.append(0)
        
        logger.info(f"✅ Extracted {len(texts)} valid samples from {file_path}")
        return texts, labels, metrics, anomalies
    
    def _load_jsonl_file(self, file_path: Path, file_type: str) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Load JSONL file with explicit UTF-8 encoding."""
        data = []
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line in {file_path}: {e}")
                        continue
        return self._process_data_by_type(data, file_type)
    
    def _load_text_file(self, file_path: Path, file_type: str) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Load text file with explicit UTF-8 encoding."""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Split by lines or paragraphs depending on content
        if '\n\n' in content:
            texts = [text.strip() for text in content.split('\n\n') if text.strip()]
        else:
            texts = [text.strip() for text in content.split('\n') if text.strip()]
        
        # Generate synthetic labels based on file type
        labels = [self._generate_label_for_type(file_type) for _ in texts]
        metrics = [self._generate_default_metrics() for _ in texts]
        anomalies = [0 for _ in texts]  # Default to no anomaly
        
        return texts, labels, metrics, anomalies
    
    def _load_csv_file(self, file_path: Path, file_type: str) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Load CSV file with explicit UTF-8 encoding."""
        import csv
        
        texts = []
        labels = []
        metrics = []
        anomalies = []
        
        with open(file_path, 'r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract text from the row - look for common text columns
                text_columns = ['text', 'message', 'description', 'content', 'query', 'log']
                text = ""
                
                for col in text_columns:
                    if col in row and row[col]:
                        text = row[col]
                        break
                
                if not text:
                    # Concatenate all non-numeric columns
                    text = " ".join([str(v) for k, v in row.items() if v and not k.lower().endswith('_id')])
                
                if text.strip():
                    texts.append(text.strip())
                    labels.append(int(row.get('label', self._generate_label_for_type(file_type))))
                    
                    # Extract metrics if available
                    metric_values = []
                    for key, value in row.items():
                        if key.lower() in ['cpu', 'memory', 'disk', 'network', 'response_time', 'error_rate']:
                            try:
                                metric_values.append(float(value))
                            except (ValueError, TypeError):
                                continue
                    
                    if not metric_values:
                        metric_values = self._generate_default_metrics()
                    
                    metrics.append(metric_values)
                    anomalies.append(int(row.get('anomaly', 0)))
        
        return texts, labels, metrics, anomalies
    
    def _load_with_encoding(self, file_path: Path, file_type: str, encoding: str) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Fallback method to load with specific encoding."""
        if file_path.suffix == '.json':
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                data = json.load(f)
            return self._process_data_by_type(data, file_type)
        elif file_path.suffix == '.txt':
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            texts = [text.strip() for text in content.split('\n') if text.strip()]
            labels = [self._generate_label_for_type(file_type) for _ in texts]
            metrics = [self._generate_default_metrics() for _ in texts]
            anomalies = [0 for _ in texts]
            return texts, labels, metrics, anomalies
        else:
            raise ValueError(f"Encoding fallback not implemented for {file_path.suffix}")

    def _generate_label_for_type(self, file_type: str) -> int:
        """Generate appropriate label based on file type."""
        type_labels = {
            'language': 0,
            'metrics': 1, 
            'splunk_logs': 2,
            'jira_tickets': 3,
            'confluence_docs': 4,
            'spectrum_logs': 5,
            'vemkd_logs': 6,
            'general': 7
        }
        return type_labels.get(file_type, 7)
    
    def _generate_default_metrics(self) -> List[float]:
        """Generate default metrics - 10 values to match metrics dataset structure."""
        return [
            1000.0,    # class_count
            0.5,       # gc_time  
            50.0,      # heap_usage
            100.0,     # thread_count
            25.0,      # cpu_usage
            1000.0,    # disk_io
            30.0,      # disk_usage
            1.0,       # load_average
            40.0,      # memory_usage
            500.0      # network_io
        ]
    
    def _process_data_by_type(self, data: List[Dict], file_type: str) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Process loaded data based on file type - now handles correct structures."""
        texts = []
        labels = []
        metrics = []
        anomalies = []
        
        for item in data:
            if isinstance(item, dict):
                # Extract text based on type
                text = ""
                
                if file_type == 'language':
                    # Language samples have 'response', 'explanation', or 'answer'
                    text = item.get('response') or item.get('explanation') or item.get('answer')
                    if not text:
                        # Fallback: combine term and question
                        parts = []
                        if item.get('term'):
                            parts.append(f"Term: {item['term']}")
                        if item.get('question'):
                            parts.append(f"Question: {item['question']}")
                        text = " ".join(parts)
                
                elif file_type == 'metrics':
                    # Metrics samples need metrics data
                    if 'metrics' in item:
                        metric_parts = []
                        metric_values = []
                        
                        # Take only first 5 metrics
                        for key, value in list(item['metrics'].items())[:5]:
                            metric_parts.append(f"{key}: {value}")
                            metric_values.append(float(value))
                        
                        text = f"Metrics: {', '.join(metric_parts)}. Status: {item.get('status', 'unknown')}"
                        
                        # Ensure exactly 5 values
                        while len(metric_values) < 5:
                            metric_values.append(0.0)
                        metric_values = metric_values[:5]
                        
                        metrics.append(metric_values)
                    else:
                        text = str(item)
                        metrics.append(self._generate_default_metrics())
                
                else:
                    # Generic extraction
                    text = item.get('text') or item.get('message') or item.get('content') or str(item)
                    metrics.append(self._generate_default_metrics())
                
                if text and text.strip():
                    texts.append(str(text).strip())
                    labels.append(self._get_label_for_sample(item, file_type))
                    
                    # Handle metrics if not already set
                    if file_type != 'metrics':
                        metrics.append(self._generate_default_metrics())
                    
                    # Determine anomaly status
                    anomalies.append(1 if item.get('status') == 'anomaly' else 0)
        
        return texts, labels, metrics, anomalies

    def _get_label_for_sample(self, sample: Dict, file_type: str) -> int:
        """Get appropriate label for a sample."""
        if file_type == 'metrics':
            return 2 if sample.get('status') == 'anomaly' else 0
        elif sample.get('type') == 'error_interpretation':
            return 1
        else:
            return self._generate_label_for_type(file_type)

class ContinualLearningEngine:
    """Engine for continual learning and metric adjustment."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.feedback_buffer = []
        self.metric_adjustments = {}
        self.learning_history = []
        
        # Load existing adjustments
        self._load_adjustments()
    
    def record_feedback(self, prediction: Dict, actual_outcome: str, user_feedback: Dict = None):
        """Record feedback for continual learning."""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'user_feedback': user_feedback,
            'metrics': prediction.get('input_metrics', {}),
            'prediction_confidence': prediction.get('anomaly_probability', 0)
        }
        
        self.feedback_buffer.append(feedback_entry)
        
        # Process feedback for immediate adjustments
        self._process_immediate_feedback(feedback_entry)
        
        # Trigger learning if buffer is full
        if len(self.feedback_buffer) >= CONFIG.get('learning_batch_size', 50):
            self._trigger_learning_update()
    
    def adjust_thresholds(self, metric_name: str, adjustment_factor: float):
        """Dynamically adjust metric thresholds based on learning."""
        if metric_name not in self.metric_adjustments:
            self.metric_adjustments[metric_name] = {
                'original_threshold': CONFIG.get('alert_thresholds', {}).get(metric_name, 80.0),
                'current_threshold': CONFIG.get('alert_thresholds', {}).get(metric_name, 80.0),
                'adjustment_history': []
            }
        
        current = self.metric_adjustments[metric_name]['current_threshold']
        new_threshold = current * adjustment_factor
        
        # Apply bounds checking
        original = self.metric_adjustments[metric_name]['original_threshold']
        min_threshold = original * 0.5  # Don't go below 50% of original
        max_threshold = original * 2.0  # Don't go above 200% of original
        
        new_threshold = max(min_threshold, min(max_threshold, new_threshold))
        
        self.metric_adjustments[metric_name]['current_threshold'] = new_threshold
        self.metric_adjustments[metric_name]['adjustment_history'].append({
            'timestamp': datetime.now().isoformat(),
            'old_threshold': current,
            'new_threshold': new_threshold,
            'adjustment_factor': adjustment_factor
        })
        
        logger.info(f"📊 Adjusted {metric_name} threshold: {current:.1f} → {new_threshold:.1f}")
        
        # Save adjustments
        self._save_adjustments()
    
    def _process_immediate_feedback(self, feedback: Dict):
        """Process feedback for immediate threshold adjustments."""
        prediction = feedback['prediction']
        actual = feedback['actual_outcome']
        metrics = feedback['metrics']
        
        # Check for false positives/negatives
        predicted_anomaly = prediction.get('final_anomaly', False)
        actual_anomaly = actual in ['anomaly', 'error', 'critical']
        
        if predicted_anomaly and not actual_anomaly:
            # False positive - relax thresholds slightly
            for metric_name, value in metrics.items():
                if value > CONFIG.get('alert_thresholds', {}).get(metric_name, 80):
                    self.adjust_thresholds(metric_name, 1.05)  # Increase threshold by 5%
        
        elif not predicted_anomaly and actual_anomaly:
            # False negative - tighten thresholds
            for metric_name, value in metrics.items():
                if value > CONFIG.get('alert_thresholds', {}).get(metric_name, 50):
                    self.adjust_thresholds(metric_name, 0.95)  # Decrease threshold by 5%

def main():
    """Main training function."""
    try:
        # Check if transformers is available
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Please install it.")
            return
        
        trainer = DistilledModelTrainer()
        trainer.train_model()
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()