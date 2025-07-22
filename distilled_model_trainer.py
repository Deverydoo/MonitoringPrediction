# distilled_model_trainer.py
import os
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple

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
    
    def __init__(self, base_model_name: str, num_labels: int = 3):
        super().__init__()
        self.num_labels = num_labels
        
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
        
    def forward(self, input_ids, attention_mask=None, labels=None, 
                metrics_targets=None, anomaly_targets=None):
        
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use [CLS] token representation
        if hasattr(outputs, 'last_hidden_state'):
            pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        elif hasattr(outputs, 'pooler_output'):
            pooled_output = outputs.pooler_output
        else:
            # Fallback: mean pooling
            pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        pooled_output = self.dropout(pooled_output)
        
        # Classification logits
        classification_logits = self.classifier(pooled_output)
        
        # Regression outputs
        metrics_predictions = self.regressor(pooled_output)
        
        # Anomaly detection
        anomaly_logits = self.anomaly_detector(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(classification_logits, labels)
            
            total_loss = classification_loss
            
            # Add regression loss if targets provided
            if metrics_targets is not None:
                mse_loss = nn.MSELoss()
                regression_loss = mse_loss(metrics_predictions, metrics_targets)
                total_loss += 0.5 * regression_loss
            
            # Add anomaly detection loss if targets provided
            if anomaly_targets is not None:
                bce_loss = nn.BCEWithLogitsLoss()
                anomaly_loss = bce_loss(anomaly_logits.squeeze(), anomaly_targets.float())
                total_loss += 0.3 * anomaly_loss
            
            loss = total_loss
        
        return {
            'loss': loss,
            'classification_logits': classification_logits,
            'metrics_predictions': metrics_predictions,
            'anomaly_logits': anomaly_logits
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
        """Load tokenizer with PyTorch backend."""
        try:
            # Try HuggingFace first
            tokenizer = AutoTokenizer.from_pretrained(
                CONFIG['model_name'],
                cache_dir=CONFIG['hf_cache_dir'],
                trust_remote_code=CONFIG.get('trust_remote_code', False),
                use_auth_token=CONFIG.get('use_auth_token', False)
            )
        except Exception as e:
            logger.warning(f"Could not load tokenizer from HuggingFace: {e}")
            try:
                # Try local cache
                local_path = os.path.join(CONFIG['local_model_cache'], CONFIG['model_name'])
                tokenizer = AutoTokenizer.from_pretrained(local_path)
            except Exception as e2:
                logger.error(f"Could not load tokenizer from local cache: {e2}")
                raise RuntimeError("No tokenizer available")
        
        # Add pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
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
    
    def __init__(self, training_dir: str):
        self.training_dir = Path(training_dir)
        self.supported_formats = ['.json', '.jsonl', '.csv', '.txt']
        
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
    
    def load_flexible_training_data(self) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Load training data from all discovered files."""
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
                    
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        
        logger.info(f"📊 Loaded {len(texts)} total training samples")
        return texts, labels, metrics, anomalies
    
    def _load_single_file(self, file_path: Path, file_type: str) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Load data from a single file based on its type."""
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