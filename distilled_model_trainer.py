#!/usr/bin/env python3
"""
Distilled Model Trainer - Refactored and Streamlined
Technology-specific predictive LLM for monitoring and troubleshooting
"""

import os
import json
import logging
import torch
# Completely disable dynamo/compilation before any model operations
try:
    torch._dynamo.config.disable = True
    torch._dynamo.config.suppress_errors = True
except:
    pass

import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import traceback
from collections import defaultdict

# Configure encoding for Windows/Linux compatibility
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('./logs/training.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class TrainingEnvironment:
    """Manages training environment with CUDA->Spark->CPU fallback"""
    
    def __init__(self):
        self.device = self._detect_best_device()
        self.env_type = self._get_environment_type()
        self._setup_environment()
    
    def _detect_best_device(self) -> str:
        """Detect best available device with fallback chain"""
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            gpu_name = torch.cuda.get_device_name()
            logger.info(f"🎮 CUDA GPU: {gpu_name}")
            return device
        
        # Check for Spark environment
        try:
            from pyspark import SparkContext
            if SparkContext._active_spark_context:
                logger.info("⚡ Spark environment detected")
                return "spark"
        except ImportError:
            pass
        
        logger.info("💻 Using CPU")
        return "cpu"
    
    def _get_environment_type(self) -> str:
        """Get environment type for optimization"""
        if "cuda" in self.device:
            return "cuda"
        elif self.device == "spark":
            return "spark"
        return "cpu"
    
    def _setup_environment(self):
        """Setup environment-specific optimizations"""
        if self.env_type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("🚀 CUDA optimizations enabled")
        
        elif self.env_type == "cpu":
            torch.set_num_threads(os.cpu_count())
            logger.info(f"🔧 CPU threads: {os.cpu_count()}")

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

class MonitoringModel(nn.Module):
    """Multi-task monitoring model with technical focus"""
    
    def __init__(self, base_model, num_labels: int = 8, num_metrics: int = 10):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = base_model.config.hidden_size
        
        # Multi-task heads
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.anomaly_detector = nn.Linear(self.hidden_size, 2)  # Binary anomaly detection
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

class DistilledModelTrainer:
    """Main trainer class with progress tracking and optimization"""
    
    def __init__(self, config: Dict[str, Any], resume_training: bool = False):
        """Initialize trainer with optional resume capability"""
        self.config = config
        self.env = TrainingEnvironment()
        self.loader = DatasetLoader(config['training_dir'])  # This is the correct class name
        self.device = torch.device(self.env.device if 'cuda' in self.env.device else 'cpu')
        
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
        """Setup model and tokenizer with local caching"""
        from transformers import AutoTokenizer, AutoModel
        from config import CONFIG
        
        model_name = self.config.get('model_name', 'bert-base-uncased')
        
        # Try local model first
        local_path = Path(CONFIG['pretrained_dir']) / model_name
        if local_path.exists():
            logger.info(f"📁 Loading from local: {local_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(local_path), local_files_only=True
                )
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
        self.model.to(self.device)
        
        # Enable compilation if supported
        if hasattr(torch, 'compile') and self.env.env_type == "cuda":
            try:
                self.model = torch.compile(self.model)
                logger.info("🚀 Model compilation enabled")
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
        
        # Tokenize
        logger.info("🔤 Tokenizing training data...")
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.get('max_length', 512),
            return_tensors="pt"
        )
        
        # Create dataset
        from torch.utils.data import TensorDataset
        dataset = TensorDataset(
            encoded['input_ids'],
            encoded['attention_mask'],
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(metrics, dtype=torch.float32),
            torch.tensor(anomalies, dtype=torch.long)
        )
        
        logger.info(f"✅ Training dataset ready: {len(dataset)} samples")
        return dataset
    
    def train(self):
        """Main training loop with progress tracking"""
        try:
            self.training_progress['start_time'] = datetime.now()
            
            # Setup
            logger.info("🏋️ Starting model training...")
            self.setup_model_and_tokenizer()
            dataset = self.prepare_training_data()
            
            # Create data loader
            from torch.utils.data import DataLoader
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.get('batch_size', 8), 
                shuffle=True
            )
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 2e-5),
                weight_decay=self.config.get('weight_decay', 0.01)
            )
            
            # Calculate total steps
            epochs = self.config.get('epochs', 3)
            self.training_progress['total_steps'] = len(dataloader) * epochs
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                self.training_progress['current_epoch'] = epoch + 1
                epoch_loss = 0.0
                
                logger.info(f"📚 Epoch {epoch + 1}/{epochs}")
                
                for step, batch in enumerate(dataloader):
                    self.training_progress['current_step'] += 1
                    
                    # Move batch to device
                    input_ids, attention_mask, labels, metrics, anomalies = [
                        b.to(self.device) for b in batch
                    ]
                    
                    # Forward pass
                    optimizer.zero_grad()
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        metrics=metrics,
                        anomalies=anomalies
                    )
                    
                    loss = outputs['loss']
                    epoch_loss += loss.item()
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    # Progress logging
                    if step % 50 == 0:
                        progress_pct = (self.training_progress['current_step'] / 
                                      self.training_progress['total_steps']) * 100
                        logger.info(f"  Step {step}: Loss={loss.item():.4f} ({progress_pct:.1f}% complete)")
                
                avg_loss = epoch_loss / len(dataloader)
                self.training_progress['losses'].append(avg_loss)
                
                if avg_loss < self.training_progress['best_loss']:
                    self.training_progress['best_loss'] = avg_loss
                    self._save_checkpoint('best_model')
                
                logger.info(f"✅ Epoch {epoch + 1} completed: Avg Loss={avg_loss:.4f}")
            
            # Save final model
            final_path = self._save_final_model()
            
            # Training summary
            duration = datetime.now() - self.training_progress['start_time']
            logger.info(f"🎉 Training completed in {duration}")
            logger.info(f"   Best loss: {self.training_progress['best_loss']:.4f}")
            logger.info(f"   Final model: {final_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.get('checkpoints_dir', './checkpoints/'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_progress': self.training_progress,
            'config': self.config
        }, checkpoint_path)
    
    def _save_final_model(self) -> str:
        """Save final trained model"""
        from config import CONFIG
        import numpy as np
        
        models_dir = Path(CONFIG['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"distilled_monitoring_{timestamp}"
        model_path.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        self.model.base_model.save_pretrained(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
    
        # Convert config to JSON-serializable format
        def make_json_serializable(obj):
            """Convert numpy/torch types to JSON-serializable types"""
            import numpy as np
            import torch
            
            if obj is None:
                return None
            elif hasattr(obj, 'dtype'):
                # Handle numpy dtypes specifically
                if str(type(obj)).startswith("<class 'numpy.dtype"):
                    return str(obj)
                # Handle numpy arrays and tensors
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                else:
                    return str(obj)
            elif isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
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
                return str(obj)  # Convert objects to string representation
            else:
                try:
                    # Test if it's JSON serializable
                    import json
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Clean config for JSON serialization
        clean_config = make_json_serializable(self.config)
        
        # Save training metadata
        metadata = {
            'model_type': 'distilled_monitoring',
            'base_model': self.config.get('model_name'),
            'training_samples': int(self.training_progress.get('total_steps', 0)),
            'best_loss': float(self.training_progress.get('best_loss', 0.0)),
            'training_time': str(datetime.now() - self.training_progress['start_time']),
            'config': clean_config,
            'timestamp': timestamp,
            'model_path': str(model_path)
        }
        
        with open(model_path / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Model saved to: {model_path}")
        # Congrats!
        return str(model_path)

    def find_latest_model(self) -> Optional[str]:
        """Find the most recent trained model for resuming"""
        from config import CONFIG
        
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
        required_files = ['pytorch_model.bin', 'config.json', 'training_metadata.json']
        if all((latest_model / f).exists() for f in required_files):
            logger.info(f"🔍 Found latest model: {latest_model}")
            return str(latest_model)
        
        logger.warning(f"⚠️ Latest model incomplete: {latest_model}")
        return None

    def load_from_checkpoint(self, model_path: str) -> bool:
        """Load model from previous training checkpoint"""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            model_path = Path(model_path)
            
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            
            # Load the base model
            base_model = AutoModel.from_pretrained(str(model_path))
            
            # Create monitoring model
            self.model = MonitoringModel(base_model)
            self.model.to(self.device)
            
            # Load training metadata if available
            metadata_path = model_path / 'training_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    logger.info(f"📊 Previous training: {metadata.get('training_samples', 0)} samples")
                    logger.info(f"📈 Previous best loss: {metadata.get('best_loss', 'unknown')}")
            
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