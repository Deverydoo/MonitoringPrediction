#!/usr/bin/env python3
"""
standalone_trainer.py
Streamlined command-line PyTorch trainer for distilled monitoring model
Optimized for PyTorch 2.0.1 + Transformers 4.26.1
"""

import os
import json
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np

os.environ['NUMEXPR_MAX_THREADS'] = '16'  # Set before any NumExpr usage

### --- Deprecated but keeping in just in case I missed something. 
### The unified logger is now in the common_utils.
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PyTorch ecosystem imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        get_linear_schedule_with_warmup
    )
    logger.info(f"✅ PyTorch {torch.__version__} loaded")
except ImportError as e:
    logger.error(f"❌ Failed to import PyTorch dependencies: {e}")
    raise

# Import project modules
try:
    from config import CONFIG
    from common_utils import (
        load_dataset_file, analyze_existing_datasets, 
        check_models_like_trainer, get_dataset_paths, log_message, get_optimal_workers
    )
    logger.info("✅ Project modules loaded")
except ImportError as e:
    logger.error(f"❌ Failed to import project modules: {e}")
    raise


class MonitoringModel(nn.Module):
    """Streamlined multi-task monitoring model for PyTorch."""
    
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
        """Initialize task-specific heads."""
        for module in [self.classifier, self.anomaly_detector, self.metrics_regressor]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask, labels=None, metrics=None, anomalies=None):
        """Forward pass with multi-task outputs."""
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


class MonitoringDataset(Dataset):
    """Efficient PyTorch dataset for monitoring data."""
    
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


class DatasetLoader:
    """Streamlined dataset loader with efficient processing."""
    
    def __init__(self, training_dir: str):
        self.training_dir = Path(training_dir)
    
    def load_training_data(self) -> Tuple[List[str], List[int], List[List[float]], List[int]]:
        """Load and process all training data efficiently."""
        log_message("📊 Loading training datasets...")
        
        texts, labels, metrics, anomalies = [], [], [], []
        dataset_paths = get_dataset_paths(self.training_dir)
        
        # Load language dataset
        if dataset_paths['language_dataset'].exists():
            lang_data = load_dataset_file(dataset_paths['language_dataset'])
            if lang_data and 'samples' in lang_data:
                for sample in lang_data['samples']:
                    if isinstance(sample, dict) and 'explanation' in sample:
                        texts.append(sample['explanation'])
                        labels.append(0)  # Default technical label
                        metrics.append([0.0] * 10)  # Default metrics
                        anomalies.append(0)  # Default normal
                
                logger.info(f"✅ Loaded {len(lang_data['samples'])} language samples")
        
        # Load metrics dataset
        if dataset_paths['metrics_dataset'].exists():
            metrics_data = load_dataset_file(dataset_paths['metrics_dataset'])
            if metrics_data and 'training_samples' in metrics_data:
                for sample in metrics_data['training_samples']:
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
                
                log_message(f"✅ Loaded {len(metrics_data['training_samples'])} metrics samples")
        
        # Clean data
        clean_texts, clean_labels, clean_metrics, clean_anomalies = [], [], [], []
        
        for i, text in enumerate(texts):
            if len(text.strip()) >= 10:  # Minimum text length
                clean_text = text.strip().replace('\x00', '').replace('\ufffd', '')
                if len(clean_text) >= 10:
                    clean_texts.append(clean_text)
                    clean_labels.append(labels[i])
                    clean_metrics.append(metrics[i])
                    clean_anomalies.append(anomalies[i])
        
        log_message(f"🎯 Final dataset: {len(clean_texts)} samples")
        log_message(f"   Labels: {len(set(clean_labels))} unique")
        log_message(f"   Anomalies: {sum(clean_anomalies)} anomaly samples")
        
        return clean_texts, clean_labels, clean_metrics, clean_anomalies


class StandaloneTrainer:
    """Streamlined standalone PyTorch trainer."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.tokenizer = None
        
        # Training state
        self.training_stats = {
            'start_time': None,
            'best_loss': float('inf'),
            'total_steps': 0,
            'current_step': 0
        }
        
        logger.info(f"🔧 Standalone trainer initialized")
        logger.info(f"🎮 Device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device with proper configuration."""
        if self.config.get('force_cpu', False):
            device = torch.device('cpu')
            logger.info("💻 Forced CPU mode")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            log_message(f"🎮 CUDA GPU: {gpu_name} ({gpu_memory}GB)")
            
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            log_message("🍎 Apple Silicon (MPS)")
        else:
            device = torch.device('cpu')
            torch.set_num_threads(os.cpu_count())
            log_message(f"💻 CPU ({os.cpu_count()} threads)")
        
        return device
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer with caching."""
        model_name = self.config.get('model_name', 'bert-base-uncased')
        
        # Try local model first
        local_path = Path(self.config['pretrained_dir']) / model_name
        if local_path.exists():
            log_message(f"📁 Loading from local cache: {local_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(local_path), local_files_only=True
                )
                base_model = AutoModel.from_pretrained(
                    str(local_path), local_files_only=True
                )
                log_message(f"✅ Loaded cached model: {model_name}")
            except Exception as e:
                log_message(f"❌ Local model load failed: {e}")
                raise
        else:
            # Download and cache
            log_message(f"📥 Downloading {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=self.config['hf_cache_dir']
            )
            base_model = AutoModel.from_pretrained(
                model_name, cache_dir=self.config['hf_cache_dir']
            )
            
            # Save locally for next time
            local_path.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(str(local_path))
            base_model.save_pretrained(str(local_path))
            logger.info(f"💾 Cached model locally: {local_path}")
        
        # Create monitoring model
        self.model = MonitoringModel(base_model)
        self.model.to(self.device)
        
        # Enable compilation if supported and configured
        if (self.config.get('torch_compile', False) and 
            hasattr(torch, 'compile') and 
            self.device.type == 'cuda'):
            try:
                self.model = torch.compile(self.model)
                logger.info("🚀 PyTorch model compilation enabled")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    def train(self) -> bool:
        """Main training function with proper optimizer/scheduler ordering."""
        logger.info("🏋️ Starting standalone model training")
        
        try:
            # Setup model and tokenizer
            log_message("🤖 Loading model and tokenizer...")
            self._load_model_and_tokenizer()
            
            # Load training data
            log_message("📊 Loading training data...")
            loader = DatasetLoader(self.config['training_dir'])
            texts, labels, metrics, anomalies = loader.load_training_data()
            
            if not texts:
                log_message("❌ No training data found")
                return False
            
            # Create dataset and dataloader
            dataset = MonitoringDataset(
                texts, labels, metrics, anomalies,
                self.tokenizer, self.config.get('max_length', 512)
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.get('batch_size', 16),
                shuffle=True,
                num_workers=get_optimal_workers(),
                pin_memory=(self.device.type == 'cuda')
            )
            
            # Setup optimizer and scheduler with proper initialization
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('learning_rate', 2e-5),
                weight_decay=self.config.get('weight_decay', 0.01)
            )
            
            total_steps = len(dataloader) * self.config.get('epochs', 3)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.get('warmup_steps', 100),
                num_training_steps=total_steps
            )
            
            # Mixed precision setup
            scaler = None
            if (self.config.get('mixed_precision', True) and 
                self.device.type == 'cuda'):
                scaler = torch.cuda.amp.GradScaler()
                log_message("🚀 Mixed precision training enabled")
            
            # Training setup
            self.training_stats['start_time'] = datetime.now()
            self.training_stats['total_steps'] = total_steps
            
            log_message(f"📈 Training configuration:")
            log_message(f"   Samples: {len(texts)}")
            log_message(f"   Epochs: {self.config.get('epochs', 3)}")
            log_message(f"   Batch size: {self.config.get('batch_size', 16)}")
            log_message(f"   Learning rate: {self.config.get('learning_rate', 2e-5)}")
            log_message(f"   Total steps: {total_steps}")
            
            # Training loop
            self.model.train()
            
            for epoch in range(self.config.get('epochs', 3)):
                epoch_loss = 0.0
                log_message(f"🔄 Starting epoch {epoch + 1}/{self.config.get('epochs', 3)}")
                
                for step, batch in enumerate(dataloader):
                    self.training_stats['current_step'] += 1
                    
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    metrics_batch = batch['metrics'].to(self.device)
                    anomalies = batch['anomalies'].to(self.device)
                    
                    # Forward pass with optional mixed precision
                    optimizer.zero_grad()
                    
                    if scaler:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                metrics=metrics_batch,
                                anomalies=anomalies
                            )
                            loss = outputs['loss']
                        
                        # Backward pass with scaling
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)  # Optimizer step BEFORE scheduler
                        scaler.update()
                    else:
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            metrics=metrics_batch,
                            anomalies=anomalies
                        )
                        loss = outputs['loss']
                        
                        # Standard backward pass
                        loss.backward()
                        optimizer.step()  # Optimizer step BEFORE scheduler
                    
                    # Scheduler step AFTER optimizer step
                    scheduler.step()
                    
                    epoch_loss += loss.item()
                    
                    # Progress logging
                    if step % 100 == 0:
                        progress_pct = (self.training_stats['current_step'] / 
                                      self.training_stats['total_steps']) * 100
                        current_lr = optimizer.param_groups[0]['lr']
                        log_message(f"   Step {step}: Loss={loss.item():.4f}, "
                                  f"LR={current_lr:.2e} ({progress_pct:.1f}%)")
                    
                    # Save checkpoint periodically
                    if step % 500 == 0 and step > 0:
                        self._save_checkpoint(f"checkpoint_epoch_{epoch+1}_step_{step}")
                
                # End of epoch processing
                avg_loss = epoch_loss / len(dataloader)
                log_message(f"✅ Epoch {epoch + 1} completed: Avg Loss={avg_loss:.4f}")
                
                # Save best model
                if avg_loss < self.training_stats['best_loss']:
                    self.training_stats['best_loss'] = avg_loss
                    self._save_final_model()
                    log_message(f"🏆 New best model saved: Loss={avg_loss:.4f}")
            
            # Training completed
            duration = datetime.now() - self.training_stats['start_time']
            log_messageo(f"🎉 Training completed successfully!")
            log_message(f"⏱️ Duration: {duration}")
            log_message(f"📈 Best loss: {self.training_stats['best_loss']:.4f}")
            
            return True
            
        except Exception as e:
            log_message(f"❌ Training failed: {e}")
            log_message(f"🔍 Traceback: {traceback.format_exc()}")
            return False
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoints_dir', './checkpoints/'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config
        }, checkpoint_path)
        
        logger.debug(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self) -> str:
        """Save final trained model."""
        models_dir = Path(self.config['models_dir'])
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"distilled_monitoring_{timestamp}"
        model_path.mkdir(exist_ok=True)
        
        # Save model using transformers format
        self.model.base_model.save_pretrained(str(model_path))
        self.tokenizer.save_pretrained(str(model_path))
        
        # Save custom heads
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
        
        # Save training metadata
        metadata = {
            'model_type': 'distilled_monitoring',
            'framework': 'pytorch',
            'base_model': self.config.get('model_name'),
            'training_samples': len(self.training_stats.get('samples', [])),
            'best_loss': float(self.training_stats['best_loss']),
            'training_duration': str(datetime.now() - self.training_stats['start_time']),
            'device_type': str(self.device),
            'torch_version': torch.__version__,
            'config': {k: v for k, v in self.config.items() 
                      if not callable(v) and k != 'tokenizer'},
            'timestamp': timestamp,
            'model_path': str(model_path)
        }
        
        with open(model_path / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        log_message(f"💾 Final model saved to: {model_path}")
        return str(model_path)


def main():
    """Main entry point for standalone training."""
    parser = argparse.ArgumentParser(
        description="Standalone PyTorch trainer for distilled monitoring model"
    )
    parser.add_argument(
        '--config', '-c', 
        type=str, 
        default='config.py',
        help='Configuration file path'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--force-cpu',
        action='store_true',
        help='Force CPU training'
    )
    parser.add_argument(
        '--no-compile',
        action='store_true',
        help='Disable torch.compile'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    try:
        config = CONFIG.copy()  # Use global config
        
        # Apply command line overrides
        if args.epochs:
            config['epochs'] = args.epochs
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if args.learning_rate:
            config['learning_rate'] = args.learning_rate
        if args.force_cpu:
            config['force_cpu'] = True
        if args.no_compile:
            config['torch_compile'] = False
            
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Check for datasets
    analysis = analyze_existing_datasets(Path(config['training_dir']))
    if not (analysis['language_dataset']['exists'] and analysis['metrics_dataset']['exists']):
        log_message("❌ Required datasets not found. Run dataset generation first.")
        log_message("   Language dataset: " + 
                    ("✅" if analysis['language_dataset']['exists'] else "❌"))
        log_message("   Metrics dataset: " + 
                    ("✅" if analysis['metrics_dataset']['exists'] else "❌"))
        return 1
    
    # Initialize and run trainer
    log_message("🚀 STANDALONE DISTILLED MODEL TRAINER")
    log_message("=" * 50)
    log_message(f"PyTorch: {torch.__version__}")
    log_message(f"CUDA available: {torch.cuda.is_available()}")
    log_message(f"Configuration: {len(config)} settings loaded")
    
    trainer = StandaloneTrainer(config)
    success = trainer.train()
    
    if success:
        log_message("🎉 Training completed successfully!")
        
        # Verify model was saved
        if check_models_like_trainer(Path(config['models_dir'])):
            log_message("✅ Model verification passed")
        else:
            log_message("⚠️  Model verification failed")
        
        return 0
    else:
        log_message("❌ Training failed!")
        return 1


if __name__ == "__main__":
    exit(main())