#!/usr/bin/env python3
# Your python environment goes above.
"""
training_core.py - Because code reuse rocks. 
Unified training library originally designed for distilled monitoring model
Shared by distilled_model_trainer.py and standalone_trainer.py
Optimized for PyTorch 2.0.1 + Transformers 4.26.1 with Dask GPU support
"""

import os
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np

os.environ['NUMEXPR_MAX_THREADS'] = '16'

# PyTorch ecosystem imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer, AutoModel, AutoConfig,
        get_linear_schedule_with_warmup
    )
    
    # Try to import Dask for GPU acceleration
    try:
        import dask
        from dask.distributed import Client, as_completed
        DASK_AVAILABLE = True
    except ImportError:
        DASK_AVAILABLE = False
    
except ImportError as e:
    raise ImportError(f"Failed to import PyTorch dependencies: {e}")

# Import project modules
from common_utils import (
    load_dataset_file, analyze_existing_datasets, 
    check_models_like_trainer, get_dataset_paths, log_message, get_optimal_workers
)


class MonitoringModel(nn.Module):
    """Unified multi-task monitoring model for PyTorch."""
    
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


class UnifiedDatasetLoader:
    """Unified dataset loader with efficient processing."""
    
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
                
                log_message(f"✅ Loaded {len(lang_data['samples'])} language samples")
        
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


class TrainingEnvironment:
    """Unified training environment with device optimization and Dask support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = self._setup_device()
        self.dask_client = None
        
        # Setup Dask for GPU training if available
        if DASK_AVAILABLE and self.device.type == 'cuda' and config.get('use_dask_gpu', True):
            self._setup_dask_gpu()
        
        log_message(f"🔧 Training environment initialized")
        log_message(f"🎮 Device: {self.device}")
        if self.dask_client:
            log_message(f"⚡ Dask GPU cluster active")
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device with proper configuration."""
        if self.config.get('force_cpu', False):
            device = torch.device('cpu')
            log_message("💻 Forced CPU mode")
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
    
    def _setup_dask_gpu(self):
        """Setup Dask GPU cluster for accelerated training."""
        try:
            # Check if we're in a distributed environment
            if 'DASK_SCHEDULER_ADDRESS' in os.environ:
                # Connect to existing cluster
                scheduler_address = os.environ['DASK_SCHEDULER_ADDRESS']
                self.dask_client = Client(scheduler_address)
                log_message(f"⚡ Connected to Dask cluster: {scheduler_address}")
            else:
                # Start local GPU cluster
                from dask.distributed import LocalCluster
                
                # Configure for GPU training
                cluster = LocalCluster(
                    n_workers=1,
                    threads_per_worker=2,
                    memory_limit='auto',
                    dashboard_address=':8787'
                )
                self.dask_client = Client(cluster)
                log_message("⚡ Started local Dask GPU cluster")
            
            # Test cluster
            future = self.dask_client.submit(lambda: torch.cuda.is_available())
            gpu_available = future.result()
            
            if gpu_available:
                log_message("✅ Dask GPU cluster verified")
            else:
                log_message("⚠️  Dask cluster running but no GPU access")
                self.dask_client.close()
                self.dask_client = None
                
        except Exception as e:
            log_message(f"⚠️  Failed to setup Dask GPU: {e}")
            self.dask_client = None
    
    def cleanup(self):
        """Cleanup training environment."""
        if self.dask_client:
            self.dask_client.close()
            log_message("🧹 Dask client closed")


class UnifiedTrainer:
    """Unified training core with GPU/Dask optimization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env = TrainingEnvironment(config)
        self.device = self.env.device
        self.model = None
        self.tokenizer = None
        
        # Training state
        self.training_stats = {
            'start_time': None,
            'best_loss': float('inf'),
            'total_steps': 0,
            'current_step': 0
        }
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with caching."""
        model_name = self.config.get('model_name', 'bert-base-uncased')
        
        # Try local model first
        local_path = Path(self.config.get('pretrained_dir', './pretrained/')) / model_name
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
                model_name, cache_dir=self.config.get('hf_cache_dir', './hf_cache/')
            )
            base_model = AutoModel.from_pretrained(
                model_name, cache_dir=self.config.get('hf_cache_dir', './hf_cache/')
            )
            
            # Save locally for next time
            local_path.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(str(local_path))
            base_model.save_pretrained(str(local_path))
            log_message(f"💾 Cached model locally: {local_path}")
        
        # Create monitoring model
        self.model = MonitoringModel(base_model)
        self.model.to(self.device)
        
        # Enable compilation if supported and configured
        if (self.config.get('torch_compile', False) and 
            hasattr(torch, 'compile') and 
            self.device.type == 'cuda'):
            try:
                self.model = torch.compile(self.model)
                log_message("🚀 PyTorch model compilation enabled")
            except Exception as e:
                log_message(f"⚠️  Model compilation failed: {e}")
    
    def prepare_training_data(self):
        """Prepare training data with efficient loading."""
        log_message("📊 Preparing training data...")
        
        # Load training data
        loader = UnifiedDatasetLoader(self.config.get('training_dir', './training/'))
        texts, labels, metrics, anomalies = loader.load_training_data()
        
        if not texts:
            raise ValueError("No training data found")
        
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
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=self.config.get('persistent_workers', True)
        )
        
        return dataloader, len(texts)
    
    def create_optimizer_and_scheduler(self, dataloader):
        """Create optimizer and scheduler with proper initialization."""
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
        
        return optimizer, scheduler, total_steps
    
    def train_epoch_gpu(self, dataloader, optimizer, scheduler, scaler, epoch):
        """GPU-optimized training epoch with optional Dask acceleration."""
        epoch_loss = 0.0
        
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
                
                # FIXED: Correct order - backward, then optimizer, then scheduler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Now correctly after optimizer.step()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    metrics=metrics_batch,
                    anomalies=anomalies
                )
                loss = outputs['loss']
                
                # FIXED: Correct order - backward, then optimizer, then scheduler
                loss.backward()
                optimizer.step()
                scheduler.step()  # Now correctly after optimizer.step()
            
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
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}_step_{step}")
        
        return epoch_loss / len(dataloader)

    def train_epoch_cpu(self, dataloader, optimizer, scheduler, epoch):
        """CPU-optimized training epoch."""
        epoch_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            self.training_stats['current_step'] += 1
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            metrics_batch = batch['metrics'].to(self.device)
            anomalies = batch['anomalies'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                metrics=metrics_batch,
                anomalies=anomalies
            )
            loss = outputs['loss']
            
            # FIXED: Correct order - backward, then optimizer, then scheduler  
            loss.backward()
            optimizer.step()
            scheduler.step()  # Now correctly after optimizer.step()
            
            epoch_loss += loss.item()
            
            # Progress logging
            if step % 100 == 0:
                progress_pct = (self.training_stats['current_step'] / 
                              self.training_stats['total_steps']) * 100
                current_lr = optimizer.param_groups[0]['lr']
                log_message(f"   Step {step}: Loss={loss.item():.4f}, "
                          f"LR={current_lr:.2e} ({progress_pct:.1f}%)")
        
        return epoch_loss / len(dataloader)

    
    def train(self) -> bool:
        """Main unified training function."""
        log_message("🏋️ Starting unified model training")
        
        try:
            # Setup model and tokenizer
            log_message("🤖 Loading model and tokenizer...")
            self.load_model_and_tokenizer()
            
            # Prepare training data
            dataloader, sample_count = self.prepare_training_data()
            
            # Setup optimizer and scheduler
            optimizer, scheduler, total_steps = self.create_optimizer_and_scheduler(dataloader)
            
            # Mixed precision setup for GPU
            scaler = None
            if (self.config.get('mixed_precision', True) and 
                self.device.type == 'cuda'):
                scaler = torch.cuda.amp.GradScaler()
                log_message("🚀 Mixed precision training enabled")
            
            # Training setup
            self.training_stats['start_time'] = datetime.now()
            self.training_stats['total_steps'] = total_steps
            
            log_message(f"📈 Training configuration:")
            log_message(f"   Samples: {sample_count}")
            log_message(f"   Epochs: {self.config.get('epochs', 3)}")
            log_message(f"   Batch size: {self.config.get('batch_size', 16)}")
            log_message(f"   Learning rate: {self.config.get('learning_rate', 2e-5)}")
            log_message(f"   Total steps: {total_steps}")
            log_message(f"   Device: {self.device}")
            
            # Training loop
            self.model.train()
            
            for epoch in range(self.config.get('epochs', 3)):
                log_message(f"🔄 Starting epoch {epoch + 1}/{self.config.get('epochs', 3)}")
                
                # Use appropriate training method based on device
                if self.device.type == 'cuda':
                    avg_loss = self.train_epoch_gpu(dataloader, optimizer, scheduler, scaler, epoch)
                else:
                    avg_loss = self.train_epoch_cpu(dataloader, optimizer, scheduler, epoch)
                
                log_message(f"✅ Epoch {epoch + 1} completed: Avg Loss={avg_loss:.4f}")
                
                # Save best model
                if avg_loss < self.training_stats['best_loss']:
                    self.training_stats['best_loss'] = avg_loss
                    self.save_final_model()
                    log_message(f"🏆 New best model saved: Loss={avg_loss:.4f}")

                # CLEANUP: Clean up checkpoints after each epoch
                if epoch > 0:  # Don't clean up after first epoch
                    self.cleanup_training_artifacts()
            
            # Training completed
            duration = datetime.now() - self.training_stats['start_time']
            log_message(f"🎉 Training completed successfully!")
            log_message(f"⏱️ Duration: {duration}")
            log_message(f"📈 Best loss: {self.training_stats['best_loss']:.4f}")
            
            return True
            
        except Exception as e:
            log_message(f"❌ Training failed: {e}")
            log_message(f"🔍 Traceback: {traceback.format_exc()}")
            return False
        finally:
            # Cleanup
            self.env.cleanup()

    def cleanup_training_artifacts(self):
        """Clean up training artifacts during and after training."""
        from common_utils import manage_training_checkpoints, periodic_cleanup
        
        # Clean up checkpoints
        checkpoints_dir = Path(self.config.get('checkpoints_dir', './checkpoints/'))
        models_dir = Path(self.config.get('models_dir', './models/'))
        manage_training_checkpoints(checkpoints_dir, models_dir)
        
        # Run periodic cleanup
        periodic_cleanup(self.config)
    
    def save_final_model(self) -> str:
        """Save final trained model using safetensors format."""
        models_dir = Path(self.config.get('models_dir', './models/'))
        models_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = models_dir / f"distilled_monitoring_{timestamp}"
        model_path.mkdir(exist_ok=True)
        
        try:
            from safetensors.torch import save_file
            
            # Save base model using safetensors
            base_model_state = self.model.base_model.state_dict()
            save_file(base_model_state, model_path / "model.safetensors")
            
            # Save tokenizer (still uses JSON/text files)
            self.tokenizer.save_pretrained(str(model_path))
            
            # Save model config
            self.model.base_model.config.save_pretrained(str(model_path))
            
            # Save custom heads using safetensors
            custom_heads_state = {
                'classifier.weight': self.model.classifier.weight,
                'classifier.bias': self.model.classifier.bias,
                'anomaly_detector.weight': self.model.anomaly_detector.weight,
                'anomaly_detector.bias': self.model.anomaly_detector.bias,
                'metrics_regressor.weight': self.model.metrics_regressor.weight,  
                'metrics_regressor.bias': self.model.metrics_regressor.bias,
            }
            save_file(custom_heads_state, model_path / 'custom_heads.safetensors')
            
            log_message("💾 Model saved in safetensors format")
            
        except ImportError:
            log_message("⚠️  safetensors not available, falling back to PyTorch format")
            # Fallback to original PyTorch format
            self.model.base_model.save_pretrained(str(model_path))
            
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
        
        # Save model configuration metadata
        model_config = {
            'model_type': 'distilled_monitoring',
            'num_labels': 8,
            'num_metrics': 10,
            'hidden_size': self.model.hidden_size,
            'architecture': 'multi_task_transformer',
            'base_model_name': self.config.get('model_name'),
            'format': 'safetensors'  # Indicate the format used
        }
        
        with open(model_path / 'model_config.json', 'w') as f:
            json.dump(model_config, f, indent=2)
        
        # Save training metadata
        metadata = {
            'model_type': 'distilled_monitoring',
            'framework': 'pytorch',
            'format': 'safetensors',
            'base_model': self.config.get('model_name'),
            'training_samples': getattr(self, 'sample_count', 0),
            'best_loss': float(self.training_stats['best_loss']),
            'training_duration': str(datetime.now() - self.training_stats['start_time']),
            'device_type': str(self.device),
            'torch_version': torch.__version__,
            'safetensors_used': True,
            'dask_used': self.env.dask_client is not None,
            'config': {k: v for k, v in self.config.items() 
                      if not callable(v) and k != 'tokenizer'},
            'timestamp': timestamp,
            'model_path': str(model_path)
        }
        
        with open(model_path / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        log_message(f"💾 Final model saved to: {model_path}")
        return str(model_path)
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint using safetensors when possible."""
        checkpoint_dir = Path(self.config.get('checkpoints_dir', './checkpoints/'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from safetensors.torch import save_file
            
            # Save model state using safetensors
            model_state_path = checkpoint_dir / f"{name}_model.safetensors"
            save_file(self.model.state_dict(), model_state_path)
            
            # Save training metadata separately (JSON)
            metadata_path = checkpoint_dir / f"{name}_metadata.json"
            metadata = {
                'training_stats': self.training_stats,
                'config': {k: v for k, v in self.config.items() if not callable(v)},
                'format': 'safetensors',
                'model_file': f"{name}_model.safetensors"
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            log_message(f"💾 Checkpoint saved: {name} (safetensors)")
            
        except ImportError:
            # Fallback to PyTorch format
            checkpoint_path = checkpoint_dir / f"{name}.pt"
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'training_stats': self.training_stats,
                'config': self.config
            }, checkpoint_path)
            log_message(f"💾 Checkpoint saved: {name} (pytorch)")


# Utility functions for backward compatibility
def create_trainer(config: Dict[str, Any]) -> UnifiedTrainer:
    """Factory function to create a unified trainer."""
    return UnifiedTrainer(config)


def validate_training_environment(config: Dict[str, Any]) -> bool:
    """Validate training environment and datasets."""
    # Check for datasets
    analysis = analyze_existing_datasets(Path(config.get('training_dir', './training/')))
    if not (analysis['language_dataset']['exists'] and analysis['metrics_dataset']['exists']):
        log_message("❌ Required datasets not found")
        log_message("   Language dataset: " + 
                    ("✅" if analysis['language_dataset']['exists'] else "❌"))
        log_message("   Metrics dataset: " + 
                    ("✅" if analysis['metrics_dataset']['exists'] else "❌"))
        return False
    
    # Check PyTorch availability
    if not torch.cuda.is_available() and not config.get('force_cpu', False):
        log_message("⚠️  CUDA not available, will use CPU training")
    
    return True