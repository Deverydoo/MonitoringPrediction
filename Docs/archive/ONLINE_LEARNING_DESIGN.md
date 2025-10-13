# Online Learning for TFT Monitoring System

**Date:** 2025-10-08
**Architecture:** Temporal Fusion Transformer (PyTorch Forecasting)
**Goal:** Enable continuous learning during production deployment

---

## 🎯 Overview

**Yes, online/continual learning is absolutely possible** with the TFT architecture. This document outlines strategies and implementation approaches.

---

## 📋 Table of Contents

1. [Why Online Learning?](#why-online-learning)
2. [Challenges & Solutions](#challenges--solutions)
3. [Implementation Strategies](#implementation-strategies)
4. [Architecture Options](#architecture-options)
5. [Code Implementation](#code-implementation)
6. [Configuration](#configuration)
7. [Monitoring & Safety](#monitoring--safety)

---

## 🤔 Why Online Learning?

### **Benefits:**

1. **Adaptation to Drift**
   - Server workload patterns change over time
   - New applications deployed
   - Hardware upgrades/changes
   - Seasonal patterns evolve

2. **Self-Improving System**
   - Learns from recent incidents
   - Adapts to new normal baselines
   - Reduces false positives over time

3. **No Retraining Downtime**
   - Continuous improvement
   - No service interruption
   - Always up-to-date predictions

4. **Efficient Resource Use**
   - Update only on new data
   - No need to retrain from scratch
   - Incremental improvements

---

## ⚠️ Challenges & Solutions

### **Challenge 1: Catastrophic Forgetting**
**Problem:** Model forgets old patterns when learning new ones

**Solutions:**
- ✅ **Replay Buffer** - Keep representative samples from past
- ✅ **Elastic Weight Consolidation (EWC)** - Protect important weights
- ✅ **Knowledge Distillation** - Teacher-student approach
- ✅ **Experience Replay** - Mix old and new data in updates

### **Challenge 2: Distribution Shift**
**Problem:** New data may be from different distribution

**Solutions:**
- ✅ **Drift Detection** - Monitor prediction error trends
- ✅ **Adaptive Learning Rate** - Reduce LR when drift detected
- ✅ **Ensemble Methods** - Keep multiple model versions
- ✅ **Validation Set** - Maintain hold-out set for quality checks

### **Challenge 3: Training Stability**
**Problem:** Frequent updates can destabilize model

**Solutions:**
- ✅ **Micro-batching** - Small, frequent updates
- ✅ **Warm Restarts** - Periodic LR resets
- ✅ **Gradient Clipping** - Prevent extreme updates
- ✅ **Checkpointing** - Rollback if performance degrades

### **Challenge 4: Computational Resources**
**Problem:** Training during inference time

**Solutions:**
- ✅ **Async Updates** - Background training thread
- ✅ **Scheduled Windows** - Update during low-traffic periods
- ✅ **Mini-batch SGD** - Fast, incremental updates
- ✅ **Model Quantization** - Lighter inference model

---

## 🏗️ Implementation Strategies

### **Strategy 1: Periodic Retraining (Simplest)**

**Approach:** Retrain model on accumulating data at fixed intervals

**Pros:**
- Simple to implement
- Proven approach
- Easy to rollback

**Cons:**
- Not truly "online"
- Delay in adaptation
- Resource spikes

**Implementation:**
```python
# Every N hours, retrain on last M days of data
RETRAIN_INTERVAL_HOURS = 24
LOOKBACK_DAYS = 30

if hours_since_last_training >= RETRAIN_INTERVAL_HOURS:
    new_data = load_data(lookback_days=LOOKBACK_DAYS)
    model = retrain_model(new_data)
    save_model(model, versioned=True)
```

---

### **Strategy 2: Sliding Window Updates (Recommended)**

**Approach:** Continuously update on sliding window of recent data

**Pros:**
- True online learning
- Smooth adaptation
- Resource efficient

**Cons:**
- Needs careful tuning
- Complexity increase

**Implementation:**
```python
# Maintain sliding window buffer
WINDOW_SIZE_HOURS = 168  # 1 week
UPDATE_FREQUENCY_MIN = 60  # Update every hour

# Circular buffer of recent data
buffer = SlidingWindowBuffer(max_hours=WINDOW_SIZE_HOURS)

# On new data arrival
buffer.add(new_metrics)

# Every update interval
if time_for_update():
    recent_data = buffer.get_window()
    model.partial_fit(recent_data, learning_rate=ONLINE_LR)
```

---

### **Strategy 3: Experience Replay (Advanced)**

**Approach:** Mix new data with representative old samples

**Pros:**
- Best catastrophic forgetting prevention
- Balanced learning
- Maintains long-term patterns

**Cons:**
- Most complex
- Memory overhead

**Implementation:**
```python
# Maintain replay buffer with diverse samples
replay_buffer = ExperienceReplayBuffer(
    capacity=100000,
    sampling_strategy='prioritized'
)

# On update
new_batch = get_new_data()
replay_batch = replay_buffer.sample(batch_size=32)
combined_batch = concatenate(new_batch, replay_batch)

model.update(combined_batch)
replay_buffer.add(new_batch)  # Store for future
```

---

### **Strategy 4: Ensemble with Model Versioning**

**Approach:** Maintain multiple model versions, ensemble predictions

**Pros:**
- Safe fallback
- Graceful degradation
- A/B testing built-in

**Cons:**
- Inference overhead
- Storage requirements

**Implementation:**
```python
# Keep last N models
model_ensemble = [
    load_model("v1.0"),  # Baseline
    load_model("v2.0"),  # Week 1 update
    load_model("v3.0"),  # Week 2 update (current)
]

# Weighted ensemble prediction
predictions = []
weights = [0.2, 0.3, 0.5]  # More weight to recent

for model, weight in zip(model_ensemble, weights):
    pred = model.predict(X)
    predictions.append(pred * weight)

final_prediction = sum(predictions)
```

---

## 🏛️ Architecture Options

### **Option 1: Dual Model System**

```
┌─────────────────┐
│  Production     │ ← Serves predictions (frozen)
│  Model (v1.0)   │
└─────────────────┘
         ↑
         │ Switch when validated
         │
┌─────────────────┐
│  Learning       │ ← Updates on new data
│  Model (v1.1)   │
└─────────────────┘
         ↑
         │ Continuous data
         │
┌─────────────────┐
│  Data Stream    │
└─────────────────┘
```

**Implementation:**
- Production model handles inference
- Learning model trains in background
- Validation gate before promotion
- Safe rollback if issues

---

### **Option 2: Active Learning Pipeline**

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Metrics   │ ──> │ Uncertainty  │ ──> │  High-Value │
│   Stream    │     │  Estimation  │     │   Samples   │
└─────────────┘     └──────────────┘     └─────────────┘
                                                  │
                                                  ↓
                                         ┌─────────────┐
                                         │  Learning   │
                                         │   Buffer    │
                                         └─────────────┘
                                                  │
                                                  ↓
                                         ┌─────────────┐
                                         │   Update    │
                                         │    Model    │
                                         └─────────────┘
```

**Implementation:**
- Only learn from uncertain/novel samples
- Reduces update frequency
- More efficient learning

---

### **Option 3: Federated Learning (Multi-Server)**

```
Server A ──┐
Server B ──┤
Server C ──┼──> Aggregate Updates ──> Global Model
Server D ──┤
Server E ──┘
```

**Implementation:**
- Each server has local model
- Periodic aggregation of updates
- Preserves server-specific patterns
- Distributed learning

---

## 💻 Code Implementation

### **1. Online Learning Configuration**

Add to `config.py`:

```python
# Online Learning Configuration
CONFIG["online_learning"] = {
    "enabled": False,  # Enable online learning
    "strategy": "sliding_window",  # "periodic", "sliding_window", "replay", "ensemble"

    # Update schedule
    "update_frequency_minutes": 60,  # How often to update
    "min_samples_for_update": 1000,  # Minimum samples before update

    # Data management
    "sliding_window_hours": 168,  # 1 week of data
    "replay_buffer_size": 50000,  # Max samples in replay buffer
    "replay_sample_ratio": 0.3,  # 30% old data in each update

    # Training parameters
    "online_learning_rate": 0.001,  # Lower than initial training
    "online_epochs": 5,  # Quick updates
    "online_batch_size": 64,

    # Safety mechanisms
    "enable_validation": True,  # Validate before deployment
    "validation_threshold": 0.05,  # Max 5% performance degradation
    "enable_rollback": True,  # Auto-rollback on failure
    "keep_model_versions": 5,  # Number of versions to keep

    # Drift detection
    "drift_detection_enabled": True,
    "drift_window_size": 1000,  # Samples for drift check
    "drift_threshold": 0.15,  # Trigger retraining if exceeded

    # Active learning
    "active_learning": False,
    "uncertainty_threshold": 0.7,  # Only learn from uncertain samples
}
```

---

### **2. Sliding Window Buffer**

Create `online_learning/sliding_buffer.py`:

```python
import pandas as pd
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

class SlidingWindowBuffer:
    """Maintains a sliding window of recent data for online learning."""

    def __init__(self, window_hours: int = 168, max_samples: Optional[int] = None):
        self.window_hours = window_hours
        self.max_samples = max_samples
        self.buffer = deque(maxlen=max_samples)
        self.timestamps = deque(maxlen=max_samples)

    def add(self, data: pd.DataFrame) -> None:
        """Add new data to buffer."""
        for _, row in data.iterrows():
            self.buffer.append(row)
            self.timestamps.append(row['timestamp'])

        # Remove old data outside window
        self._cleanup_old_data()

    def _cleanup_old_data(self) -> None:
        """Remove data older than window."""
        if len(self.timestamps) == 0:
            return

        cutoff_time = datetime.now() - timedelta(hours=self.window_hours)

        while self.timestamps and self.timestamps[0] < cutoff_time:
            self.buffer.popleft()
            self.timestamps.popleft()

    def get_window(self) -> pd.DataFrame:
        """Get current window as DataFrame."""
        return pd.DataFrame(list(self.buffer))

    def size(self) -> int:
        """Current buffer size."""
        return len(self.buffer)

    def is_ready(self, min_samples: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size() >= min_samples
```

---

### **3. Online Trainer**

Create `online_learning/online_trainer.py`:

```python
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import shutil

from pytorch_forecasting import TemporalFusionTransformer
from config import CONFIG

class OnlineTrainer:
    """Manages online learning updates for TFT model."""

    def __init__(self, base_model_path: str):
        self.base_model_path = Path(base_model_path)
        self.config = CONFIG["online_learning"]
        self.model = self._load_model()
        self.version_history = []

    def _load_model(self) -> TemporalFusionTransformer:
        """Load the base model."""
        from safetensors.torch import load_file

        # Load model architecture and weights
        # ... (implementation details)

        return model

    def update(self, new_data: pd.DataFrame) -> Dict:
        """Perform online learning update."""
        print(f"🔄 Online learning update starting...")

        # Create backup of current model
        backup_path = self._backup_current_model()

        try:
            # Prepare data
            train_dataloader = self._prepare_dataloader(new_data)

            # Set model to training mode
            self.model.train()

            # Lower learning rate for fine-tuning
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["online_learning_rate"]
            )

            # Quick update (few epochs)
            initial_loss = None
            final_loss = None

            for epoch in range(self.config["online_epochs"]):
                epoch_loss = 0
                batch_count = 0

                for batch in train_dataloader:
                    optimizer.zero_grad()
                    loss = self.model.training_step(batch, batch_count)
                    loss.backward()

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=CONFIG["gradient_clip_val"]
                    )

                    optimizer.step()
                    epoch_loss += loss.item()
                    batch_count += 1

                avg_loss = epoch_loss / batch_count
                if initial_loss is None:
                    initial_loss = avg_loss
                final_loss = avg_loss

                print(f"   Epoch {epoch+1}/{self.config['online_epochs']}: Loss = {avg_loss:.4f}")

            # Validate updated model
            if self.config["enable_validation"]:
                is_valid = self._validate_update(backup_path)
                if not is_valid:
                    print("⚠️  Validation failed - rolling back")
                    self._rollback(backup_path)
                    return {"success": False, "reason": "validation_failed"}

            # Save updated model
            self._save_model()

            # Update version history
            self.version_history.append({
                "timestamp": datetime.now(),
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "samples": len(new_data)
            })

            print(f"✅ Online learning update complete")
            print(f"   Loss improvement: {initial_loss:.4f} → {final_loss:.4f}")

            return {
                "success": True,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "improvement": initial_loss - final_loss
            }

        except Exception as e:
            print(f"❌ Online learning failed: {e}")
            self._rollback(backup_path)
            return {"success": False, "reason": str(e)}

    def _backup_current_model(self) -> Path:
        """Create backup of current model."""
        backup_dir = Path(CONFIG["checkpoints_dir"]) / "online_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"backup_{timestamp}"

        shutil.copytree(self.base_model_path, backup_path)

        return backup_path

    def _validate_update(self, backup_path: Path) -> bool:
        """Validate that update didn't degrade performance."""
        # Compare performance on validation set
        # ... (implementation)

        return True  # Placeholder

    def _rollback(self, backup_path: Path) -> None:
        """Rollback to backup model."""
        if backup_path.exists():
            shutil.rmtree(self.base_model_path)
            shutil.copytree(backup_path, self.base_model_path)
            self.model = self._load_model()

    def _save_model(self) -> None:
        """Save updated model."""
        from safetensors.torch import save_file

        # Save model weights
        # ... (implementation)

        pass
```

---

### **4. Drift Detector**

Create `online_learning/drift_detector.py`:

```python
import numpy as np
from collections import deque
from typing import Tuple, Optional

class DriftDetector:
    """Detects concept drift in prediction errors."""

    def __init__(self, window_size: int = 1000, threshold: float = 0.15):
        self.window_size = window_size
        self.threshold = threshold
        self.error_history = deque(maxlen=window_size)
        self.baseline_mean = None
        self.baseline_std = None

    def set_baseline(self, errors: np.ndarray) -> None:
        """Set baseline error statistics."""
        self.baseline_mean = np.mean(errors)
        self.baseline_std = np.std(errors)

    def add_error(self, error: float) -> bool:
        """Add new error and check for drift."""
        self.error_history.append(error)

        if len(self.error_history) < self.window_size:
            return False  # Not enough data yet

        current_mean = np.mean(self.error_history)
        current_std = np.std(self.error_history)

        if self.baseline_mean is None:
            self.set_baseline(np.array(self.error_history))
            return False

        # Check if current stats deviate significantly from baseline
        mean_change = abs(current_mean - self.baseline_mean) / self.baseline_mean
        std_change = abs(current_std - self.baseline_std) / self.baseline_std

        drift_detected = mean_change > self.threshold or std_change > self.threshold

        if drift_detected:
            print(f"⚠️  Concept drift detected!")
            print(f"   Mean change: {mean_change:.2%}")
            print(f"   Std change: {std_change:.2%}")

        return drift_detected

    def get_stats(self) -> Dict:
        """Get current drift statistics."""
        if len(self.error_history) == 0:
            return {}

        return {
            "current_mean": np.mean(self.error_history),
            "current_std": np.std(self.error_history),
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "samples": len(self.error_history)
        }
```

---

### **5. Integration with Inference**

Modify `tft_inference.py` to collect data and trigger updates:

```python
class TFTInferenceWithOnlineLearning(TFTInference):
    """TFT Inference with online learning capability."""

    def __init__(self, model_path: Optional[str] = None):
        super().__init__(model_path)

        if CONFIG["online_learning"]["enabled"]:
            from online_learning.sliding_buffer import SlidingWindowBuffer
            from online_learning.online_trainer import OnlineTrainer
            from online_learning.drift_detector import DriftDetector

            self.buffer = SlidingWindowBuffer(
                window_hours=CONFIG["online_learning"]["sliding_window_hours"]
            )
            self.trainer = OnlineTrainer(self.model_dir)
            self.drift_detector = DriftDetector(
                window_size=CONFIG["online_learning"]["drift_window_size"],
                threshold=CONFIG["online_learning"]["drift_threshold"]
            )
            self.last_update_time = datetime.now()

    def predict(self, data: pd.DataFrame) -> Dict:
        """Make predictions and optionally trigger learning."""
        # Make prediction
        predictions = super().predict(data)

        # If online learning enabled
        if CONFIG["online_learning"]["enabled"]:
            # Store data for learning
            self.buffer.add(data)

            # Check if it's time to update
            minutes_since_update = (datetime.now() - self.last_update_time).seconds / 60

            if minutes_since_update >= CONFIG["online_learning"]["update_frequency_minutes"]:
                if self.buffer.is_ready(CONFIG["online_learning"]["min_samples_for_update"]):
                    # Trigger async update
                    self._trigger_update()

        return predictions

    def record_actual(self, actual_values: pd.DataFrame) -> None:
        """Record actual values for drift detection."""
        if not CONFIG["online_learning"]["enabled"]:
            return

        # Calculate prediction error
        # ... calculate error between predictions and actuals

        # Update drift detector
        drift = self.drift_detector.add_error(error)

        if drift:
            print("🔄 Drift detected - triggering immediate update")
            self._trigger_update()

    def _trigger_update(self) -> None:
        """Trigger online learning update."""
        import threading

        window_data = self.buffer.get_window()

        # Run update in background thread
        def update_task():
            result = self.trainer.update(window_data)
            if result["success"]:
                # Reload model with updated weights
                self.model = self.trainer.model

        thread = threading.Thread(target=update_task, daemon=True)
        thread.start()

        self.last_update_time = datetime.now()
```

---

## ⚙️ Configuration Examples

### **Conservative Setup (Recommended for Production)**

```python
CONFIG["online_learning"] = {
    "enabled": True,
    "strategy": "sliding_window",
    "update_frequency_minutes": 360,  # Every 6 hours
    "sliding_window_hours": 336,  # 2 weeks
    "online_learning_rate": 0.0005,  # Very conservative
    "online_epochs": 3,  # Quick updates
    "enable_validation": True,
    "validation_threshold": 0.03,  # Max 3% degradation
    "enable_rollback": True,
    "drift_detection_enabled": True,
}
```

### **Aggressive Setup (For Fast-Changing Environments)**

```python
CONFIG["online_learning"] = {
    "enabled": True,
    "strategy": "replay",
    "update_frequency_minutes": 60,  # Every hour
    "sliding_window_hours": 168,  # 1 week
    "replay_buffer_size": 100000,
    "online_learning_rate": 0.001,
    "online_epochs": 5,
    "enable_validation": True,
    "drift_detection_enabled": True,
}
```

---

## 📊 Monitoring & Safety

### **Key Metrics to Track:**

1. **Model Performance:**
   - Validation loss trend
   - Prediction error (MAE, RMSE)
   - Quantile calibration

2. **Learning Health:**
   - Update frequency
   - Loss improvement per update
   - Rollback frequency

3. **Drift Metrics:**
   - Error distribution shift
   - Feature distribution changes
   - Prediction confidence trends

4. **System Health:**
   - Update latency
   - Memory usage
   - CPU/GPU utilization during updates

### **Safety Mechanisms:**

1. **Validation Gate**
   - Always validate on hold-out set
   - Block deployment if degraded

2. **Auto-Rollback**
   - Automatic reversion on failure
   - Keep N previous versions

3. **Circuit Breaker**
   - Stop updates if too many failures
   - Alert human operators

4. **Shadow Mode**
   - Run new model in parallel
   - Compare before switching

---

## 🎓 Best Practices

1. **Start Conservative**
   - Low learning rate
   - Infrequent updates
   - Strong validation

2. **Monitor Closely**
   - Dashboard for online learning metrics
   - Alerts on anomalies
   - Audit trail of updates

3. **Use Experience Replay**
   - Prevents catastrophic forgetting
   - Maintains diverse patterns

4. **Implement Rollback**
   - Always keep working fallback
   - Test rollback procedure

5. **Gradual Rollout**
   - Test on subset of servers first
   - Canary deployments

---

## 📈 Expected Results

### **Benefits:**
- **Adaptation Time:** < 24 hours (vs weeks for full retraining)
- **Resource Usage:** 10-20% overhead (vs 100% for batch retraining)
- **Accuracy Improvement:** 5-15% in dynamic environments
- **False Positives:** 20-30% reduction over time

### **Challenges:**
- Initial complexity increase
- Monitoring overhead
- Potential for instability if misconfigured

---

## ✅ Recommendation

**For your TFT monitoring system, I recommend:**

**Phase 1 (Immediate):**
- Start with **periodic retraining** (every 24 hours)
- Collect actual vs predicted metrics
- Build drift detection capabilities

**Phase 2 (1-2 months):**
- Implement **sliding window** approach
- Add validation gates
- Enable rollback mechanism

**Phase 3 (3+ months):**
- Add **experience replay** if needed
- Implement active learning
- Optimize update frequency

This provides a gradual, safe path to full online learning capability.

---

**Status:** Design Complete
**Implementation Effort:** Medium (2-4 weeks)
**Risk:** Low (with proper validation)
**ROI:** High (in dynamic environments)
