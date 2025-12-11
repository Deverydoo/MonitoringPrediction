# Tachyon Argus - Training Guide

Complete guide to generating training data and training TFT prediction models with **fleet-level features** for cascading failure detection.

## Overview

The Tachyon Argus system uses a Temporal Fusion Transformer (TFT) model for server failure prediction. Training involves:

1. **Generate synthetic data** or collect real metrics
2. **Train the model** using standard or streaming mode (now with fleet features)
3. **Deploy the model** to the inference daemon

### New in v2.1: Fleet-Level Features

The training pipeline now automatically computes **18 fleet-level features** that enable the model to:
- Learn cross-server correlations
- Detect environment-wide issues (cascading failures)
- Flag "green" servers that are part of a fleet-wide problem

## Prerequisites

- Python 3.10+ with conda environment activated
- CUDA GPU recommended (training is ~10x faster)
- 8GB+ RAM (16GB+ for large datasets)

```bash
cd Argus
conda activate tachyon
```

---

## Step 1: Generate Training Data

### Basic Generation

```bash
# Generate 30 days of data for 20 servers
python src/training/main.py generate --servers 20 --hours 720

# Generate 2 weeks of data for 45 servers
python src/training/main.py generate --servers 45 --hours 336

# Custom output directory
python src/training/main.py generate --servers 20 --hours 720 --output ./my_training_data/
```

### Data Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--servers` | 20 | Number of servers to simulate |
| `--hours` | 720 | Hours of data (720 = 30 days) |
| `--output` | `./training/` | Output directory |

### Output Format

Data is saved as time-partitioned Parquet files for memory-efficient streaming:

```
training/
├── server_metrics_partitioned/
│   ├── chunk_20251201_00.parquet   # 2-hour chunk
│   ├── chunk_20251201_02.parquet
│   ├── chunk_20251201_04.parquet
│   ├── ...
│   └── chunk_manifest.json         # Chunk metadata
└── metrics_metadata.json           # Dataset info
```

### Using Real Data

If you have real metrics data, format it as Parquet with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | ISO 8601 format |
| `server_name` | string | Unique server ID |
| `status` | string | Server state |
| `cpu_user_pct` | float | User CPU % (0-100) |
| `cpu_sys_pct` | float | System CPU % (0-100) |
| `cpu_iowait_pct` | float | I/O wait % (0-100) |
| `cpu_idle_pct` | float | Idle CPU % (0-100) |
| `java_cpu_pct` | float | Java process CPU % |
| `mem_used_pct` | float | Memory usage % |
| `swap_used_pct` | float | Swap usage % |
| `disk_usage_pct` | float | Disk usage % |
| `net_in_mb_s` | float | Network in (MB/s) |
| `net_out_mb_s` | float | Network out (MB/s) |
| `back_close_wait` | int | Backend CLOSE_WAIT conns |
| `front_close_wait` | int | Frontend CLOSE_WAIT conns |
| `load_average` | float | System load average |
| `uptime_days` | int | Days since reboot |

See [METRICS_FEED_GUIDE.md](METRICS_FEED_GUIDE.md) for complete schema details.

---

## Step 2: Train the Model

### Standard Training (Small Datasets)

For datasets that fit in memory (~30 days, 50 servers):

```bash
# Train with default settings (20 epochs)
python src/training/main.py train

# Train for specific number of epochs
python src/training/main.py train --epochs 10

# Train on specific dataset
python src/training/main.py train --dataset ./my_data/
```

### Streaming Training (Large Datasets)

For large datasets (weeks/months of data, 100+ servers):

```bash
# Memory-efficient streaming mode
python src/training/main.py train --streaming

# Streaming with custom epochs
python src/training/main.py train --streaming --epochs 5
```

**Streaming mode benefits:**
- Processes data in 2-hour chunks
- Memory usage stays bounded (~2-4 GB)
- Supports datasets of any size
- Includes checkpoint support for resume
- **Computes fleet features per chunk for cascading detection**

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 20 | Number of training epochs |
| `--dataset` | `./training/` | Dataset directory |
| `--streaming` | false | Use streaming mode |
| `--incremental` | true | Continue from existing model |
| `--fresh` | false | Start fresh (ignore existing model) |

### Incremental Training

By default, training continues from the last model:

```bash
# Add 5 more epochs to existing model
python src/training/main.py train --epochs 5 --incremental

# Start fresh training (ignore existing model)
python src/training/main.py train --epochs 20 --fresh
```

---

## Fleet-Level Feature Engineering

### What Gets Computed

During training, the pipeline automatically computes **18 fleet-level features** for each timestamp:

```
[FLEET] Computing fleet-level features for cascading failure detection...
[FLEET] Added 18 fleet-level features
```

### Feature Categories

| Category | Features | Description |
|----------|----------|-------------|
| **Fleet Averages** | `fleet_cpu_user_pct_mean`, `fleet_mem_used_pct_mean`, etc. | Average across all servers at each timestamp |
| **Fleet Variability** | `fleet_cpu_user_pct_std`, `fleet_mem_used_pct_std`, etc. | Standard deviation - detects synchronized changes |
| **Fleet Peaks** | `fleet_cpu_user_pct_max`, `fleet_mem_used_pct_max`, etc. | Maximum value across fleet |
| **Stress Indicators** | `fleet_pct_high_cpu`, `fleet_pct_high_mem`, `fleet_pct_high_iowait` | % of servers above warning thresholds |
| **Server Deviation** | `cpu_vs_fleet`, `mem_vs_fleet`, `iowait_vs_fleet` | How each server differs from fleet average |

### How It Enables Cascading Detection

```
Example scenario:
┌────────────────────────────────────────────────────────────────┐
│ Time 10:00: fleet_pct_high_cpu = 5% (normal)                   │
│ Time 10:15: fleet_pct_high_cpu = 15% (rising)                  │
│ Time 10:30: fleet_pct_high_cpu = 40% (CASCADE!)                │
│                                                                 │
│ The model learns: "When fleet_pct_high_cpu rises quickly,      │
│ ALL servers are at risk, even those currently 'green'"         │
└────────────────────────────────────────────────────────────────┘
```

### Training Output with Fleet Features

```
[PREP] Preparing data for TFT training...
[INFO] Original columns: ['timestamp', 'server_name', 'cpu_user_pct', ...]
[FLEET] Computing fleet-level features for cascading failure detection...
[FLEET] Added 18 fleet-level features
[FLEET] Features: ['fleet_cpu_user_pct_mean', 'fleet_cpu_user_pct_std', ...]
[TRANSFER] Profile feature enabled - model will learn per-profile patterns
[FLEET] Added 18 fleet features to model input
[INFO] Multi-target mode: ['cpu_user_pct', 'cpu_iowait_pct', 'mem_used_pct', 'swap_used_pct', 'load_average']
```

---

## Multi-Target Prediction

### Enabled by Default

The model now predicts **5 metrics simultaneously**:

| Target | Description |
|--------|-------------|
| `cpu_user_pct` | User CPU utilization |
| `cpu_iowait_pct` | I/O wait percentage |
| `mem_used_pct` | Memory utilization |
| `swap_used_pct` | Swap utilization |
| `load_average` | System load |

### Benefits

- **Holistic view**: Single prediction captures all key health metrics
- **Correlated predictions**: Model learns relationships between metrics
- **Better risk scoring**: Multiple signals improve accuracy

---

## Step 3: Monitor Training Progress

### Check Status

```bash
python src/training/main.py status
```

Shows:
- Available datasets (Parquet files)
- Trained models
- Training configuration

### Training Output (Standard Mode)

During standard training you'll see:

```
[EPOCH 1/20] Starting epoch...
============================================================
[EPOCH 1/20] Completed in 45.2s
  Train Loss: 0.0234
  Val Loss: 0.0198 (BEST)
  LR: 0.001
  ETA: 15.1 minutes remaining
============================================================
```

### Training Output (Streaming Mode)

Streaming mode shows comprehensive progress tracking with ETA:

```
[STREAM] Found 60 time chunks (2 hours each)
[STREAM] Will process 60 chunks per epoch
[STREAM] Total epochs: 3
[STREAM] Total chunks to process: 180
[STREAM] Checkpoint every 5 chunks
======================================================================

[EPOCH 1/3] Starting streaming epoch...
============================================================

[CHUNK 1/60] Loading 20251201_00...
[PROGRESS] Overall: 0/180 (0.0%) | ETA: calculating...
...training output...
[CHUNK] 20251201_00 done in 38.2s | Loss: 0.0342
[PROGRESS] 1/180 (0.6%) | Avg: 38.2s/chunk | ETA: 1h 54m

[CHUNK 2/60] Loading 20251201_02...
[PROGRESS] Overall: 1/180 (0.6%) | ETA: 1h 54m
...training output...
[CHUNK] 20251201_02 done in 41.5s | Loss: 0.0298
[PROGRESS] 2/180 (1.1%) | Avg: 39.9s/chunk | ETA: 1h 58m
```

### Epoch Summary

After each epoch completes:

```
============================================================
[EPOCH 1/3] COMPLETE [BEST]
============================================================
   Epoch time:  38.2 min
   Avg loss:    0.0245
   Best loss:   0.0245
   Progress:    60/180 chunks (33.3%)
   ETA:         1h 16m (120 chunks @ 38.2s avg)
============================================================
```

### Training Complete

```
======================================================================
[OK] STREAMING TRAINING COMPLETE
======================================================================
   Total time:       115.3 minutes (1.92 hours)
   Chunks processed: 180
   Epochs completed: 3
   Best loss:        0.0198
   Avg chunk time:   38.4s
======================================================================
```

### TensorBoard Logs

Training logs are saved for visualization:

```bash
tensorboard --logdir=lightning_logs/
```

---

## Checkpoint Support & Resume (Streaming Mode)

Streaming training automatically saves checkpoints every 5 chunks. If training is interrupted, simply restart with the same command to resume exactly where you left off.

### Automatic Resume

```bash
# Just restart - it will auto-resume from checkpoint
python src/training/main.py train --streaming
```

### Resume Output

When resuming from a checkpoint, you'll see detailed progress information:

```
============================================================
[CHECKPOINT] RESUMING FROM SAVED CHECKPOINT
============================================================
[CHECKPOINT] Saved at: 2025-01-15T10:30:00
[CHECKPOINT] Epoch 2, Chunk 45
[CHECKPOINT] Best loss so far: 0.0198
[CHECKPOINT] Progress: 165/360 chunks (45.8%)
[CHECKPOINT] Avg chunk time: 42.3s (from 50 samples)
============================================================

[RESUME] Restoring from checkpoint...
[RESUME] Continuing from epoch 2, chunk 46
[RESUME] Best loss: 0.0198
[RESUME] Overall progress: 165/360 chunks (45.8%)
[RESUME] Estimated time remaining: 2h 17m (195 chunks @ 42.3s avg)
```

### What's Saved in Checkpoints

| Data | Purpose |
|------|---------|
| Model weights | Full model state for exact resume |
| Epoch/chunk position | Resume from exact location |
| Best loss | Continue tracking improvements |
| Chunk order | Maintain shuffle consistency within epoch |
| Chunk timing history | Accurate ETA calculations |
| Overall progress | Total chunks completed across all epochs |

### Checkpoint Location

Checkpoint files are stored in `models/streaming_checkpoint.pt` and automatically cleared after successful completion.

---

## Model Output

Trained models are saved to `models/tft_model_YYYYMMDD_HHMMSS/`:

```
models/tft_model_20251215_143022/
├── model.safetensors          # Model weights
├── config.json                # Model configuration
├── dataset_parameters.pkl     # Data encoders (includes fleet feature encoders)
├── server_mapping.json        # Server ID hash mapping
└── training_info.json         # Training metadata
```

**Important:** The `dataset_parameters.pkl` file is required for inference. It contains the data encoders (including fleet features) and must be kept with the model.

---

## Training Modes Comparison

| Mode | Memory | Speed | Use Case |
|------|--------|-------|----------|
| Standard | High (loads all data) | Fast | <30 days, <50 servers |
| Streaming | Low (2-hour chunks) | Slower | Large datasets, limited RAM |
| Incremental | Varies | Fast | Adding epochs to existing model |

---

## Best Practices

### Data Quality
- **Minimum data**: 7 days for basic patterns, 30 days recommended
- **Server count**: 10+ servers for good generalization (more servers = better fleet features)
- **Anomalies**: Include some failure scenarios for better prediction
- **Cascading events**: Include data with correlated server issues for cascade detection

### Training Duration
- **Quick test**: 5 epochs (~10 minutes)
- **Standard**: 20 epochs (~1 hour)
- **Production**: 30-50 epochs (~2-4 hours)

### GPU vs CPU
- GPU (RTX 3080+): ~2 minutes per epoch
- CPU: ~20-30 minutes per epoch

### Memory Management
- Standard mode: ~8GB RAM for 30 days × 50 servers
- Streaming mode: ~2-4GB RAM regardless of dataset size

### Cascading Failure Detection
- Train with **all servers together** (not individual models)
- Fleet features require multiple servers at each timestamp
- Include periods with fleet-wide stress for better learning

---

## Troubleshooting

### "CUDA out of memory"
- Use streaming mode: `--streaming`
- Reduce batch size in config
- Use a smaller dataset for initial tests

### "No data found"
- Check dataset path exists
- Verify Parquet files are present
- Run `python src/training/main.py status`

### Training stalls
- Check GPU temperature (throttling?)
- Monitor memory usage
- Try streaming mode for large datasets

### Poor model performance
- Increase training epochs
- Ensure data quality (no gaps, realistic values)
- Include more failure scenarios in training data

### Fleet features not appearing
- Ensure data has multiple servers per timestamp
- Check that `timestamp` column is properly formatted
- Verify fleet feature columns exist after preparation

---

## Advanced: Direct Trainer Usage

For more control, use the trainer directly:

```python
from training.tft_trainer import TFTTrainer, train_model

# Quick training
model_path = train_model(
    dataset_path="./training/",
    epochs=10,
    streaming=True
)

# Or with full control
trainer = TFTTrainer(epochs=20)
trainer.train_streaming("./training/")
```

---

## Continuous Learning

After initial training, the system supports **automatic retraining**:

### Drift-Triggered Retraining

The inference daemon monitors model performance and automatically triggers retraining when drift is detected:

```
┌────────────────────────────────────────────────────────────────┐
│  1. Inference daemon tracks prediction accuracy                │
│  2. Drift monitor calculates: PER, DSS, FDS, Anomaly Rate      │
│  3. If any metric exceeds threshold → trigger retraining       │
│  4. 5-epoch incremental training (non-blocking)               │
│  5. Hot-reload new model without restart                       │
└────────────────────────────────────────────────────────────────┘
```

### Manual Retraining Trigger

```bash
# Via API
curl -X POST http://localhost:8000/admin/trigger-training \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"epochs": 5, "incremental": true}'

# Check status
curl -H "X-API-Key: your-key" http://localhost:8000/admin/training-status
```

### Scheduled Retraining

Use the provided script for weekly retraining:

```bash
# Add to crontab
0 2 * * 0 /path/to/Argus/bin/weekly_retrain.sh
```

---

## Next Steps

After training:
1. **Deploy model**: Copy to production or use hot-reload API
2. **Start inference**: `python src/daemons/tft_inference_daemon.py`
3. **Monitor cascade detection**: Check `/cascade/status` endpoint
4. **Track drift**: Monitor `/drift/status` for model health
5. **Auto-retraining**: Enabled by default when drift detected

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for production deployment.
