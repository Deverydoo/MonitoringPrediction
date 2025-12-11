# Tachyon Argus - Training Guide

Complete guide to generating training data and training TFT prediction models.

## Overview

The Tachyon Argus system uses a Temporal Fusion Transformer (TFT) model for server failure prediction. Training involves:

1. **Generate synthetic data** or collect real metrics
2. **Train the model** using standard or streaming mode
3. **Deploy the model** to the inference daemon

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

## Step 3: Monitor Training Progress

### Check Status

```bash
python src/training/main.py status
```

Shows:
- Available datasets (Parquet files)
- Trained models
- Training configuration

### Training Output

During training you'll see:

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

### TensorBoard Logs

Training logs are saved for visualization:

```bash
tensorboard --logdir=lightning_logs/
```

---

## Checkpoint Support (Streaming Mode)

Streaming training automatically saves checkpoints every 5 chunks (~20 minutes of training). If training is interrupted:

```bash
# Just restart - it will auto-resume from checkpoint
python src/training/main.py train --streaming
```

Output when resuming:
```
[CHECKPOINT] Found checkpoint from 2025-01-15T10:30:00
[CHECKPOINT] Epoch 2, Chunk 45
[CHECKPOINT] Best loss so far: 0.0198
[STREAMING] Resuming from epoch 2, chunk 45...
```

Checkpoint files are stored in `models/streaming_checkpoint.pt` and automatically cleared after successful completion.

---

## Model Output

Trained models are saved to `models/tft_model_YYYYMMDD_HHMMSS/`:

```
models/tft_model_20251215_143022/
├── model.safetensors          # Model weights
├── config.json                # Model configuration
├── dataset_parameters.pkl     # Data encoders (CRITICAL)
├── server_mapping.json        # Server ID hash mapping
└── training_info.json         # Training metadata
```

**Important:** The `dataset_parameters.pkl` file is required for inference. It contains the data encoders and must be kept with the model.

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
- **Server count**: 10+ servers for good generalization
- **Anomalies**: Include some failure scenarios for better prediction

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

## Next Steps

After training:
1. **Deploy model**: Copy to production or use hot-reload API
2. **Start inference**: `python src/daemons/tft_inference_daemon.py`
3. **Monitor performance**: Track prediction accuracy in dashboard
4. **Retrain periodically**: Weekly or when drift is detected

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for production deployment.
