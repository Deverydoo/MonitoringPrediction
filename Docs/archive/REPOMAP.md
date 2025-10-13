# Repository Map - MonitoringPrediction

**Last Updated:** 2025-10-08
**Project:** TFT-Based Server Monitoring & Prediction System

---

## 📁 Project Structure

```
MonitoringPrediction/
├── 🔧 Core Configuration
│   └── config.py                      # Central configuration for TFT system
│
├── 🎯 Main Entry Points
│   ├── main.py                        # CLI interface for generate/train/predict
│   └── run_demo.py                    # Convenient demo runner (data + dashboard)
│
├── 📊 Data Generation
│   ├── metrics_generator.py           # Production-level server metrics generator
│   └── demo_data_generator.py         # Reproducible demo dataset with incident pattern
│
├── 🤖 Model Training
│   └── tft_trainer.py                 # TFT trainer with parquet support, progress tracking
│
├── 🔮 Model Inference
│   └── tft_inference.py               # TFT inference engine (Parquet/CSV support)
│
├── 📈 Visualization & Dashboards
│   ├── tft_dashboard_refactored.py    # Production dashboard (file-based, no random data)
│   └── tft_dashboard.py               # Legacy dashboard
│
├── 🛠️ Utilities
│   └── common_utils.py                # Shared utility functions
│
├── 🧪 Testing & Validation
│   ├── test_phase1_improvements.py    # Phase 1: Data loading improvements
│   ├── test_phase2_improvements.py    # Phase 2: Training enhancements
│   └── test_phase3_improvements.py    # Phase 3: Advanced optimizations
│
├── 📚 Documentation
│   ├── README.md                      # Main project documentation
│   ├── DEMO_README.md                 # Demo setup and usage guide
│   ├── SETUP_DEMO.md                  # Demo setup instructions
│   ├── CHANGELOG.md                   # Version history and changes
│   │
│   ├── Implementation Docs
│   │   ├── IMPLEMENTATION_COMPLETE.md     # Overall implementation summary
│   │   ├── ALL_PHASES_COMPLETE.md         # All development phases summary
│   │   ├── PHASE1_COMPLETE.md             # Phase 1: Data loading
│   │   ├── PHASE2_COMPLETE.md             # Phase 2: Training enhancements
│   │   ├── PHASE2_SUMMARY.md              # Phase 2 summary
│   │   ├── PHASE3_COMPLETE.md             # Phase 3: Advanced optimizations
│   │   └── PHASE3_SUMMARY.md              # Phase 3 summary
│   │
│   ├── Technical Analysis
│   │   ├── TRAINING_IMPROVEMENTS_ANALYSIS.md  # Training analysis
│   │   ├── TRAINING_QUICK_WINS.md             # Quick training improvements
│   │   ├── DATA_LOADING_IMPROVEMENTS.md       # Data loading enhancements
│   │   ├── CHANGES_SUMMARY.md                 # Overall changes summary
│   │   ├── PARQUET_UPDATE_SUMMARY.md          # Parquet integration summary
│   │   ├── PARQUET_DEFAULT_SUMMARY.md         # Parquet as default format
│   │   └── DEFAULT_PARQUET_UPDATE.md          # Default parquet update details
│
├── 📂 Data Directories
│   ├── training/                      # Training datasets (CSV/Parquet)
│   ├── models/                        # Trained model artifacts
│   ├── checkpoints/                   # Training checkpoints
│   ├── logs/                          # Training and system logs
│   ├── lightning_logs/                # Lightning framework logs
│   └── data_config/                   # Data configuration files
│
├── 🔍 Notebooks
│   └── _StartHere.ipynb               # Getting started notebook
│
└── ⚙️ Configuration
    ├── .gitignore                     # Git ignore patterns
    ├── .gitattributes                 # Git attributes
    └── tft_config_adjusted.json       # Adjusted TFT configuration
```

---

## 🎯 Core Scripts Reference

### Main Entry Points

#### [main.py](main.py)
**Purpose:** Primary CLI interface for the TFT monitoring system
**Key Features:**
- Setup and environment validation
- Dataset generation orchestration
- Model training initiation
- Inference execution
- System status checking

**Usage:**
```bash
python main.py setup          # Validate environment
python main.py generate       # Generate training data
python main.py train          # Train TFT model
python main.py predict        # Run predictions
python main.py status         # Check system status
```

---

#### [run_demo.py](run_demo.py)
**Purpose:** Convenient one-command demo launcher
**Key Features:**
- Generates demo dataset automatically
- Launches interactive dashboard
- Handles data regeneration
- Configurable fleet size and duration

**Usage:**
```bash
python run_demo.py                                    # Run with defaults
python run_demo.py --duration 10 --fleet-size 20     # Custom settings
python run_demo.py --regenerate                       # Force new data
```

**Arguments:**
- `--duration`: Demo duration in minutes (default: 5)
- `--fleet-size`: Number of servers (default: 10)
- `--seed`: Random seed (default: 42)
- `--refresh`: Dashboard refresh interval in seconds (default: 5)
- `--regenerate`: Force regeneration of demo data

---

### Data Generation

#### [metrics_generator.py](metrics_generator.py)
**Purpose:** Production-level server metrics data generator
**Key Features:**
- Fleet-wide 5-second polling simulation
- Realistic server profiles (production, staging, compute, service, container)
- Operational state modeling (idle, healthy, heavy_load, critical, maintenance)
- Time-based patterns (diurnal, weekly cycles)
- Problem children modeling (servers with chronic issues)
- Dual output: CSV + Parquet (Parquet prioritized)
- Streaming-friendly sequential format

**Server Profiles:**
- `PRODUCTION`: High-traffic production servers
- `STAGING`: Pre-production testing environments
- `COMPUTE`: Batch processing and compute-heavy tasks
- `SERVICE`: API and service endpoints
- `CONTAINER`: Containerized workloads

**Server States:**
- `IDLE`: Low activity baseline
- `HEALTHY`: Normal operational state
- `MORNING_SPIKE`: Peak usage periods
- `HEAVY_LOAD`: High utilization
- `CRITICAL_ISSUE`: System problems
- `MAINTENANCE`: Scheduled maintenance
- `RECOVERY`: Post-incident recovery
- `OFFLINE`: Server unavailable

**Usage:**
```bash
python metrics_generator.py --servers 15 --hours 720 --output ./training/
```

**Output Format:**
```
timestamp,server_name,cpu_percent,memory_percent,disk_percent,load_average,
java_heap_usage,network_errors,anomaly_score,hour,day_of_week,day_of_month,
month,quarter,is_weekend,is_business_hours,status,timeframe,service_type,
datacenter,environment
```

---

#### [demo_data_generator.py](demo_data_generator.py)
**Purpose:** Reproducible demo dataset with predictable incident pattern
**Key Features:**
- Fixed 5-minute duration with incident timeline
- Reproducible with seed control
- Predictable incident pattern for demos
- 4-phase incident timeline

**Incident Timeline:**
- `0:00-1:30`: Stable baseline (healthy operations)
- `1:30-2:30`: Gradual escalation (warning signs appear)
- `2:30-3:30`: Incident peak (critical state)
- `3:30-5:00`: Recovery to stable (gradual normalization)

**Usage:**
```bash
python demo_data_generator.py --duration 5 --fleet-size 10 --seed 42
```

---

### Model Training

#### [tft_trainer.py](tft_trainer.py)
**Purpose:** Fixed TFT model trainer with enhanced features
**Key Features:**
- Prioritizes Parquet over CSV/JSON for better performance
- Per-server model training capability
- Enhanced progress reporting with ETA
- Learning rate monitoring
- Automatic checkpoint management
- TensorBoard integration
- Safetensors model serialization

**Training Phases:**
- **Phase 1**: Data loading improvements (Parquet support)
- **Phase 2**: Training enhancements (LR monitoring, progress tracking)
- **Phase 3**: Advanced optimizations (mixed precision, gradient accumulation)

**Usage:**
```bash
# Train single model
python tft_trainer.py --training-dir ./training/ --output-dir ./models/

# Train per-server models
python tft_trainer.py --per-server --training-dir ./training/ --output-dir ./models/

# Custom epochs and batch size
python tft_trainer.py --epochs 30 --batch-size 64
```

**Training Callbacks:**
- `TrainingProgressCallback`: Enhanced progress reporting with ETA
- `EarlyStopping`: Prevents overfitting (patience: 8 epochs)
- `ModelCheckpoint`: Saves best models
- `LearningRateMonitor`: Tracks learning rate changes

---

### Model Inference

#### [tft_inference.py](tft_inference.py)
**Purpose:** Primary TFT inference engine
**Key Features:**
- Automatic model discovery (finds latest trained model)
- Safetensors model loading
- Batch prediction support
- GPU acceleration support
- Config-driven inference

**Usage:**
```bash
python tft_inference.py --model-path ./models/tft_model_20251008/ --data ./data/recent.parquet
```

---

### Visualization

#### [tft_dashboard_refactored.py](tft_dashboard_refactored.py)
**Purpose:** Production-level file-based monitoring dashboard
**Key Features:**
- NO random data generation - reads from actual data files
- Supports CSV and Parquet formats
- Auto-detects data format from file extension
- Real-time metric visualization
- Rolling time windows
- Fleet-wide and per-server views
- Anomaly highlighting
- Configurable refresh intervals
- Optional plot saving

**Dashboard Views:**
1. **Fleet Overview**: Aggregate metrics across all servers
2. **Critical Servers**: Servers with anomaly scores above threshold
3. **Per-Server Trends**: Individual server metric timelines
4. **Anomaly Detection**: Real-time anomaly scoring

**Usage:**
```bash
# Run with demo data
python tft_dashboard_refactored.py --data ./demo_data/demo_dataset.parquet

# Run with production data
python tft_dashboard_refactored.py --data ./training/server_metrics.parquet --refresh 10

# Save plots
python tft_dashboard_refactored.py --data ./data/metrics.csv --save-plots
```

---

### Configuration

#### [config.py](config.py)
**Purpose:** Centralized configuration for entire TFT system
**Key Sections:**

**Model Architecture:**
```python
hidden_size: 32
attention_heads: 8
dropout: 0.15
output_size: 7  # 7 target metrics
```

**Training Parameters:**
```python
epochs: 20
batch_size: 32
learning_rate: 0.01
early_stopping_patience: 8
```

**Time Series Configuration:**
```python
prediction_horizon: 96      # 8 hours (96 * 5min)
context_length: 288         # 24 hours lookback
min_encoder_length: 144     # 12 hours minimum
```

**Target Metrics:**
- `cpu_percent`: CPU utilization
- `memory_percent`: Memory utilization
- `disk_percent`: Disk utilization
- `load_average`: System load average
- `java_heap_usage`: JVM heap utilization
- `network_errors`: Network error count
- `anomaly_score`: Composite anomaly indicator

**Alert Thresholds:**
- Warning levels: 75-85% depending on metric
- Critical levels: 90-95% depending on metric

**Phase-specific Features:**
- Phase 2: Auto LR find, enhanced logging
- Phase 3: Mixed precision, gradient accumulation, multi-target support

---

### Testing & Validation

#### [test_phase1_improvements.py](test_phase1_improvements.py)
**Purpose:** Validate Phase 1 data loading improvements
**Tests:**
- Parquet file generation and loading
- Data format consistency
- Performance benchmarks
- Schema validation

---

#### [test_phase2_improvements.py](test_phase2_improvements.py)
**Purpose:** Validate Phase 2 training enhancements
**Tests:**
- Learning rate monitoring
- Progress callback functionality
- Training metrics logging
- Checkpoint creation

---

#### [test_phase3_improvements.py](test_phase3_improvements.py)
**Purpose:** Validate Phase 3 advanced optimizations
**Tests:**
- Mixed precision training
- Gradient accumulation
- Multi-target prediction support
- Performance improvements

---

## 🔄 Typical Workflows

### 1. Quick Demo (Recommended for first-time users)
```bash
# One command to run everything
python run_demo.py
```

### 2. Production Training Pipeline
```bash
# Step 1: Generate production data
python metrics_generator.py --servers 15 --hours 720 --output ./training/

# Step 2: Train model
python tft_trainer.py --training-dir ./training/ --output-dir ./models/ --epochs 20

# Step 3: Run predictions
python tft_inference.py --model-path ./models/tft_model_latest/ --data ./data/current.parquet
```

### 3. Per-Server Model Training
```bash
# Generate data
python metrics_generator.py --servers 20 --hours 1440 --output ./training/

# Train individual models per server
python tft_trainer.py --per-server --training-dir ./training/ --output-dir ./models/per_server/

# This creates separate models for each server for specialized predictions
```

### 4. Dashboard Monitoring
```bash
# With live production data
python tft_dashboard_refactored.py --data ./training/server_metrics.parquet --refresh 5

# With demo data
python tft_dashboard_refactored.py --data ./demo_data/demo_dataset.parquet
```

---

## 📦 Data Formats

### Parquet Files (Preferred)
- **Location:** `./training/*.parquet`, `./demo_data/*.parquet`
- **Advantages:** Faster loading, smaller size, columnar format
- **Default:** All new data generation uses Parquet

### CSV Files (Legacy)
- **Location:** `./training/*.csv`
- **Status:** Still supported for backward compatibility
- **Performance:** Slower than Parquet for large datasets

### Model Artifacts
- **Format:** Safetensors (`.safetensors`)
- **Location:** `./models/tft_model_*/`
- **Contents:**
  - `model.safetensors`: Model weights
  - `config.json`: Model configuration
  - `training_info.json`: Training metadata

---

## 🔑 Key Dependencies

```
python >= 3.8
torch >= 2.0.0
lightning >= 2.0.0
pytorch-forecasting == 1.0.0
pandas >= 1.5.0
numpy >= 1.23.0
safetensors >= 0.3.0
pyarrow >= 10.0.0  (for Parquet support)
```

---

## 🚀 Recent Improvements

### Phase 1: Data Loading (COMPLETE)
- ✅ Parquet format integration
- ✅ Faster data loading (3-5x speedup)
- ✅ Reduced memory footprint
- ✅ Backward compatibility with CSV

### Phase 2: Training Enhancements (COMPLETE)
- ✅ Learning rate monitoring
- ✅ Enhanced progress tracking with ETA
- ✅ Improved logging (every N steps)
- ✅ Auto LR finder support

### Phase 3: Advanced Optimizations (COMPLETE)
- ✅ Mixed precision training (FP16/BF16)
- ✅ Gradient accumulation
- ✅ Multi-target prediction support
- ✅ Per-server model training

---

## 📊 Model Architecture

**Temporal Fusion Transformer (TFT)**
- Variable selection networks for feature importance
- Gated Residual Networks (GRN) for information processing
- Multi-head attention for temporal relationships
- Quantile regression for probabilistic forecasting

**Input Features:**
- **Time-varying known**: hour, day_of_week, day_of_month, month, quarter, is_weekend, is_business_hours
- **Time-varying unknown**: cpu_percent, memory_percent, disk_percent, load_average, java_heap_usage, network_errors, anomaly_score
- **Static categorical**: server_name, status, timeframe, service_type, datacenter, environment

**Output:**
- Multi-horizon forecasts (96 steps = 8 hours)
- Quantile predictions (p10, p50, p90) for uncertainty quantification

---

## 🐛 Troubleshooting

### Common Issues

**1. Import Error: `pytorch_lightning` not found**
- Solution: Use `lightning` package instead (Lightning 2.0+)
- Fixed in: `tft_trainer.py`

**2. Data Loading Slow**
- Solution: Use Parquet format instead of CSV
- Command: `python metrics_generator.py --output ./training/` (generates Parquet by default)

**3. Out of Memory During Training**
- Solution: Reduce batch size in `config.py`
- Alternative: Enable gradient accumulation (Phase 3 feature)

**4. Model Not Found During Inference**
- Solution: Check `./models/` directory for trained models
- Ensure model path points to directory containing `model.safetensors`

---

## 📝 Git Status

**Current Branch:** main
**Modified Files:**
- `config.py` - Enhanced configuration
- `metrics_generator.py` - Production improvements
- `tft_trainer.py` - Per-server training support

**Untracked Documentation:**
- Phase completion summaries
- Implementation guides
- Training analysis reports

---

## 🎓 Learning Resources

- **Start Here:** [_StartHere.ipynb](_StartHere.ipynb)
- **Demo Guide:** [DEMO_README.md](DEMO_README.md)
- **Setup Instructions:** [SETUP_DEMO.md](SETUP_DEMO.md)
- **Phase Summaries:** See `PHASE*_COMPLETE.md` files
- **Training Analysis:** [TRAINING_IMPROVEMENTS_ANALYSIS.md](TRAINING_IMPROVEMENTS_ANALYSIS.md)

---

**Repository Map Version:** 1.0
**Generated:** 2025-10-08
**Maintainer:** Claude Code Assistant
