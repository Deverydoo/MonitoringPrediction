# TFT Monitoring Prediction System

Temporal Fusion Transformer (TFT) based predictive monitoring for server fleet management.

## 🎯 What This Does

Predicts server incidents 30 minutes to 8 hours in advance using time series forecasting. Built for:
- **Demonstrations**: Reproducible incident scenarios
- **Production**: Real-time monitoring dashboards
- **Training**: Generate synthetic data for model development

## 🚀 Quick Demo

```bash
# One command to see everything
python run_demo.py
```

This generates a 5-minute incident scenario and launches the monitoring dashboard. You'll see:
- Baseline healthy state
- Gradual escalation of metrics
- Critical incident peak
- Recovery back to normal

**[See SETUP_DEMO.md for detailed instructions](SETUP_DEMO.md)**

## 📊 System Components

### Demo System (NEW - Recommended for Getting Started)
- **[demo_data_generator.py](demo_data_generator.py)** - Reproducible incident scenarios
- **[tft_dashboard_refactored.py](tft_dashboard_refactored.py)** - File-based monitoring dashboard
- **[run_demo.py](run_demo.py)** - One-command demo runner
- **[DEMO_README.md](DEMO_README.md)** - Complete demo documentation
- **[SETUP_DEMO.md](SETUP_DEMO.md)** - Quick start guide

### Core System (Production Ready)
- **[config.py](config.py)** - System configuration
- **[metrics_generator.py](metrics_generator.py)** - Full-scale training data generator
- **[tft_trainer.py](tft_trainer.py)** - TFT model training
- **[tft_inference.py](tft_inference.py)** - Real-time inference
- **[common_utils.py](common_utils.py)** - Shared utilities
- **[main.py](main.py)** - CLI interface

### Legacy
- **[tft_dashboard.py](tft_dashboard.py)** - Original dashboard (random data)

## 📖 Documentation

| Document | Purpose |
|----------|---------|
| **[SETUP_DEMO.md](SETUP_DEMO.md)** | Quick start - get demo running in 5 minutes |
| **[DEMO_README.md](DEMO_README.md)** | Complete demo guide with examples |
| **[DATA_LOADING_IMPROVEMENTS.md](DATA_LOADING_IMPROVEMENTS.md)** | Parquet optimization (10-100x faster) ⚡ |
| **[PARQUET_UPDATE_SUMMARY.md](PARQUET_UPDATE_SUMMARY.md)** | Quick Parquet guide |
| **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)** | What changed in the refactor and why |
| **[CHANGELOG.md](CHANGELOG.md)** | Complete version history |
| **README.md** | This file - project overview |

## 🎬 Usage Examples

### Demo Mode (Recommended First Step)
```bash
# Generate demo data
python demo_data_generator.py

# Run dashboard
python tft_dashboard_refactored.py demo_data/demo_dataset.parquet

# Or do both at once
python run_demo.py
```

### Training Mode
```bash
# Generate training data
python metrics_generator.py --hours 72 --num_prod 40

# Train model
python tft_trainer.py --dataset ./training/ --epochs 20

# Check status
python main.py status
```

### Production Mode
```python
from tft_dashboard_refactored import DataSource, LiveDashboard

# Connect to your production metrics
data_source = DataSource("/path/to/production/metrics.parquet")
dashboard = LiveDashboard(data_source)
dashboard.run()
```

## 🔧 Installation

```bash
# Clone repository
git clone <repo-url>
cd MonitoringPrediction

# Install dependencies
pip install torch lightning pytorch-forecasting safetensors
pip install pandas numpy matplotlib pyarrow ipython

# Verify setup
python main.py setup
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  DATA SOURCES                    │
├─────────────────────────────────────────────────┤
│  Demo Data Generator  │  Metrics Generator      │
│  (5min scenarios)     │  (training datasets)    │
│         ↓             │         ↓               │
│  demo_dataset.parquet │  training/*.parquet     │
└──────────┬────────────┴────────────┬────────────┘
           │                         │
           ↓                         ↓
    ┌─────────────┐          ┌──────────────┐
    │  Dashboard  │          │  TFT Trainer │
    │  (monitor)  │          │  (learn)     │
    └─────────────┘          └──────┬───────┘
                                    │
                                    ↓
                            ┌───────────────┐
                            │ TFT Inference │
                            │ (predict)     │
                            └───────────────┘
```

## 📋 Features

### Demo System
✅ Reproducible incident scenarios
✅ Explainable patterns
✅ Multiple output formats (CSV, Parquet)
✅ Configurable fleet size and duration
✅ Predictable timeline with phases

### Monitoring Dashboard
✅ Real-time visualization (5 figures)
✅ File-based data sources
✅ Risk scoring per server
✅ Environment-wide incident probability
✅ Interactive time series plots

### Training Pipeline
✅ Synthetic data generation
✅ TFT model training with PyTorch Lightning
✅ Safetensors model storage
✅ Early stopping and checkpointing
✅ GPU acceleration support

### Inference Engine
✅ Real-time predictions
✅ 30-minute and 8-hour horizons
✅ Alert generation with thresholds
✅ Trend analysis

## 🎯 Use Cases

### For Demos/Presentations
Generate reproducible incidents to demonstrate:
- Predictive capabilities
- Early warning detection
- Alert generation
- Risk scoring

### For Development
Test and develop against:
- Consistent datasets
- Known patterns
- Controlled scenarios

### For Production
Monitor real systems:
- Connect to production data
- Real-time dashboards
- Predictive alerts
- Capacity planning

## 📊 Demo Data Structure

The demo generates a predictable 5-minute incident:

```
Time    Phase       Intensity  Behavior
────────────────────────────────────────────
0:00    Stable      0%         Normal operations
  ↓
1:30    Escalation  30-70%     Warnings appear
  ↓
2:30    Peak        80-100%    Critical alerts
  ↓
3:30    Recovery    70-0%      Return to normal
  ↓
5:00    Stable      0%         Resolved
```

**Affected Servers**: First server of each type (web-001, api-001, db-001, cache-001)
**Unaffected Servers**: Remain stable for comparison
**Metrics**: CPU, memory, latency, error rate, network, disk I/O

## 🛠️ Configuration

### Demo Generator
```python
# demo_data_generator.py
duration_minutes = 5      # Length of scenario
fleet_size = 10          # Number of servers
seed = 42                # Reproducibility
tick_seconds = 5         # Data point interval
```

### Dashboard
```python
# tft_dashboard_refactored.py
REFRESH_SECONDS = 5      # Display update rate
ROLLING_WINDOW_MIN = 5   # Historical data shown
SAVE_PLOTS = False       # Save figures to disk
```

### Training
```python
# config.py
epochs = 20
batch_size = 32
hidden_size = 32
attention_heads = 8
prediction_horizon = 96  # 8 hours
context_length = 288     # 24 hours
```

## 🧪 Testing

```bash
# Generate demo data
python demo_data_generator.py

# Verify files created
ls demo_data/

# Check metadata
cat demo_data/demo_dataset_metadata.json

# Run dashboard
python run_demo.py
```

## 📈 Dashboard Visualizations

The dashboard displays 5 real-time figures:

1. **KPI Header**
   - Environment health status (GREEN/YELLOW/RED)
   - Current incident phase
   - 30-min and 8-hour probabilities
   - Fleet status and progress

2. **Top 5 Problem Servers**
   - Risk-ranked bar chart
   - Server profile and state
   - Color-coded severity

3. **Incident Probability Trend**
   - Time series of predictions
   - 30-minute vs 8-hour forecasts
   - Phase boundary markers

4. **Fleet Risk Heat Map**
   - At-a-glance health strip
   - Green/yellow/red color coding
   - All servers visible

5. **Rolling Metrics**
   - CPU usage (median)
   - Latency P95
   - Error rate (mean)
   - Warning/critical thresholds

## 🔄 Workflow

### Demo Workflow
```
1. python demo_data_generator.py
   → Generates demo_dataset.parquet

2. python tft_dashboard_refactored.py demo_data/demo_dataset.parquet
   → Displays incident scenario

3. Present and explain the predictable pattern
```

### Development Workflow
```
1. python metrics_generator.py --hours 72
   → Generates training/*.parquet

2. python tft_trainer.py --epochs 20
   → Trains model → models/tft_model_*/

3. python tft_inference.py --model models/latest/
   → Makes predictions

4. Integrate with production monitoring
```

## 🎓 Learning Path

1. **Start Here**: Run `python run_demo.py` to see the system in action
2. **Understand**: Read [DEMO_README.md](DEMO_README.md) to learn about the components
3. **Customize**: Modify `demo_data_generator.py` to create new scenarios
4. **Scale Up**: Use `metrics_generator.py` for training datasets
5. **Train**: Run `tft_trainer.py` to build your model
6. **Deploy**: Connect dashboard to production data

## 🤝 Contributing

Key extension points:
- **New incident patterns**: Modify `demo_data_generator.py` phases
- **Additional visualizations**: Add figures to dashboard
- **Custom metrics**: Extend schema in data generators
- **Model improvements**: Adjust TFT architecture in `tft_trainer.py`
- **Alert logic**: Enhance `ModelAdapter` in dashboard

## 📝 Requirements

### Python Packages
```
torch>=2.0
lightning>=2.0
pytorch-forecasting
safetensors
pandas
numpy
matplotlib
pyarrow (optional, for Parquet)
IPython (optional, for notebooks)
```

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- GPU optional (for training)

## 🐛 Troubleshooting

**"No module named 'numpy'"**
```bash
pip install numpy pandas matplotlib
```

**"Data source not found"**
```bash
python demo_data_generator.py
```

**"PyArrow not available"**
```bash
pip install pyarrow
# Or use CSV: --format csv
```

**Dashboard closes immediately**
- Run from Jupyter notebook
- Or check Python environment

See [DEMO_README.md](DEMO_README.md) for complete troubleshooting guide.

## 📄 License

[Your License Here]

## 🙋 Support

- Check [SETUP_DEMO.md](SETUP_DEMO.md) for quick start
- Read [DEMO_README.md](DEMO_README.md) for detailed guide
- Review [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md) for architecture details

## 🎯 Project Status

✅ Demo system - Production ready
✅ Data generators - Fully functional
✅ Dashboard - File-based, reproducible
✅ Training pipeline - Complete
🔄 Model integration - In progress
🔄 Production deployment - Coming soon

---

**Quick Links:**
- [Quick Start](SETUP_DEMO.md)
- [Complete Guide](DEMO_README.md)
- [What Changed](CHANGES_SUMMARY.md)
- [Run Demo](run_demo.py)
