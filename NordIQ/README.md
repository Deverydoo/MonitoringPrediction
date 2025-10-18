# NordIQ AI Systems - Predictive Infrastructure Monitoring

**Nordic precision, AI intelligence**

Copyright © 2025 NordIQ AI, LLC. All rights reserved.

---

## 🚀 Quick Start

### 1. Start the System

**Windows:**
```bash
start_all.bat
```

**Linux/Mac:**
```bash
./start_all.sh
```

The system will automatically:
- Generate/load API keys
- Start inference daemon (port 8000)
- Start metrics generator  
- Start web dashboard (port 8501)

### 2. Access the Dashboard

Open your browser to: **http://localhost:8501**

---

## 📁 Directory Structure

```
NordIQ/
├── start_all.bat/sh         # Start all services
├── stop_all.bat/sh          # Stop all services
├── bin/                     # Utility scripts
│   ├── generate_api_key.py  # API key management
│   └── setup_api_key.*      # API key setup helpers
├── src/                     # Application source code
│   ├── daemons/             # Background services
│   │   ├── tft_inference_daemon.py
│   │   ├── metrics_generator_daemon.py
│   │   └── adaptive_retraining_daemon.py
│   ├── dashboard/           # Web dashboard
│   │   ├── tft_dashboard_web.py
│   │   └── Dashboard/       # Modular components
│   ├── training/            # Model training
│   │   ├── main.py          # CLI interface
│   │   ├── tft_trainer.py   # Training logic
│   │   └── precompile.py    # Optimization
│   ├── core/                # Shared libraries
│   │   ├── config/          # Configuration
│   │   ├── utils/           # Utilities
│   │   ├── adapters/        # Production adapters
│   │   └── explainers/      # XAI components
│   └── generators/          # Data generation
├── models/                  # Trained models
├── data/                    # Runtime data
├── logs/                    # Application logs
└── .streamlit/              # Dashboard config
```

---

## 🎓 Training a Model

### Option 1: Use the CLI (Recommended)

```bash
# 1. Generate training data (30 days, 20 servers)
python src/training/main.py generate --hours 720 --servers 20

# 2. Train model (20 epochs)
python src/training/main.py train --epochs 20

# 3. Check status
python src/training/main.py status
```

### Option 2: Direct Training

```bash
# Generate data
python src/generators/metrics_generator.py --hours 720 --servers 20

# Train model
python src/training/tft_trainer.py --epochs 20
```

---

## 🔧 Configuration

### API Keys

API keys are automatically generated on first run. To regenerate:

```bash
python bin/generate_api_key.py --force
```

### Models

Models are stored in `models/` directory. The system automatically uses the most recent model.

To use a specific model, set in `.streamlit/config.toml`:
```toml
[server]
model_path = "models/tft_model_YYYYMMDD_HHMMSS"
```

### Environment Variables

Create `.env` file in NordIQ root:
```bash
TFT_API_KEY=your_api_key_here
CUDA_VISIBLE_DEVICES=0
```

---

## 📊 Monitoring

### Service Endpoints

- **Dashboard**: http://localhost:8501
- **Inference API**: http://localhost:8000
- **Metrics API**: http://localhost:8001
- **Health Check**: http://localhost:8000/health

### Logs

Application logs are stored in `logs/` directory:
- `logs/inference.log` - Inference daemon
- `logs/metrics.log` - Metrics generator
- `logs/dashboard.log` - Dashboard

---

## 🚢 Deployment

### Production Deployment

1. **Copy NordIQ folder** to production server
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure production adapters** in `src/core/adapters/`
4. **Start services**:
   ```bash
   ./start_all.sh
   ```

### Docker Deployment

```dockerfile
FROM python:3.10
COPY NordIQ/ /app/
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["./start_all.sh"]
```

### Systemd Service (Linux)

Create `/etc/systemd/system/nordiq.service`:
```ini
[Unit]
Description=NordIQ AI Monitoring System
After=network.target

[Service]
Type=forking
User=nordiq
WorkingDirectory=/opt/NordIQ
ExecStart=/opt/NordIQ/start_all.sh
ExecStop=/opt/NordIQ/stop_all.sh

[Install]
WantedBy=multi-user.target
```

---

## 🛠️ Troubleshooting

### Services won't start

1. Check conda environment:
   ```bash
   conda activate py310
   python --version  # Should be 3.10
   ```

2. Check dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Check logs in `logs/` directory

### Dashboard shows "Connection Failed"

1. Ensure inference daemon is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Check API key matches in:
   - `.env` file
   - `.streamlit/secrets.toml`

### No predictions appearing

1. Ensure metrics generator is streaming:
   ```bash
   curl http://localhost:8001/status
   ```

2. Check model exists:
   ```bash
   ls models/
   ```

---

## 📞 Support

- **Documentation**: ../Docs/
- **Issues**: GitHub Issues
- **Email**: hello@nordiqai.io
- **Website**: https://nordiqai.io

---

## 📄 License

Business Source License 1.1

Copyright © 2025 NordIQ AI, LLC. All rights reserved.

See LICENSE file for details.

---

**Nordic precision, AI intelligence** 🧭
