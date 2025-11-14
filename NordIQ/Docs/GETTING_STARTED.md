# Getting Started with NordIQ TFT Monitoring System

**Version**: 2.0 (Contract-Based Architecture)
**Last Updated**: November 14, 2025
**Target Audience**: Engineers, System Administrators, Data Scientists

---

## Quick Start (30 Seconds)

Get the system running immediately with these commands:

```bash
# 1. Activate the Python environment
conda activate py310

# 2. Start the inference daemon (uses latest trained model)
python tft_inference.py --daemon --port 8000

# 3. Launch the web dashboard (in a new terminal)
python dash_app.py

# 4. Open your browser to http://localhost:8501
```

That's it! Your monitoring system is now running.

---

## Prerequisites

Before getting started, ensure you have the following installed:

### System Requirements

- **Python**: 3.10+ (via Anaconda/Miniconda recommended)
- **RAM**: 8GB minimum (16GB recommended for training)
- **GPU**: Optional but strongly recommended (NVIDIA with CUDA support for faster training)
- **Disk Space**: 10GB minimum (training data + models)

### Software Requirements

- **Conda**: For environment management
- **Git**: For version control
- **Bash/CMD**: Terminal access
- **Web Browser**: Any modern browser (Chrome, Firefox, Safari, Edge)

### Required Python Packages

The `py310` environment should already include:

- PyTorch 2.0+
- PyTorch Lightning 2.0+
- PyTorch Forecasting 1.0+
- Pandas 2.2+
- PyArrow
- NumPy
- Matplotlib
- Safetensors
- Streamlit

To verify all dependencies are installed, run:

```bash
conda activate py310
python -c "from main import setup; setup()"
```

---

## Installation Steps

### Step 1: Environment Activation

Always activate the dedicated Python environment before running any commands:

#### Windows

```bash
conda activate py310
# or
py310\Scripts\activate
```

#### Linux/macOS

```bash
conda activate py310
# or
source py310/bin/activate
```

You should see `(py310)` at the start of your terminal prompt.

### Step 2: Verify Environment

Confirm you're using the correct Python version:

```bash
# Verify Python version
python --version
# Should output: Python 3.10.x

# Verify environment path
which python  # Linux/macOS
# or
where python  # Windows
```

### Step 3: Set Up Configuration Files

The system requires configuration for API authentication. Two options:

#### Recommended: Automated Setup

Run the automated setup script (creates `.env` and Streamlit secrets):

**Windows:**
```bash
setup_api_key.bat
```

**Linux/macOS:**
```bash
./setup_api_key.sh
```

#### Manual Setup

If automation doesn't work, proceed to the API Key Setup section below.

---

## API Key Setup

The NordIQ system uses API key authentication to secure communication between the dashboard and the inference daemon. This prevents unauthorized access to your monitoring data.

### How API Keys Work

1. **Dashboard** reads API key from `.streamlit/secrets.toml`
2. **Dashboard** sends requests with `X-API-Key` header
3. **Daemon** validates the key against `TFT_API_KEY` environment variable
4. **Daemon** returns data if valid, rejects if invalid

### Configuration

#### For Development (Recommended)

Use environment variables and .env files for flexibility:

1. **Copy the example environment file:**

```bash
cp .env.example .env
```

2. **The `.env` file contains:**

```bash
TFT_API_KEY=bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO
```

3. **Load it before starting the daemon:**

**Linux/macOS:**
```bash
# Option A: Export directly
export TFT_API_KEY="bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO"

# Option B: Source from .env file
export $(cat .env | xargs)
```

**Windows:**
```cmd
set TFT_API_KEY=bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO
```

4. **Configure Streamlit (Dashboard):**

The dashboard API key is stored in `.streamlit/secrets.toml` (automatically created):

```toml
[daemon]
api_key = "bwAeXnR2VbQvuVIhGXLm5uvOXcFhnNi32Kvc9UXKzQMmoMm3c1JLW30tICf95OsO"
```

### File Locations

```
MonitoringPrediction/
├── .streamlit/
│   └── secrets.toml          # Dashboard API key (do NOT commit)
├── .env                       # Daemon API key (do NOT commit)
├── .env.example              # Template (safe to commit)
└── .gitignore                # Protects secrets
```

### Important Security Notes

**Always do:**
- Use different API keys for dev/staging/production
- Rotate API keys quarterly
- Store keys in secret management systems for production
- Use `.gitignore` to prevent committing secrets
- Use HTTPS in production

**Never do:**
- Commit `.streamlit/secrets.toml` or `.env` to git
- Share API keys via email or chat
- Reuse keys across environments
- Log API keys in application logs

### Generating a New API Key

If you need to rotate your API key for security:

```bash
python -c "
import secrets
import string
alphabet = string.ascii_letters + string.digits
api_key = ''.join(secrets.choice(alphabet) for _ in range(64))
print(f'New API Key: {api_key}')
"
```

Then update:
1. `.streamlit/secrets.toml` (dashboard)
2. `.env` or environment variable (daemon)

### Development Mode (No Authentication)

For local development without API key protection:

1. Do NOT set `TFT_API_KEY` environment variable
2. Do NOT configure `.streamlit/secrets.toml`
3. Both services will run without authentication

**Warning**: Development mode is insecure. Only use on localhost!

### Production Deployment

#### Systemd Service (Linux)

Add the API key to your systemd service file:

```ini
[Service]
Environment="TFT_API_KEY=your-api-key-here"
ExecStart=/path/to/venv/bin/python tft_inference.py
```

#### Docker

```bash
docker run -e TFT_API_KEY="your-api-key-here" ...
```

Or in `docker-compose.yml`:

```yaml
services:
  tft-daemon:
    environment:
      - TFT_API_KEY=your-api-key-here
```

#### Cloud Platforms

**AWS/Azure/GCP**: Use native secret management services and inject via environment variables

**Kubernetes**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: tft-api-key
type: Opaque
stringData:
  api_key: your-api-key-here
```

---

## Python Environment Setup

### Understanding the py310 Environment

The project uses a dedicated Python 3.10 environment named `py310`. This ensures all necessary packages are installed and version conflicts are avoided.

### Activation

**Always activate before working:**

#### Windows
```bash
conda activate py310
```

#### Linux/macOS
```bash
conda activate py310
# Alternative if using venv instead of conda
source py310/bin/activate
```

You'll see `(py310)` in your terminal prompt when activated.

### Using with Jupyter Notebooks

The notebook `_StartHere.ipynb` uses the `py310` kernel:

1. **Open the notebook:**
   ```bash
   jupyter notebook _StartHere.ipynb
   ```

2. **Verify the kernel:**
   - Click the kernel name (top-right)
   - Select `py310` from the dropdown
   - If not available, run:
     ```bash
     python -m ipykernel install --user --name py310
     ```

3. **Run cells in order:**
   - Cell 6: Generate training data
   - Cell 7: Train the model
   - Cell 9: Start dashboard

### Running Scripts

Always ensure the environment is active:

```bash
conda activate py310

# Then run any Python script
python metrics_generator.py --servers 25 --hours 24 --output ./training/
python tft_trainer.py --dataset ./training/ --epochs 20
python tft_inference.py --daemon --port 8000
python dash_app.py
```

### Verifying Dependencies

Check that all required packages are available:

```bash
conda activate py310

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Verify PyTorch Forecasting
python -c "import pytorch_forecasting; print('PyTorch Forecasting: OK')"

# Verify Pandas
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

# Run full system check
python -c "from main import setup; setup()"
```

### Common Environment Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: No module named 'numpy'` | Wrong environment active | Run `conda activate py310` first |
| Script uses wrong Python version | Using system Python instead | Verify with `which python` or `where python` |
| Import errors for PyTorch | Missing CUDA or dependency | Reinstall: `conda install pytorch::pytorch` |
| Kernel not found in Jupyter | py310 not registered | Run `python -m ipykernel install --user --name py310` |

---

## Full Setup Process (First Time Only)

If you need to train a new model from scratch, follow these steps in order:

### Step 1: Generate Training Data

```bash
conda activate py310

# Generate 24 hours of metrics for 25 servers
python metrics_generator.py --servers 25 --hours 24 --output ./training/
```

**Creates:**
- `training/server_metrics.parquet` - Training dataset
- `training/server_mapping.json` - Server name-to-hash encoder
- `training/metrics_metadata.json` - Dataset information

**Time**: ~30-60 seconds

**For better models, generate more data:**

```bash
# 1 week of data
python metrics_generator.py --servers 25 --hours 168 --output ./training/

# 30 days of data (recommended for best performance)
python metrics_generator.py --servers 25 --hours 720 --output ./training/
```

### Step 2: Train the Model

```bash
# Train for 20 epochs (recommended)
python tft_trainer.py --dataset ./training/ --epochs 20
```

**Creates:**
- `models/tft_model_YYYYMMDD_HHMMSS/model.safetensors` - Trained weights
- `models/tft_model_YYYYMMDD_HHMMSS/server_mapping.json` - Server encoder (copied)
- `models/tft_model_YYYYMMDD_HHMMSS/training_info.json` - Contract version and metadata
- `models/tft_model_YYYYMMDD_HHMMSS/config.json` - Model architecture

**Time**: ~30-40 minutes on RTX 4090 GPU, longer on CPU

### Step 3: Start the Inference Daemon

```bash
python tft_inference.py --daemon --port 8000
```

**What it does:**
- Loads the latest trained model
- Validates contract compatibility
- Starts REST API server on port 8000
- Provides real-time predictions
- Runs continuously until stopped (Ctrl+C)

### Step 4: Launch the Web Dashboard

Open a **new terminal** and run:

```bash
conda activate py310
python dash_app.py
```

**Dashboard shows:**
- Fleet health overview
- Server risk heatmap
- Top problem servers with predictions
- Historical trend analysis
- Model information and settings

**Access at**: http://localhost:8501

### Complete Workflow Diagram

```
┌─────────────────────────────────────────────┐
│  1. Generate Data                           │
│  python metrics_generator.py                │
│  → training/server_metrics.parquet          │
│  → training/server_mapping.json             │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  2. Train Model                             │
│  python tft_trainer.py                      │
│  → models/tft_model_*/model.safetensors     │
│  → models/tft_model_*/server_mapping.json   │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  3. Start Daemon (Terminal 1)               │
│  python tft_inference.py --daemon           │
│  → REST API: http://localhost:8000          │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  4. Launch Dashboard (Terminal 2)           │
│  python dash_app.py                         │
│  → Web UI: http://localhost:8501            │
└─────────────────────────────────────────────┘
```

---

## Verification

### Verify Installation Success

Run these commands to confirm everything is working:

```bash
conda activate py310

# Check Python version
python --version
# Expected: Python 3.10.x

# Check system status
python main.py status
```

Output should show:
- Available trained models
- Training data location
- GPU status (if applicable)
- All dependencies installed

### Test the Inference API

Once the daemon is running (port 8000), test the API:

```bash
# Check daemon health
curl http://localhost:8000/health

# Get current predictions
curl http://localhost:8000/predictions/current

# Get active alerts
curl http://localhost:8000/alerts/active
```

### Verify Daemon and Dashboard Communication

1. Start daemon: `python tft_inference.py --daemon --port 8000`
2. Start dashboard: `python dash_app.py`
3. Open http://localhost:8501
4. Look for "Connected" status indicator
5. Navigate to Advanced tab to verify daemon info

### Check GPU Acceleration (Optional)

If you have an NVIDIA GPU:

```bash
conda activate py310

python -c "
import torch
print(f'GPU Available: {torch.cuda.is_available()}')
print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')
"
```

---

## Troubleshooting

### Environment and Setup Issues

#### "Cannot find py310 environment"

**Cause**: Environment not created or not accessible

**Solutions:**
```bash
# List all environments
conda env list

# Create py310 if missing
conda create -n py310 python=3.10

# Update conda
conda update conda

# Try activating again
conda activate py310
```

#### "ModuleNotFoundError: No module named 'X'"

**Cause**: Wrong Python environment active or missing dependency

**Solution:**
```bash
conda activate py310
python -c "from main import setup; setup()"
```

#### "Cannot connect to daemon"

**Cause**: Daemon not running or listening on wrong port

**Solution:**
```bash
# Check if daemon is running
curl http://localhost:8000/health

# If not running, start it
python tft_inference.py --daemon --port 8000

# If port 8000 is busy, use a different port
python tft_inference.py --daemon --port 8001
```

### Model and Data Issues

#### "Model not found"

**Cause**: No trained models exist

**Solution:**
```bash
# Check models directory
ls models/

# If empty, generate and train
python metrics_generator.py --servers 25 --hours 24 --output ./training/
python tft_trainer.py --dataset ./training/ --epochs 20
```

#### "server_mapping.json not found"

**Cause**: Model was trained with older version before contract implementation

**Solution:**
```bash
# Regenerate training data with current version
python metrics_generator.py --servers 25 --hours 24 --output ./training/

# Retrain model
python tft_trainer.py --dataset ./training/ --epochs 20

# Restart daemon to load new model
python tft_inference.py --daemon --port 8000
```

#### "Contract version mismatch"

**Cause**: Model trained with different contract version than current code

**Solution:**
```bash
# Retrain with current code
python tft_trainer.py --dataset ./training/ --epochs 20

# Restart daemon
python tft_inference.py --daemon --port 8000
```

### Dashboard Issues

#### "Dashboard shows connection errors (401/403)"

**Cause**: API key mismatch between dashboard and daemon

**Solution:**
1. Verify `.streamlit/secrets.toml` has correct API key
2. Verify daemon environment has `TFT_API_KEY` set correctly
3. Ensure both keys match exactly
4. Restart both services

```bash
# Check daemon has API key
echo $TFT_API_KEY  # Should output your key

# Restart daemon
python tft_inference.py --daemon --port 8000

# Restart dashboard in new terminal
python dash_app.py
```

#### "Dashboard won't load"

**Cause**: Port 8501 in use, or Streamlit issue

**Solution:**
```bash
# Try killing process on port 8501
lsof -ti:8501 | xargs kill -9  # Linux/macOS

# Try different port
streamlit run dash_app.py --server.port 8502

# Restart and clear cache
streamlit run dash_app.py --logger.level=debug
```

#### "No API key configured" message

**Cause**: Dashboard cannot find API key

**Solution:**
```bash
# Run setup script
./setup_api_key.bat    # Windows
./setup_api_key.sh     # Linux/macOS

# Or manually set
export TFT_API_KEY="your-key-here"
```

### Training Issues

#### "Out of memory" during training

**Cause**: Insufficient GPU/CPU memory

**Solutions:**
```bash
# Reduce batch size (if applicable)
python tft_trainer.py --dataset ./training/ --epochs 20 --batch-size 16

# Use smaller dataset
python metrics_generator.py --servers 10 --hours 24 --output ./training/

# Enable CPU-only (slower but less memory)
# (Check tft_trainer.py for CPU flag options)
```

#### Training is very slow

**Cause**: Using CPU instead of GPU, or GPU memory issues

**Solution:**
```bash
# Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, install GPU support:
# For NVIDIA GPUs with CUDA 11.8
conda install pytorch::pytorch pytorch::pytorch-cuda=11.8 -c pytorch -c nvidia

# Then try training again
python tft_trainer.py --dataset ./training/ --epochs 20
```

### API Key Issues

#### Daemon not reading .env file

**Cause**: python-dotenv not installed or .env not found

**Solution:**
```bash
# Option 1: Manually export
export TFT_API_KEY="your-key-here"
python tft_inference.py --daemon --port 8000

# Option 2: Install python-dotenv
pip install python-dotenv

# Option 3: Source .env before running
export $(cat .env | xargs)
python tft_inference.py --daemon --port 8000
```

---

## Common Tasks

### Check System Status

```bash
python main.py status
```

Shows available models, training data, GPU status, and dependency versions.

### Test Prediction API

```bash
# Health check
curl http://localhost:8000/health

# Current predictions
curl http://localhost:8000/predictions/current

# Specific server predictions
curl http://localhost:8000/predictions/server/ppvra00a01

# Active alerts
curl http://localhost:8000/alerts/active
```

### Retrain Model with New Data

```bash
conda activate py310

# Generate new training data
python metrics_generator.py --servers 25 --hours 720 --output ./training/

# Retrain model
python tft_trainer.py --dataset ./training/ --epochs 20

# Restart daemon to load new model
# (Stop with Ctrl+C, then restart)
python tft_inference.py --daemon --port 8000
```

### Enable Demo Mode

For presentations or testing without real data:

1. Start dashboard: `python dash_app.py`
2. Click "Enable Demo Mode" in sidebar
3. Choose scenario:
   - **Stable** - Healthy baseline system
   - **Degrading** - Gradual resource exhaustion
   - **Critical** - Acute failure conditions
4. Click "Start Demo"
5. Watch predictions evolve in real-time!

### Use Jupyter Notebook

For interactive exploration and analysis:

```bash
conda activate py310
jupyter notebook _StartHere.ipynb
```

Run cells in order:
- Cell 6: Generate training data
- Cell 7: Train model
- Cell 9: Start dashboard

---

## Dashboard Features

### Main Tabs

1. **Overview** - Fleet status and incident probabilities
2. **Heatmap** - Visual risk grid for all servers
3. **Top Servers** - Problem servers with TFT predictions
4. **Historical** - Trend charts and historical analysis
5. **Advanced** - Settings, debug mode, model information

### Key Features

- **Real-time monitoring**: Live metrics from all servers
- **Risk predictions**: AI-powered incident probability forecasts
- **Heatmap visualization**: Quick visual assessment of fleet health
- **Historical analysis**: Trend identification and pattern analysis
- **Demo mode**: Safe testing environment with synthetic data
- **API integration**: REST API for external system integration

---

## What's New in Version 2.0

### Hash-Based Server Encoding

**Before v2.0:**
```python
server_id = 0, 1, 2, 3...  # Sequential - breaks when fleet changes
```

**After v2.0:**
```python
server_id = hash('ppvra00a01') → '957601'  # Stable, deterministic
```

Benefits:
- Stable across fleet changes
- Deterministic hashing
- No retraining needed for new servers

### Contract Validation

All components now validate against a data contract:
- 8 valid server states (no more schema mismatches!)
- Server mapping saved with trained models
- Contract version tracking
- Automatic validation on model load

### Improved Unknown Server Support

- New servers automatically encoded via hash
- TFT handles unknowns gracefully
- Predictions decoded back to server names
- No retraining needed for individual server additions

---

## Pro Tips

1. **Use Web Dashboard** - Much better UX than terminal version
2. **Train with 720 hours of data** - Best model performance
3. **Enable Demo Mode** - Great for demonstrations and testing
4. **Always validate contract** - Check DATA_CONTRACT.md before training
5. **Save old models** - Keep as backups for comparison
6. **Monitor GPU usage** - Training uses significant GPU resources
7. **Use Systemd** - Run daemon as service for production
8. **Rotate API keys** - Change quarterly for security

---

## Next Steps

1. **Quick Start**: Follow the 30-second quick start at the top of this guide
2. **Full Setup**: If training a new model, follow the Full Setup Process section
3. **Explore Dashboard**: Navigate the web UI at http://localhost:8501
4. **Check Documentation**: Read related guides for deeper knowledge
5. **Customize Configuration**: Adjust parameters for your specific needs

---

## Related Documentation

- **[DATA_CONTRACT.md](DATA_CONTRACT.md)** - Data schema and validation rules (essential reading)
- **[DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md)** - Detailed dashboard feature guide
- **[UNKNOWN_SERVER_HANDLING.md](UNKNOWN_SERVER_HANDLING.md)** - How unknown servers are handled
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete system architecture overview
- **[CONTRACT_IMPLEMENTATION_PLAN.md](CONTRACT_IMPLEMENTATION_PLAN.md)** - Implementation details
- **[PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)** - Production deployment guide
- **[AUTHENTICATION_IMPLEMENTATION_GUIDE.md](AUTHENTICATION_IMPLEMENTATION_GUIDE.md)** - Authentication options

---

## Support and Questions

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review **Related Documentation**
3. Verify **Prerequisites** are met
4. Run **Verification** steps to diagnose issues
5. Check daemon logs: `tail -f daemon.log`

---

Built by **Craig Giannelli** and **Claude Code**

Last Updated: November 14, 2025
