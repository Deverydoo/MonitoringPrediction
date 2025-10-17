# Dashboard Guide

**Last Updated:** 2025-10-11
**Recommended Dashboard:** tft_dashboard_web.py ⭐

---

## 🎨 Web Dashboard (RECOMMENDED)

### Quick Start

```bash
# 1. Start the TFT inference daemon
python tft_inference.py --daemon --port 8000

# 2. Launch web dashboard
streamlit run tft_dashboard_web.py

# 3. Open browser to http://localhost:8501
```

### Features

✅ **Beautiful Streamlit UI** - Professional web interface
✅ **Real-time Predictions** - Connects to TFT daemon via REST API
✅ **Auto-refresh** - Updates every 30 seconds
✅ **Demo Mode** - Built-in simulation scenarios
✅ **Contract Compliant** - Uses hash-based server encoding

### Dashboard Tabs

1. **📊 Overview** - Fleet health status, risk distribution
2. **🔥 Heatmap** - Visual server risk grid
3. **⚠️ Top Servers** - Problem servers with predictions
4. **📈 Historical** - Trend analysis and charts
5. **⚙️ Advanced** - Settings, debug info, model details

### Configuration

In the Streamlit sidebar:
- **Daemon URL**: Default `http://localhost:8000`
- **Refresh Interval**: Default 30 seconds
- **Demo Mode**: Enable for simulation scenarios
- **Scenario**: Choose stable/degrading/critical patterns

### Requirements

```bash
pip install streamlit plotly requests pandas
```

---

## 📊 Terminal Dashboard (Legacy)

### tft_dashboard.py

⚠️ **DEPRECATED** - Use web dashboard instead

This is the matplotlib-based terminal dashboard. Still works but lacks:
- Interactive UI
- Modern visualizations
- Easy configuration

**Only use if:**
- Running headless without browser
- Need command-line only interface
- Debugging data pipelines

**Usage:**
```bash
python tft_dashboard.py training/server_metrics.parquet \
  --daemon-url http://localhost:8000 \
  --tick-interval 5 \
  --refresh 30
```

---

## 🔄 Complete Workflow

### For Web Dashboard (Recommended)

```bash
# 1. Generate training data (if needed)
python metrics_generator.py --servers 25 --hours 24 --output ./training/

# 2. Train model (if needed, or wait for current training)
python tft_trainer.py --dataset ./training/ --epochs 20

# 3. Start TFT daemon
python tft_inference.py --daemon --port 8000
# Wait for: [SUCCESS] TFT model loaded successfully!

# 4. Launch web dashboard
streamlit run tft_dashboard_web.py

# 5. Open browser to http://localhost:8501
```

### Demo Mode (Without Training)

```bash
# 1. Start daemon in simulation mode
python tft_inference.py --daemon --port 8000 --simulation business_hours

# 2. Launch web dashboard
streamlit run tft_dashboard_web.py

# 3. Enable "Demo Mode" in sidebar
#    - Choose scenario (stable/degrading/critical)
#    - Click "Start Demo"
```

---

## 🎯 Dashboard Features Comparison

| Feature | Web Dashboard | Terminal Dashboard |
|---------|--------------|-------------------|
| Interface | Streamlit (browser) | Matplotlib (terminal) |
| Status | ✅ **Recommended** | ⚠️ Deprecated |
| Real-time Updates | ✅ Auto-refresh | ✅ Auto-refresh |
| TFT Predictions | ✅ Via daemon API | ✅ Via daemon API |
| Demo Mode | ✅ Built-in | ❌ File-based only |
| Interactive | ✅ Click/select | ❌ View only |
| Modern UI | ✅ Professional | ⚠️ Basic plots |
| Configuration | ✅ Sidebar | ⚠️ CLI args |
| Mobile Friendly | ✅ Responsive | ❌ Terminal only |
| Export | ✅ Download CSV | ⚠️ Save plots |
| Server Details | ✅ Drill-down | ❌ Overview only |

---

## 📖 Web Dashboard Quick Reference

### Main Metrics

**Fleet Health Status**
- 🟢 **GOOD**: Incident probability < 20%
- 🟡 **WARNING**: Incident probability 20-50%
- 🔴 **CRITICAL**: Incident probability > 50%

**Risk Levels (Per Server)**
- **Low**: 0.0 - 0.3
- **Medium**: 0.3 - 0.6
- **High**: 0.6 - 1.0

### Keyboard Shortcuts

- `Ctrl+R` - Refresh dashboard
- `R` - Rerun app (in Streamlit)
- `C` - Clear cache

### Troubleshooting

**"Cannot connect to daemon"**
```bash
# Check if daemon is running
curl http://localhost:8000/health

# If not, start it:
python tft_inference.py --daemon --port 8000
```

**"No predictions available"**
- Wait 30 seconds for first refresh
- Check daemon logs for errors
- Verify model trained successfully

**"Contract version mismatch"**
- Model needs retraining with new contract
- See DATA_CONTRACT.md for details

---

## 🚀 Production Deployment

### Docker Deployment (Future)

```dockerfile
# Dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Start daemon and web dashboard
CMD ["sh", "-c", "python tft_inference.py --daemon & streamlit run tft_dashboard_web.py"]
```

### Cloud Deployment

**Streamlit Cloud:**
1. Push code to GitHub
2. Connect Streamlit Cloud to repo
3. Set daemon URL in Streamlit secrets
4. Deploy!

**Requirements for Production:**
- Persistent storage for models
- Load balancer for multiple instances
- Monitoring/alerting integration
- Authentication layer

---

## 📝 Summary

**Use This:**
```bash
streamlit run tft_dashboard_web.py
```

**Don't Use:**
```bash
python tft_dashboard.py  # Deprecated
```

**Perfect For:**
- Production monitoring
- Stakeholder demos
- Real-time incident response
- Model validation
- Team collaboration

---

**Guide Version:** 1.0
**Last Updated:** 2025-10-11
**Maintained By:** Project Team
