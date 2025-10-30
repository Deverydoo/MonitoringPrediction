# NordIQ AI - Documentation

**Nordic precision, AI intelligence**

Welcome to the NordIQ AI documentation! This folder contains everything you need to deploy, integrate, and use the NordIQ Predictive Infrastructure Monitoring platform.

---

## 🚀 Quick Start

**New to NordIQ?** Start here:

1. **[Quick Start Guide](getting-started/QUICK_START.md)** - Get up and running in 10 minutes
2. **[API Key Setup](getting-started/API_KEY_SETUP.md)** - Configure authentication
3. **[Python Environment](getting-started/PYTHON_ENV.md)** - Set up your environment

---

## 📚 Documentation Categories

### 🎯 Getting Started

Everything you need to get NordIQ up and running:

- **[Quick Start Guide](getting-started/QUICK_START.md)** - 10-minute setup guide
- **[API Key Setup](getting-started/API_KEY_SETUP.md)** - Authentication configuration
- **[Python Environment](getting-started/PYTHON_ENV.md)** - Environment setup (conda/pip)

**Time to deploy:** 10-15 minutes

---

### 🔌 Integration

Connect NordIQ to your existing tools and workflows:

- **[Integration Guide](integration/INTEGRATION_GUIDE.md)** (800+ lines)
  - Complete REST API reference
  - Python client examples
  - JavaScript/React integration
  - Grafana JSON API integration
  - Custom dashboard examples

- **[Integration Quick Start](integration/INTEGRATION_QUICKSTART.md)** - 5-minute integration guide

- **[Production Integration](integration/PRODUCTION_INTEGRATION_GUIDE.md)** - Enterprise deployment

- **[Production Data Adapters](integration/PRODUCTION_DATA_ADAPTERS.md)** - Connect to your data sources
  - Elasticsearch adapter
  - MongoDB adapter
  - Custom adapter development

- **[API Reference](integration/QUICK_REFERENCE_API.md)** - REST API quick reference

**Use cases:**
- Feed predictions to Grafana dashboards
- Trigger Slack/Teams alerts
- Build custom visualizations
- Integrate with existing monitoring tools

---

### ⚙️ Operations

Run and manage NordIQ in production:

- **[Daemon Management](operations/DAEMON_MANAGEMENT.md)** (700+ lines)
  - systemd service setup
  - Docker deployment
  - nginx reverse proxy
  - Health monitoring
  - Log management
  - Troubleshooting

- **[Inference Service](operations/INFERENCE_README.md)** - Inference daemon operations
  - REST API on port 8000
  - WebSocket streaming
  - Model management
  - Performance tuning

**Production tools:**
- `daemon.bat` (Windows) / `daemon.sh` (Linux) - Service management
- `start_all.bat/sh` - One-command startup
- `stop_all.bat/sh` - Graceful shutdown

---

### 🔐 Authentication

Secure your NordIQ deployment:

- **[Authentication Guide](authentication/AUTHENTICATION_IMPLEMENTATION_GUIDE.md)**
  - API key authentication
  - Session-based auth
  - Token-based auth
  - Implementation examples (2-8 hours)

- **[Okta SSO Integration](authentication/OKTA_SSO_INTEGRATION.md)**
  - Corporate single sign-on
  - "Weird passthrough" SSO
  - Configuration guide
  - Testing procedures

**Security levels:**
- Development: API key (quick)
- Production: Okta SSO (enterprise)
- Custom: Token-based auth

---

### 🧠 Understanding NordIQ

Learn how the system works and why it's better:

- **[How Predictions Work](understanding/HOW_PREDICTIONS_WORK.md)**
  - Temporal Fusion Transformer (TFT) explained
  - 30-60 minute early warning
  - Prediction horizons (30min, 1hr, 8hr)

- **[Why TFT?](understanding/WHY_TFT.md)**
  - Model selection rationale
  - vs LSTM, GRU, Prophet, ARIMA
  - Attention mechanisms
  - Multi-horizon forecasting

- **[Contextual Risk Intelligence](understanding/CONTEXTUAL_RISK_INTELLIGENCE.md)**
  - Fuzzy logic risk scoring
  - Profile-aware thresholds
  - Trend analysis
  - Multi-metric correlation

- **[Server Profiles](understanding/SERVER_PROFILES.md)**
  - 7 server profile types
  - Transfer learning benefits
  - Profile-specific thresholds
  - Custom profile creation

- **[Alert Levels](understanding/ALERT_LEVELS.md)**
  - 7 graduated severity levels
  - SLA response times
  - Escalation policies
  - Alert routing

**Key Innovation:** Predicts incidents 30-60 minutes in advance using contextual intelligence, not just static thresholds.

---

### 📊 Marketing & Business

Understand the value proposition and economics:

- **[Project Summary](marketing/PROJECT_SUMMARY.md)**
  - Product overview
  - Key features
  - Technical specifications
  - Value proposition

- **[Managed Hosting Economics](marketing/MANAGED_HOSTING_ECONOMICS.md)**
  - Cost analysis
  - Pricing models
  - ROI calculations
  - Deployment options

- **[Future Roadmap](marketing/FUTURE_ROADMAP.md)**
  - Planned features
  - Enhancement timeline
  - XAI integration
  - Multi-datacenter support

- **[Customer Branding Guide](marketing/CUSTOMER_BRANDING_GUIDE.md)**
  - Custom themes (Wells Fargo, etc.)
  - White-label options
  - Logo placement
  - Color schemes

**Business Value:**
- 15-60 minute early warning
- $50K-75K annual savings
- 5-8x faster development
- Context-aware intelligence

---

## 🎯 Common Use Cases

### Deploying for a New Customer

1. Read **[Quick Start](getting-started/QUICK_START.md)**
2. Follow **[Production Integration](integration/PRODUCTION_INTEGRATION_GUIDE.md)**
3. Set up **[Authentication](authentication/AUTHENTICATION_IMPLEMENTATION_GUIDE.md)**
4. Configure **[Daemon Management](operations/DAEMON_MANAGEMENT.md)**
5. Apply **[Customer Branding](marketing/CUSTOMER_BRANDING_GUIDE.md)**

**Total time:** 2-4 hours for basic deployment, 1-2 days for enterprise

---

### Integrating with Existing Tools

1. Read **[Integration Quick Start](integration/INTEGRATION_QUICKSTART.md)** (5 minutes)
2. Review **[API Reference](integration/QUICK_REFERENCE_API.md)**
3. Follow **[Integration Guide](integration/INTEGRATION_GUIDE.md)** for your use case:
   - Grafana dashboard
   - Slack alerts
   - Custom web app
   - React/Vue frontend

**Total time:** 30 minutes to 2 hours depending on complexity

---

### Understanding the Technology

1. **[How Predictions Work](understanding/HOW_PREDICTIONS_WORK.md)** - Core technology
2. **[Why TFT?](understanding/WHY_TFT.md)** - Model selection
3. **[Contextual Risk Intelligence](understanding/CONTEXTUAL_RISK_INTELLIGENCE.md)** - Risk scoring
4. **[Server Profiles](understanding/SERVER_PROFILES.md)** - Transfer learning

**For:** Technical stakeholders, data scientists, architects

---

### Running in Production

1. **[Daemon Management](operations/DAEMON_MANAGEMENT.md)** - systemd/Docker setup
2. **[Authentication Guide](authentication/AUTHENTICATION_IMPLEMENTATION_GUIDE.md)** - Secure deployment
3. **[Production Integration](integration/PRODUCTION_INTEGRATION_GUIDE.md)** - Data sources
4. **[Inference Service](operations/INFERENCE_README.md)** - Service operations

**For:** DevOps, SRE, system administrators

---

## 📦 What's Included in NordIQ/

```
NordIQ/
├── Docs/                    # This folder - All client documentation
├── src/                     # Application source code
│   ├── daemons/            # Services (inference, metrics)
│   ├── dashboard/          # Streamlit web UI
│   ├── core/               # Shared libraries
│   ├── generators/         # Data/demo generators
│   └── training/           # Model training
├── models/                  # Trained TFT models
├── data/                    # Runtime data
├── logs/                    # Log files
├── bin/                     # Utilities
├── .streamlit/             # Streamlit config
├── dash_app.py             # Dash production dashboard (15× faster)
├── dash_config.py          # Customer branding + config
├── daemon.bat/sh           # Daemon manager
├── start_all.bat/sh        # One-command startup
└── stop_all.bat/sh         # Graceful shutdown
```

**This entire folder is self-contained and ready to deploy!**

---

## 🌐 Dashboard Options

### Streamlit Dashboard (Legacy)
- **Location:** `src/dashboard/tft_dashboard_web.py`
- **Launch:** `streamlit run src/dashboard/tft_dashboard_web.py`
- **Port:** 8501
- **Features:** Full-featured, 10 tabs, comprehensive
- **Performance:** ~1200ms render time

### Dash Dashboard (Production)
- **Location:** `dash_app.py`
- **Launch:** `python dash_app.py`
- **Port:** 8050
- **Features:** High-performance, customer branding, WebGL
- **Performance:** ~78ms render time (15× faster)
- **Recommended for:** Production deployments

---

## 🔧 Quick Commands

### Start Everything
```bash
# Windows
start_all.bat

# Linux/Mac
./start_all.sh
```

### Manage Daemons
```bash
# Windows
daemon.bat start inference
daemon.bat start metrics
daemon.bat stop all

# Linux/Mac
./daemon.sh start inference
./daemon.sh start metrics
./daemon.sh stop all
```

### Access Dashboard
- Streamlit: http://localhost:8501
- Dash: http://localhost:8050

### API Health Check
```bash
curl http://localhost:8000/health
```

---

## 📞 Support

### Documentation Issues
- Check the relevant guide in this folder
- Review troubleshooting sections
- See operations/DAEMON_MANAGEMENT.md for common issues

### Integration Help
- Start with integration/INTEGRATION_QUICKSTART.md
- Review API examples in integration/INTEGRATION_GUIDE.md
- Check integration/QUICK_REFERENCE_API.md for endpoint details

### Technical Questions
- Read understanding/ docs for system concepts
- Review marketing/PROJECT_SUMMARY.md for overview
- Check operations/ docs for production guidance

---

## 🎓 Recommended Reading Order

### For Developers
1. getting-started/QUICK_START.md
2. understanding/HOW_PREDICTIONS_WORK.md
3. integration/INTEGRATION_GUIDE.md
4. integration/QUICK_REFERENCE_API.md

### For DevOps/SRE
1. getting-started/QUICK_START.md
2. operations/DAEMON_MANAGEMENT.md
3. operations/INFERENCE_README.md
4. authentication/AUTHENTICATION_IMPLEMENTATION_GUIDE.md

### For Business/Sales
1. marketing/PROJECT_SUMMARY.md
2. understanding/HOW_PREDICTIONS_WORK.md
3. marketing/MANAGED_HOSTING_ECONOMICS.md
4. marketing/FUTURE_ROADMAP.md

### For Data Scientists
1. understanding/WHY_TFT.md
2. understanding/CONTEXTUAL_RISK_INTELLIGENCE.md
3. understanding/SERVER_PROFILES.md
4. integration/PRODUCTION_DATA_ADAPTERS.md

---

## 📊 Documentation Statistics

```
Total Documents: 21 files
Total Lines: ~11,000 lines
Categories: 6
Coverage:
  ✅ Getting Started (3 guides)
  ✅ Integration (5 guides)
  ✅ Operations (2 guides)
  ✅ Authentication (2 guides)
  ✅ Understanding (5 guides)
  ✅ Marketing (4 guides)
```

---

## 🚀 Next Steps

1. **New to NordIQ?** → [Quick Start Guide](getting-started/QUICK_START.md)
2. **Integrating?** → [Integration Quick Start](integration/INTEGRATION_QUICKSTART.md)
3. **Deploying to production?** → [Daemon Management](operations/DAEMON_MANAGEMENT.md)
4. **Questions about the tech?** → [How Predictions Work](understanding/HOW_PREDICTIONS_WORK.md)

---

**Version:** 1.0.0
**Company:** NordIQ AI Systems, LLC
**Tagline:** Nordic precision, AI intelligence
**License:** Business Source License 1.1

© 2025 NordIQ AI, LLC. All rights reserved.
