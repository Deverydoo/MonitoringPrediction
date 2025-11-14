# ArgusAI - Documentation

**Predictive System Monitoring**

Welcome to the ArgusAI documentation! This folder contains everything you need to deploy, integrate, and use the NordIQ Predictive Infrastructure Monitoring platform.

---

## 🚀 Quick Start

**New to NordIQ?** Start here:

1. **[Getting Started Guide](GETTING_STARTED.md)** - Complete setup covering authentication, environment, and quick start (10-15 minutes)

---

## 📚 Documentation by Audience

### 👨‍💻 For Developers

**Building custom integrations and dashboards:**

- **[API Reference](for-developers/API_REFERENCE.md)** - Complete REST API documentation
  - All endpoints (health, predictions, alerts, XAI)
  - Request/response formats
  - Authentication & rate limiting
  - Python & JavaScript client examples

- **[Data Format Specification](for-developers/DATA_FORMAT_SPEC.md)** - Complete schema reference
  - Input metrics (14 LINBORG metrics)
  - Output predictions (risk scores, forecasts)
  - JSON schemas & TypeScript interfaces
  - Validation rules

**Time to integrate:** 1-2 hours

---

### 📊 For Business Intelligence & Analytics

**Connecting to BI tools:**

- **[Grafana Integration Guide](for-business-intelligence/GRAFANA_INTEGRATION.md)** - Complete Grafana setup
  - JSON API plugin installation
  - 6 panel configurations (time series, stat, table, gauge)
  - Variables & filtering
  - Alert configuration
  - Best practices

- **Power BI Integration** *(Coming soon)*
- **Tableau Integration** *(Coming soon)*

**Time to visualize:** 30-45 minutes

---

### ⚙️ For Production / DevOps

**Deploying and operating in production:**

- **[Production Integration Guide](PRODUCTION_INTEGRATION.md)** ⭐ **CRITICAL**
  - Comprehensive production deployment covering all integration scenarios
  - Stop using demo data & connect production systems
  - Real Data Integration (Elasticsearch, Prometheus, MongoDB)
  - Data Ingestion API specification (POST `/feed/data`)
  - Elasticsearch adapter setup & optimization
  - MongoDB adapter setup & performance guidance
  - Data transformation requirements
  - Warmup & verification procedures
  - Code examples (Python, Node.js, curl)

**Time to production:** 1-2 hours

---

### 🎯 Getting Started (Everyone)

Essential setup guide:

- **[Getting Started Guide](GETTING_STARTED.md)** - Complete setup including quick start (10 min), API key setup, and Python environment configuration

**Time to deploy:** 10-15 minutes

---

### ⚙️ Operations & Management

Run and manage NordIQ in production:

- **[Daemon Management](operations/DAEMON_MANAGEMENT.md)** (700+ lines)
  - systemd service setup
  - Docker deployment
  - nginx reverse proxy
  - Health monitoring
  - Log management
  - Troubleshooting

**Production tools:**
- `daemon.bat` (Windows) / `daemon.sh` (Linux) - Service management
- `start_all.bat/sh` - One-command startup
- `stop_all.bat/sh` - Graceful shutdown

---

### 🔐 Authentication & Security

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

### I need to connect my production data (stop using demo)

1. **[Production Integration Guide](PRODUCTION_INTEGRATION.md)** - Complete walkthrough for all scenarios
   - Real Data Integration (Elasticsearch, Prometheus, MongoDB)
   - Data Ingestion API specification
   - Elasticsearch/MongoDB setup guides
2. **[Data Format Specification](for-developers/DATA_FORMAT_SPEC.md)** - Schema reference

**Total time:** 1-2 hours

---

### I need to build a custom dashboard

1. **[API Reference](for-developers/API_REFERENCE.md)** - All available endpoints
2. **[Data Format Specification](for-developers/DATA_FORMAT_SPEC.md)** - JSON schemas
3. **[Getting Started Guide](GETTING_STARTED.md)** - Get system running

**Total time:** 2-4 hours

---

### I need to visualize in Grafana

1. **[Grafana Integration Guide](for-business-intelligence/GRAFANA_INTEGRATION.md)** - Complete setup
2. **[API Reference](for-developers/API_REFERENCE.md)** - Endpoint details

**Total time:** 30-45 minutes

---

### I need to deploy for a customer

1. **[Getting Started Guide](GETTING_STARTED.md)** - Basic setup
2. **[Production Integration Guide](PRODUCTION_INTEGRATION.md)** - Connect their systems
3. **[Authentication Guide](authentication/AUTHENTICATION_IMPLEMENTATION_GUIDE.md)** - Secure deployment
4. **[Daemon Management](operations/DAEMON_MANAGEMENT.md)** - Production operations
5. **[Customer Branding Guide](marketing/CUSTOMER_BRANDING_GUIDE.md)** - Customize appearance

**Total time:** 2-4 hours for basic deployment, 1-2 days for enterprise

---

### I need to understand the technology

1. **[How Predictions Work](understanding/HOW_PREDICTIONS_WORK.md)** - Core technology
2. **[Why TFT?](understanding/WHY_TFT.md)** - Model selection
3. **[Contextual Risk Intelligence](understanding/CONTEXTUAL_RISK_INTELLIGENCE.md)** - Risk scoring
4. **[Server Profiles](understanding/SERVER_PROFILES.md)** - Transfer learning

**For:** Technical stakeholders, data scientists, architects

---

## 📦 What's Included in NordIQ/

```
NordIQ/
├── Docs/                          # This folder - All documentation
│   ├── GETTING_STARTED.md        # Complete setup guide (consolidated)
│   ├── PRODUCTION_INTEGRATION.md # Production deployment guide (consolidated)
│   ├── for-developers/           # API reference, schemas, dev guides
│   ├── for-business-intelligence/ # BI tools (Grafana, Power BI)
│   ├── operations/               # Daemon management, troubleshooting
│   ├── authentication/           # Security & SSO
│   ├── understanding/            # How it works, concepts
│   └── marketing/                # Business value, ROI
│
├── src/                           # Application source code
│   ├── daemons/                  # Services (inference, metrics)
│   ├── dashboard/                # Streamlit web UI
│   ├── core/                     # Shared libraries
│   ├── generators/               # Data/demo generators
│   └── training/                 # Model training
│
├── models/                        # Trained TFT models
├── data/                          # Runtime data
├── logs/                          # Log files
├── bin/                           # Utilities
├── .streamlit/                   # Streamlit config
├── dash_app.py                   # Dash production dashboard (15× faster)
├── dash_config.py                # Customer branding + config
├── daemon.bat/sh                 # Daemon manager
├── start_all.bat/sh              # One-command startup
└── stop_all.bat/sh               # Graceful shutdown
```

**This entire folder is self-contained and ready to deploy!**

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
- Review troubleshooting sections in operations/

### Integration Help
- **Developers:** See for-developers/ folder
- **BI Tools:** See for-business-intelligence/ folder
- **Production:** See for-production/ folder

### Technical Questions
- Read understanding/ docs for system concepts
- Review marketing/PROJECT_SUMMARY.md for overview
- Check operations/ docs for production guidance

---

## 🎓 Recommended Reading Order

### For Developers
1. GETTING_STARTED.md
2. for-developers/API_REFERENCE.md
3. for-developers/DATA_FORMAT_SPEC.md
4. understanding/HOW_PREDICTIONS_WORK.md

### For DevOps/SRE
1. GETTING_STARTED.md
2. PRODUCTION_INTEGRATION.md
3. operations/DAEMON_MANAGEMENT.md
4. authentication/AUTHENTICATION_IMPLEMENTATION_GUIDE.md

### For BI Analysts
1. GETTING_STARTED.md
2. for-business-intelligence/GRAFANA_INTEGRATION.md
3. for-developers/API_REFERENCE.md (API endpoints)

### For Business/Sales
1. marketing/PROJECT_SUMMARY.md
2. understanding/HOW_PREDICTIONS_WORK.md
3. marketing/MANAGED_HOSTING_ECONOMICS.md
4. marketing/FUTURE_ROADMAP.md

### For Data Scientists
1. understanding/WHY_TFT.md
2. understanding/CONTEXTUAL_RISK_INTELLIGENCE.md
3. understanding/SERVER_PROFILES.md
4. PRODUCTION_INTEGRATION.md

---

## 📊 Documentation Statistics

```
Total Documents: 25+ files
Total Lines: ~22,000+ lines
Categories: 7

✅ Core Getting Started (1 consolidated guide)
✅ Core Production Integration (1 consolidated guide)
✅ For Developers (4 guides) - API, Schemas, Adapters, References
✅ For Business Intelligence (1 guide) - Grafana
✅ Operations (1 guide) - Daemon management
✅ Authentication (2 guides) - Security
✅ Understanding (5 guides) - Concepts
✅ Marketing (4 guides) - Business value
```

**Consolidated Structure Benefits:**
- ✅ Streamlined client documentation (2 primary guides)
- ✅ Clear separation by audience (dev/BI/ops)
- ✅ All getting started content in one accessible file
- ✅ All production integration scenarios consolidated
- ✅ Zero redundancy between documents
- ✅ Each doc has ONE clear purpose
- ✅ Find what you need in < 30 seconds

---

## 🚀 Next Steps

1. **New to NordIQ?** → [Getting Started Guide](GETTING_STARTED.md)
2. **Connecting real data?** → [Production Integration Guide](PRODUCTION_INTEGRATION.md)
3. **Building custom dashboard?** → [API Reference](for-developers/API_REFERENCE.md)
4. **Setting up Grafana?** → [Grafana Integration](for-business-intelligence/GRAFANA_INTEGRATION.md)
5. **Deploying to production?** → [Daemon Management](operations/DAEMON_MANAGEMENT.md)
6. **Questions about the tech?** → [How Predictions Work](understanding/HOW_PREDICTIONS_WORK.md)

---

**Version:** 2.0.0 (Documentation Restructure)
**Company:** ArgusAI, LLC
**Tagline:** Predictive System Monitoring
**License:** Business Source License 1.1

Built by Craig Giannelli and Claude Code
