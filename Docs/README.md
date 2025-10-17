# Documentation - TFT Monitoring Prediction System

**Last Updated**: October 17, 2025
**Status**: Production Ready with Modular Architecture

---

## 📚 Quick Navigation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Fast setup guide (30 seconds to running system)
- **[QUICK_START.md](QUICK_START.md)** - Alternative full setup walkthrough
- **[PYTHON_ENV.md](PYTHON_ENV.md)** - Environment setup

### Core System Documentation
- **[DATA_CONTRACT.md](DATA_CONTRACT.md)** ⭐ - Schema specification (14 LINBORG metrics)
- **[SERVER_PROFILES.md](SERVER_PROFILES.md)** - 7 server profiles for transfer learning
- **[HOW_PREDICTIONS_WORK.md](HOW_PREDICTIONS_WORK.md)** - Technical explanation of TFT predictions
- **[CONTEXTUAL_RISK_INTELLIGENCE.md](CONTEXTUAL_RISK_INTELLIGENCE.md)** - Design philosophy

### Operations & Training
- **[MODEL_TRAINING_GUIDELINES.md](MODEL_TRAINING_GUIDELINES.md)** - How to train/retrain models
- **[RETRAINING_PIPELINE.md](RETRAINING_PIPELINE.md)** - Operational retraining procedures
- **[PRODUCTION_INTEGRATION_GUIDE.md](PRODUCTION_INTEGRATION_GUIDE.md)** - Integrating real production data
- **[INFERENCE_README.md](INFERENCE_README.md)** - Inference daemon details

### Security & Authentication
- **[AUTHENTICATION_IMPLEMENTATION_GUIDE.md](AUTHENTICATION_IMPLEMENTATION_GUIDE.md)** - Auth options (2-8 hours)
- **[OKTA_SSO_INTEGRATION.md](OKTA_SSO_INTEGRATION.md)** - Corporate SSO integration

### Planning & Future
- **[FUTURE_ROADMAP.md](FUTURE_ROADMAP.md)** - Planned enhancements and Phase 2+
- **[HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md)** - Team handoff document

### Reference & Technical Details
- **[QUICK_REFERENCE_API.md](QUICK_REFERENCE_API.md)** - API endpoints reference
- **[UNKNOWN_SERVER_HANDLING.md](UNKNOWN_SERVER_HANDLING.md)** - Hash-based server encoding
- **[SPARSE_DATA_HANDLING.md](SPARSE_DATA_HANDLING.md)** - Handling offline servers
- **[GPU_AUTO_CONFIGURATION.md](GPU_AUTO_CONFIGURATION.md)** - GPU setup and optimization
- **[WHY_TFT.md](WHY_TFT.md)** - Why Temporal Fusion Transformer?

### Project Meta
- **[INDEX.md](INDEX.md)** - Detailed navigation
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete system overview
- **[HUMAN_VS_AI_TIMELINE.md](HUMAN_VS_AI_TIMELINE.md)** - Development velocity analysis
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

### For AI Assistants
- **[RAG/](RAG/)** - Context documents for AI sessions
  - **[CURRENT_STATE.md](RAG/CURRENT_STATE.md)** ⭐ - Single source of truth
  - **[PROJECT_CODEX.md](RAG/PROJECT_CODEX.md)** - Development rules
  - **[CLAUDE_SESSION_GUIDELINES.md](RAG/CLAUDE_SESSION_GUIDELINES.md)** - Session management

---

## 🎯 Common Tasks - Quick Links

### I want to...

**...get the system running quickly**
→ [QUICKSTART.md](QUICKSTART.md)

**...understand the LINBORG metrics**
→ [DATA_CONTRACT.md](DATA_CONTRACT.md)

**...train a new model**
→ [MODEL_TRAINING_GUIDELINES.md](MODEL_TRAINING_GUIDELINES.md)

**...integrate with production**
→ [PRODUCTION_INTEGRATION_GUIDE.md](PRODUCTION_INTEGRATION_GUIDE.md)

**...add authentication**
→ [AUTHENTICATION_IMPLEMENTATION_GUIDE.md](AUTHENTICATION_IMPLEMENTATION_GUIDE.md)

**...understand how predictions work**
→ [HOW_PREDICTIONS_WORK.md](HOW_PREDICTIONS_WORK.md)

**...see future plans**
→ [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md)

**...hand off to another team**
→ [HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md)

**...contribute code**
→ [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📦 Document Categories

### Essential (Read First)
1. **QUICKSTART.md** - Get running in 30 seconds
2. **DATA_CONTRACT.md** - Understand the 14 LINBORG metrics
3. **CONTEXTUAL_RISK_INTELLIGENCE.md** - Understand the design philosophy
4. **HOW_PREDICTIONS_WORK.md** - Understand the technical approach

### Operations
- MODEL_TRAINING_GUIDELINES.md
- RETRAINING_PIPELINE.md
- PRODUCTION_INTEGRATION_GUIDE.md
- INFERENCE_README.md

### Security
- AUTHENTICATION_IMPLEMENTATION_GUIDE.md
- OKTA_SSO_INTEGRATION.md

### Technical Reference
- SERVER_PROFILES.md
- UNKNOWN_SERVER_HANDLING.md
- SPARSE_DATA_HANDLING.md
- GPU_AUTO_CONFIGURATION.md
- WHY_TFT.md
- QUICK_REFERENCE_API.md

### Planning
- FUTURE_ROADMAP.md
- HANDOFF_SUMMARY.md
- HUMAN_VS_AI_TIMELINE.md

---

## 🏗️ System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   TFT Monitoring System                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [Metrics Generator] → [Training Data] → [TFT Model]   │
│         ↓                                      ↓        │
│  [Synthetic/Real]         [Parquet]      [Safetensors]  │
│                                                ↓        │
│                          [Inference Daemon] ← Model      │
│                                  ↓                      │
│                          REST API (8000)                │
│                                  ↓                      │
│                          [Web Dashboard]                │
│                              Streamlit                  │
│                          http://localhost:8501          │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**
- **14 LINBORG Metrics**: Production-grade monitoring metrics
- **7 Server Profiles**: Profile-based transfer learning
- **Contextual Risk Scoring**: Multi-factor intelligent alerts
- **Graduated Severity**: 7 levels from Healthy to Imminent Failure

---

## 🚨 Critical Information

### LINBORG Metrics (Required)
The system uses **14 production LINBORG metrics**. Old 4-metric system is deprecated.

```python
LINBORG_METRICS = [
    'cpu_user_pct', 'cpu_sys_pct', 'cpu_iowait_pct',  # CPU components
    'cpu_idle_pct', 'java_cpu_pct',                    # CPU (cont.)
    'mem_used_pct', 'swap_used_pct',                   # Memory
    'disk_usage_pct',                                   # Disk
    'net_in_mb_s', 'net_out_mb_s',                     # Network
    'back_close_wait', 'front_close_wait',             # TCP connections
    'load_average', 'uptime_days'                      # System
]
```

See [DATA_CONTRACT.md](DATA_CONTRACT.md) for details.

### Server Profiles (Transfer Learning)
7 profiles enable predictions for new servers without retraining:
- ML_COMPUTE (ppml####)
- DATABASE (ppdb###)
- WEB_API (ppweb###)
- CONDUCTOR_MGMT (ppcon##)
- DATA_INGEST (ppdi###)
- RISK_ANALYTICS (ppra###)
- GENERIC (ppsrv###)

See [SERVER_PROFILES.md](SERVER_PROFILES.md) for details.

---

## 📊 Documentation Stats

**Total Documents**: 24 core documents (down from 52)
**Categories**: 6 main categories
**Archive**: 26+ historical documents moved to archive/

### Recent Cleanup (Oct 17, 2025)
- ✅ Removed presentation/demo docs (6 files)
- ✅ Archived completion reports (13 files)
- ✅ Archived implementation plans (4 files)
- ✅ Removed silly/irrelevant docs (3 files)
- ✅ Result: 54% reduction in doc count

---

## 🔄 Documentation Maintenance

### When to Update
- **After major features**: Update relevant technical docs and CURRENT_STATE.md
- **Schema changes**: Update DATA_CONTRACT.md first, then related docs
- **New profiles**: Update SERVER_PROFILES.md
- **Security changes**: Update authentication guides
- **Future plans**: Update FUTURE_ROADMAP.md

### Archive Policy
Move to `archive/` folder:
- Session notes (after information is incorporated)
- Completion reports (after work is done)
- Implementation plans (after implementation)
- Presentation materials (after presentation)
- Obsolete guides (after replacement)

---

## 🎓 For New Team Members

**Start here:**
1. Read [QUICKSTART.md](QUICKSTART.md) - Get the system running
2. Read [DATA_CONTRACT.md](DATA_CONTRACT.md) - Understand the data
3. Read [HOW_PREDICTIONS_WORK.md](HOW_PREDICTIONS_WORK.md) - Understand predictions
4. Read [CONTEXTUAL_RISK_INTELLIGENCE.md](CONTEXTUAL_RISK_INTELLIGENCE.md) - Understand design
5. Explore the dashboard and try different scenarios
6. Read [HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md) - Full context

**For development:**
- Read [RAG/PROJECT_CODEX.md](RAG/PROJECT_CODEX.md) - Development rules
- Read [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- Check [FUTURE_ROADMAP.md](FUTURE_ROADMAP.md) - See what's planned

---

## 🔗 External Resources

### Frameworks & Libraries
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) - TFT implementation
- [Streamlit](https://streamlit.io/) - Dashboard framework
- [FastAPI](https://fastapi.tiangolo.com/) - REST API framework

### Research Papers
- [Temporal Fusion Transformers (2019)](https://arxiv.org/abs/1912.09363) - Original TFT paper
- "Attention is All You Need" (2017) - Transformer architecture

### Corporate Resources
- Linborg Monitoring Documentation (internal)
- Okta SSO Setup Guide (internal)
- Spectrum Platform Overview (internal)

---

## ❓ Need Help?

**Technical Issues:**
- Check [QUICKSTART.md](QUICKSTART.md) troubleshooting section
- Check [INDEX.md](INDEX.md) for detailed navigation
- Review [RAG/CURRENT_STATE.md](RAG/CURRENT_STATE.md) for current state

**Process Questions:**
- Check [CONTRIBUTING.md](CONTRIBUTING.md)
- Check [HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md)

**Design Questions:**
- Check [CONTEXTUAL_RISK_INTELLIGENCE.md](CONTEXTUAL_RISK_INTELLIGENCE.md)
- Check [WHY_TFT.md](WHY_TFT.md)

---

**Maintained By**: Project Team
**Last Major Cleanup**: October 17, 2025
**Next Review**: After Phase 2 completion

**Welcome to the TFT Monitoring Prediction System documentation!** 🚀
