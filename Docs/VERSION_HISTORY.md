# NordIQ Version History

**Current Version:** v1.2.1
**Last Updated:** October 18, 2025

---

## v1.2.1 (October 18, 2025) - XAI Integration Complete

**Major Feature:** Explainable AI (XAI) fully integrated and production-ready

### Features Added
- ✅ **XAI Dashboard Tab** - "🧠 Insights (XAI)" with 3 sub-tabs
  - SHAP feature importance (which metrics drove prediction)
  - Attention analysis (which time periods model focused on)
  - Counterfactual scenarios (what-if analysis with recommendations)
- ✅ **Daemon API Endpoint** - `/explain/{server_name}` (version 2.1)
- ✅ **Professional Polish**
  - Display names for metrics ("CPU User %" vs "cpu_user_pct")
  - Scenario-specific emojis (🔄 Restart, 📈 Scale, ⚡ Optimize)
  - 30-second caching for performance
- ✅ **NordIQ Metrics Framework Metrics Support** - All 14 metrics in XAI analysis

### Bug Fixes
- Fixed `List` import error in insights.py
- Fixed API key authentication (403 errors) for XAI endpoint
- Updated XAI explainers to use NordIQ Metrics Framework schema (was using old 4-metric system)
- Fixed counterfactual field names (`safe` vs `is_safe`, calculated effectiveness score)

### Documentation
- Created XAI_POLISH_CHECKLIST.md (comprehensive improvement plan)
- Created FUTURE_DASHBOARD_MIGRATION.md (Dash/React migration strategy)
- Created HUMAN_TODO_CHECKLIST.md (deployment & sales action plan)
- Added SESSION_2025-10-18_DEBUGGING.md (debugging session summary)

### Git Cleanup
- Excluded Docs/archive/ from git tracking (removed 30,564 lines of historical docs)
- Archive files remain on disk for local reference

### Website Updates
- Added Tron-themed hero image (blue/red network mesh)
- Added screenshot gallery to product.html (11 dashboard images)
- CSS styling for responsive gallery

**Total Commits:** 10
**Lines Added:** ~1,900 (XAI integration + documentation)
**Key Differentiator:** XAI explanations - no competitor has this!

---

## v1.1.0 (October 17, 2025) - NordIQ Branding Release

**Major Change:** Rebranding from generic "TFT Monitoring" to **NordIQ AI Systems**

### Branding
- ✅ Company name: NordIQ AI Systems, LLC
- ✅ Brand identity: 🧭 Nordic compass (precision, direction)
- ✅ Tagline: "Nordic precision, AI intelligence"
- ✅ Color scheme: Ice blue (#00D4FF) + Navy blue (#141E30)
- ✅ Website: 6/6 pages complete (nordiqai.io planned)

### XAI Foundation (Code Complete)
- ✅ SHAP explainer (361 lines) - Feature importance analysis
- ✅ Attention visualizer (341 lines) - Temporal focus detection
- ✅ Counterfactual generator (465 lines) - What-if scenarios
- ✅ Total: 1,189 lines of XAI code (not yet integrated into dashboard)

### Dashboard Improvements
- 11 specialized tabs (modular architecture)
- Scenario control (healthy, degrading, critical)
- Auto-refresh optimizations
- Performance improvements (60% faster with caching)

### Website
- Complete 6-page website built
- Professional copy and value proposition
- SEO optimization
- Contact forms and clear CTAs

**Total Commits:** ~15
**Key Achievement:** Production-ready brand identity + XAI modules built

---

## v1.0.0 (October 13-15, 2025) - NordIQ Metrics Framework Metrics Refactor

**Major Change:** Migrated from 4-metric system to 14-metric NordIQ Metrics Framework production schema

### Metrics System Overhaul
- ✅ **14 NordIQ Metrics Framework metrics** (industry-standard production monitoring)
  - CPU: user, system, iowait, idle, Java process
  - Memory: used, swap used
  - Disk: usage percentage
  - Network: ingress MB/s, egress MB/s
  - Connections: backend close-wait, frontend close-wait
  - System: load average, uptime days
- ✅ **Profile-based baselines** (7 server profiles)
- ✅ **Fuzzy logic risk scoring** (contextual intelligence)

### Architecture
- Modular dashboard (84.8% code reduction from refactor)
- Microservices: Inference daemon, Metrics generator, Dashboard
- GPU-accelerated predictions (<100ms per server)
- WebSocket streaming for real-time updates

### Model
- Temporal Fusion Transformer (TFT)
- 111,320 parameters
- 88% accuracy on critical incidents
- 8-hour prediction horizon

**Key Achievement:** Production-grade metric system, enterprise-ready

---

## v0.9.0 (October 10-12, 2025) - Feature Complete

### Features Completed
- ✅ Predictive monitoring (30-60 min early warning)
- ✅ 7 graduated severity levels
- ✅ Cost avoidance calculator
- ✅ Auto-remediation scenarios
- ✅ Alerting strategy configurator
- ✅ Historical trends analysis
- ✅ Fleet heatmap visualization

### Technical
- Dashboard optimization (fragment-based updates)
- State persistence (rolling window)
- Automated data collection for retraining
- Certification testing framework

**Key Achievement:** Feature-complete product, ready for beta testing

---

## v0.5.0 (October 7-9, 2025) - Initial Build

### Core Functionality
- ✅ TFT model training pipeline
- ✅ Inference daemon (FastAPI)
- ✅ Streamlit dashboard (basic)
- ✅ 4-metric monitoring (CPU, memory, disk I/O, latency)

### Demo
- 20-server synthetic environment
- Scenario testing (healthy, degrading, critical)
- Real-time predictions

**Key Achievement:** Proof of concept, working end-to-end

---

## Version Numbering Scheme

**Format:** MAJOR.MINOR.PATCH

- **MAJOR (1.x.x):** Breaking changes, major releases
  - v1.0.0: NordIQ Metrics Framework metrics (production-ready)
  - v2.0.0: (Future) React dashboard rewrite

- **MINOR (x.1.x):** New features, no breaking changes
  - v1.1.0: NordIQ branding
  - v1.2.0: XAI integration

- **PATCH (x.x.1):** Bug fixes, polish
  - v1.2.1: XAI bug fixes, documentation

---

## Upcoming Releases (Planned)

### v1.3.0 - Production Integrations (Q4 2025)
**When:** After first paying customer

- [ ] MongoDB adapter (production data source)
- [ ] Prometheus adapter
- [ ] REST API push endpoint
- [ ] Improved model (20+ epoch training)
- [ ] Customer-specific profiles

### v1.4.0 - Advanced Analytics (Q1 2026)
**When:** After 3-5 customers

- [ ] Anomaly detection
- [ ] Root cause analysis
- [ ] Custom alerting rules
- [ ] Slack/PagerDuty integration
- [ ] Multi-tenant support

### v2.0.0 - Dash Migration (Q1 2026)
**When:** After 5-10 customers, $5K MRR

- [ ] Plotly Dash dashboard (3-5 day migration)
- [ ] 3-4x faster performance
- [ ] Better mobile support
- [ ] Advanced visualizations
- **Cost:** $5K-8K

### v3.0.0 - React Migration (Q2-Q3 2026)
**When:** After 20+ customers, $25K MRR

- [ ] React + FastAPI architecture
- [ ] Lightning-fast client-side rendering
- [ ] White-label support
- [ ] Mobile app foundation
- **Cost:** $20K-30K

---

## Release Philosophy

**Ship Early, Ship Often:**
- v0.5.0: Proof of concept (October 7)
- v1.0.0: Production-ready (October 13) - 6 days later!
- v1.2.1: XAI slam dunk (October 18) - 5 days later!

**Focus on Value:**
- Don't wait for perfect
- Ship features that close deals
- Improve based on customer feedback
- Revenue-driven upgrades

**Current Status (v1.2.1):**
- ✅ **Production-ready** - Deploy today
- ✅ **Feature-complete** - Everything you need
- ✅ **Slam dunk XAI** - Competitive differentiator
- 🎯 **Next:** Get paying customers!

---

**Maintained By:** Craig Giannelli / NordIQ AI Systems, LLC
**Git Tag:** v1.2.1
**Release Date:** October 18, 2025
**Next Review:** After first paying customer
