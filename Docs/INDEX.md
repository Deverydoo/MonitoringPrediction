# Documentation Index

**Quick Reference for the TFT Monitoring Prediction System**

---

## 🚀 Start Here

New to the project? Start with these in order:

1. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** ⭐ **START HERE** - Complete current state, architecture, workflows
2. **[SETUP_DEMO.md](SETUP_DEMO.md)** - Get demo running in 5 minutes
3. **[DEMO_README.md](DEMO_README.md)** - Complete demo guide with examples
4. **[PYTHON_ENV.md](PYTHON_ENV.md)** - Environment setup (py310)

---

## 📖 By Use Case

### I want to run a demo
- **[SETUP_DEMO.md](SETUP_DEMO.md)** - Quick start
- **[SCENARIO_GUIDE.md](SCENARIO_GUIDE.md)** - Three demo scenarios (healthy/degrading/critical)
- **[DEMO_README.md](DEMO_README.md)** - Detailed demo guide

### I want to understand the architecture
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - System architecture and data flow
- **[TFT_MODEL_INTEGRATION.md](TFT_MODEL_INTEGRATION.md)** - How the TFT model works
- **[SESSION_INTEGRATION_COMPLETE.md](SESSION_INTEGRATION_COMPLETE.md)** - Latest integration work
- **[REPOMAP.md](REPOMAP.md)** - Complete repository map

### I want to train a model
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - See "Workflow 2: Training New Model"
- **[DATA_LOADING_IMPROVEMENTS.md](DATA_LOADING_IMPROVEMENTS.md)** - Parquet optimization (10-100x faster)
- **[ALL_PHASES_COMPLETE.md](ALL_PHASES_COMPLETE.md)** - Training improvements summary

### I want to run in production
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - See "Workflow 3: Production Monitoring"
- **[OPERATIONAL_MAINTENANCE_GUIDE.md](OPERATIONAL_MAINTENANCE_GUIDE.md)** - Maintaining model effectiveness
- **[MAINTENANCE_QUICK_REFERENCE.md](MAINTENANCE_QUICK_REFERENCE.md)** - Quick ops reference
- **[TFT_MODEL_INTEGRATION.md](TFT_MODEL_INTEGRATION.md)** - Model loading and inference

### I want to understand what changed
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[PROJECT_CLEANUP_SUMMARY.md](PROJECT_CLEANUP_SUMMARY.md)** - File cleanup decisions
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - See "Change Notes" section

---

## 📚 All Documentation (Alphabetical)

### Active Documentation (Primary Reference)

| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[ALL_PHASES_COMPLETE.md](ALL_PHASES_COMPLETE.md)** | Development phases summary | Understanding training improvements |
| **[CHANGELOG.md](CHANGELOG.md)** | Version history | Understanding what changed when |
| **[CLAUDE_SESSION_GUIDELINES.md](CLAUDE_SESSION_GUIDELINES.md)** | 🤖 Claude assistant session protocol | For Claude: session start/end, time tracking |
| **[DATA_LOADING_IMPROVEMENTS.md](DATA_LOADING_IMPROVEMENTS.md)** | Parquet optimization guide | Need fast data loading |
| **[DEMO_README.md](DEMO_README.md)** | Complete demo guide | Running demos, presentations |
| **[INDEX.md](INDEX.md)** | This file - navigation guide | Finding the right documentation |
| **[MAINTENANCE_QUICK_REFERENCE.md](MAINTENANCE_QUICK_REFERENCE.md)** | Quick ops reference | Production maintenance |
| **[ONLINE_LEARNING_DESIGN.md](ONLINE_LEARNING_DESIGN.md)** | Online learning design (future) | Planning incremental updates |
| **[OPERATIONAL_MAINTENANCE_GUIDE.md](OPERATIONAL_MAINTENANCE_GUIDE.md)** | Ops and maintenance guide | Maintaining model effectiveness |
| **[PROJECT_CLEANUP_SUMMARY.md](PROJECT_CLEANUP_SUMMARY.md)** | File cleanup decisions | Understanding file structure |
| **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** | ⭐ **MAIN REFERENCE** - Current state | Always start here |
| **[PYTHON_ENV.md](PYTHON_ENV.md)** | Python environment setup | Setting up py310 environment |
| **[README.md](README.md)** | Project overview | High-level understanding |
| **[REPOMAP.md](REPOMAP.md)** | Repository structure map | Finding files and understanding organization |
| **[SCENARIO_GUIDE.md](SCENARIO_GUIDE.md)** | Demo scenario usage | Creating custom demo scenarios |
| **[SESSION_INTEGRATION_COMPLETE.md](SESSION_INTEGRATION_COMPLETE.md)** | Latest session work | Understanding recent changes |
| **[SETUP_DEMO.md](SETUP_DEMO.md)** | Quick start guide | Getting started quickly |
| **[TFT_MODEL_INTEGRATION.md](TFT_MODEL_INTEGRATION.md)** | Model integration details | Understanding how model works |

### Archived Documentation

Historical documents moved to [archive/](archive/) directory. See [archive/README.md](archive/README.md) for details.

---

## 🎯 Documentation by Topic

### Getting Started
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - Main reference
- [SETUP_DEMO.md](SETUP_DEMO.md) - Quick start
- [PYTHON_ENV.md](PYTHON_ENV.md) - Environment setup

### Demos & Scenarios
- [DEMO_README.md](DEMO_README.md) - Demo guide
- [SCENARIO_GUIDE.md](SCENARIO_GUIDE.md) - Scenario usage

### Model & Training
- [TFT_MODEL_INTEGRATION.md](TFT_MODEL_INTEGRATION.md) - Model integration
- [DATA_LOADING_IMPROVEMENTS.md](DATA_LOADING_IMPROVEMENTS.md) - Performance optimization
- [ALL_PHASES_COMPLETE.md](ALL_PHASES_COMPLETE.md) - Training improvements

### Operations & Maintenance
- [OPERATIONAL_MAINTENANCE_GUIDE.md](OPERATIONAL_MAINTENANCE_GUIDE.md) - Full ops guide
- [MAINTENANCE_QUICK_REFERENCE.md](MAINTENANCE_QUICK_REFERENCE.md) - Quick reference
- [ONLINE_LEARNING_DESIGN.md](ONLINE_LEARNING_DESIGN.md) - Future enhancements

### Project Management
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [REPOMAP.md](REPOMAP.md) - Repository map
- [PROJECT_CLEANUP_SUMMARY.md](PROJECT_CLEANUP_SUMMARY.md) - Cleanup decisions
- [SESSION_INTEGRATION_COMPLETE.md](SESSION_INTEGRATION_COMPLETE.md) - Latest work

### For Claude Assistant 🤖
- [CLAUDE_SESSION_GUIDELINES.md](CLAUDE_SESSION_GUIDELINES.md) - Session management protocol
  - **Must read at session start and end**
  - Time tracking requirements
  - Session summary templates
  - Handoff procedures

---

## 📊 Documentation Stats

- **Active Documents:** 18
- **Archived Documents:** 14
- **Total Documentation:** ~200 KB
- **Last Updated:** 2025-10-10

---

## 🔍 Search Tips

Looking for something specific?

- **Model loading** → [TFT_MODEL_INTEGRATION.md](TFT_MODEL_INTEGRATION.md)
- **Fast data loading** → [DATA_LOADING_IMPROVEMENTS.md](DATA_LOADING_IMPROVEMENTS.md)
- **Demo scenarios** → [SCENARIO_GUIDE.md](SCENARIO_GUIDE.md)
- **Production deployment** → [OPERATIONAL_MAINTENANCE_GUIDE.md](OPERATIONAL_MAINTENANCE_GUIDE.md)
- **Environment issues** → [PYTHON_ENV.md](PYTHON_ENV.md)
- **What changed** → [CHANGELOG.md](CHANGELOG.md)
- **File structure** → [REPOMAP.md](REPOMAP.md)
- **Everything else** → [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

---

## 📝 Contributing to Documentation

When adding new documentation:

1. Update this index
2. Add link in appropriate section
3. Update [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) if it affects current state
4. Update [CHANGELOG.md](CHANGELOG.md) for version changes

---

**Last Updated:** 2025-10-10
**Maintainer:** Claude Code Assistant
