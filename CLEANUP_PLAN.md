# Repository Cleanup Plan

**Date:** 2025-10-19
**Version:** 1.0.0
**Status:** Ready to Execute
**Estimated Time:** 2-5 minutes
**Space Savings:** 3.7 GB

---

## Quick Start

**To execute cleanup:**

```bash
# Run the automated cleanup script
CLEANUP_REPO.bat

# This will:
# 1. Delete all duplicate files (3.7 GB)
# 2. Consolidate documentation
# 3. Create deprecation notice
# Total time: 2-5 minutes
```

**Safety:** Git tag `v1.1.0-pre-cleanup` created for rollback.

---

## What Will Be Removed

### 1. Duplicate Models Directory (2.1 GB)
```
❌ models/                          (entire directory)
   ├── tft_model_20251013_100205/  (~500 MB)
   ├── tft_model_20251014_131232/  (~500 MB)
   ├── tft_model_20251015_080653/  (~600 MB)
   └── tft_model_20251017_122454/  (~600 MB)

✅ KEEP: NordIQ/models/ (has same files)
```

### 2. Duplicate Python Files (21 files, ~500 KB)
```
❌ Root Directory Files:
   ├── adaptive_retraining_daemon.py    → NordIQ/src/daemons/
   ├── constants.py                     → NordIQ/src/core/
   ├── data_buffer.py                   → NordIQ/src/core/
   ├── data_validator.py                → NordIQ/src/core/
   ├── demo_data_generator.py           → NordIQ/src/generators/
   ├── demo_stream_generator.py         → NordIQ/src/generators/
   ├── drift_monitor.py                 → NordIQ/src/core/
   ├── generate_api_key.py              → NordIQ/bin/
   ├── gpu_profiles.py                  → NordIQ/src/core/
   ├── main.py                          → NordIQ/src/training/
   ├── metrics_generator.py             → NordIQ/src/generators/
   ├── metrics_generator_daemon.py      → NordIQ/src/daemons/
   ├── precompile.py                    → NordIQ/src/training/
   ├── scenario_demo_generator.py       → NordIQ/src/generators/
   ├── server_encoder.py                → NordIQ/src/core/
   ├── server_profiles.py               → NordIQ/src/core/
   ├── tft_dashboard.py                 (OLD - replaced)
   ├── tft_dashboard_web.py             → NordIQ/src/dashboard/
   ├── tft_inference.py                 (OLD - replaced with daemon)
   ├── tft_inference_daemon.py          → NordIQ/src/daemons/
   └── tft_trainer.py                   → NordIQ/src/training/
```

### 3. Deprecated/Old Files
```
❌ linborg_schema.py                (old schema, deprecated)
❌ run_demo.py                      (replaced by scenario_demo_generator)
❌ tft_dashboard_web.py.backup      (backup file, 138 KB)
```

### 4. Duplicate Directories
```
❌ Dashboard/       → NordIQ/src/dashboard/Dashboard/
❌ adapters/        → NordIQ/src/core/adapters/
❌ explainers/      → NordIQ/src/core/explainers/
❌ tabs/            → NordIQ/src/dashboard/Dashboard/tabs/
❌ utils/           → NordIQ/src/dashboard/Dashboard/utils/
```

### 5. Duplicate Scripts
```
❌ run_daemon.bat           → NordIQ/bin/
❌ setup_api_key.bat/sh     → NordIQ/bin/
❌ start_all.bat/sh         → NordIQ/
❌ stop_all.bat/sh          → NordIQ/
```

### 6. One-Off Validation Scripts
```
❌ run_certification.bat    (one-time use)
❌ validate_pipeline.bat    (one-time use)
❌ validate_schema.bat      (one-time use)
```

### 7. Artifacts
```
❌ nul                              (error artifact, 844 B)
❌ inference_rolling_window.pkl     (old format, have .parquet, 1.7 MB)
```

### 8. Old Model Versions in NordIQ (1.1 GB)
```
Keep only latest 2:
✅ tft_model_20251013_100205  (demo model)
✅ tft_model_20251017_122454  (latest production)

Delete old versions:
❌ tft_model_20251014_131232  (~500 MB)
❌ tft_model_20251015_080653  (~600 MB)
```

---

## What Will Be Moved

### Documentation Consolidation

**Move to Docs/configuration/:**
```
→ CONFIG_GUIDE.md
```

**Move to Docs/security/:**
```
→ DASHBOARD_SECURITY_AUDIT.md
→ SECURITY_ANALYSIS.md
```

**Move to Docs/deployment/:**
```
→ SECURE_DEPLOYMENT_GUIDE.md
→ PRODUCTION_DEPLOYMENT.md
→ STARTUP_GUIDE_CORPORATE.md
```

**Move to Docs/archive/:**
```
→ CONFIGURATION_MIGRATION_COMPLETE.md
→ CLEANUP_COMPLETE.md
→ CORPORATE_LAUNCHER_COMPLETE.md
→ REFACTORING_SUMMARY.md
→ SECURITY_IMPROVEMENTS_COMPLETE.md
→ SILENT_DAEMON_MODE_COMPLETE.md
```

**Move to Docs/:**
```
→ PARQUET_VS_PICKLE_VS_JSON.md
→ GPU_PROFILER_INTEGRATION.md
→ CORPORATE_BROWSER_FIX.md
```

---

## What Will Be Created

### README.DEPRECATED.md
- Explains what was removed
- Provides rollback instructions
- Documents new clean structure

---

## Post-Cleanup Structure

```
MonitoringPrediction/
├── NordIQ/                    # ✅ PRIMARY APPLICATION
│   ├── bin/                   # Utilities
│   ├── src/                   # All source code
│   │   ├── core/              # Shared libraries
│   │   ├── daemons/           # Services
│   │   ├── dashboard/         # Web UI
│   │   ├── generators/        # Data generation
│   │   └── training/          # Model training
│   ├── models/                # Trained models (2 latest only)
│   ├── start_all.bat/sh       # Startup scripts
│   └── stop_all.bat/sh        # Shutdown scripts
│
├── NordIQ-Website/            # Business website
├── Docs/                      # All documentation (organized)
├── BusinessPlanning/          # Business strategy (confidential)
│
├── Development Artifacts:
│   ├── checkpoints/           # Training checkpoints
│   ├── lightning_logs/        # Training logs
│   ├── training/              # Training data
│   ├── logs/                  # Runtime logs
│   ├── data_buffer/           # Buffer data
│   └── plots/                 # Analysis plots
│
├── Repository Files:
│   ├── README.md              # Main README
│   ├── README.DEPRECATED.md   # NEW - Cleanup documentation
│   ├── REPOMAP.md             # Repository map
│   ├── LICENSE                # Business Source License
│   ├── VERSION                # Version number
│   ├── CHANGELOG.md           # Version history
│   ├── _StartHere.ipynb       # Getting started
│   ├── environment.yml        # Conda environment
│   └── .gitignore             # Git configuration
│
└── Utilities (kept):
    ├── test_env.bat           # Environment testing
    ├── install_security_deps.* # Security setup
    └── kill_daemon.ps1        # Daemon utility
```

---

## Safety Measures

### Before Cleanup
✅ **COMPLETED:**
- [x] Created REPOMAP.md (complete file inventory)
- [x] Committed all current work
- [x] Created git tag: `v1.1.0-pre-cleanup`

### Rollback Instructions

If anything goes wrong:

```bash
# Restore all deleted files
git reset --hard v1.1.0-pre-cleanup

# Or restore specific files
git checkout v1.1.0-pre-cleanup -- <file-path>
```

---

## Execution Steps

### 1. Run Cleanup Script
```bash
CLEANUP_REPO.bat
```

**What it does:**
- Deletes duplicate files (automated)
- Moves documentation (automated)
- Creates deprecation notice (automated)
- Shows progress and summary

**Duration:** 2-5 minutes

### 2. Verify Cleanup
```bash
cd NordIQ
start_all.bat
```

**Check:**
- [ ] Inference daemon starts (port 8000)
- [ ] Metrics generator starts (port 8001)
- [ ] Dashboard opens (http://localhost:8501)
- [ ] All 10 tabs load correctly
- [ ] No import errors in console

### 3. Test Application
- [ ] Switch scenarios (healthy → degrading → critical)
- [ ] Verify predictions display
- [ ] Check risk scores calculate correctly
- [ ] Verify no errors in logs

### 4. Commit Cleanup (if successful)
```bash
git add -A
git commit -m "cleanup: remove 3.7GB duplicate files, consolidate on NordIQ structure

- Deleted duplicate models/ directory (2.1 GB)
- Deleted 21 duplicate Python files (500 KB)
- Deleted 5 duplicate directories
- Deleted 10 duplicate scripts
- Deleted 2 old model versions (1.1 GB)
- Cleaned up artifacts (1.9 MB)
- Consolidated documentation (moved 15+ files)
- Created README.DEPRECATED.md

Total space saved: 3.7 GB
See REPOMAP.md for complete cleanup details.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

git tag v1.1.1-post-cleanup
```

---

## Verification Checklist

After cleanup, verify:

### File System
- [ ] `models/` directory deleted (root)
- [ ] All duplicate .py files deleted (root)
- [ ] Duplicate directories deleted (Dashboard/, adapters/, etc.)
- [ ] NordIQ/ directory intact and complete
- [ ] Documentation moved to Docs/ subdirectories
- [ ] README.DEPRECATED.md created

### Application Functionality
- [ ] NordIQ/start_all.bat works
- [ ] Inference daemon loads model correctly
- [ ] Metrics generator streams data
- [ ] Dashboard displays all tabs
- [ ] No import errors
- [ ] Predictions calculate correctly

### Space Savings
```bash
# Check disk usage before/after
du -sh .
# Should be ~3.7 GB smaller
```

---

## Troubleshooting

### If cleanup fails:
```bash
git reset --hard v1.1.0-pre-cleanup
```

### If application won't start after cleanup:
1. Check console for specific errors
2. Verify NordIQ/src/ directory is complete
3. Check that .env file exists in NordIQ/
4. Rollback if needed: `git reset --hard v1.1.0-pre-cleanup`

### If imports fail:
- NordIQ files should have path: `NordIQ/src/...`
- Check that `_path_setup.py` exists in `NordIQ/src/core/`
- Verify __init__.py files exist in all packages

---

## Questions Answered

Based on REPOMAP questions:

1. **Corporate scripts?** → Deleted (NordIQ/start_all.* is sufficient)
2. **Production forwarder?** → Kept as template in root (low priority to move)
3. **Old models?** → Deleted versions before Oct 17
4. **Config directory?** → Kept for reference (not deleted, not moved)
5. **_StartHere.ipynb?** → Will update in next phase (not part of this cleanup)

---

## Next Steps After Cleanup

1. **Update _StartHere.ipynb** - Update paths to reference NordIQ/
2. **Update README.md** - Reflect new structure
3. **Test end-to-end** - Full workflow verification
4. **Create session summary** - Document cleanup work

---

## Summary

**Before Cleanup:**
- 295 files
- ~4.2 GB repository
- Severe duplication (2 complete copies)
- Scattered documentation

**After Cleanup:**
- ~245 files (50 removed)
- ~500 MB repository (3.7 GB saved)
- Single source of truth (NordIQ/)
- Organized documentation

**Risk:** LOW (git tag created, fully reversible)

**Benefit:** HIGH (3.7 GB saved, eliminates confusion, cleaner structure)

---

**Ready to execute!** Run `CLEANUP_REPO.bat` when ready.

**Safety:** Rollback available via `git reset --hard v1.1.0-pre-cleanup`
