# Deprecated Scripts Archive

**Archive Date:** 2025-10-17
**Version:** v1.0.0 Release Cleanup

---

## 📁 What's Here

This directory contains scripts that are **no longer needed** after the v1.0.0 release and refactoring. These scripts served their purpose during development and validation but are no longer part of the active production system.

---

## 🗂️ Directory Structure

### `validation/` - One-Off Validation Scripts

These scripts were used during development to validate the system but are no longer needed now that the system is stable and certified.

| Script | Purpose | When Used | Status |
|--------|---------|-----------|--------|
| `debug_data_flow.py` | Debug data pipeline flow | During v1.0.0 refactor | ✅ Validated |
| `debug_live_feed.py` | Debug live streaming | During daemon development | ✅ Works |
| `verify_linborg_streaming.py` | Verify LINBORG metrics streaming | During LINBORG integration | ✅ Certified |
| `validate_linborg_schema.py` | Validate LINBORG schema compliance | During schema design | ✅ Compliant |
| `verify_refactor.py` | Verify v1.0.0 modular refactor | Post-refactor validation | ✅ Verified |
| `PIPELINE_VALIDATION.py` | End-to-end pipeline validation | During integration testing | ✅ Passed |
| `end_to_end_certification.py` | Full system certification | Pre-production certification | ✅ Certified |

### `security/` - Already Applied Security Patches

| Script | Purpose | When Applied | Status |
|--------|---------|--------------|--------|
| `apply_security_fixes.py` | Applied security hardening patches | 2025-10-17 | ✅ Applied |

**Security patches included:**
- Input validation for all daemon endpoints
- Path traversal prevention
- API key authentication (X-API-Key header)
- Rate limiting and request size limits
- Safe file operations with restricted paths

**All patches are now part of the production code** in:
- `tft_inference_daemon.py`
- `metrics_generator_daemon.py`
- `Dashboard/` modules

---

## ⚠️ Do Not Use These Scripts

These scripts are **archived only** and should not be used in production:

❌ **Reasons:**
1. **Already Validated** - Validation scripts already did their job
2. **Patches Applied** - Security fixes are already in production code
3. **Obsolete** - System architecture has evolved beyond these scripts
4. **Unmaintained** - These scripts are no longer kept up to date

✅ **Use Instead:**
- **For validation:** Run `python main.py status` or use the production test suite
- **For debugging:** Check production logs and monitoring in the dashboard
- **For security:** All security features are built into the daemons

---

## 🔄 If You Need to Reference These Scripts

**Purpose:** Historical reference only

**Use Cases:**
- Understanding how the system was validated during development
- Reference for security patches that were applied
- Learning how the pipeline was debugged

**Do NOT:**
- Run these scripts against the production system
- Rely on these scripts for current validation
- Use these instead of production monitoring

---

## 📚 Modern Replacements

| Old Script (archived) | Modern Replacement |
|-----------------------|-------------------|
| `debug_data_flow.py` | Dashboard → Advanced tab → System diagnostics |
| `debug_live_feed.py` | Dashboard → Overview tab → Live metrics |
| `verify_linborg_streaming.py` | Dashboard → Overview tab → Metrics display |
| `validate_linborg_schema.py` | `data_validator.py` (still in root) |
| `verify_refactor.py` | N/A - Refactor complete, tests pass |
| `PIPELINE_VALIDATION.py` | `python main.py status` |
| `end_to_end_certification.py` | Production monitoring + tests |
| `apply_security_fixes.py` | Security built into daemons |

---

## 🗑️ Future Cleanup

**Retention Policy:**
- Keep archived for **2 release cycles** (v1.0.0 → v2.0.0)
- After v2.0.0: Consider permanent removal
- Git history will preserve these files forever if needed

**To Restore (if needed):**
```bash
# Copy from archive
cp scripts/deprecated/validation/debug_data_flow.py .

# Or restore from git
git log --all --full-history -- "**/debug_data_flow.py"
git checkout <commit> -- debug_data_flow.py
```

---

## ✅ System Status After Archiving

**Production System:** ✅ Fully Operational

- Inference Daemon: ✅ Running
- Metrics Generator: ✅ Running
- Dashboard: ✅ Running
- All Tests: ✅ Passing
- Security: ✅ Hardened
- Validation: ✅ Complete

**No production functionality was removed** - only one-off validation scripts.

---

## 📖 Documentation

For current system documentation, see:
- [README.md](../../README.md) - Project overview
- [Docs/SCRIPT_DEPRECATION_ANALYSIS.md](../../Docs/SCRIPT_DEPRECATION_ANALYSIS.md) - Full deprecation analysis
- [_StartHere.ipynb](../../_StartHere.ipynb) - Training and setup guide

---

**Archive maintained by:** TFT Monitoring System Team
**Last Updated:** 2025-10-17
**Version:** v1.0.0+
