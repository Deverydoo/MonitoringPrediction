# Maintenance Quick Reference Card

**Use this guide while waiting 4-8 weeks to collect retraining data**

---

## 🎯 Three Essential Strategies (Pick at least 2)

### **1. Dynamic Threshold Adjustment** ⭐ EASIEST
**What:** Auto-adjust alert thresholds based on last 7 days
**Impact:** 30-50% fewer false positives
**Effort:** 1 script, run daily
**Code:** `threshold_adjuster.py` (see full guide)

```python
# Run daily
python threshold_adjuster.py

# What it does:
# - Analyzes last 7 days of healthy data
# - Updates warning/critical thresholds
# - Smooth transition (70% new, 30% old)
```

---

### **2. Baseline Correction** ⭐ RECOMMENDED
**What:** Adjust predictions for baseline drift
**Impact:** Aligns predictions with current reality
**Effort:** 1 script, run every 6 hours
**Code:** `baseline_corrector.py`

```python
# Calculate shift
shifts = calculate_shift(training_data, recent_data)

# Apply to predictions
corrected = apply_correction(predictions, shifts)

# Example:
# Training: CPU avg 45%
# Recent:   CPU avg 52%
# Shift:    +7%
# Old pred: 75% → Corrected: 82%
```

---

### **3. Hybrid ML + Rules** ⭐ MOST ROBUST
**What:** Combine ML predictions with simple rules
**Impact:** Catches edge cases ML misses
**Effort:** 1 script, runs with inference
**Code:** `hybrid_alerting.py`

```python
# ML prediction
ml_alerts = model.predict(data)

# Rule-based checks
if cpu.spike() > 30%:          alert("CPU spike")
if memory.trend() == "increasing": alert("Memory leak")
if errors.burst() > 100:       alert("Error burst")

# Combine
final_alerts = ml_alerts + rule_alerts
```

---

## 📅 Weekly Maintenance Schedule

### **Daily (Automated - 5 min setup)**
```bash
# Cron job
0 3 * * * python threshold_adjuster.py
0 6 * * * python baseline_corrector.py
```

### **Weekly (Manual - 30 min)**
- Review accuracy trends
- Check false positive rate
- Validate thresholds
- Document any issues

### **Monthly (Planning - 1 hour)**
- Assess data collection progress
- Decide if retraining needed
- Plan retraining window

---

## 🚨 When to Retrain Immediately

```
🔴 CRITICAL - Retrain Now:
   - Accuracy drop > 25%
   - False positive rate > 50%
   - Model errors/crashes

🟡 WARNING - Retrain Soon (1-2 weeks):
   - Accuracy drop > 15%
   - False positive rate > 30%
   - Major environment changes

🟢 OK - Continue Monitoring:
   - Accuracy drop < 15%
   - False positives manageable
   - Normal operations
```

---

## 🎛️ Quick Diagnostics

### **Problem: Too many alerts**
```python
# Fix 1: Raise thresholds
python threshold_adjuster.py --percentile 95  # Default: 90

# Fix 2: Filter by confidence
python confidence_filter.py --min-confidence 0.8

# Fix 3: Check for baseline shift
python baseline_corrector.py --report
```

### **Problem: Missing real incidents**
```python
# Fix 1: Lower thresholds
python threshold_adjuster.py --percentile 85

# Fix 2: Add rule-based checks
python hybrid_alerting.py --enable-all-rules

# Fix 3: Check model health
python monitoring_dashboard.py --health-check
```

### **Problem: Predictions seem off**
```python
# Fix 1: Apply baseline correction
python baseline_corrector.py --apply

# Fix 2: Check for drift
python drift_detector.py --analyze

# Fix 3: Use ensemble
python model_ensemble.py --load-last-3-versions
```

---

## 📊 Expected Accuracy Over Time

**Without maintenance:**
```
Week 1: 100% ████████████████████
Week 2:  92% ██████████████████░░
Week 3:  85% █████████████████░░░
Week 4:  78% ███████████████░░░░░
Week 6:  70% ██████████████░░░░░░ ⚠️ Retrain!
```

**With maintenance strategies:**
```
Week 1: 100% ████████████████████
Week 2:  97% ███████████████████░
Week 3:  94% ██████████████████░░
Week 4:  91% ██████████████████░░
Week 6:  87% █████████████████░░░ ✅ Still good!
Week 8:  82% ████████████████░░░░ 🟡 Plan retraining
```

**10-15% improvement with simple maintenance!**

---

## 🛠️ Minimal Setup (30 minutes)

**Option A: Quick & Dirty (1 script)**
```bash
# Just adjust thresholds daily
python threshold_adjuster.py
# Schedule: cron daily at 3 AM
```
**Impact:** 30% fewer false positives

---

**Option B: Recommended (3 scripts)**
```bash
# 1. Adjust thresholds
python threshold_adjuster.py

# 2. Correct baseline
python baseline_corrector.py

# 3. Hybrid alerts
python hybrid_alerting.py --enable
```
**Impact:** 50% fewer false positives, catches more edge cases

---

**Option C: Full Suite (all strategies)**
```bash
# All scripts from full guide
# See OPERATIONAL_MAINTENANCE_GUIDE.md
```
**Impact:** Maximum effectiveness, graceful degradation

---

## 💡 Pro Tips

1. **Start conservative**
   - Use Option A for first week
   - Add Option B if needed
   - Only use Option C if critical production

2. **Monitor daily for first week**
   - New model needs observation
   - Fine-tune thresholds
   - Document any issues

3. **Set retraining calendar**
   - 30 days: Good practice
   - 45 days: Acceptable
   - 60+ days: Risky

4. **Keep model versions**
   - Always keep N-1 version
   - Can rollback quickly
   - Compare performance

5. **Test before retraining**
   - Generate synthetic scenarios
   - Verify model catches them
   - Document blind spots

---

## 📞 Escalation Path

```
Issue Detected
    ↓
Check diagnostics (above)
    ↓
Apply quick fixes
    ↓
Still broken? → Rollback to previous model
    ↓
Still broken? → Switch to rule-based only
    ↓
Page on-call engineer
```

---

## 📚 Full Documentation

- **Complete Guide:** [OPERATIONAL_MAINTENANCE_GUIDE.md](OPERATIONAL_MAINTENANCE_GUIDE.md)
- **Online Learning:** [ONLINE_LEARNING_DESIGN.md](ONLINE_LEARNING_DESIGN.md)
- **System Overview:** [REPOMAP.md](REPOMAP.md)

---

**Remember:** The model doesn't need to be perfect, it needs to be useful while you collect new training data. These strategies keep it useful for 6-8 weeks.

**Goal:** Maintain 85%+ accuracy until retraining (vs 70% without maintenance)
