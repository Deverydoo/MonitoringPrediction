# Dash PoC Performance Optimization Log

**Target:** <500ms page loads (vs 12s Streamlit)
**Initial Result:** 5642ms (11× slower than target!)

---

## Optimization Round 1: Eliminate Duplicate Risk Calculations

### Problem Identified
Each render function was recalculating risk scores independently:
- `render_overview()` calculated 90 risk scores
- `render_heatmap()` calculated 90 risk scores
- `render_top_risks()` calculated 90 risk scores

**Result:** 270 risk calculations per page load (3× wasteful!)

### Fix Applied
1. Changed render functions to accept pre-calculated `risk_scores` parameter
2. Calculate risk scores ONCE in `render_tab()` callback
3. Pass pre-calculated scores to all render functions

**Files Modified:**
- `dash_poc.py` lines 376, 483, 532 (function signatures)
- `dash_poc.py` lines 337-355 (central calculation)

**Expected Improvement:** 3× faster (1800ms → 600ms estimated)

---

## Optimization Round 2: Added Performance Instrumentation

### Instrumentation Added
Added timing checkpoints to identify bottleneck:

```python
[PERF] Risk calculation: XXXms for N servers
[PERF] Tab rendering (overview): XXXms
[PERF] TOTAL render time: XXXms (Target: <500ms)
```

**Purpose:** Identify whether bottleneck is:
- Risk score calculation (CPU-bound loop)
- Chart creation (Plotly figure generation)
- DataFrame operations (Pandas processing)

---

## Testing Instructions

### Run Optimized Version
```bash
cd NordIQ
python dash_poc.py
```

### Check Performance Logs
Terminal will show timing breakdown:
```
[PERF] Risk calculation: 450ms for 90 servers
[PERF] Tab rendering (overview): 200ms
[PERF] TOTAL render time: 650ms (Target: <500ms)
```

### Interpret Results

**If risk calculation is slow (>300ms):**
- 90 servers × 5ms each = 450ms baseline
- Options:
  1. Optimize risk calculation logic
  2. Calculate only visible servers (top 15/30)
  3. Use faster computation (numba, vectorization)

**If tab rendering is slow (>200ms):**
- Likely Plotly chart creation bottleneck
- Options:
  1. Reduce data points (top 15 instead of 30)
  2. Simplify chart layouts
  3. Use Plotly's `Scattergl` for large datasets

**If both are reasonable but total >500ms:**
- Cumulative overhead from both
- May need to accept 600-800ms (still 15× faster than Streamlit!)

---

## Performance Targets (Revised)

| Scenario | Target | Acceptable | Unacceptable |
|----------|--------|------------|--------------|
| Risk Calculation (90 servers) | <200ms | <400ms | >500ms |
| Tab Rendering (Overview) | <200ms | <300ms | >400ms |
| **TOTAL Page Load** | **<500ms** | **<800ms** | **>1000ms** |

**Note:** Even 800ms is **15× faster** than Streamlit's 12s, so migration still valuable if we hit 500-800ms range.

---

## Next Optimizations (If Needed)

### Option 1: Lazy Risk Calculation
Only calculate risk scores for servers actually displayed:
- Overview: Top 15 for bar chart
- Heatmap: Top 30 for heatmap
- Top Risks: Top 10 only

**Expected Gain:** Calculate 30 instead of 90 = 3× faster risk calc

### Option 2: Vectorized Risk Calculation
Use NumPy arrays instead of Python loops:
```python
# Convert to NumPy arrays
cpu_values = np.array([extract_cpu_used(p) for p in preds.values()])
risk_scores = vectorized_risk_calculation(cpu_values, mem_values, ...)
```

**Expected Gain:** 5-10× faster risk calculation

### Option 3: Simplify Charts
- Reduce bar chart from 15 servers to 10
- Use simpler Plotly layouts (no fancy formatting)
- Pre-compute chart data structures

**Expected Gain:** 20-30% faster rendering

### Option 4: Web Workers (Advanced)
Move risk calculation to JavaScript web worker:
- Calculate risk scores in browser
- Dash callback only renders charts

**Expected Gain:** 50%+ faster (parallel processing)

---

## Decision Matrix

**If we achieve <500ms:** ✅ Proceed with full Dash migration (3-4 weeks)

**If we achieve 500-800ms:** ⚠️ Discuss with user - still 15× faster, worth migrating?

**If we're still >1000ms:** ❌ Re-evaluate - maybe data volume is the issue, not framework

---

## Status: Testing Required

**Changes Committed:** ✅ Optimization Round 1 + Instrumentation
**Testing Status:** ⏳ Awaiting user test run
**Next Step:** Analyze performance logs to identify bottleneck

---

**Document Version:** 1.0
**Date:** October 29, 2025
**Last Updated:** Optimization Round 1 complete
