# Dashboard Metric Consistency Analysis
**Date**: 2025-10-14
**Purpose**: Pre-presentation executive review - ensure all metrics align

---

## Executive Summary

**CRITICAL ISSUE FOUND**: Heatmap tab (Tab 2) uses **legacy metric names** that don't exist in LINBORG system.

**Impact**: Heatmap will show all zeros during executive presentation, which would undermine credibility.

**Status**:
- ✅ Overview Tab (Tab 1): CORRECT - Uses LINBORG metrics
- 🔴 Heatmap Tab (Tab 2): **BROKEN** - Uses legacy metrics
- ✅ Top 5 Problem Servers Tab (Tab 3): CORRECT - Uses LINBORG metrics

---

## Detailed Findings

### ✅ TAB 1: Overview (CORRECT)

**Location**: Lines 833-1310

**Metrics Used**:
```python
# Active Alerts Table (lines 1023-1081)
cpu_idle_pct ✅
cpu_user_pct ✅
cpu_sys_pct ✅
cpu_iowait_pct ✅
mem_used_pct ✅
swap_used_pct ✅
load_average ✅

# Top 5 Busiest Servers (lines 1223-1288)
cpu_idle_pct ✅
cpu_user_pct ✅
cpu_sys_pct ✅
cpu_iowait_pct ✅
mem_used_pct ✅
swap_used_pct ✅
load_average ✅
```

**Risk Calculation Function** (lines 281-435):
```python
cpu_idle_pct ✅
cpu_user_pct ✅
cpu_sys_pct ✅
cpu_iowait_pct ✅
mem_used_pct ✅
swap_used_pct ✅
disk_usage_pct ✅
load_average ✅
```

**Status**: ✅ FULLY ALIGNED WITH LINBORG

**Display Logic**: Correctly calculates "CPU Used = 100 - cpu_idle_pct" for human-friendly display

---

### 🔴 TAB 2: Heatmap (BROKEN)

**Location**: Lines 1312-1439

**CRITICAL ISSUE** (lines 1340-1346):
```python
# WRONG - These metrics DON'T EXIST in LINBORG
elif mk == 'cpu':
    cpu = server_pred.get('cpu_percent', {})  # ❌ DOES NOT EXIST
    p90 = cpu.get('p90', [])
    value = max(p90[:6]) if len(p90) >= 6 else (max(p90) if p90 else 0)
elif mk == 'memory':
    mem = server_pred.get('memory_percent', {})  # ❌ DOES NOT EXIST
    p90 = mem.get('p90', [])
    value = max(p90[:6]) if len(p90) >= 6 else (max(p90) if p90 else 0)
```

**What Exists in LINBORG**:
```python
# These are the ACTUAL metric names
cpu_idle_pct   # (display as: 100 - cpu_idle_pct)
cpu_user_pct
cpu_sys_pct
cpu_iowait_pct
mem_used_pct   # NOT memory_percent
```

**Impact**:
- Heatmap will display **all zeros** because `cpu_percent` and `memory_percent` don't exist
- During executive presentation, this will look like **broken/non-functional system**
- Contradicts working data shown in Tab 1

**Fix Required**: Change to LINBORG metric names

---

### ✅ TAB 3: Top 5 Problem Servers (CORRECT)

**Location**: Lines 1440-1639

**Metrics Used** (lines 1499-1568):
```python
# Server Detail Comparison Table
cpu_idle_pct ✅
cpu_user_pct ✅
cpu_sys_pct ✅
cpu_iowait_pct ✅
mem_used_pct ✅
load_average ✅
```

**Forecast Chart** (lines 1584-1636):
```python
cpu_idle_pct ✅  # Correctly inverted to show CPU Used
```

**Status**: ✅ FULLY ALIGNED WITH LINBORG

**Display Logic**: Correctly calculates "CPU Used = 100 - cpu_idle_pct" for charts

---

## Metric Naming Consistency Analysis

### Overview Tab (Tab 1)
| Section | Metric Display Name | Underlying LINBORG Key | Status |
|---------|-------------------|----------------------|--------|
| Active Alerts | CPU Now | cpu_idle_pct (as 100-idle) | ✅ |
| Active Alerts | I/O Wait Now | cpu_iowait_pct | ✅ |
| Active Alerts | Mem Now | mem_used_pct | ✅ |
| Busiest Servers | CPU | cpu_idle_pct (as 100-idle) | ✅ |
| Busiest Servers | Memory | mem_used_pct | ✅ |
| Busiest Servers | I/O Wait | cpu_iowait_pct | ✅ |
| Busiest Servers | Swap | swap_used_pct | ✅ |
| Busiest Servers | Load | load_average | ✅ |

### Heatmap Tab (Tab 2)
| Section | Metric Display Name | Code Reference | Status |
|---------|-------------------|----------------|--------|
| Heatmap | Risk Score | calculate_server_risk_score() | ✅ |
| Heatmap | CPU (p90) | cpu_percent | 🔴 WRONG |
| Heatmap | Memory (p90) | memory_percent | 🔴 WRONG |
| Heatmap | Latency (p90) | load_average | ✅ |

### Top 5 Problem Servers (Tab 3)
| Section | Metric Display Name | Underlying LINBORG Key | Status |
|---------|-------------------|----------------------|--------|
| Detail View | CPU Used | cpu_idle_pct (as 100-idle) | ✅ |
| Detail View | I/O Wait | cpu_iowait_pct | ✅ |
| Detail View | Memory | mem_used_pct | ✅ |
| Detail View | Load Avg | load_average | ✅ |
| Forecast Chart | CPU Forecast | cpu_idle_pct (inverted) | ✅ |

---

## Color Coding Consistency

**Function**: `get_metric_color_indicator()` (lines 233-279)

**Thresholds Used**:
```python
# CPU
🟡 Warning: >= 90%
🟠 Danger:  >= 95%
🔴 Critical: >= 98%

# I/O Wait (CRITICAL metric)
🟡 Warning: >= 10%
🟠 Danger:  >= 20%
🔴 Critical: >= 30%

# Memory (Generic)
🟡 Warning: >= 90%
🟠 Danger:  >= 95%
🔴 Critical: >= 98%

# Memory (Database profile)
🟡 Warning: >= 95%  # Databases run hot on memory
🟠 Danger:  >= 98%
🔴 Critical: >= 99.5%

# Swap
🟡 Warning: >= 10%
🟠 Danger:  >= 25%
🔴 Critical: >= 50%

# Load Average
🟡 Warning: >= 8.0
🟠 Danger:  >= 12.0
🔴 Critical: >= 16.0
```

**Status**: ✅ CONSISTENT across all tabs that use it (Tab 1 and Tab 3)

**Note**: Tab 2 (Heatmap) uses different color logic (green/yellow/orange/red gradient based on value) which is appropriate for heatmap visualization.

---

## Recommendations

### CRITICAL - Must Fix Before Presentation

1. **Fix Heatmap Tab Metric References** (Line 1340-1346)
   - Change `cpu_percent` → `cpu_idle_pct` (calculate as 100-idle for p90)
   - Change `memory_percent` → `mem_used_pct`
   - Ensure p90 calculations work correctly

### HIGH - Strongly Recommended

2. **Add Metric Tooltips to Heatmap**
   - Explain what p90 means ("90th percentile - worst-case scenario")
   - Clarify CPU is "% Used" not "% Idle"

3. **Test Heatmap with Live Data**
   - Verify heatmap populates correctly after fix
   - Confirm color gradients work as expected

### MEDIUM - Nice to Have

4. **Add Consistency Validation Script**
   - Script to verify all metric references match LINBORG schema
   - Run before any presentation/demo

---

## All LINBORG Metrics Coverage

### Metrics USED in Dashboard
✅ cpu_user_pct - Used in calculation
✅ cpu_sys_pct - Used in calculation
✅ cpu_iowait_pct - **PROMINENTLY DISPLAYED** (CRITICAL metric)
✅ cpu_idle_pct - **CORE DISPLAY** (as 100-idle = CPU Used)
❓ java_cpu_pct - NOT displayed (acceptable - internal detail)
✅ mem_used_pct - **PROMINENTLY DISPLAYED**
✅ swap_used_pct - **PROMINENTLY DISPLAYED**
❓ disk_usage_pct - Used in risk calc, not directly displayed
❓ net_in_mb_s - NOT displayed (acceptable - not critical for demo)
❓ net_out_mb_s - NOT displayed (acceptable - not critical for demo)
❓ back_close_wait - NOT displayed (acceptable - internal detail)
❓ front_close_wait - NOT displayed (acceptable - internal detail)
✅ load_average - **PROMINENTLY DISPLAYED**
❓ uptime_days - NOT displayed (acceptable - informational only)

**Coverage**: 8/14 metrics prominently displayed, 3/14 used in calculations, 3/14 not displayed but acceptable for executive demo

### Metrics NOT in Dashboard but Should Consider
- Network metrics (net_in_mb_s, net_out_mb_s) - Could add to "busiest servers" view
- Disk usage (disk_usage_pct) - Could add to detail view
- Connection states (back_close_wait, front_close_wait) - More for troubleshooting than executive view

---

## Testing Checklist

Before presentation, verify:

- [ ] Heatmap Tab shows actual data (not all zeros)
- [ ] Heatmap CPU values match Overview tab CPU values
- [ ] Heatmap Memory values match Overview tab Memory values
- [ ] Color coding is consistent between tabs
- [ ] All metric labels are clear and professional
- [ ] "CPU" always means "CPU Used" (not idle) in display
- [ ] I/O Wait is highlighted as CRITICAL metric
- [ ] Risk scores are consistent across all tabs
- [ ] No references to old metric names (cpu_pct, memory_pct, cpu_percent, memory_percent)

---

**Prepared for Corporate Presentation Review**
**Next Action**: Fix Heatmap tab metric references (lines 1340-1346)
