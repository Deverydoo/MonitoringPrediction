# Alert Levels - ArgusAI

**Version:** 1.2.1
**Single Source of Truth:** `NordIQ/src/core/alert_levels.py`
**Updated:** 2025-10-18

## Overview

ArgusAI uses a **4-level graduated severity system** for server health alerts. This provides clear, actionable intelligence without alert fatigue.

Risk scores (0-100) are calculated from:
- **70% Current State** - "What's on fire NOW?"
- **30% Predictions** - "What will be on fire soon?"

---

## The Four Alert Levels

### 🔴 Critical (70-100)
**Immediate action required - page on-call engineer**

- **Color:** Red `#ff4444` / `rgb(255, 68, 68)`
- **Emoji:** 🔴
- **Threshold:** Risk Score >= 70
- **Response Time:** 5-15 minutes
- **Action:** Page on-call, escalate to CTO if needed

**What it means:**
- Server will fail imminently (within 30 minutes)
- OOM kill, CPU throttling, or disk full imminent
- Compound stress (CPU + Memory + I/O all critical)

**Examples:**
- Memory 98% + climbing + swap active
- CPU 98% + I/O wait 30%
- Disk 99% full + write-heavy workload

---

### 🟠 Warning (40-69)
**Needs attention - investigate within 1 hour**

- **Color:** Orange `#ff9900` / `rgb(255, 153, 0)`
- **Emoji:** 🟠
- **Threshold:** Risk Score >= 40
- **Response Time:** 30-60 minutes
- **Action:** Investigate root cause, prepare remediation

**What it means:**
- Server degrading, will become critical without intervention
- Memory leak detected
- Sustained high load with upward trend

**Examples:**
- Memory 72% → predicted 94% in 30min (memory leak)
- CPU 85% + Memory 90% (compound stress building)
- I/O wait 20% sustained (storage bottleneck)
- **Risk Score 58 = 🟠 Warning** ← Your example!

---

### 🟡 Watch (20-39)
**Minor concerns - trending upward**

- **Color:** Yellow `#ffcc00` / `rgb(255, 204, 0)`
- **Emoji:** 🟡
- **Threshold:** Risk Score >= 20
- **Response Time:** 1-2 hours (background monitoring)
- **Action:** Monitor for escalation, check dashboards

**What it means:**
- Early warning indicators
- Metrics elevated but stable
- Worth watching, not yet actionable

**Examples:**
- CPU 70% steady (batch job running)
- Memory 85% on database (normal - page cache)
- Load average 8 (high but not critical)

---

### 🟢 Healthy (0-19)
**Normal operations - no action needed**

- **Color:** Green `#44ff44` / `rgb(68, 255, 68)`
- **Emoji:** 🟢
- **Threshold:** Risk Score >= 0
- **Response Time:** N/A
- **Action:** None

**What it means:**
- Server operating normally
- All metrics within acceptable ranges
- No alerts, no concerns

---

## Usage in Code

### Import the Module

```python
from core.alert_levels import (
    get_alert_level,
    get_alert_color,
    get_alert_emoji,
    get_alert_label,
    format_risk_display,
    AlertLevel
)
```

### Get Alert Level

```python
risk_score = 58
level = get_alert_level(risk_score)
# Returns: AlertLevel.WARNING
```

### Get Color

```python
# Hex format (default)
color = get_alert_color(58)
# Returns: '#ff9900'

# RGB format
color = get_alert_color(58, format="rgb")
# Returns: (255, 153, 0)
```

### Get Emoji

```python
emoji = get_alert_emoji(58)
# Returns: '🟠'
```

### Get Label

```python
# Default style
label = get_alert_label(58)
# Returns: 'Warning'

# Short style
label = get_alert_label(58, style="short")
# Returns: 'WARN'

# Action style
label = get_alert_label(58, style="action")
# Returns: 'Investigate'
```

### Format for Display

```python
# Full format with emoji
display = format_risk_display(58)
# Returns: '🟠 Warning'

# Without emoji
display = format_risk_display(58, include_emoji=False)
# Returns: 'Warning'

# Action-oriented
display = format_risk_display(58, label_style="action")
# Returns: '🟠 Investigate'
```

---

## Integration Points

### Dashboard (`NordIQ/src/dashboard/`)
All dashboard tabs should use `get_alert_color()` for risk score visualization:

```python
from core.alert_levels import get_alert_color, format_risk_display

# Color coding
st.markdown(f"<div style='color: {get_alert_color(risk_score)}'>{risk_score}</div>")

# Full display
st.write(format_risk_display(risk_score))
```

### Risk Scoring (`NordIQ/src/dashboard/Dashboard/utils/risk_scoring.py`)
The `get_risk_color()` function now delegates to the centralized system for backward compatibility.

### Website (`NordIQ-Website/how-it-works.html`)
Documentation matches the 4-level system:
- Critical: 70-100
- Warning: 40-69
- Watch: 20-39
- Healthy: 0-19

---

## Why 4 Levels?

### ✅ Clear Escalation Path
- Healthy → Watch → Warning → Critical
- Each level has distinct actions
- No ambiguity about response time

### ✅ Avoids Alert Fatigue
- Only 2 actionable levels (Warning, Critical)
- Watch = informational, not urgent
- Healthy = silence (no noise)

### ✅ Executive-Friendly
- Simple color coding (traffic light + orange)
- Consistent across all views
- Easy to explain to non-technical stakeholders

### ❌ Why Not 7 Levels?
The website previously showed 7 levels (Healthy, Watch, Degrading, Warning, Danger, Critical, Imminent). This caused:
- **Confusion** - Too many levels to remember
- **Inconsistency** - Code used 4 levels, website showed 7
- **Alert fatigue** - Too many "yellow" states
- **Overlapping definitions** - Degrading vs Warning vs Danger

---

## Historical Context

### Previous Inconsistency (Fixed Oct 18, 2025)
- **Code:** 4 levels (Healthy, Watch, Warning, Critical)
- **Website:** 7 levels (caused confusion)
- **Example Bug:** "Risk Score: 58 (🟢 Degrading)" showed GREEN instead of ORANGE

### Current System (v1.2.1)
- **Single Source:** `core/alert_levels.py`
- **Consistent:** Code + Website + Docs all match
- **Validated:** All thresholds tested
- **Correct:** Risk Score 58 = 🟠 Warning

---

## Testing

Run validation:
```bash
cd NordIQ/src
python -m core.alert_levels
```

Output:
```
======================================================================
NordIQ Alert Levels System v1.2.1
======================================================================
[OK] Alert system validation passed

======================================================================
ALERT LEVELS:
======================================================================
🟢 Healthy     >=   0 | Color: #44ff44
🟡 Watch       >=  20 | Color: #ffcc00
🟠 Warning     >=  40 | Color: #ff9900
🔴 Critical    >=  70 | Color: #ff4444

======================================================================
EXAMPLE: Risk Score 58 (User's Question)
======================================================================
Level: AlertLevel.WARNING
Color: #ff9900
Emoji: 🟠
Label: Warning
Display: 🟠 Warning
Display (action): 🟠 Investigate
```

---

## References

- **Implementation:** `NordIQ/src/core/alert_levels.py`
- **Risk Calculation:** `NordIQ/src/dashboard/Dashboard/utils/risk_scoring.py`
- **Website Docs:** `NordIQ-Website/how-it-works.html` (Step 4)
- **Dashboard Integration:** All tabs under `NordIQ/src/dashboard/Dashboard/tabs/`

---

## FAQ

**Q: Why is Risk Score 58 orange, not green?**
A: Risk Score 58 falls in the Warning range (40-69). This indicates significant issues requiring investigation within 1 hour. Green is only for scores 0-19 (Healthy).

**Q: Can I customize the thresholds?**
A: Thresholds are hardcoded in `alert_levels.py`. Changing them requires updating the module and retraining documentation.

**Q: What if I want more granular levels?**
A: The 4-level system is intentionally simple. If you need more detail, use the raw risk score (0-100) and create custom logic in your integration.

**Q: Why are databases different at 98% memory?**
A: Database servers use high memory for page cache (normal behavior). The risk scoring engine in `risk_scoring.py` has profile-specific thresholds to handle this context.

---

**Last Updated:** 2025-10-18
**Maintainer:** Craig Giannelli (ArgusAI)
**Status:** Production - Single Source of Truth
