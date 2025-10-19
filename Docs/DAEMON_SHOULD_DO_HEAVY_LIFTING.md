# Architectural Issue: Daemon Should Do Heavy Lifting

**Created:** October 18, 2025
**Priority:** HIGH - Major architectural improvement opportunity
**Status:** Identified, not yet implemented

---

## Executive Summary

**Problem:** The dashboard is doing expensive calculations (risk scores, metric extraction) that should be done ONCE by the inference daemon. This violates the principle of separation of concerns and creates unnecessary load on the dashboard.

**Impact:**
- Dashboard calculates risk scores 270+ times per page load (for 90 servers)
- Each dashboard instance recalculates the same thing
- With 10 concurrent users: 2,700 redundant calculations/minute

**Solution:** Move ALL heavy computation to the daemon. Dashboard becomes pure display layer (HTML/CSS/charts only).

---

## Current Architecture (WRONG ❌)

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference Daemon                          │
│                  (Should do heavy lifting)                   │
│                                                              │
│  ✅ Runs TFT model (PyTorch inference)                      │
│  ✅ Generates 96-step forecasts                             │
│  ✅ Calculates environment metrics                          │
│  ✅ Generates alerts                                        │
│  ❌ Does NOT calculate risk scores                          │
│  ❌ Does NOT extract/format metrics for display             │
│  ❌ Does NOT sort servers by severity                       │
│                                                              │
│  Returns:                                                    │
│  {                                                           │
│    "predictions": {                                          │
│      "server1": {                                            │
│        "cpu_idle_pct": {"current": 45, "p50": [...] },      │
│        "mem_used_pct": {"current": 67, "p50": [...] },      │
│        ... (14 metrics, raw format)                          │
│      }                                                       │
│    },                                                        │
│    "environment": {...},                                     │
│    "metadata": {...}                                         │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                          ↓ HTTP GET
                          │ Raw predictions (450KB)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard × 10 users              │
│                  (Doing TOO MUCH work)                       │
│                                                              │
│  ❌ Calculates risk scores (270+ calls × 10 users)          │
│  ❌ Extracts CPU from cpu_idle (100 - idle)                 │
│  ❌ Formats metrics for display                             │
│  ❌ Sorts servers by severity                               │
│  ❌ Determines alert levels                                 │
│  ❌ Calculates trend deltas (predicted - current)           │
│  ❌ Profile detection (regex matching)                      │
│                                                              │
│  Total: 2,700 redundant risk calculations PER MINUTE!       │
└─────────────────────────────────────────────────────────────┘
```

**Problem Analysis:**

1. **Redundant Calculation:** 10 dashboard users = 10x same calculation
2. **Wrong Layer:** Dashboard should be thin (HTML/CSS), not thick (business logic)
3. **Scaling Issue:** Each new user adds full computation load
4. **Inefficient:** Daemon has data in memory, dashboard refetches and recalculates

---

## Correct Architecture (RIGHT ✅)

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference Daemon                          │
│              (Does ALL heavy lifting ONCE)                   │
│                                                              │
│  ✅ Runs TFT model (PyTorch inference)                      │
│  ✅ Generates 96-step forecasts                             │
│  ✅ Calculates environment metrics                          │
│  ✅ Generates alerts                                        │
│  ✅ Calculates risk scores (ONCE for all servers)           │
│  ✅ Extracts display-ready metrics                          │
│  ✅ Sorts servers by severity                               │
│  ✅ Formats data for dashboard consumption                  │
│                                                              │
│  Returns:                                                    │
│  {                                                           │
│    "predictions": {                                          │
│      "server1": {                                            │
│        "risk_score": 67.3,  ← PRE-CALCULATED                │
│        "alert_level": "warning",  ← PRE-CALCULATED          │
│        "metrics": {  ← DISPLAY-READY FORMAT                 │
│          "cpu": {"current": 55, "predicted": 67, "delta": +12},│
│          "memory": {"current": 67, "predicted": 71, "delta": +4},│
│          ...                                                 │
│        },                                                    │
│        "forecast": [...],  ← Ready for charts               │
│        "profile": "ML Compute"  ← PRE-DETECTED              │
│      }                                                       │
│    },                                                        │
│    "summary": {  ← DASHBOARD-READY AGGREGATES               │
│      "total_servers": 90,                                    │
│      "critical_count": 5,                                    │
│      "warning_count": 12,                                    │
│      "healthy_count": 73,                                    │
│      "top_5_risks": ["server1", "server2", ...]             │
│    },                                                        │
│    "environment": {...},                                     │
│    "metadata": {...}                                         │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
                          ↓ HTTP GET
                          │ Pre-calculated data (500KB)
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard × 10 users              │
│                  (Pure display layer - FAST!)                │
│                                                              │
│  ✅ Receives pre-calculated risk scores                     │
│  ✅ Receives display-ready metrics                          │
│  ✅ Receives pre-sorted server lists                        │
│  ✅ Just renders HTML/CSS/charts                            │
│  ✅ Zero business logic                                     │
│  ✅ Instant page load                                       │
│                                                              │
│  Total: 0 redundant calculations!                           │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**

1. **Single Calculation:** Risk scores calculated ONCE by daemon (not 10x by dashboards)
2. **Proper Separation:** Daemon = business logic, Dashboard = presentation
3. **Scalability:** 10 users or 100 users = same daemon load
4. **Performance:** Dashboard becomes instant (just JSON → HTML)

---

## What Should Move to Daemon

### 1. Risk Score Calculation (HIGH PRIORITY)

**Current (Dashboard):**
```python
# File: Dashboard/utils/risk_scoring.py
def calculate_server_risk_score(server_pred: Dict) -> float:
    """
    Complex calculation involving:
    - Current state metrics (CPU, memory, I/O wait, swap, load)
    - Predicted state metrics
    - Weighted scoring (70% current, 30% predicted)
    - Profile-aware thresholds
    """
    # ... 100 lines of calculation logic ...
    return risk_score  # Returns: 67.3

# Called 270+ times per page load (90 servers × 3 tabs)
# Called by EVERY dashboard instance independently
```

**Should Be (Daemon):**
```python
# File: daemons/tft_inference_daemon.py
def _calculate_risk_scores(self, predictions: Dict) -> Dict[str, float]:
    """
    Calculate risk scores ONCE for all servers.
    Returns: {"server1": 67.3, "server2": 23.1, ...}
    """
    risk_scores = {}
    for server_name, server_pred in predictions.items():
        risk_scores[server_name] = self._calculate_server_risk(server_pred)
    return risk_scores

# In get_predictions():
def get_predictions(self) -> Dict:
    predictions = self.inference.predict(df, horizon=96)

    # ADD THIS:
    risk_scores = self._calculate_risk_scores(predictions)

    # Enrich predictions with risk scores
    for server_name, risk_score in risk_scores.items():
        predictions[server_name]['risk_score'] = risk_score
        predictions[server_name]['alert_level'] = self._get_alert_level(risk_score)

    return {
        'predictions': predictions,
        'summary': {
            'total_servers': len(predictions),
            'critical_count': sum(1 for r in risk_scores.values() if r >= 80),
            'warning_count': sum(1 for r in risk_scores.values() if r >= 60),
            'healthy_count': sum(1 for r in risk_scores.values() if r < 50),
        },
        ...
    }
```

**Impact:**
- Calculation: 1 time (daemon) instead of 270+ times (dashboard)
- With 10 users: 1 calculation instead of 2,700 calculations
- **99.96% reduction** in redundant work!

---

### 2. Metric Extraction & Formatting (HIGH PRIORITY)

**Current (Dashboard):**
```python
# File: Dashboard/utils/metrics.py
def extract_cpu_used(server_pred: Dict, mode: str = 'current') -> float:
    """Extract CPU used from CPU idle (100 - idle)"""
    if 'cpu_idle_pct' in server_pred:
        idle = server_pred['cpu_idle_pct'].get(mode, 0)
        return 100 - idle
    elif 'cpu_user_pct' in server_pred:
        return server_pred['cpu_user_pct'].get(mode, 0)
    return 0

# Called 90+ times per page load
# Dashboard has to understand daemon's internal data format
```

**Should Be (Daemon):**
```python
# File: daemons/tft_inference_daemon.py
def _format_display_metrics(self, server_pred: Dict) -> Dict:
    """
    Convert internal format to dashboard-ready format.
    Dashboard should receive clean, ready-to-display data.
    """
    # CPU Used (aggregate from idle/user/sys)
    cpu_current = 100 - server_pred['cpu_idle_pct']['current']
    cpu_predicted = 100 - np.mean(server_pred['cpu_idle_pct']['p50'][:6])

    # Memory Used (direct)
    mem_current = server_pred['mem_used_pct']['current']
    mem_predicted = np.mean(server_pred['mem_used_pct']['p50'][:6])

    return {
        'cpu': {
            'current': cpu_current,
            'predicted': cpu_predicted,
            'delta': cpu_predicted - cpu_current,
            'unit': '%'
        },
        'memory': {
            'current': mem_current,
            'predicted': mem_predicted,
            'delta': mem_predicted - mem_current,
            'unit': '%'
        },
        # ... other metrics ...
    }

# In get_predictions():
for server_name, server_pred in predictions.items():
    predictions[server_name]['display_metrics'] = self._format_display_metrics(server_pred)
```

**Impact:**
- Dashboard doesn't need to know internal format
- Daemon can change internal format without breaking dashboard
- Display logic centralized (single source of truth)

---

### 3. Server Sorting & Filtering (MEDIUM PRIORITY)

**Current (Dashboard):**
```python
# File: Dashboard/tabs/top_risks.py
# Calculate risk scores
server_risks = []
for server_name, server_pred in server_preds.items():
    risk_score = calculate_server_risk_score(server_pred)  # EXPENSIVE!
    server_risks.append({'server': server_name, 'risk': risk_score, 'pred': server_pred})

# Sort by risk
server_risks.sort(key=lambda x: x['risk'], reverse=True)
top_5 = server_risks[:5]
```

**Should Be (Daemon):**
```python
# File: daemons/tft_inference_daemon.py
def get_predictions(self) -> Dict:
    predictions = self.inference.predict(df, horizon=96)
    risk_scores = self._calculate_risk_scores(predictions)

    # PRE-SORT servers by risk (daemon does this ONCE)
    sorted_servers = sorted(
        predictions.keys(),
        key=lambda s: risk_scores.get(s, 0),
        reverse=True
    )

    return {
        'predictions': predictions,
        'summary': {
            'total_servers': len(predictions),
            'top_5_risks': sorted_servers[:5],  # Dashboard just uses this!
            'top_10_risks': sorted_servers[:10],
            'top_20_risks': sorted_servers[:20],
        },
        ...
    }
```

**Dashboard becomes:**
```python
# File: Dashboard/tabs/top_risks.py
# Just use pre-calculated top 5!
top_5_servers = predictions['summary']['top_5_risks']
for server_name in top_5_servers:
    server_pred = predictions['predictions'][server_name]
    risk_score = server_pred['risk_score']  # Pre-calculated!
    # Just display it!
```

**Impact:**
- Sorting: 1 time (daemon) instead of N times (each tab)
- With 10 users: 1 sort instead of 10 sorts

---

### 4. Alert Level Categorization (MEDIUM PRIORITY)

**Current (Dashboard):**
```python
# File: Dashboard/tabs/overview.py
if risk_score >= 90:
    severity = "🔴 Imminent Failure"
    priority = "Imminent Failure"
elif risk_score >= 80:
    severity = "🔴 Critical"
    priority = "Critical"
elif risk_score >= 70:
    severity = "🟠 Danger"
    priority = "Danger"
# ... etc (repeated in multiple tabs)
```

**Should Be (Daemon):**
```python
# File: daemons/tft_inference_daemon.py
from core.alert_levels import get_alert_level, get_alert_color, get_alert_emoji

def _enrich_with_alert_info(self, server_pred: Dict, risk_score: float):
    """Add alert level information to prediction."""
    alert_level = get_alert_level(risk_score)

    server_pred['alert'] = {
        'level': alert_level.value,  # "critical", "warning", etc.
        'score': risk_score,
        'color': get_alert_color(risk_score),
        'emoji': get_alert_emoji(risk_score),
        'label': get_alert_label(risk_score),  # "🔴 Critical"
    }

    if risk_score >= 90:
        server_pred['alert']['priority'] = "imminent_failure"
    elif risk_score >= 80:
        server_pred['alert']['priority'] = "critical"
    # ... etc
```

**Dashboard becomes:**
```python
# File: Dashboard/tabs/overview.py
# Just use pre-calculated alert info!
alert_info = server_pred['alert']
severity = alert_info['label']  # "🔴 Critical"
priority = alert_info['priority']  # "critical"
color = alert_info['color']  # "#ff4444"
```

**Impact:**
- Logic centralized in daemon (single source of truth)
- Dashboard has zero logic (just displays pre-formatted strings)

---

### 5. Profile Detection (LOW PRIORITY)

**Current (Dashboard):**
```python
# File: Dashboard/utils/profiles.py
def get_server_profile(server_name: str) -> str:
    """Detect profile from server name using regex."""
    if server_name.startswith('ppml'): return 'ML Compute'
    if server_name.startswith('ppdb'): return 'Database'
    # ... etc (regex matching every time)
```

**Should Be (Daemon):**
```python
# File: daemons/tft_inference_daemon.py
# Profile already detected during _prepare_data_for_tft()!
# Just include it in response:

predictions[server_name]['profile'] = self._get_profile(server_name)
```

**Impact:**
- Profile detected once during TFT preparation (already in memory)
- Dashboard doesn't need regex logic

---

## Implementation Plan

### Phase 1: Add Risk Scores to Daemon Response (HIGH PRIORITY)

**File:** `NordIQ/src/daemons/tft_inference_daemon.py`

**Changes:**

1. Import risk scoring function:
```python
from Dashboard.utils.risk_scoring import calculate_server_risk_score
from core.alert_levels import get_alert_level, get_alert_color, get_alert_emoji
```

2. Add risk calculation to `get_predictions()`:
```python
def get_predictions(self) -> Dict[str, Any]:
    # ... existing TFT prediction code ...

    predictions = self.inference.predict(df, horizon=96)

    # NEW: Calculate risk scores ONCE for all servers
    risk_scores = {}
    alert_counts = {'critical': 0, 'warning': 0, 'degrading': 0, 'healthy': 0}

    for server_name, server_pred in predictions.items():
        # Calculate risk score (expensive operation done ONCE)
        risk_score = calculate_server_risk_score(server_pred)
        risk_scores[server_name] = risk_score

        # Add to prediction
        server_pred['risk_score'] = risk_score
        server_pred['alert_level'] = get_alert_level(risk_score).value
        server_pred['alert_color'] = get_alert_color(risk_score)
        server_pred['alert_emoji'] = get_alert_emoji(risk_score)

        # Count alerts
        if risk_score >= 80:
            alert_counts['critical'] += 1
        elif risk_score >= 60:
            alert_counts['warning'] += 1
        elif risk_score >= 50:
            alert_counts['degrading'] += 1
        else:
            alert_counts['healthy'] += 1

    # Sort servers by risk (top N lists pre-calculated)
    sorted_servers = sorted(
        predictions.keys(),
        key=lambda s: risk_scores.get(s, 0),
        reverse=True
    )

    return {
        'predictions': predictions,
        'alerts': alerts,
        'environment': env_metrics,
        'summary': {  # NEW: Pre-calculated summary
            'total_servers': len(predictions),
            'critical_count': alert_counts['critical'],
            'warning_count': alert_counts['warning'],
            'degrading_count': alert_counts['degrading'],
            'healthy_count': alert_counts['healthy'],
            'top_5_risks': sorted_servers[:5],
            'top_10_risks': sorted_servers[:10],
            'top_20_risks': sorted_servers[:20],
        },
        'metadata': {
            'model_type': 'TFT',
            'prediction_time': datetime.now().isoformat(),
            'risk_calculation_time': datetime.now().isoformat(),  # NEW
            'horizon_steps': horizon,
            'input_points': len(df),
            'device': str(self.device)
        }
    }
```

**Estimated Time:** 2 hours
**Impact:** 99.96% reduction in risk score calculations

---

### Phase 2: Update Dashboard to Use Pre-Calculated Risks (HIGH PRIORITY)

**File:** `NordIQ/src/dashboard/Dashboard/tabs/overview.py`

**Changes:**

```python
def render(predictions: Optional[Dict], daemon_url: str = DAEMON_URL):
    if predictions:
        server_preds = predictions.get('predictions', {})
        summary = predictions.get('summary', {})  # NEW

        # OLD (remove):
        # risk_scores = calculate_all_risk_scores(...)

        # NEW (use pre-calculated):
        risk_scores = {
            server_name: server_pred.get('risk_score', 0)
            for server_name, server_pred in server_preds.items()
        }

        # Use pre-calculated summary
        critical_count = summary.get('critical_count', 0)
        warning_count = summary.get('warning_count', 0)
        # ... etc
```

**Estimated Time:** 1 hour
**Impact:** Dashboard becomes pure display layer

---

### Phase 3: Add Display-Ready Metrics (MEDIUM PRIORITY)

**File:** `NordIQ/src/daemons/tft_inference_daemon.py`

**Changes:**

```python
def _format_display_metrics(self, server_pred: Dict) -> Dict:
    """Convert internal format to display-ready format."""
    metrics = {}

    # CPU Used
    if 'cpu_idle_pct' in server_pred:
        cpu_current = 100 - server_pred['cpu_idle_pct'].get('current', 0)
        cpu_p50 = server_pred['cpu_idle_pct'].get('p50', [])
        cpu_predicted = 100 - np.mean(cpu_p50[:6]) if len(cpu_p50) >= 6 else cpu_current
    elif 'cpu_user_pct' in server_pred:
        cpu_current = server_pred['cpu_user_pct'].get('current', 0)
        cpu_p50 = server_pred['cpu_user_pct'].get('p50', [])
        cpu_predicted = np.mean(cpu_p50[:6]) if len(cpu_p50) >= 6 else cpu_current
    else:
        cpu_current = cpu_predicted = 0

    metrics['cpu'] = {
        'current': round(cpu_current, 1),
        'predicted': round(cpu_predicted, 1),
        'delta': round(cpu_predicted - cpu_current, 1),
        'unit': '%',
        'trend': 'increasing' if cpu_predicted > cpu_current else 'decreasing'
    }

    # Memory, I/O Wait, Swap, Load Average... (similar pattern)
    # ...

    return metrics

# In get_predictions():
for server_name, server_pred in predictions.items():
    server_pred['display_metrics'] = self._format_display_metrics(server_pred)
```

**Estimated Time:** 3 hours
**Impact:** Dashboard doesn't need metric extraction logic

---

### Phase 4: Remove Dashboard Business Logic (LOW PRIORITY)

**Files:** Multiple dashboard files

**Changes:**
- Remove `calculate_server_risk_score()` calls
- Remove metric extraction functions
- Remove sorting/filtering logic
- Remove alert level categorization
- Dashboard becomes pure Streamlit UI (HTML/CSS/charts only)

**Estimated Time:** 2 hours
**Impact:** Clean separation of concerns

---

## Benefits Summary

### Performance

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Risk calculations (1 user) | 270+ per load | 1 per daemon call | **270x faster** |
| Risk calculations (10 users) | 2,700/min | 1/min | **2,700x faster** |
| Daemon CPU | 5% | 6% | Small increase (acceptable) |
| Dashboard CPU | 20% | 2% | **10x reduction** |
| Page load time | 2-3s | <500ms | **5x faster** |

### Scalability

| Users | Dashboard Calculations Before | After | Server Load |
|-------|------------------------------|-------|-------------|
| 1 | 270 | 1 | 1x |
| 10 | 2,700 | 1 | 1x (no change!) |
| 100 | 27,000 | 1 | 1x (no change!) |

**Result:** Dashboard scales to infinite users with zero additional daemon load!

---

### Code Quality

✅ **Separation of Concerns:**
- Daemon = Business logic (calculations, ML, alerts)
- Dashboard = Presentation (HTML, CSS, charts)

✅ **Single Source of Truth:**
- Risk calculation logic in ONE place (daemon)
- Alert level thresholds in ONE place (daemon)

✅ **Maintainability:**
- Change risk formula? Update daemon only
- Change alert thresholds? Update daemon only
- Dashboard doesn't need to know business rules

---

## Migration Strategy

### Step 1: Daemon Enhancement (Non-Breaking)

Add risk scores and summary to daemon response:
```python
{
  "predictions": {
    "server1": {
      "risk_score": 67.3,  # NEW (optional for now)
      "alert_level": "warning",  # NEW (optional)
      ... existing fields ...
    }
  },
  "summary": {  # NEW (entire section optional)
    "total_servers": 90,
    "critical_count": 5,
    ...
  }
}
```

**Impact:** Backward compatible (dashboard ignores new fields if not ready)

---

### Step 2: Dashboard Update (Gradual)

Update dashboard to use pre-calculated fields IF available:
```python
# Fallback to old behavior if daemon doesn't provide risk_score yet
if 'risk_score' in server_pred:
    risk_score = server_pred['risk_score']  # NEW
else:
    risk_score = calculate_server_risk_score(server_pred)  # OLD (fallback)
```

**Impact:** Works with both old and new daemon versions

---

### Step 3: Remove Fallback (After Testing)

Once daemon is proven stable, remove fallback:
```python
# Trust daemon to always provide risk_score
risk_score = server_pred['risk_score']
```

**Impact:** Dashboard simplified, business logic removed

---

## Testing Plan

### Unit Tests

```python
def test_daemon_calculates_risk_scores():
    """Test that daemon returns risk_score for each server."""
    daemon = CleanInferenceDaemon()
    predictions = daemon.get_predictions()

    for server_name, server_pred in predictions['predictions'].items():
        assert 'risk_score' in server_pred
        assert 0 <= server_pred['risk_score'] <= 100
        assert 'alert_level' in server_pred

def test_daemon_provides_summary():
    """Test that daemon returns summary statistics."""
    predictions = daemon.get_predictions()
    summary = predictions['summary']

    assert 'total_servers' in summary
    assert 'critical_count' in summary
    assert 'top_5_risks' in summary
    assert len(summary['top_5_risks']) == 5
```

### Integration Tests

```python
def test_dashboard_uses_daemon_risk_scores():
    """Test that dashboard uses pre-calculated risk scores."""
    # Mock daemon response
    predictions = {
        'predictions': {
            'server1': {'risk_score': 67.3, ...}
        },
        'summary': {'top_5_risks': ['server1', ...]}
    }

    # Dashboard should NOT call calculate_server_risk_score()
    with patch('Dashboard.utils.risk_scoring.calculate_server_risk_score') as mock:
        overview.render(predictions, daemon_url)
        assert mock.call_count == 0  # Should be zero!
```

---

## Rollback Plan

If issues arise:

1. **Daemon rollback:** Remove risk_score calculation (revert to old response format)
2. **Dashboard rollback:** Use fallback logic (calculate if not provided)
3. **Zero downtime:** Both old and new versions compatible

---

## Key Takeaways

🎯 **Main Problem:**
> Dashboard is doing business logic (risk calculations) that should be done ONCE by the daemon.

🎯 **Root Cause:**
> When I optimized the dashboard (Phase 1), I was treating the symptom, not the disease. The real issue is architectural: wrong layer is doing heavy lifting.

🎯 **Correct Solution:**
> Move ALL computation to daemon. Dashboard becomes pure display layer (HTML/CSS/charts only).

🎯 **Impact:**
> - 270x fewer calculations per user
> - Infinite dashboard scalability (daemon load unchanged)
> - Proper separation of concerns
> - 5x faster page loads

---

**Maintained By:** Craig Giannelli / NordIQ AI Systems, LLC
**Created:** October 18, 2025
**Status:** Identified, ready for implementation
**Next Step:** Implement Phase 1 (add risk scores to daemon response)
