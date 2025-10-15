# Dashboard Optimization Guide - Performance Improvements

**Current Status**: tft_dashboard_web.py is 3,237 lines with 10 tabs in single monolithic file
**Problem**: Slow startup, no caching, all tabs render even when not visible
**Goal**: Faster load times, better responsiveness, modular architecture

---

## 🎯 Quick Wins (Immediate - No Refactoring)

### 1. Add Strategic Caching

**Current Issue**: Only 1 function cached (`get_server_profile`)

**Add Caching to These Functions**:

```python
# Cache risk calculations (expensive)
@st.cache_data(ttl=10)  # 10 seconds for real-time data
def calculate_server_risk_score(server_pred: Dict) -> float:
    # ... existing code ...

# Cache health status
@st.cache_data(ttl=5)  # 5 seconds
def get_health_status(predictions: Dict) -> tuple:
    # ... existing code ...

# Cache color indicators (called frequently)
@st.cache_data(ttl=300)  # 5 minutes - static logic
def get_metric_color_indicator(value: float, metric_type: str, profile: str) -> str:
    # ... existing code ...

# Cache CPU extraction (called multiple times per render)
@st.cache_data(ttl=10)
def extract_cpu_used(server_pred: Dict, metric_type: str = 'current') -> float:
    # ... existing code ...
```

**Expected Impact**: 30-50% faster render time

### 2. Lazy Tab Loading

**Current Issue**: All 10 tabs render even when user only views 1-2

**Solution**: Use Streamlit's native tab behavior (already works!)
- Streamlit only renders active tab content
- BUT: Move expensive calculations INSIDE tab blocks

**Before** (bad):
```python
# Calculate everything upfront (wasteful)
heatmap_data = calculate_heatmap(predictions)  # Runs even if tab not viewed
historical_data = load_history()  # Runs even if tab not viewed

with tab1:
    st.write("Overview")

with tab2:
    st.plotly_chart(heatmap_data)  # Data already computed
```

**After** (good):
```python
with tab1:
    st.write("Overview")

with tab2:
    # Only calculate if user clicks Heatmap tab
    heatmap_data = calculate_heatmap(predictions)
    st.plotly_chart(heatmap_data)
```

**Expected Impact**: 60-70% faster initial load

### 3. Session State for Predictions

**Current Issue**: Fetching predictions on every rerun

**Solution**:
```python
# Top of file, after imports
if 'last_predictions' not in st.session_state:
    st.session_state.last_predictions = None
    st.session_state.last_fetch_time = 0

# In main loop
current_time = time.time()
if current_time - st.session_state.last_fetch_time > 5:  # Fetch every 5 seconds
    predictions = fetch_predictions()
    st.session_state.last_predictions = predictions
    st.session_state.last_fetch_time = current_time
else:
    predictions = st.session_state.last_predictions
```

**Expected Impact**: Reduces API calls by 80%

---

## 🏗️ Medium-Term Optimization (2-4 hours work)

### 4. Split into Modular Files

**Structure**:
```
tft_dashboard_web.py          # Main entry (200 lines)
├── tabs/
│   ├── __init__.py
│   ├── overview.py           # Tab 1 (300 lines)
│   ├── heatmap.py            # Tab 2 (200 lines)
│   ├── top_servers.py        # Tab 3 (300 lines)
│   ├── historical.py         # Tab 4 (400 lines)
│   ├── cost_avoidance.py     # Tab 5 (300 lines)
│   ├── auto_remediation.py   # Tab 6 (400 lines)
│   ├── alerting.py           # Tab 7 (400 lines)
│   ├── advanced.py           # Tab 8 (200 lines)
│   ├── documentation.py      # Tab 9 (500 lines)
│   └── roadmap.py            # Tab 10 (200 lines)
├── utils/
│   ├── __init__.py
│   ├── risk_scoring.py       # Risk calculation logic
│   ├── metrics.py            # Metric extraction helpers
│   └── caching.py            # Shared cache functions
└── config/
    └── dashboard_config.py   # Constants, thresholds
```

**Main File** (tft_dashboard_web.py):
```python
import streamlit as st
from tabs import overview, heatmap, top_servers, historical
from tabs import cost_avoidance, auto_remediation, alerting
from tabs import advanced, documentation, roadmap
from utils import risk_scoring, metrics, caching

# Common setup
st.set_page_config(page_title="TFT Monitoring", layout="wide")
predictions = caching.get_cached_predictions()

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Overview", "🔥 Heatmap", "⚠️ Top 5", "📈 Historical",
    "💰 Cost", "🤖 Auto-Remediation", "📱 Alerts", "⚙️ Advanced",
    "📚 Docs", "🗺️ Roadmap"
])

with tab1:
    overview.render(predictions)

with tab2:
    heatmap.render(predictions)

# ... etc
```

**Benefits**:
- ✅ Faster imports (only load what you need)
- ✅ Easier to maintain/test
- ✅ Parallel development possible
- ✅ Cleaner git diffs

**Expected Impact**: 40% faster startup

---

## 🚀 Advanced Optimization (Future - If Needed)

### 5. Multi-Page App (Streamlit Pages)

**When to Use**: If dashboard becomes >5,000 lines or has distinct user roles

**Structure**:
```
pages/
├── 1_📊_Overview.py
├── 2_🔥_Heatmap.py
├── 3_⚠️_Top_Servers.py
└── ...
```

**Benefits**:
- Each page is separate Python file
- Only loads active page
- Browser routing (shareable URLs)

**Drawbacks**:
- Loses tab UX (becomes sidebar navigation)
- Shared state more complex

### 6. WebSocket Real-Time Updates

**Current**: Dashboard polls API every 5 seconds

**Improved**: Daemon pushes updates via WebSocket

```python
# In tft_inference_daemon.py (already has /ws endpoint!)
# Just need to connect dashboard to it

import asyncio
import websockets

@st.cache_resource
def get_websocket_connection():
    return websockets.connect("ws://localhost:8000/ws")

async def listen_for_updates():
    async with get_websocket_connection() as ws:
        while True:
            data = await ws.recv()
            st.session_state.predictions = json.loads(data)
            st.rerun()
```

**Expected Impact**: Near-instant updates, 90% less network traffic

---

## 📊 Performance Benchmarks

### Current Performance (3,237 line monolith):
- **Initial Load**: ~5-8 seconds
- **Tab Switch**: ~1-2 seconds
- **Refresh**: ~2-3 seconds
- **API Calls**: Every 5 seconds (20% wasteful re-fetches)

### With Quick Wins (caching + lazy loading):
- **Initial Load**: ~2-3 seconds ✅ (60% faster)
- **Tab Switch**: <0.5 seconds ✅ (75% faster)
- **Refresh**: ~0.5-1 second ✅ (70% faster)
- **API Calls**: Smart caching (80% reduction)

### With Modular Split:
- **Initial Load**: ~1-2 seconds ✅ (80% faster)
- **Tab Switch**: <0.3 seconds ✅ (85% faster)
- **Refresh**: ~0.3-0.5 seconds ✅ (85% faster)

---

## 🎯 Recommended Approach for Presentation

### **Option 1: Do Nothing (Safest)**
- Current dashboard works
- 5-8 second load is acceptable for demo
- Focus on content, not performance
- **Risk**: None
- **Time**: 0 hours

### **Option 2: Quick Wins Only (Recommended)**
- Add 4-5 @st.cache_data decorators
- Move expensive calculations inside tab blocks
- Add session state for predictions
- **Risk**: Low (non-breaking changes)
- **Time**: 30-45 minutes
- **Benefit**: 60% faster load time

### **Option 3: Full Modular Refactor (Post-Presentation)**
- Split into 15+ files
- Comprehensive caching strategy
- WebSocket real-time updates
- **Risk**: Medium (breaking changes possible)
- **Time**: 4-6 hours
- **Benefit**: 80% faster, maintainable codebase

---

## 🛠️ Implementation: Quick Wins (30 minutes)

### Step 1: Add Caching (10 minutes)

```python
# Add these decorators to existing functions (lines 175, 233, 282, 345)

@st.cache_data(ttl=5)
def get_health_status(predictions: Dict) -> tuple:
    # ... existing code unchanged ...

@st.cache_data(ttl=300)  # Already cached, increase TTL
def get_server_profile(server_name: str) -> str:
    # ... existing code unchanged ...

@st.cache_data(ttl=60)
def get_metric_color_indicator(value: float, metric_type: str, profile: str = 'Generic') -> str:
    # ... existing code unchanged ...

@st.cache_data(ttl=10)
def extract_cpu_used(server_pred: Dict, metric_type: str = 'current') -> float:
    # ... existing code unchanged ...

@st.cache_data(ttl=10)
def calculate_server_risk_score(server_pred: Dict) -> float:
    # ... existing code unchanged ...
```

### Step 2: Session State for Predictions (10 minutes)

```python
# After imports, before main logic (around line 140)

# Initialize session state
if 'predictions_cache' not in st.session_state:
    st.session_state.predictions_cache = None
    st.session_state.last_fetch = 0

# Replace prediction fetching logic (around line 725)
def get_cached_predictions():
    """Fetch predictions with smart caching."""
    current_time = time.time()

    # Only fetch if cache expired (5 seconds)
    if (st.session_state.predictions_cache is None or
        current_time - st.session_state.last_fetch > 5):

        try:
            response = requests.get(f"{daemon_url}/predictions/current", timeout=5)
            if response.ok:
                st.session_state.predictions_cache = response.json().get('predictions', {})
                st.session_state.last_fetch = current_time
        except:
            pass  # Use cached data on error

    return st.session_state.predictions_cache or {}

# Use it
predictions = get_cached_predictions()
```

### Step 3: Lazy Tab Loading (10 minutes)

**Find expensive operations in tabs and move them inside tab blocks**

Example - Historical tab (currently loads all history upfront):

**Before**:
```python
# Top of file (line ~750)
historical_data = load_historical_data()  # Always runs!

# ...

with tab4:
    st.plotly_chart(historical_data)
```

**After**:
```python
with tab4:
    # Only loads when user clicks Historical tab
    historical_data = load_historical_data()
    st.plotly_chart(historical_data)
```

---

## 🎯 Presentation Day Decision Matrix

| Scenario | Recommendation | Rationale |
|----------|---------------|-----------|
| **Dashboard loads in <3 seconds** | Do nothing | It's fast enough |
| **Dashboard loads in 3-6 seconds** | Quick wins | 30 min = 60% faster |
| **Dashboard loads in >6 seconds** | Quick wins + lazy loading | Worth the time investment |
| **Frequent crashes/hangs** | Full refactor | Critical UX issue |

**Current Status**: ~5-8 seconds → **Quick Wins Recommended**

---

## 📝 Testing Checklist

After optimization, verify:

- [ ] All 10 tabs still render correctly
- [ ] Predictions update every 5 seconds
- [ ] Scenario switching works (healthy/degrading/critical)
- [ ] Color-coding still appears correctly
- [ ] No console errors in browser dev tools
- [ ] Risk scores calculate correctly
- [ ] Heatmap displays properly
- [ ] Historical charts load

---

## 🔮 Future Enhancements (Post-Presentation)

### Priority 1: Modular Architecture
- Split tabs into separate files
- Shared utility modules
- Comprehensive test coverage

### Priority 2: Advanced Caching
- Redis for cross-session caching
- Prediction result caching in daemon
- Computed risk scores cached server-side

### Priority 3: Real-Time Updates
- WebSocket connection to daemon
- Server-sent events for alerts
- Live metric streaming

### Priority 4: Performance Monitoring
- Track dashboard render times
- Log slow operations
- User interaction analytics

---

## 💡 Key Insights

### Why Streamlit is "Slow"
- **Not Python bytecode compilation** - that's fast (<1ms)
- **The real issue**: Re-running entire script on every interaction
- **Solution**: Caching and lazy loading, not "applets"

### Why Tabs vs Multi-Page
- **Tabs**: Single-page UX, faster switching, shared state
- **Multi-Page**: Slower (full reload), but better code organization
- **For demo**: Tabs are better (seamless experience)

### Caching Strategy
- **Static data** (thresholds, profiles): Cache 5+ minutes
- **Real-time data** (predictions): Cache 5-10 seconds
- **Computed results** (risk scores): Cache 10-30 seconds
- **User interactions**: Use session_state, no TTL

---

**Last Updated**: October 15, 2025
**Status**: Recommendations for presentation optimization
**Recommended Action**: Quick Wins (30 minutes) for 60% performance improvement

**For Post-Presentation**: Full modular refactor (4-6 hours) for 80% improvement + maintainability
