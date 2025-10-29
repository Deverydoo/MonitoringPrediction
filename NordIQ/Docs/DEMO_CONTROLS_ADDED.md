# Demo Controls and System Status Added
**Date:** 2025-10-29
**Status:** ✅ Complete
**Impact:** Professional demo/development features with full system visibility

---

## Features Added

### 1. Connection Status Display

**What:** Real-time indicator showing daemon connection status

**Visual:**
```
┌────────────────────────────────────────────────────┐
│ 🟢 Connected                    ✅ Model Ready     │
│ 47 servers | Last update: 14:23:45                 │
│                                                     │
│ All systems operational                            │
└────────────────────────────────────────────────────┘
```

**When Connected:**
- 🟢 Green alert box
- Shows number of servers being monitored
- Displays last update timestamp (HH:MM:SS format)

**When Disconnected:**
- 🔴 Red alert box
- Shows instructions: "Start daemon: python src/daemons/tft_inference_daemon.py"

**Implementation:**
- Updates automatically with predictions data
- No manual refresh needed
- Located below refresh interval slider

### 2. Warmup Progress Display

**What:** Shows daemon model warming up status with progress bar

**Visual (During Warmup):**
```
┌────────────────────────────────────────┐
│ ⏳ Model warming up (2/10 servers)    │
│ ████████████░░░░░░░░░░░ 65%          │
└────────────────────────────────────────┘
```

**Visual (After Warmup):**
```
┌────────────────────────────────────────┐
│ ✅ Model Ready                         │
│ All systems operational                │
└────────────────────────────────────────┘
```

**Features:**
- Progress bar shows completion percentage
- Updates every refresh cycle
- Disappears when model is fully warmed up
- Yellow alert during warmup, green when ready

### 3. Demo Scenario Controls

**What:** Three buttons to control metrics generator behavior for demos

**Visual:**
```
┌──────────────────────────────────────────────────────┐
│ 🎬 Demo Scenario Controls                           │
│ Control the metrics generator behavior (port 8001)  │
│                                                       │
│ ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│ │🟢 Healthy│  │🟡Degrading│  │🔴 Critical│         │
│ └──────────┘  └──────────┘  └──────────┘          │
│                                                       │
│ 🟢 Current Scenario: HEALTHY                        │
│ Affected servers: 0 | Tick count: 142               │
└──────────────────────────────────────────────────────┘
```

**Buttons:**

1. **🟢 Healthy** (Green)
   - Sets all servers to healthy state
   - No warnings or critical alerts
   - Used to reset after testing failure scenarios

2. **🟡 Degrading** (Yellow/Orange)
   - Simulates gradual performance degradation
   - Some servers show warning-level metrics
   - Tests dashboard's warning detection

3. **🔴 Critical** (Red)
   - Simulates multiple critical failures
   - Many servers in critical state
   - Tests dashboard's alert system and escalation

**Status Display:**
- Shows current scenario (HEALTHY/DEGRADING/CRITICAL)
- Number of affected servers
- Tick count (how many data cycles generated)
- Color-coded based on scenario severity

---

## Implementation Details

### File Modified: dash_app.py

**Lines 138-187:** Added UI components
```python
# Connection Status & Demo Controls
dbc.Card([
    dbc.CardBody([
        dbc.Row([
            # Connection Status
            dbc.Col([html.Div(id='connection-status-display')], width=6),
            # Warmup Status
            dbc.Col([html.Div(id='warmup-status-display')], width=6)
        ], className="mb-3"),

        # Demo Scenario Controls
        html.Hr(),
        html.H6("🎬 Demo Scenario Controls", className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Button("🟢 Healthy", id='scenario-healthy-btn', ...)], width=4),
            dbc.Col([dbc.Button("🟡 Degrading", id='scenario-degrading-btn', ...)], width=4),
            dbc.Col([dbc.Button("🔴 Critical", id='scenario-critical-btn', ...)], width=4)
        ]),
        html.Div(id='scenario-status-display', className="mt-2")
    ])
])
```

**Lines 273-332:** Connection Status & Warmup Callback
```python
@app.callback(
    [Output('connection-status-display', 'children'),
     Output('warmup-status-display', 'children')],
    Input('predictions-store', 'data')
)
def update_connection_status(predictions):
    # Parse predictions data
    # Check daemon health
    # Query warmup status from /status endpoint
    # Return formatted alerts
```

**Lines 335-437:** Demo Scenario Controls Callback
```python
@app.callback(
    Output('scenario-status-display', 'children'),
    [Input('scenario-healthy-btn', 'n_clicks'),
     Input('scenario-degrading-btn', 'n_clicks'),
     Input('scenario-critical-btn', 'n_clicks'),
     Input('refresh-interval', 'n_intervals')],
    prevent_initial_call=False
)
def handle_scenario_controls(...):
    # Detect which button was clicked
    # POST to metrics generator /scenario/set endpoint
    # Return success/error message
    # Also poll /scenario/status on page load
```

---

## API Integration

### Daemon Endpoints Used

**1. `/status` - Health and Warmup Status**

**Request:**
```
GET http://localhost:8000/status
```

**Response (During Warmup):**
```json
{
  "status": "warming_up",
  "warmup": {
    "is_warmed_up": false,
    "progress_percent": 65,
    "message": "Model warming up (2/10 servers)",
    "servers_ready": 2,
    "servers_total": 10
  }
}
```

**Response (After Warmup):**
```json
{
  "status": "ready",
  "warmup": {
    "is_warmed_up": true,
    "progress_percent": 100,
    "message": "All systems operational"
  }
}
```

### Metrics Generator Endpoints Used

**2. `/scenario/set` - Change Scenario**

**Request:**
```
POST http://localhost:8001/scenario/set
Content-Type: application/json

{
  "scenario": "healthy"  // or "degrading" or "critical"
}
```

**Response:**
```json
{
  "status": "success",
  "scenario": "healthy",
  "affected_servers": 0,
  "message": "Scenario changed to healthy"
}
```

**3. `/scenario/status` - Get Current Scenario**

**Request:**
```
GET http://localhost:8001/scenario/status
```

**Response:**
```json
{
  "scenario": "healthy",
  "total_affected": 0,
  "tick_count": 142,
  "uptime_seconds": 3567
}
```

---

## Use Cases

### 1. Development & Testing

**Scenario:** Developer wants to test dashboard with different failure scenarios

**Steps:**
1. Start metrics generator: `python src/daemons/metrics_generator_daemon.py`
2. Open dashboard at http://localhost:8050
3. Click **🟡 Degrading** button
4. Observe dashboard updating with warning-level servers
5. Click **🔴 Critical** button
6. Observe dashboard showing critical alerts
7. Click **🟢 Healthy** button to reset

**Value:** Quick scenario switching without restarting services or editing configs

### 2. Demo/Presentation

**Scenario:** Sales demo showing AI prediction capabilities

**Steps:**
1. Start in **🟢 Healthy** mode (all systems green)
2. During demo: "Let me show you what happens when servers start degrading..."
3. Click **🟡 Degrading** button
4. Dashboard updates to show warnings, predictions increase
5. "And here's a critical situation..."
6. Click **🔴 Critical** button
7. Dashboard shows multiple alerts, high risk scores, What-If scenarios activate

**Value:** Live, interactive demo without pre-recorded data

### 3. Training

**Scenario:** Training operations team on using the dashboard

**Steps:**
1. Show healthy state: "This is what normal looks like"
2. Switch to degrading: "These yellow indicators mean we should investigate"
3. Show What-If scenarios: "The AI suggests restarting this service"
4. Switch to critical: "Red alerts require immediate action"
5. Demonstrate alerting tab: "Here's how you'd configure notifications"

**Value:** Safe, controlled environment for learning without affecting production

### 4. Performance Testing

**Scenario:** Testing dashboard performance under different loads

**Steps:**
1. Start in **🟢 Healthy** (low load, few alerts)
2. Switch to **🔴 Critical** (high load, many alerts)
3. Measure render times, check for performance degradation
4. Verify all tabs still load quickly

**Value:** Stress testing with realistic failure scenarios

---

## Visual Design

### Card Layout

```
┌────────────────────────────────────────────────────────────┐
│                    TOP SECTION (Connection Status)          │
│ ┌─────────────────────────┬─────────────────────────┐     │
│ │ 🟢 Connected            │ ✅ Model Ready          │     │
│ │ 47 servers | 14:23:45   │ All systems operational │     │
│ └─────────────────────────┴─────────────────────────┘     │
│                                                             │
│ ─────────────────────────────────────────────────────────  │
│                                                             │
│                 DEMO SCENARIO CONTROLS                      │
│ 🎬 Demo Scenario Controls                                  │
│ Control the metrics generator behavior (port 8001)         │
│                                                             │
│ ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│ │ 🟢 Healthy │  │ 🟡Degrading│  │ 🔴 Critical│           │
│ └────────────┘  └────────────┘  └────────────┘           │
│                                                             │
│ 🟢 Current Scenario: HEALTHY                               │
│ Affected servers: 0 | Tick count: 142                      │
└────────────────────────────────────────────────────────────┘
```

**Colors:**
- Connection: Green (success) or Red (danger)
- Warmup: Yellow (warning) during warmup, Green (success) when ready
- Healthy button: Green outline
- Degrading button: Yellow/Orange outline
- Critical button: Red outline
- Status display: Matches current scenario color

**Spacing:**
- Card has `mb-3` (margin-bottom 3 units)
- Sections separated by `<hr>` divider
- Buttons have full width within columns (`w-100`)
- Status display has top margin (`mt-2`)

---

## Error Handling

### Metrics Generator Not Running

**Scenario:** User clicks scenario button but generator is not running

**Response:**
```
┌────────────────────────────────────────────────────────┐
│ ⚠️ Cannot connect to metrics generator (port 8001)    │
└────────────────────────────────────────────────────────┘
```

**User Action:** Start metrics generator: `python src/daemons/metrics_generator_daemon.py`

### Daemon Not Running

**Scenario:** Daemon is not running, no predictions available

**Response:**
```
┌────────────────────────────────────────────────────────┐
│ 🔴 Disconnected                                        │
│ Start daemon: python src/daemons/tft_inference_daemon.py │
└────────────────────────────────────────────────────────┘
```

**User Action:** Start inference daemon

### Network Timeout

**Scenario:** Request to generator/daemon times out

**Behavior:**
- Connection status shows "Disconnected"
- Scenario status shows: "💡 Start metrics generator on port 8001 to use demo controls"
- No error popup, graceful degradation

**User Action:** Check if services are running

---

## Advantages Over Streamlit

### Streamlit Implementation

**Location:** Sidebar with `@st.fragment` wrapper

**Issues:**
- Sidebar not always visible (can be collapsed)
- Fragment reduces but doesn't eliminate full app reruns
- Button clicks still trigger state updates
- Status updates require manual polling

### Dash Implementation

**Location:** Main content area, below settings

**Advantages:**
- ✅ Always visible (part of main content)
- ✅ True callback-based (no app reruns)
- ✅ Button clicks only update status display
- ✅ Status auto-updates with predictions data
- ✅ Better visual design (Bootstrap components)
- ✅ Color-coded alerts (success/warning/danger)

---

## Configuration

### Environment Variables

**Daemon URL:**
```bash
export DAEMON_URL="http://localhost:8000"
```

**Metrics Generator URL:**
```python
# Currently hardcoded in callback
generator_url = "http://localhost:8001"
```

**To change generator URL:**
Edit `dash_app.py` line 354:
```python
generator_url = "http://localhost:8001"  # Change port if needed
```

### Disabling Demo Controls (Production)

To hide demo controls in production, comment out lines 152-185 in `dash_app.py`:

```python
# Demo Scenario Controls (COMMENT OUT FOR PRODUCTION)
# html.Hr(),
# html.H6("🎬 Demo Scenario Controls", className="mb-3"),
# ...
```

Or wrap in environment check:
```python
import os
if os.getenv('ENVIRONMENT') == 'development':
    # Show demo controls
    ...
```

---

## Testing Instructions

### Test 1: Connection Status

**Steps:**
1. Stop inference daemon (if running)
2. Open dashboard at http://localhost:8050
3. Observe: **🔴 Disconnected** with instructions
4. Start daemon: `python src/daemons/tft_inference_daemon.py`
5. Wait 30-60 seconds for first predictions
6. Observe: **🟢 Connected** with server count

**Expected Result:**
- Status automatically updates from red to green
- Shows number of servers and last update time

### Test 2: Warmup Progress

**Steps:**
1. Stop inference daemon
2. Start daemon: `python src/daemons/tft_inference_daemon.py`
3. Immediately open dashboard at http://localhost:8050
4. Observe warmup status (if model is still warming up)

**Expected Result:**
- Shows **⏳ Model warming up** with progress bar
- Progress bar increases from 0% to 100%
- When complete, shows **✅ Model Ready**

**Note:** May not see warmup if daemon starts quickly (small dataset)

### Test 3: Demo Scenario Controls

**Steps:**
1. Start metrics generator: `python src/daemons/metrics_generator_daemon.py`
2. Open dashboard at http://localhost:8050
3. Observe initial status (usually HEALTHY)
4. Click **🟡 Degrading** button
5. Observe: Status updates, shows affected servers
6. Wait 30-60 seconds, check dashboard tabs
7. Click **🔴 Critical** button
8. Observe: Multiple servers in critical state
9. Click **🟢 Healthy** button
10. Observe: System returns to normal

**Expected Result:**
- Each button click updates status display immediately
- Dashboard data updates within 30 seconds (after generator produces new metrics)
- Status display shows current scenario, affected servers, tick count

### Test 4: Generator Not Running

**Steps:**
1. Stop metrics generator (if running)
2. Open dashboard at http://localhost:8050
3. Click any scenario button

**Expected Result:**
- Shows: **💡 Start metrics generator on port 8001 to use demo controls**
- No error popup or crash
- Other dashboard features still work

---

## Summary

**Features Added:**
- ✅ Connection status indicator (real-time daemon health)
- ✅ Warmup progress display (model initialization tracking)
- ✅ Demo scenario controls (3 buttons for Healthy/Degrading/Critical)
- ✅ Current scenario status (shows active scenario and stats)

**Benefits:**
- Professional development/demo environment
- Full system visibility (connection, warmup, scenario)
- Interactive testing capabilities
- Safe scenario switching without config changes
- Better than Streamlit (always visible, true callbacks, better design)

**Production Ready:**
- Can be easily hidden for production (comment out lines 152-185)
- No impact on production features
- Graceful degradation when generator not available

The Dash dashboard now has all the development/demo features from Streamlit, implemented in a superior way with better visibility, performance, and design! 🚀
