# Interactive Demo System - Final Integration Steps

## Status: 90% Complete

### ✅ Completed:
1. **ScenarioDemoGenerator** created (`scenario_demo_generator.py`)
   - Reads training data for server fleet
   - Auto-detects fleet size
   - Three scenarios: healthy, degrading, critical
   - Smooth real-time transitions

2. **Import added** to `tft_inference.py`
   - Line 24: `from scenario_demo_generator import ScenarioDemoGenerator, ScenarioMode`

### 🔄 Remaining Steps (15 minutes):

#### Step 1: Replace SimulationGenerator in InferenceDaemon
**File:** `tft_inference.py`, Line ~1040

**Current:**
```python
self.generator = SimulationGenerator(fleet_size, seed, mode)
```

**Change to:**
```python
# Use interactive scenario generator (reads training data)
self.generator = ScenarioDemoGenerator(
    training_data_path="./training/server_metrics.parquet",
    seed=seed
)
print(f"[DEMO] Loaded {self.generator.fleet_size} servers from training data")
```

#### Step 2: Add Scenario Control Method to InferenceDaemon
**File:** `tft_inference.py`, After line ~1050

**Add:**
```python
def set_scenario(self, mode: str, affected_count: Optional[int] = None):
    """
    Set demo scenario - called from dashboard button.

    Args:
        mode: 'healthy', 'degrading', or 'critical'
        affected_count: Number of servers to affect (default: random 1-5)
    """
    self.generator.set_scenario(mode, affected_count)
    print(f"[SCENARIO] Switched to: {mode.upper()}")

def get_scenario_status(self) -> Dict:
    """Get current scenario status."""
    return self.generator.get_status()
```

#### Step 3: Add REST Endpoint for Scenario Control
**File:** `tft_inference.py`, After line ~1220 (in create_api_app function)

**Add:**
```python
@app.post("/scenario/set")
async def set_scenario(request: Dict):
    """
    Interactive demo scenario control.

    Expected: {"mode": "healthy|degrading|critical", "affected_count": 3}
    """
    try:
        mode = request.get('mode', 'healthy')
        affected_count = request.get('affected_count')

        daemon.set_scenario(mode, affected_count)

        return {
            "status": "success",
            "scenario": mode,
            "affected_servers": len(daemon.generator.affected_servers)
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.get("/scenario/status")
async def get_scenario_status():
    """Get current scenario status."""
    return daemon.get_scenario_status()
```

#### Step 4: Update Dashboard Buttons
**File:** `tft_dashboard_web.py`, Around line ~270

**Add:**
```python
st.subheader("Interactive Demo Control")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🟢 Healthy", use_container_width=True):
        response = requests.post(
            f"{daemon_url}/scenario/set",
            json={"mode": "healthy"}
        )
        if response.ok:
            st.success("Scenario: Healthy - All servers recovering")
        else:
            st.error(f"Failed: {response.text}")

with col2:
    if st.button("🟡 Degrading", use_container_width=True):
        response = requests.post(
            f"{daemon_url}/scenario/set",
            json={"mode": "degrading"}
        )
        if response.ok:
            result = response.json()
            st.warning(f"Scenario: Degrading - {result['affected_servers']} servers affected")
        else:
            st.error(f"Failed: {response.text}")

with col3:
    if st.button("🔴 Critical", use_container_width=True):
        response = requests.post(
            f"{daemon_url}/scenario/set",
            json={"mode": "critical"}
        )
        if response.ok:
            result = response.json()
            st.error(f"Scenario: Critical - {result['affected_servers']} servers in crisis!")
        else:
            st.error(f"Failed: {response.text}")

# Show current scenario status
try:
    scenario_response = requests.get(f"{daemon_url}/scenario/status", timeout=1)
    if scenario_response.ok:
        status = scenario_response.json()
        st.info(f"Current: {status['scenario'].upper()} | "
                f"Affected: {len(status['affected_servers'])} servers | "
                f"Progress: {status['transition_progress']*100:.0f}%")
except:
    pass
```

### Testing Sequence:

1. **Start daemon:**
   ```bash
   python tft_inference.py --daemon --port 8000
   ```
   - Should see: `[DEMO] Loaded 20 servers from training data`

2. **Start dashboard:**
   ```bash
   streamlit run tft_dashboard_web.py
   ```

3. **Test scenarios:**
   - Click "Healthy" → All servers stable
   - Click "Degrading" → 1-5 servers gradually degrade over 5 minutes
   - Click "Critical" → 1-5 servers rapid spike
   - Click "Healthy" → Recovery mode

4. **Watch TFT predictions** respond in real-time to scenario changes!

### Key Benefits:

✅ **Demo isolation** - Easy to swap for production data (just replace generator)
✅ **Training data consistency** - Auto-reads from parquet file
✅ **Real-time control** - Dashboard buttons instantly trigger scenarios
✅ **Cinema-grade** - Smooth transitions, realistic degradation patterns
✅ **No fleet_size config needed** - Auto-detected from training data

---

**Ready to complete the integration! 🎬🚀**
