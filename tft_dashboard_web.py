#!/usr/bin/env python3
"""
TFT Monitoring Dashboard - Production Web Interface
Built with Streamlit for easy deployment and professional UI

Usage:
    streamlit run tft_dashboard_web.py

Requirements:
    pip install streamlit plotly requests pandas
"""

import streamlit as st
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import from modular Dashboard package
from Dashboard.utils import DaemonClient
from Dashboard.config.dashboard_config import DAEMON_URL, REFRESH_INTERVAL, DAEMON_API_KEY
from Dashboard.tabs import (
    overview,
    heatmap,
    top_risks,
    historical,
    cost_avoidance,
    auto_remediation,
    alerting,
    advanced,
    documentation,
    roadmap
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="TFT Monitoring Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "TFT Monitoring Dashboard - Temporal Fusion Transformer for Server Monitoring"
    }
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if 'daemon_url' not in st.session_state:
    st.session_state.daemon_url = DAEMON_URL

if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = REFRESH_INTERVAL

if 'history' not in st.session_state:
    st.session_state.history = []

if 'last_update' not in st.session_state:
    st.session_state.last_update = None

if 'daemon_connected' not in st.session_state:
    st.session_state.daemon_connected = False

if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

if 'demo_running' not in st.session_state:
    st.session_state.demo_running = False

if 'demo_data' not in st.session_state:
    st.session_state.demo_data = None

if 'demo_index' not in st.session_state:
    st.session_state.demo_index = 0

if 'demo_scenario' not in st.session_state:
    st.session_state.demo_scenario = None

if 'demo_start_time' not in st.session_state:
    st.session_state.demo_start_time = None

if 'demo_generator' not in st.session_state:
    st.session_state.demo_generator = None

if 'last_demo_tick' not in st.session_state:
    st.session_state.last_demo_tick = None

# =============================================================================
# DEMO MODE FUNCTIONS (Legacy - kept for backwards compatibility)
# =============================================================================

def initialize_demo_generator(scenario: str, fleet_size: int = 90):
    """Initialize streaming demo data generator (legacy function)."""
    # Import here to avoid circular dependency
    from metrics_generator import DemoGenerator

    st.session_state.demo_generator = DemoGenerator(scenario=scenario, fleet_size=fleet_size)
    st.session_state.demo_running = True
    st.session_state.demo_scenario = scenario
    st.session_state.demo_start_time = datetime.now()
    st.session_state.last_demo_tick = None

def stop_demo():
    """Stop demo mode."""
    st.session_state.demo_running = False
    st.session_state.demo_generator = None
    st.session_state.demo_scenario = None
    st.session_state.demo_start_time = None
    st.session_state.last_demo_tick = None

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("⚙️ Dashboard Settings")

    # Daemon connection
    st.subheader("Daemon Connection")
    daemon_url = st.text_input(
        "Daemon URL",
        value=st.session_state.daemon_url,
        help="URL of the TFT inference daemon"
    )
    st.session_state.daemon_url = daemon_url

    # Test connection (pass API key for authenticated requests)
    client = DaemonClient(daemon_url, api_key=DAEMON_API_KEY)
    health = client.check_health()

    if health['connected']:
        st.success("✅ Connected to daemon")
        st.session_state.daemon_connected = True
        health_data = health.get('data', {})
        st.caption(f"Status: {health_data.get('status', 'Unknown')}")

        # Show warmup status if available
        try:
            status_response = requests.get(f"{daemon_url}/status", timeout=2)
            if status_response.ok:
                status_data = status_response.json()
                warmup = status_data.get('warmup', {})
                if not warmup.get('is_warmed_up', True):
                    progress = warmup.get('progress_percent', 0)
                    st.warning(f"⏳ {warmup.get('message', 'Model warming up')} ({progress:.0f}%)")
                    st.progress(progress / 100)
                else:
                    st.caption(f"✅ {warmup.get('message', 'Model ready')}")
        except:
            pass
    else:
        st.error(f"❌ Not connected: {health.get('error', 'Unknown error')}")
        st.session_state.daemon_connected = False
        st.info("Start daemon with:\n```\npython tft_inference_daemon.py\n```")

    st.divider()

    # Refresh settings
    st.subheader("Refresh Settings")
    refresh_interval = st.slider(
        "Auto-refresh interval (seconds)",
        min_value=5,
        max_value=300,
        value=st.session_state.refresh_interval,
        step=5,
        help="How often to fetch new predictions"
    )
    st.session_state.refresh_interval = refresh_interval

    auto_refresh = st.checkbox("Enable auto-refresh", value=True)

    if st.button("🔄 Refresh Now", use_container_width=True):
        # Force cache clear
        if 'cached_predictions' in st.session_state:
            del st.session_state['cached_predictions']
        if 'cached_alerts' in st.session_state:
            del st.session_state['cached_alerts']
        st.session_state.last_update = None
        st.rerun()

    st.divider()

    # Interactive Demo Control (Scenario Switcher)
    st.subheader("🎬 Interactive Demo Control")

    @st.fragment
    def render_scenario_controls():
        """Scenario control buttons - isolated to prevent full app rerun"""
        if st.session_state.daemon_connected:
            st.markdown("**Scenario Control:** (Changes metrics generator behavior)")

            col1, col2, col3 = st.columns(3)

            # Metrics generator URL (port 8001)
            generator_url = "http://localhost:8001"

            with col1:
                if st.button("🟢 Healthy", use_container_width=True, key="scenario_healthy"):
                    try:
                        response = requests.post(
                            f"{generator_url}/scenario/set",
                            json={"scenario": "healthy"},
                            timeout=2
                        )
                        if response.ok:
                            st.success("✅ Scenario: Healthy")
                        else:
                            st.error(f"Failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"Cannot connect to generator")

            with col2:
                if st.button("🟡 Degrading", use_container_width=True, key="scenario_degrading"):
                    try:
                        response = requests.post(
                            f"{generator_url}/scenario/set",
                            json={"scenario": "degrading"},
                            timeout=2
                        )
                        if response.ok:
                            result = response.json()
                            st.warning(f"⚠️ Degrading - {result.get('affected_servers', 0)} affected")
                        else:
                            st.error(f"Failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"Cannot connect to generator")

            with col3:
                if st.button("🔴 Critical", use_container_width=True, key="scenario_critical"):
                    try:
                        response = requests.post(
                            f"{generator_url}/scenario/set",
                            json={"scenario": "critical"},
                            timeout=2
                        )
                        if response.ok:
                            result = response.json()
                            st.error(f"🔴 Critical - {result.get('affected_servers', 0)} in crisis!")
                        else:
                            st.error(f"Failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"Cannot connect to generator")

            # Show current scenario status
            try:
                scenario_response = requests.get(f"{generator_url}/scenario/status", timeout=1)
                if scenario_response.ok:
                    status = scenario_response.json()
                    st.info(f"📊 Current: {status['scenario'].upper()} | "
                            f"Affected: {status.get('total_affected', 0)} servers | "
                            f"Tick: {status.get('tick_count', 0)}")
            except:
                st.caption("💡 Tip: Start metrics generator on port 8001")
        else:
            st.warning("⚠️ Connect to daemon to use scenario control")

    # Render the fragment
    render_scenario_controls()

    st.divider()

    # Demo Mode (Legacy)
    st.subheader("🎬 Demo Mode (Legacy)")

    demo_enabled = st.checkbox(
        "Enable Demo Mode",
        value=st.session_state.demo_mode,
        help="Generate and stream demo scenarios in real-time"
    )
    st.session_state.demo_mode = demo_enabled

    if demo_enabled:
        st.markdown("**Select Scenario:**")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🟢 Stable", use_container_width=True, disabled=st.session_state.demo_running):
                initialize_demo_generator('stable', fleet_size=90)
                st.success("✅ Stable scenario initialized")

        with col2:
            if st.button("🟡 Degrading", use_container_width=True, disabled=st.session_state.demo_running):
                initialize_demo_generator('degrading', fleet_size=90)
                st.success("✅ Degrading scenario initialized")

        with col3:
            if st.button("🔴 Critical", use_container_width=True, disabled=st.session_state.demo_running):
                initialize_demo_generator('critical', fleet_size=90)
                st.success("✅ Critical scenario initialized")

        # Demo status
        if st.session_state.demo_running and st.session_state.demo_generator is not None:
            scenario_info = st.session_state.demo_generator.get_scenario_info()
            st.info(f"🎬 Running: {st.session_state.demo_scenario.upper()}")

            # Show scenario progress
            tick_count = scenario_info['tick_count']
            elapsed = (datetime.now() - st.session_state.demo_start_time).total_seconds()
            st.caption(f"Ticks: {tick_count} | Elapsed: {elapsed:.0f}s")

            # Show degrading servers
            if scenario_info['degrading_servers'] > 0:
                st.caption(f"⚠️ {scenario_info['degrading_servers']} servers degrading")

            if st.button("⏹️ Stop Demo", use_container_width=True):
                stop_demo()
                st.success("✅ Demo stopped")

    st.divider()

    # Info
    st.subheader("📊 Dashboard Info")
    if st.session_state.last_update:
        time_since = (datetime.now() - st.session_state.last_update).total_seconds()
        if time_since < 2:
            st.success(f"🔄 Updated: {st.session_state.last_update.strftime('%H:%M:%S')} (just now)")
        else:
            st.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')} ({int(time_since)}s ago)")
    st.caption(f"Refresh interval: {refresh_interval}s")

    # Status indicator
    st.divider()
    if st.session_state.daemon_connected:
        if st.session_state.demo_running:
            st.info("🎬 Demo Mode Active")
        else:
            st.success("🟢 Dashboard Active")
    else:
        st.warning("🟡 Daemon Offline")

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

st.title("🔮 TFT Monitoring Dashboard")

# Add refresh indicator
col1, col2 = st.columns([4, 1])
with col1:
    st.caption("Temporal Fusion Transformer - Real-time Server Monitoring & Prediction")
with col2:
    if st.session_state.daemon_connected and st.session_state.last_update:
        time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
        if time_since_update < 2:
            st.success("🔄 Just Updated", icon="✅")
        else:
            next_update = refresh_interval - time_since_update
            if next_update > 0:
                st.info(f"⏱️ Next: {int(next_update)}s", icon="🔄")

# Demo Mode: Stream data to daemon (time-based, no blocking)
if st.session_state.demo_running and st.session_state.demo_generator is not None:
    current_time = datetime.now()

    # Only generate tick if 5 seconds have passed since last tick
    should_tick = (
        st.session_state.last_demo_tick is None or
        (current_time - st.session_state.last_demo_tick).total_seconds() >= 5
    )

    if should_tick:
        # Generate next tick of data
        tick_data = st.session_state.demo_generator.generate_tick()

        # Convert to records and feed to daemon
        records = tick_data.to_dict('records')

        # Convert timestamps to strings for JSON serialization
        for record in records:
            if 'timestamp' in record:
                record['timestamp'] = str(record['timestamp'])

        if st.session_state.daemon_connected:
            # Feed data to daemon
            success = client.feed_data(records)

            if not success:
                st.warning("⚠️ Failed to feed demo data to daemon")

        # Update last tick time
        st.session_state.last_demo_tick = current_time

# Fetch current data (with smart caching to avoid re-fetching on UI changes)
if st.session_state.daemon_connected:
    # Only fetch if refresh interval has passed or no cached data
    should_fetch = (
        'cached_predictions' not in st.session_state or
        st.session_state.last_update is None or
        (datetime.now() - st.session_state.last_update).total_seconds() >= refresh_interval
    )

    if should_fetch:
        with st.spinner('🔮 Fetching predictions from TFT model...'):
            predictions = client.get_predictions()
            alerts = client.get_alerts()
            st.session_state.last_update = datetime.now()

            # Cache for fast UI updates
            st.session_state.cached_predictions = predictions
            st.session_state.cached_alerts = alerts

        # Show brief toast notification on successful refresh
        if predictions:
            st.toast("✅ Data refreshed!", icon="🔄")

        # Store in history
        if predictions:
            st.session_state.history.append({
                'timestamp': datetime.now(),
                'predictions': predictions
            })
            # Keep last 100 entries
            st.session_state.history = st.session_state.history[-100:]
    else:
        # Use cached data (much faster for dropdown changes!)
        predictions = st.session_state.get('cached_predictions')
        alerts = st.session_state.get('cached_alerts')
else:
    predictions = None
    alerts = None
    st.warning("⚠️ Daemon not connected. Connect to see live predictions.")

# =============================================================================
# TABS - Now using modular components
# =============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Overview",
    "🔥 Heatmap",
    "⚠️ Top 5 Servers",
    "📈 Historical",
    "💰 Cost Avoidance",
    "🤖 Auto-Remediation",
    "📱 Alerting Strategy",
    "⚙️ Advanced",
    "📚 Documentation",
    "🗺️ Roadmap"
])

# Tab 1: Overview
with tab1:
    overview.render(predictions, daemon_url)

# Tab 2: Heatmap
with tab2:
    heatmap.render(predictions)

# Tab 3: Top 5 Risks
with tab3:
    top_risks.render(predictions)

# Tab 4: Historical Trends
with tab4:
    historical.render(predictions)

# Tab 5: Cost Avoidance
with tab5:
    cost_avoidance.render(predictions)

# Tab 6: Auto-Remediation
with tab6:
    auto_remediation.render(predictions)

# Tab 7: Alerting Strategy
with tab7:
    alerting.render(predictions)

# Tab 8: Advanced
with tab8:
    advanced.render(predictions, client, daemon_url)

# Tab 9: Documentation
with tab9:
    documentation.render(predictions)

# Tab 10: Roadmap
with tab10:
    roadmap.render(predictions)

# =============================================================================
# AUTO-REFRESH (Corporate-friendly: Less aggressive rerun behavior)
# =============================================================================

if auto_refresh and st.session_state.daemon_connected:
    # Check if enough time has passed since last update
    current_time = datetime.now()

    # Initialize last_update if needed
    if st.session_state.last_update is None:
        st.session_state.last_update = current_time

    time_since_update = (current_time - st.session_state.last_update).total_seconds()

    # Only rerun if the refresh interval has actually elapsed
    # NOTE: Reduced aggressiveness for corporate browser compatibility
    if st.session_state.demo_running and time_since_update >= 5:
        # Use st.empty() placeholder to avoid full page rerun
        time.sleep(5)  # Wait the full interval
        st.rerun()
    elif time_since_update >= refresh_interval:
        # Corporate-friendly: Add small buffer to prevent rapid reruns
        time.sleep(min(refresh_interval + 1, 10))  # Add 1s buffer, max 10s
        st.rerun()

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("🔮 TFT Monitoring Dashboard | Built with Streamlit | Powered by Temporal Fusion Transformer")
