"""
Advanced Tab - System diagnostics and configuration

Provides:
- System information (dashboard and daemon status)
- Alert threshold configuration (placeholder)
- Debug information and raw prediction data
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Optional


def render(predictions: Optional[Dict], client, daemon_url: str):
    """
    Render the Advanced tab.

    Args:
        predictions: Current predictions from daemon
        client: DaemonClient instance
        daemon_url: URL of the inference daemon
    """
    st.subheader("⚙️ Advanced Settings & Diagnostics")

    # System Info
    st.markdown("### System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Dashboard**")
        st.code(f"""
Python Version: 3.10+
Streamlit Version: {st.__version__}
Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
History Size: {len(st.session_state.history)} entries
        """)

    with col2:
        st.markdown("**Daemon**")
        if st.session_state.daemon_connected:
            health = client.check_health()
            health_data = health.get('data', {})
            st.code(f"""
URL: {daemon_url}
Status: Connected ✅
Model Loaded: {health_data.get('model_loaded', 'Unknown')}
Uptime: {health_data.get('uptime', 'Unknown')}
            """)
        else:
            st.code(f"""
URL: {daemon_url}
Status: Not Connected ❌
            """)

    st.divider()

    # Alert Thresholds
    st.markdown("### Alert Thresholds")

    st.info("🚧 Alert threshold configuration coming soon")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("CPU Warning Threshold (%)", value=70, min_value=0, max_value=100)
        st.number_input("Memory Warning Threshold (%)", value=80, min_value=0, max_value=100)

    with col2:
        st.number_input("CPU Critical Threshold (%)", value=90, min_value=0, max_value=100)
        st.number_input("Memory Critical Threshold (%)", value=95, min_value=0, max_value=100)

    st.divider()

    # Debug Info
    with st.expander("🔍 Debug Information"):
        st.markdown("**Session State**")
        st.json({
            'daemon_url': st.session_state.daemon_url,
            'daemon_connected': st.session_state.daemon_connected,
            'refresh_interval': st.session_state.refresh_interval,
            'history_size': len(st.session_state.history),
            'last_update': str(st.session_state.last_update)
        })

        if predictions:
            st.markdown("**Latest Predictions (Raw)**")
            st.json(predictions)
