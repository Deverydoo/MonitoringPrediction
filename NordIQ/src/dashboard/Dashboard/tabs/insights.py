"""
Insights Tab - Explainable AI (XAI) Analysis

Provides deep insights into WHY predictions happen and WHAT TO DO:
- SHAP feature importance (which metrics drove the prediction)
- Attention analysis (which time periods the model focused on)
- Counterfactual scenarios (what-if analysis with actionable recommendations)

This is a key differentiator - shows the AI's reasoning in plain English!
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional

from Dashboard.utils import (
    calculate_server_risk_score,
    extract_cpu_used,
    get_server_profile
)
from Dashboard.config.dashboard_config import DAEMON_URL, DAEMON_API_KEY
from core.alert_levels import ALERT_COLORS_HEX, AlertLevel

# Professional metric display names
METRIC_DISPLAY_NAMES = {
    'cpu_user_pct': 'CPU User %',
    'cpu_sys_pct': 'CPU System %',
    'cpu_iowait_pct': 'CPU I/O Wait %',
    'cpu_idle_pct': 'CPU Idle %',
    'java_cpu_pct': 'Java CPU %',
    'mem_used_pct': 'Memory Used %',
    'swap_used_pct': 'Swap Used %',
    'disk_usage_pct': 'Disk Usage %',
    'net_in_mb_s': 'Network In (MB/s)',
    'net_out_mb_s': 'Network Out (MB/s)',
    'back_close_wait': 'Backend Close-Wait Connections',
    'front_close_wait': 'Frontend Close-Wait Connections',
    'load_average': 'System Load Average',
    'uptime_days': 'Server Uptime (days)',
    # Legacy names (fallback)
    'cpu_pct': 'CPU %',
    'mem_pct': 'Memory %',
    'disk_io_mb_s': 'Disk I/O (MB/s)',
    'latency_ms': 'Latency (ms)'
}


def get_metric_display_name(metric_name: str) -> str:
    """Convert internal metric name to user-friendly display name."""
    return METRIC_DISPLAY_NAMES.get(
        metric_name,
        # Fallback: capitalize and replace underscores
        metric_name.replace('_', ' ').replace('pct', '%').title()
    )


@st.cache_data(ttl=30, show_spinner=False)
def fetch_explanation(server_name: str, daemon_url: str = DAEMON_URL) -> Optional[Dict]:
    """
    Fetch XAI explanation for a specific server from the daemon.

    Cached for 30 seconds to improve performance.

    Args:
        server_name: Server to explain
        daemon_url: URL of the inference daemon

    Returns:
        Dict with SHAP, attention, and counterfactual explanations, or None if error
    """
    try:
        # Prepare headers with API key
        headers = {}
        if DAEMON_API_KEY:
            headers["X-API-Key"] = DAEMON_API_KEY

        response = requests.get(
            f"{daemon_url}/explain/{server_name}",
            headers=headers,
            timeout=10
        )
        if response.ok:
            return response.json()
        else:
            st.error(f"Failed to fetch explanation: {response.status_code}")
            if response.status_code == 403:
                st.error("❌ Authentication failed - check API key configuration")
            return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out - XAI analysis can take a few seconds")
        return None
    except Exception as e:
        st.error(f"Error fetching explanation: {str(e)}")
        return None


def render_shap_explanation(shap_data: Dict):
    """
    Render SHAP feature importance analysis.

    Shows which metrics (CPU, memory, disk, network) drove the prediction.
    """
    st.subheader("📊 Feature Importance (SHAP)")
    st.markdown("""
    **What This Shows**: Which metrics are driving the prediction.
    ⭐⭐⭐ = Very high impact | ⭐⭐ = Medium impact | ⭐ = Low impact
    """)

    feature_importance = shap_data.get('feature_importance', [])

    if not feature_importance:
        st.info("No feature importance data available")
        return

    # Display summary
    summary = shap_data.get('summary', 'No summary available')
    st.markdown(f"**Summary**: {summary}")

    # Create bar chart of feature importance
    features = []
    impacts = []
    directions = []
    stars = []

    for feature, info in feature_importance:
        # Use professional display names
        feature_display = get_metric_display_name(feature)
        features.append(feature_display)
        impacts.append(info['impact'] * 100)  # Convert to percentage
        directions.append(info['direction'])
        stars.append(info.get('stars', ''))

    # Create DataFrame for display
    df = pd.DataFrame({
        'Metric': features,
        'Impact': impacts,
        'Direction': directions,
        'Importance': stars
    })

    # Plotly bar chart
    fig = go.Figure()

    # Color by direction using standardized alert colors
    # Increasing risk (bad) = red, decreasing risk (good) = green, neutral = gray
    colors = [
        ALERT_COLORS_HEX[AlertLevel.HEALTHY] if d == 'increasing'  # Green (good - risk going down)
        else ALERT_COLORS_HEX[AlertLevel.CRITICAL] if d == 'decreasing'  # Red (bad - risk going up)
        else '#6B7280'  # Gray (neutral)
        for d in directions
    ]

    fig.add_trace(go.Bar(
        x=impacts,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{imp:.1f}%" for imp in impacts],
        textposition='auto'
    ))

    fig.update_layout(
        title="Feature Impact on Prediction",
        xaxis_title="Impact (%)",
        yaxis_title="Metric",
        height=400,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    with st.expander("📋 Detailed Breakdown"):
        st.dataframe(df, use_container_width=True)


def render_attention_analysis(attention_data: Dict):
    """
    Render attention visualization showing which time periods the model focused on.
    """
    st.subheader("⏱️ Temporal Attention Analysis")
    st.markdown("""
    **What This Shows**: Which time periods the model "paid attention to" when making the prediction.
    Higher attention = more influence on the prediction.
    """)

    summary = attention_data.get('summary', 'No summary available')
    st.markdown(f"**Summary**: {summary}")

    important_periods = attention_data.get('important_periods', [])

    if important_periods:
        # Show important periods as cards
        cols = st.columns(min(len(important_periods), 3))

        for i, period in enumerate(important_periods):
            with cols[i % 3]:
                attention_pct = period['attention'] * 100
                importance = period['importance']

                # Color based on importance using standardized alert levels
                if importance == 'VERY HIGH':
                    color = ALERT_COLORS_HEX[AlertLevel.CRITICAL]  # Red
                elif importance == 'HIGH':
                    color = ALERT_COLORS_HEX[AlertLevel.WARNING]  # Orange
                elif importance == 'MEDIUM':
                    color = ALERT_COLORS_HEX[AlertLevel.WATCH]  # Yellow
                else:
                    color = '#6B7280'  # Gray (neutral/low)

                st.markdown(f"""
                <div style="background-color: {color}22; padding: 15px; border-radius: 8px; border-left: 4px solid {color};">
                    <p style="margin: 0; font-weight: bold;">{period['period']}</p>
                    <p style="margin: 5px 0 0 0; font-size: 24px; color: {color};">{attention_pct:.0f}%</p>
                    <p style="margin: 5px 0 0 0; font-size: 12px;">{importance} importance</p>
                </div>
                """, unsafe_allow_html=True)

    # Attention weights timeline (if available)
    attention_weights = attention_data.get('attention_weights', [])
    if attention_weights and len(attention_weights) > 10:
        with st.expander("📈 Attention Timeline"):
            # Create line chart
            fig = go.Figure()

            timesteps = list(range(len(attention_weights)))

            # Use Scattergl for GPU-accelerated rendering (faster for >10 points)
            fig.add_trace(go.Scattergl(
                x=timesteps,
                y=attention_weights,
                mode='lines',
                fill='tozeroy',
                line=dict(color='#0EA5E9', width=2),
                name='Attention Weight'
            ))

            fig.update_layout(
                title="Attention Weights Over Time",
                xaxis_title="Timestep (most recent = right)",
                yaxis_title="Attention Weight",
                height=300,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)


# Scenario icons for visual appeal
SCENARIO_ICONS = {
    'restart': '🔄',
    'scale': '📈',
    'stabilize': '⚖️',
    'optimize': '⚡',
    'reduce': '🧹',
    'nothing': '⏸️'
}


def get_scenario_icon(scenario_name: str) -> str:
    """Get icon for scenario based on keywords."""
    scenario_lower = scenario_name.lower()
    if 'restart' in scenario_lower:
        return SCENARIO_ICONS['restart']
    elif 'scale' in scenario_lower:
        return SCENARIO_ICONS['scale']
    elif 'stabilize' in scenario_lower:
        return SCENARIO_ICONS['stabilize']
    elif 'optimize' in scenario_lower or 'disk' in scenario_lower:
        return SCENARIO_ICONS['optimize']
    elif 'reduce' in scenario_lower or 'memory' in scenario_lower:
        return SCENARIO_ICONS['reduce']
    elif 'nothing' in scenario_lower:
        return SCENARIO_ICONS['nothing']
    else:
        return '💡'  # Default


def render_counterfactual_scenarios(counterfactual_data: List[Dict]):
    """
    Render what-if scenarios with actionable recommendations.
    """
    st.subheader("🎯 What-If Scenarios")
    st.markdown("""
    **What This Shows**: Different actions you could take and their predicted impact.
    ✅ = Safe outcome | ⚠️ = Risky outcome | 📊 = Predicted impact
    """)

    if not counterfactual_data:
        st.info("No scenarios available")
        return

    # Find best scenario (lowest predicted CPU that is safe)
    # Score = improvement (negative change) + safety bonus + low effort bonus
    best_scenario = None
    best_score = float('-inf')

    for scenario in counterfactual_data:
        predicted_cpu = scenario.get('predicted_cpu', 100)
        change = scenario.get('change', 0)
        is_safe = scenario.get('safe', False)
        effort = scenario.get('effort', 'MEDIUM')

        # Calculate score: bigger negative change = better, safe = bonus, low effort = bonus
        score = -change  # Negative change is good (reduction)
        if is_safe:
            score += 20  # Safety bonus
        if effort == 'LOW':
            score += 10
        elif effort == 'MEDIUM':
            score += 5

        if score > best_score:
            best_score = score
            best_scenario = scenario

    # Display best recommendation
    if best_scenario:
        scenario_name = best_scenario.get('scenario', 'Unknown')
        predicted_cpu = best_scenario.get('predicted_cpu', 0)
        change = best_scenario.get('change', 0)
        is_safe = best_scenario.get('safe', False)
        effort = best_scenario.get('effort', 'MEDIUM')

        safety_icon = "✅" if is_safe else "⚠️"
        scenario_icon = get_scenario_icon(scenario_name)
        # Use standardized alert colors: green for safe, red for unsafe
        color = ALERT_COLORS_HEX[AlertLevel.HEALTHY] if is_safe else ALERT_COLORS_HEX[AlertLevel.CRITICAL]

        st.markdown(f"""
        <div style="background-color: {color}22; padding: 20px; border-radius: 10px; border: 2px solid {color}; margin-bottom: 20px;">
            <h3 style="margin: 0; color: {color};">{safety_icon} Recommended Action</h3>
            <p style="font-size: 18px; margin: 10px 0; font-weight: bold;">{scenario_icon} {scenario_name}</p>
            <p style="margin: 5px 0;">Predicted CPU: <strong>{predicted_cpu:.1f}%</strong> ({change:+.1f}% change)</p>
            <p style="margin: 5px 0;">Effort: <strong>{effort}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # Display all scenarios in expandable section
    with st.expander("📋 All Scenarios"):
        for scenario in counterfactual_data:
            scenario_name = scenario.get('scenario', 'Unknown')
            predicted_cpu = scenario.get('predicted_cpu', 0)
            change = scenario.get('change', 0)
            is_safe = scenario.get('safe', False)
            effort = scenario.get('effort', 'MEDIUM')
            risk = scenario.get('risk', 'MEDIUM')

            safety_icon = "✅" if is_safe else "⚠️"
            scenario_icon = get_scenario_icon(scenario_name)

            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

            with col1:
                st.markdown(f"**{safety_icon} {scenario_icon} {scenario_name}**")
            with col2:
                st.metric("Predicted CPU", f"{predicted_cpu:.1f}%", delta=f"{change:+.1f}%")
            with col3:
                st.text(f"Effort: {effort}")
            with col4:
                st.text(f"Risk: {risk}")

            st.divider()


@st.fragment
def render(predictions: Optional[Dict], daemon_url: str = DAEMON_URL):
    """
    Render the Insights (XAI) tab.

    PHASE 4 OPTIMIZATION: Fragment-based rendering - only reruns when needed.

    Args:
        predictions: Current predictions from daemon
        daemon_url: URL of the inference daemon
    """
    st.markdown("### 🧠 Explainable AI Insights")
    st.markdown("""
    This tab provides deep insights into **WHY** the AI made its predictions and **WHAT TO DO** about them.

    **ArgusAI Advantage**: Most monitoring tools just show you numbers. We show you the **reasoning** behind them.
    """)

    if not predictions or 'predictions' not in predictions:
        st.warning("No predictions available. Start the inference daemon and metrics generator.")
        return

    # Server selection
    servers = list(predictions['predictions'].keys())

    if not servers:
        st.info("No servers being monitored")
        return

    # Sort servers by risk score (highest first)
    servers_with_risk = []
    for server in servers:
        pred = predictions['predictions'][server]
        risk = calculate_server_risk_score(pred)
        servers_with_risk.append((server, risk))

    servers_with_risk.sort(key=lambda x: x[1], reverse=True)
    sorted_servers = [s[0] for s in servers_with_risk]

    selected_server = st.selectbox(
        "Select server to analyze:",
        sorted_servers,
        format_func=lambda s: f"{s} (Risk: {calculate_server_risk_score(predictions['predictions'][s]):.0f})"
    )

    if selected_server:
        # Fetch explanation
        with st.spinner(f"🔍 Analyzing {selected_server}... This may take a few seconds."):
            explanation = fetch_explanation(selected_server, daemon_url)

        if explanation and 'error' not in explanation:
            # Show server context
            server_pred = explanation.get('prediction', {})
            profile = get_server_profile(selected_server)

            col1, col2, col3 = st.columns(3)

            with col1:
                cpu_used = extract_cpu_used(server_pred)
                st.metric("Current CPU", f"{cpu_used:.1f}%")

            with col2:
                mem_used = server_pred.get('mem_used_pct', {}).get('current', 0)
                st.metric("Current Memory", f"{mem_used:.1f}%")

            with col3:
                st.metric("Profile", profile)

            st.divider()

            # Render XAI components
            tabs = st.tabs(["📊 Feature Importance", "⏱️ Temporal Focus", "🎯 What-If Scenarios"])

            with tabs[0]:
                if 'shap' in explanation:
                    render_shap_explanation(explanation['shap'])
                else:
                    st.info("SHAP analysis not available")

            with tabs[1]:
                if 'attention' in explanation:
                    render_attention_analysis(explanation['attention'])
                else:
                    st.info("Attention analysis not available")

            with tabs[2]:
                if 'counterfactuals' in explanation:
                    render_counterfactual_scenarios(explanation['counterfactuals'])
                else:
                    st.info("Counterfactual scenarios not available")

        elif explanation and 'error' in explanation:
            st.error(f"Error generating explanation: {explanation.get('message', 'Unknown error')}")

            # Show debug info in expander
            if 'traceback' in explanation:
                with st.expander("🐛 Debug Information"):
                    st.code(explanation['traceback'])
        else:
            st.error("Failed to fetch explanation from daemon")
