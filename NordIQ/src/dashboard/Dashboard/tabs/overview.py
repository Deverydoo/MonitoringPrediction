"""
Overview Tab - Main dashboard view with KPIs, alerts, and risk distribution

Provides high-level fleet health overview with:
- Environment status and incident probabilities
- Actual vs AI prediction comparison
- Fleet risk distribution charts
- Active alerts table with actual vs predicted metrics
- Healthy server summary when no alerts
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Optional

from Dashboard.utils import (
    calculate_server_risk_score,
    extract_cpu_used,
    get_health_status,
    get_metric_color_indicator,
    get_server_profile
)
from Dashboard.config.dashboard_config import DAEMON_URL, METRICS_GENERATOR_URL


@st.cache_data(ttl=2, show_spinner=False)
def fetch_warmup_status(daemon_url: str):
    """Cached warmup status check (2s TTL to reduce load)."""
    try:
        response = requests.get(f"{daemon_url}/status", timeout=2)
        if response.ok:
            return response.json().get('warmup', {})
    except:
        pass
    return None


@st.cache_data(ttl=2, show_spinner=False)
def fetch_scenario_status(generator_url: str):
    """Cached scenario status check (2s TTL to reduce load)."""
    try:
        response = requests.get(f"{generator_url}/scenario/status", timeout=1)
        if response.ok:
            return response.json()
    except:
        pass
    return None


def render(predictions: Optional[Dict], daemon_url: str = DAEMON_URL):
    """
    Render the Overview tab.

    Args:
        predictions: Current predictions from daemon
        daemon_url: URL of the inference daemon
    """
    if predictions:
        # Check warmup status and show warning if model not ready (CACHED)
        warmup = fetch_warmup_status(daemon_url)
        if warmup and not warmup.get('is_warmed_up', True):
            progress = warmup.get('progress_percent', 0)
            st.warning(f"""
            ⏳ **Model Warming Up** ({progress:.0f}% complete)

            The model is still learning from incoming data. Predictions may be inconsistent during warm-up.
            Once warmed up, all metrics will tell a consistent story.

            **What's happening:** The model has {warmup.get('current_size', 0)}/{warmup.get('required_size', 288)} data points needed per server for reliable predictions.
            """, icon="⏳")

        # Environment status
        status_text, status_color, status_emoji = get_health_status(predictions, calculate_server_risk_score)

        # Top KPIs
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Build status explanation
            if status_text == "Critical":
                help_text = "🔴 CRITICAL: >30% of fleet is Critical/Imminent Failure (Risk 80+) OR >50% are in Danger/Warning state. Immediate action required."
            elif status_text == "Warning":
                help_text = "🟠 WARNING: >10% of fleet is Critical/Imminent Failure (Risk 80+) OR >30% are in Danger/Warning state. Monitor closely."
            elif status_text == "Caution":
                help_text = "🟡 CAUTION: >10% of fleet is Degrading (Risk 50+). Early warning - investigate soon."
            elif status_text == "Healthy":
                help_text = "🟢 HEALTHY: <10% of fleet has elevated risk. Normal operations."
            else:
                help_text = "Status unknown - waiting for predictions."

            st.metric(
                label="Environment Status",
                value=f"{status_emoji} {status_text}",
                delta=None,
                help=help_text
            )

        with col2:
            env = predictions.get('environment', {})
            prob_30m = env.get('prob_30m', 0) * 100
            st.metric(
                label="Incident Risk (30m)",
                value=f"{prob_30m:.1f}%",
                delta=None,
                help="Probability of incident in next 30 minutes"
            )

        with col3:
            prob_8h = env.get('prob_8h', 0) * 100
            st.metric(
                label="Incident Risk (8h)",
                value=f"{prob_8h:.1f}%",
                delta=None,
                help="Probability of incident in next 8 hours"
            )

        with col4:
            server_preds = predictions.get('predictions', {})
            total_servers = len(server_preds)
            # Healthy = risk < 50 (matching Active Alerts threshold)
            healthy_count = sum(1 for s, p in server_preds.items() if calculate_server_risk_score(p) < 50)
            st.metric(
                label="Fleet Status",
                value=f"{healthy_count}/{total_servers}",
                delta=None,
                help="Healthy servers (Risk < 50) / Total servers"
            )

        st.divider()

        # Actual vs Predicted Comparison (Management View)
        st.subheader("🎯 Actual State vs AI Prediction")
        st.markdown("**Show the power of predictive AI** - Current reality vs future forecast")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📍 **Actual Current State**")
            st.markdown("_(What's happening RIGHT NOW)_")

            # Try to get actual scenario from generator (CACHED)
            generator_url = METRICS_GENERATOR_URL
            status = fetch_scenario_status(generator_url)
            if status:
                actual_scenario = status['scenario'].upper()
                actual_affected = status.get('total_affected', 0)

                if actual_scenario == 'HEALTHY':
                    st.success(f"✅ **{actual_scenario}**")
                    st.metric("Affected Servers (Now)", f"{actual_affected}")
                    st.caption("Environment is currently operating normally")
                elif actual_scenario == 'DEGRADING':
                    st.warning(f"⚠️ **{actual_scenario}**")
                    st.metric("Affected Servers (Now)", f"{actual_affected}")
                    st.caption("Some servers experiencing issues")
                else:  # CRITICAL
                    st.error(f"🔴 **{actual_scenario}**")
                    st.metric("Affected Servers (Now)", f"{actual_affected}")
                    st.caption("Multiple servers in critical state")
            else:
                st.info("💡 Actual state: Unknown (connect metrics generator on port 8001)")

        with col2:
            st.markdown("### 🔮 **AI Prediction (Next 30min-8h)**")
            st.markdown("_(What the AI forecasts will happen)_")

            # Show prediction based on ACTUAL incident probabilities (not server risk scores)
            env = predictions.get('environment', {})
            prob_30m = env.get('prob_30m', 0) * 100
            prob_8h = env.get('prob_8h', 0) * 100

            # Determine prediction status based on incident probabilities
            if prob_30m > 70 or prob_8h > 85:
                predicted_status = "CRITICAL"
                st.error(f"🔴 **{predicted_status}**")
                st.caption("⚠️ High probability of incidents ahead")
            elif prob_30m > 40 or prob_8h > 60:
                predicted_status = "WARNING"
                st.warning(f"🟠 **{predicted_status}**")
                st.caption("⚠️ Elevated risk of incidents")
            elif prob_30m > 20 or prob_8h > 40:
                predicted_status = "CAUTION"
                st.warning(f"🟡 **{predicted_status}**")
                st.caption("⚠️ Minor risk indicators detected")
            else:
                predicted_status = "HEALTHY"
                st.success(f"✅ **{predicted_status}**")
                st.caption("AI predicts continued stability")

            st.metric("Predicted Incident Risk (30m)", f"{prob_30m:.1f}%")
            st.metric("Predicted Incident Risk (8h)", f"{prob_8h:.1f}%")

        # Insight box (CACHED)
        generator_url = METRICS_GENERATOR_URL
        status = fetch_scenario_status(generator_url)
        if status:
            actual_scenario = status['scenario'].upper()

            if actual_scenario == 'HEALTHY' and predicted_status in ['CRITICAL', 'WARNING']:
                st.warning(f"""
                🎯 **This is the value of Predictive AI!**

                - **Current Reality**: Environment is HEALTHY (no active issues)
                - **AI Forecast**: {prob_30m:.0f}% risk in 30m, {prob_8h:.0f}% risk in 8h
                - **Action Window**: Act NOW to prevent issues before they occur
                - **Value**: Proactive prevention vs reactive firefighting
                """)
            elif actual_scenario != 'HEALTHY' and predicted_status in ['CRITICAL', 'WARNING']:
                st.info("""
                ✅ **AI accurately detecting ongoing issues**

                The model is correctly identifying the current degradation and predicting continued problems.
                """)
            elif actual_scenario == 'HEALTHY' and predicted_status == 'HEALTHY':
                st.success("""
                ✅ **All systems stable**

                Both current state and predictions show healthy operations.
                """)

        st.divider()

        # Server Risk Distribution
        col_header1, col_header2 = st.columns([3, 1])
        with col_header1:
            st.subheader("📊 Fleet Risk Distribution")
        with col_header2:
            st.markdown("")  # Spacer
            st.caption("ℹ️ No visible data? **No news is good news!** An empty/flat chart means all servers are healthy (Risk=0).")

        if server_preds:
            # Calculate risk scores
            server_risks = []
            for server_name, server_pred in server_preds.items():
                risk_score = calculate_server_risk_score(server_pred)
                server_risks.append({
                    'Server': server_name,
                    'Risk Score': risk_score,
                    'Status': 'Critical' if risk_score >= 80 else
                             'Warning' if risk_score >= 60 else
                             'Degrading' if risk_score >= 50 else
                             'Watch' if risk_score >= 30 else 'Healthy'
                })

            risk_df = pd.DataFrame(server_risks)

            col1, col2 = st.columns([2, 1])

            with col1:
                # Bar chart of risks
                fig = px.bar(
                    risk_df.sort_values('Risk Score', ascending=False).head(15),
                    x='Server',
                    y='Risk Score',
                    color='Risk Score',
                    color_continuous_scale=['green', 'yellow', 'orange', 'red'],
                    range_color=[0, 100],
                    title="Top 15 Servers by Risk Score",
                    height=400
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            with col2:
                # Risk distribution pie
                status_counts = risk_df['Status'].value_counts()
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Server Status Distribution",
                    color=status_counts.index,
                    color_discrete_map={
                        'Healthy': 'green',
                        'Watch': 'lightblue',
                        'Degrading': 'gold',
                        'Warning': 'orange',
                        'Critical': 'red'
                    },
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.divider()

        # Active Alerts with Real vs Predicted
        st.subheader("🔔 Active Alerts")
        st.markdown("**Real-time comparison**: Current values vs AI predictions")

        if predictions and server_preds:
            # Build enhanced alert table with actual vs predicted
            alert_rows = []

            for server_name, server_pred in server_preds.items():
                risk_score = calculate_server_risk_score(server_pred)

                # Only show servers with risk >= 50 (Warning and above)
                if risk_score >= 50:
                    # Determine alert severity based on risk score
                    if risk_score >= 90:
                        severity = "🔴 Imminent Failure"
                        priority = "Imminent Failure"
                    elif risk_score >= 80:
                        severity = "🔴 Critical"
                        priority = "Critical"
                    elif risk_score >= 70:
                        severity = "🟠 Danger"
                        priority = "Danger"
                    elif risk_score >= 60:
                        severity = "🟡 Warning"
                        priority = "Warning"
                    elif risk_score >= 50:
                        severity = "🟢 Degrading"
                        priority = "Degrading"
                    else:
                        continue

                    # Get actual vs predicted for key LINBORG metrics (using helper)
                    cpu_actual = min(100.0, max(0.0, extract_cpu_used(server_pred, 'current')))
                    cpu_predicted = min(100.0, max(0.0, extract_cpu_used(server_pred, 'p50')))

                    # Still need iowait for direct access
                    cpu_iowait_cur = server_pred.get('cpu_iowait_pct', {}).get('current', 0)

                    # I/O Wait - CRITICAL metric
                    iowait_actual = min(100.0, max(0.0, cpu_iowait_cur))
                    iowait_p50 = server_pred.get('cpu_iowait_pct', {}).get('p50', [])
                    iowait_predicted = min(100.0, max(0.0, np.mean(iowait_p50[:6]) if iowait_p50 and len(iowait_p50) >= 6 else iowait_actual))

                    # Memory
                    mem_actual = min(100.0, max(0.0, server_pred.get('mem_used_pct', {}).get('current', 0)))
                    mem_p50 = server_pred.get('mem_used_pct', {}).get('p50', [])
                    mem_predicted = min(100.0, max(0.0, np.mean(mem_p50[:6]) if mem_p50 and len(mem_p50) >= 6 else mem_actual))

                    # Swap (for color-coding)
                    swap_actual = min(100.0, max(0.0, server_pred.get('swap_used_pct', {}).get('current', 0)))
                    swap_p50 = server_pred.get('swap_used_pct', {}).get('p50', [])
                    swap_predicted = min(100.0, max(0.0, np.mean(swap_p50[:6]) if swap_p50 and len(swap_p50) >= 6 else swap_actual))

                    # Load average (for color-coding)
                    load_actual = server_pred.get('load_average', {}).get('current', 0)

                    # Get profile for threshold logic
                    profile = get_server_profile(server_name)

                    # Add color indicators to problematic metrics
                    cpu_now_color = get_metric_color_indicator(cpu_actual, 'cpu', profile)
                    cpu_pred_color = get_metric_color_indicator(cpu_predicted, 'cpu', profile)
                    iowait_now_color = get_metric_color_indicator(iowait_actual, 'iowait', profile)
                    iowait_pred_color = get_metric_color_indicator(iowait_predicted, 'iowait', profile)
                    mem_now_color = get_metric_color_indicator(mem_actual, 'memory', profile)
                    mem_pred_color = get_metric_color_indicator(mem_predicted, 'memory', profile)

                    alert_rows.append({
                        'Priority': priority,
                        'Server': server_name,
                        'Profile': profile,
                        'Risk': f"{risk_score:.0f}",
                        'CPU Now': f"{cpu_now_color}{cpu_actual:.1f}%",
                        'CPU Predicted (30m)': f"{cpu_pred_color}{cpu_predicted:.1f}%",
                        'CPU Δ': f"{(cpu_predicted - cpu_actual):+.1f}%",
                        'I/O Wait Now': f"{iowait_now_color}{iowait_actual:.1f}%",
                        'I/O Wait Predicted (30m)': f"{iowait_pred_color}{iowait_predicted:.1f}%",
                        'I/O Wait Δ': f"{(iowait_predicted - iowait_actual):+.1f}%",
                        'Mem Now': f"{mem_now_color}{mem_actual:.1f}%",
                        'Mem Predicted (30m)': f"{mem_pred_color}{mem_predicted:.1f}%",
                        'Mem Δ': f"{(mem_predicted - mem_actual):+.1f}%"
                    })

            if alert_rows:
                st.markdown(f"**{len(alert_rows)} servers requiring attention**")

                # Sort by risk score (descending)
                alert_df = pd.DataFrame(alert_rows)
                alert_df = alert_df.sort_values('Risk', ascending=False)

                # Display the enhanced table
                st.dataframe(
                    alert_df,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        'Priority': st.column_config.TextColumn('Priority', width='medium', help='Imminent Failure (90+) → Critical (80-89) → Danger (70-79) → Warning (60-69) → Degrading (50-59)'),
                        'Server': st.column_config.TextColumn('Server', width='medium', help='Server hostname'),
                        'Profile': st.column_config.TextColumn('Profile', width='medium', help='Server workload type (ML Compute, Database, Web API, etc.)'),
                        'Risk': st.column_config.TextColumn('Risk', width='small', help='Overall risk score (0-100). Higher = more urgent'),
                        'CPU Now': st.column_config.TextColumn('CPU Now', width='small', help='Current CPU utilization (% Used = 100 - Idle). 🟡=90%+ 🟠=95%+ 🔴=98%+'),
                        'CPU Predicted (30m)': st.column_config.TextColumn('CPU Pred', width='small', help='AI predicted CPU in next 30 minutes. 🟡=90%+ 🟠=95%+ 🔴=98%+'),
                        'CPU Δ': st.column_config.TextColumn('CPU Δ', width='small', help='Predicted change (Delta). Positive (+) = increasing/degrading, Negative (-) = decreasing/improving'),
                        'I/O Wait Now': st.column_config.TextColumn('I/O Wait Now', width='small', help='Current I/O wait % - CRITICAL troubleshooting metric. 🟡=10%+ 🟠=20%+ 🔴=30%+'),
                        'I/O Wait Predicted (30m)': st.column_config.TextColumn('I/O Wait Pred', width='small', help='AI predicted I/O wait in next 30 minutes. 🟡=10%+ 🟠=20%+ 🔴=30%+'),
                        'I/O Wait Δ': st.column_config.TextColumn('I/O Δ', width='small', help='Predicted change (Delta). Positive (+) = increasing I/O contention'),
                        'Mem Now': st.column_config.TextColumn('Mem Now', width='small', help='Current memory utilization. 🟡=90%+ 🟠=95%+ 🔴=98%+ (DB: 🟡=95%+ 🟠=98%+)'),
                        'Mem Predicted (30m)': st.column_config.TextColumn('Mem Pred', width='small', help='AI predicted memory in next 30 minutes. 🟡=90%+ 🟠=95%+ 🔴=98%+'),
                        'Mem Δ': st.column_config.TextColumn('Mem Δ', width='small', help='Predicted change (Delta). Positive (+) = increasing/degrading, Negative (-) = decreasing/improving')
                    }
                )

                # Environment-level health assessment
                st.markdown("---")
                st.markdown("**🏢 Environment Health Assessment**")

                total_servers = len(server_preds)
                alert_count = len(alert_rows)
                alert_percentage = (alert_count / total_servers * 100) if total_servers > 0 else 0

                # Determine environment health
                if alert_percentage < 5:
                    env_status = "✅ HEALTHY"
                    env_color = "green"
                    env_msg = f"{alert_count}/{total_servers} servers alerting ({alert_percentage:.1f}%) - Within normal operational variance"
                elif alert_percentage < 15:
                    env_status = "⚠️ WATCH"
                    env_color = "orange"
                    env_msg = f"{alert_count}/{total_servers} servers alerting ({alert_percentage:.1f}%) - Elevated alert rate, monitor closely"
                elif alert_percentage < 30:
                    env_status = "🟠 DEGRADING"
                    env_color = "orange"
                    env_msg = f"{alert_count}/{total_servers} servers alerting ({alert_percentage:.1f}%) - Significant degradation, investigate root cause"
                else:
                    env_status = "🔴 CRITICAL"
                    env_color = "red"
                    env_msg = f"{alert_count}/{total_servers} servers alerting ({alert_percentage:.1f}%) - Widespread issues, potential systemic problem"

                if env_color == "green":
                    st.success(f"{env_status}: {env_msg}")
                elif env_color == "orange":
                    st.warning(f"{env_status}: {env_msg}")
                else:
                    st.error(f"{env_status}: {env_msg}")

                st.caption("💡 **Environment health** reflects the overall fleet, while individual servers may still require attention")

                # Summary metrics - Server counts by severity
                st.markdown("---")
                st.markdown("**Individual Server Alert Levels** _(breakdown of the {0} servers above)_".format(alert_count))
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    critical_count = len([r for r in alert_rows if r['Priority'] in ['Imminent Failure', 'Critical']])
                    st.metric("🔴 Critical+", critical_count, delta=None, help="Risk >= 80 - Immediate action required")

                with col2:
                    danger_count = len([r for r in alert_rows if r['Priority'] == 'Danger'])
                    st.metric("🟠 Danger", danger_count, delta=None, help="Risk 70-79 - High priority attention needed")

                with col3:
                    warning_count = len([r for r in alert_rows if r['Priority'] == 'Warning'])
                    st.metric("🟡 Warning", warning_count, delta=None, help="Risk 60-69 - Monitor closely")

                with col4:
                    degrading_count = len([r for r in alert_rows if r['Priority'] == 'Degrading'])
                    st.metric("🟢 Degrading", degrading_count, delta=None, help="Risk 50-59 - Performance declining")

                # Show healthy/watch servers in separate row
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    total_servers = len(server_preds)
                    healthy_count = total_servers - len(alert_rows)
                    st.metric("✅ Healthy", healthy_count, delta=None, help="Risk < 50 - Normal operations, not shown in alerts table")

                with col2:
                    # Watch = servers with risk 30-49 (not shown in alerts but worth noting)
                    watch_count = sum(1 for s, p in server_preds.items() if 30 <= calculate_server_risk_score(p) < 50)
                    st.metric("👁️ Watch", watch_count, delta=None, help="Risk 30-49 - Low concern, monitoring only")

                # Trend analysis (separate row, clearly marked as subset)
                st.markdown("---")
                st.markdown("**📈 Trend Analysis** (of the alerts above)")

                col1, col2 = st.columns(2)
                with col1:
                    # Calculate degrading metrics
                    degrading = sum(1 for r in alert_rows if '+' in r['CPU Δ'] or '+' in r['Mem Δ'])
                    pct_degrading = (degrading / len(alert_rows) * 100) if alert_rows else 0
                    st.metric(
                        "⬆️ Degrading",
                        f"{degrading}/{len(alert_rows)}",
                        delta=None,
                        help=f"{pct_degrading:.0f}% of alerts showing increasing CPU/Memory trends"
                    )

                with col2:
                    # Calculate improving metrics
                    improving = sum(1 for r in alert_rows if '-' in r['CPU Δ'] or '-' in r['Mem Δ'])
                    pct_improving = (improving / len(alert_rows) * 100) if alert_rows else 0
                    st.metric(
                        "⬇️ Improving",
                        f"{improving}/{len(alert_rows)}",
                        delta=None,
                        help=f"{pct_improving:.0f}% of alerts showing decreasing CPU/Memory trends"
                    )

                st.markdown("---")

                # Add insight with better context
                if critical_count > 0:
                    st.error(f"⚠️ **Action Required**: {critical_count} critical server(s) need immediate attention")
                elif danger_count > 0:
                    st.warning(f"⚠️ **High Priority**: {danger_count} server(s) in danger state")
                elif warning_count > 0:
                    st.warning(f"⚠️ **Monitor Closely**: {warning_count} server(s) showing warning signs")

                # Show summary explanation
                st.caption(f"📊 Total fleet: {total_servers} servers | Showing: {len(alert_rows)} alerts | Hidden: {healthy_count} healthy")

            else:
                # No alerts - show top 5 busiest servers instead
                st.success("✅ No active alerts - All servers healthy!")
                st.markdown("**Top 5 Busiest Servers** (even though healthy)")

                # Calculate busyness score (CPU + Memory + I/O Wait)
                busy_servers = []
                for server_name, server_pred in server_preds.items():
                    # Calculate metrics using helpers where applicable (clamp to 0-100%)
                    cpu_actual = min(100.0, max(0.0, extract_cpu_used(server_pred, 'current')))

                    # Memory
                    mem_actual = min(100.0, max(0.0, server_pred.get('mem_used_pct', {}).get('current', 0)))

                    # I/O Wait
                    iowait_actual = min(100.0, max(0.0, server_pred.get('cpu_iowait_pct', {}).get('current', 0)))

                    # Swap
                    swap_actual = min(100.0, max(0.0, server_pred.get('swap_used_pct', {}).get('current', 0)))

                    # Load average (no clamping - can exceed 100)
                    load_actual = max(0.0, server_pred.get('load_average', {}).get('current', 0))

                    # Busyness score (weighted sum)
                    busyness = (cpu_actual * 0.3) + (mem_actual * 0.25) + (iowait_actual * 0.25) + (swap_actual * 0.1) + (load_actual * 0.1)

                    busy_servers.append({
                        'Server': server_name,
                        'Profile': get_server_profile(server_name),
                        'Status': '✅ Healthy',
                        'CPU': f"{cpu_actual:.1f}%",
                        'Memory': f"{mem_actual:.1f}%",
                        'I/O Wait': f"{iowait_actual:.1f}%",
                        'Swap': f"{swap_actual:.1f}%",
                        'Load': f"{load_actual:.2f}",
                        'Busyness': busyness
                    })

                # Sort by busyness and take top 5
                busy_servers_sorted = sorted(busy_servers, key=lambda x: x['Busyness'], reverse=True)[:5]

                # Remove busyness score from display
                for server in busy_servers_sorted:
                    del server['Busyness']

                busy_df = pd.DataFrame(busy_servers_sorted)
                st.dataframe(
                    busy_df,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        'Server': st.column_config.TextColumn('Server', width='medium', help='Server hostname'),
                        'Profile': st.column_config.TextColumn('Profile', width='medium', help='Server workload type'),
                        'Status': st.column_config.TextColumn('Status', width='small', help='All servers are healthy'),
                        'CPU': st.column_config.TextColumn('CPU', width='small', help='Current CPU utilization (% Used = 100 - Idle)'),
                        'Memory': st.column_config.TextColumn('Memory', width='small', help='Current memory utilization'),
                        'I/O Wait': st.column_config.TextColumn('I/O Wait', width='small', help='I/O wait % - CRITICAL troubleshooting metric'),
                        'Swap': st.column_config.TextColumn('Swap', width='small', help='Swap usage (thrashing indicator)'),
                        'Load': st.column_config.TextColumn('Load', width='small', help='System load average')
                    }
                )

                st.caption(f"📊 Showing top 5 of {len(server_preds)} healthy servers by activity level (CPU + Memory + I/O + Swap + Load)")
        else:
            st.info("No alert data available")

    else:
        st.info("👈 Connect to daemon to see live predictions")
        st.markdown("""
        **To get started:**

        1. Start the inference daemon:
           ```bash
           python tft_inference.py --daemon --port 8000
           ```

        2. The dashboard will automatically connect and display live predictions

        3. Predictions update every few seconds
        """)
