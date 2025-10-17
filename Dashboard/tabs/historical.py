"""
Historical Trends Tab - Time-series view of metrics over time

Shows historical data collected during dashboard session:
- Configurable lookback period
- Multiple metric options (Environment Risk, Fleet Health)
- Time-series charts with statistics
- CSV download capability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Optional

from Dashboard.utils import calculate_server_risk_score


def render(predictions: Optional[Dict]):
    """
    Render the Historical Trends tab.

    Args:
        predictions: Current predictions from daemon (unused, reads from session_state.history)
    """
    st.subheader("📈 Historical Trends")

    if st.session_state.history:
        # Time range selector
        col1, col2 = st.columns(2)

        with col1:
            lookback_minutes = st.slider(
                "Lookback period (minutes)",
                min_value=5,
                max_value=60,
                value=30,
                step=5
            )

        with col2:
            metric_to_plot = st.selectbox(
                "Metric to display",
                ["Environment Risk (30m)", "Environment Risk (8h)", "Fleet Health"]
            )

        # Filter history
        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        recent_history = [h for h in st.session_state.history if h['timestamp'] >= cutoff_time]

        if recent_history:
            # Extract data
            timestamps = [h['timestamp'] for h in recent_history]

            if metric_to_plot == "Environment Risk (30m)":
                values = [h['predictions'].get('environment', {}).get('prob_30m', 0) * 100
                         for h in recent_history]
                ylabel = "Probability (%)"
                title = "Environment Incident Risk (30 minutes)"

            elif metric_to_plot == "Environment Risk (8h)":
                values = [h['predictions'].get('environment', {}).get('prob_8h', 0) * 100
                         for h in recent_history]
                ylabel = "Probability (%)"
                title = "Environment Incident Risk (8 hours)"

            else:  # Fleet Health
                values = []
                for h in recent_history:
                    preds = h['predictions'].get('predictions', {})
                    if preds:
                        healthy = sum(1 for s, p in preds.items() if calculate_server_risk_score(p) < 20)
                        total = len(preds)
                        values.append((healthy / total * 100) if total > 0 else 0)
                    else:
                        values.append(0)
                ylabel = "Healthy %"
                title = "Fleet Health Percentage"

            # Create chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name=metric_to_plot,
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))

            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=ylabel,
                height=400,
                hovermode='x'
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # Statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current", f"{values[-1]:.1f}")
            with col2:
                st.metric("Average", f"{np.mean(values):.1f}")
            with col3:
                st.metric("Min", f"{np.min(values):.1f}")
            with col4:
                st.metric("Max", f"{np.max(values):.1f}")

            # Download data
            st.divider()

            if st.button("📥 Download Historical Data (CSV)"):
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'value': values
                })
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"tft_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info(f"No data collected in the last {lookback_minutes} minutes")

    else:
        st.info("No historical data yet. Data will accumulate as the dashboard runs.")
