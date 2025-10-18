"""
Heatmap Tab - Visual fleet heatmap with color-coded server health

Displays server fleet in a grid layout with color-coded health indicators.
Supports multiple metric views: Risk Score, CPU, Memory, Load Average.
Uses caching and Streamlit fragments for optimal performance.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional

from Dashboard.utils import calculate_server_risk_score, get_risk_color


def render(predictions: Optional[Dict]):
    """
    Render the Heatmap tab.

    Args:
        predictions: Current predictions from daemon
    """
    st.subheader("🔥 Server Fleet Heatmap")

    if predictions and predictions.get('predictions'):
        server_preds = predictions['predictions']

        # Metric options
        metric_options = {
            'Risk Score': 'risk',
            'CPU (p90)': 'cpu',
            'Memory (p90)': 'memory',
            'Load Avg (p90)': 'load'
        }

        # Cache key for heatmap data (only recalculate when predictions change)
        predictions_hash = str(predictions.get('timestamp', '')) if predictions else ''
        cache_key = f'heatmap_data_{predictions_hash}'

        # Build or retrieve cached heatmap data for ALL metrics at once
        if cache_key not in st.session_state:
            # Calculate all metrics at once to avoid recalculation
            heatmap_cache = {}

            for metric_name, mk in metric_options.items():
                metric_data = []
                for server_name, server_pred in server_preds.items():
                    if mk == 'risk':
                        value = calculate_server_risk_score(server_pred)
                    elif mk == 'cpu':
                        # NordIQ Metrics Framework: Use cpu_idle_pct and convert to CPU Used (100 - idle)
                        # For p90, we want p10 of idle (since lower idle = higher CPU used)
                        cpu_idle = server_pred.get('cpu_idle_pct', {})
                        p10_idle = cpu_idle.get('p10', [])
                        if p10_idle and len(p10_idle) >= 6:
                            # p10 idle = p90 CPU used (worst case)
                            min_idle = min(p10_idle[:6])
                            value = 100 - min_idle
                        else:
                            # Fallback: use current
                            current_idle = cpu_idle.get('current', 0)
                            value = 100 - current_idle if current_idle > 0 else 0
                    elif mk == 'memory':
                        # NordIQ Metrics Framework: Use mem_used_pct directly
                        mem = server_pred.get('mem_used_pct', {})
                        p90 = mem.get('p90', [])
                        value = max(p90[:6]) if len(p90) >= 6 else (max(p90) if p90 else 0)
                    elif mk == 'load':
                        load_avg = server_pred.get('load_average', {})
                        p90 = load_avg.get('p90', [])
                        value = max(p90[:6]) if len(p90) >= 6 else (max(p90) if p90 else 0)
                    else:
                        value = 0

                    metric_data.append({
                        'Server': server_name,
                        'Value': value
                    })

                heatmap_cache[metric_name] = pd.DataFrame(metric_data).sort_values('Value', ascending=False)

            st.session_state[cache_key] = heatmap_cache

        # Fragment: Isolate heatmap rendering to prevent full app rerun on dropdown change
        @st.fragment
        def render_heatmap_fragment():
            """Render heatmap with metric selector - isolated to prevent full app rerun"""
            # Metric selector
            selected_metric = st.selectbox(
                "Select metric to display",
                options=list(metric_options.keys()),
                index=0,
                key='heatmap_metric_selector'
            )

            metric_key = metric_options[selected_metric]

            # Retrieve cached data for selected metric
            heatmap_df = st.session_state[cache_key][selected_metric]

            # Display as grid using st.columns (with @st.fragment this is now fast!)
            servers_per_row = 5
            rows = [heatmap_df.iloc[i:i+servers_per_row] for i in range(0, len(heatmap_df), servers_per_row)]

            # Render grid
            for row_data in rows:
                cols = st.columns(servers_per_row)
                for idx, (_, server_row) in enumerate(row_data.iterrows()):
                    if idx < len(cols):
                        with cols[idx]:
                            server_name = server_row['Server']
                            value = server_row['Value']

                            # Determine color
                            if metric_key == 'risk':
                                color = get_risk_color(value)
                            else:
                                # Scale color based on value
                                if value > 90:
                                    color = "#ff4444"
                                elif value > 70:
                                    color = "#ff9900"
                                elif value > 50:
                                    color = "#ffcc00"
                                else:
                                    color = "#44ff44"

                            # Display server card
                            st.markdown(
                                f'<div style="background-color: {color}; padding: 15px; border-radius: 5px; text-align: center; margin: 5px;">'
                                f'<strong style="color: #000;">{server_name}</strong><br>'
                                f'<span style="font-size: 24px; color: #000;">{value:.1f}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

            st.divider()

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Minimum", f"{heatmap_df['Value'].min():.1f}")
            with col2:
                st.metric("Average", f"{heatmap_df['Value'].mean():.1f}")
            with col3:
                st.metric("Maximum", f"{heatmap_df['Value'].max():.1f}")
            with col4:
                st.metric("Std Dev", f"{heatmap_df['Value'].std():.1f}")

        # Execute the fragment (only THIS section reruns on dropdown change!)
        render_heatmap_fragment()

    else:
        st.info("Connect to daemon to see server heatmap")
