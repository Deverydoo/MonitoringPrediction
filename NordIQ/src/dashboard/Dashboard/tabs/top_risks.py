"""
Top 5 Risks Tab - Detailed view of highest-risk servers

Shows top 5 problem servers with:
- Risk gauge visualization
- Current vs predicted metrics comparison
- 8-hour prediction timeline with confidence bands
- Degradation warnings
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional

from Dashboard.utils import calculate_server_risk_score, extract_cpu_used, get_risk_color


def render(predictions: Optional[Dict]):
    """
    Render the Top 5 Risks tab.

    Args:
        predictions: Current predictions from daemon
    """
    st.subheader("⚠️ Top 5 Problem Servers")

    if predictions and predictions.get('predictions'):
        server_preds = predictions['predictions']

        # Calculate risk scores
        server_risks = []
        for server_name, server_pred in server_preds.items():
            risk_score = calculate_server_risk_score(server_pred)
            server_risks.append({
                'server': server_name,
                'risk': risk_score,
                'pred': server_pred
            })

        # Sort by risk
        server_risks.sort(key=lambda x: x['risk'], reverse=True)
        top_5 = server_risks[:5]

        # Display each server
        for idx, server_info in enumerate(top_5):
            server_name = server_info['server']
            risk_score = server_info['risk']
            server_pred = server_info['pred']

            # Expander for each server
            with st.expander(f"#{idx+1} - {server_name} (Risk: {risk_score:.1f})", expanded=(idx == 0)):
                col1, col2 = st.columns([1, 2])

                with col1:
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_score,
                        title={'text': "Risk Score"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': get_risk_color(risk_score)},
                            'steps': [
                                {'range': [0, 20], 'color': "lightgreen"},
                                {'range': [20, 40], 'color': "yellow"},
                                {'range': [40, 70], 'color': "orange"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': risk_score
                            }
                        }
                    ))
                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"gauge_{server_name}")

                with col2:
                    # Show current vs predicted side-by-side
                    st.markdown("**Current State vs Predictions (30min ahead)**")

                    # Create comparison table - NordIQ Metrics Framework metrics
                    metric_rows = []

                    # CPU Used (using helper)
                    cpu_current = extract_cpu_used(server_pred, 'current')
                    cpu_future = extract_cpu_used(server_pred, 'p50')

                    cpu_delta = cpu_future - cpu_current
                    cpu_delta_str = f"+{cpu_delta:.1f}%" if cpu_delta > 0 else f"{cpu_delta:.1f}%"
                    metric_rows.append({
                        'Metric': 'CPU Used',
                        'Current': f"{cpu_current:.1f}%",
                        'Predicted': f"{cpu_future:.1f}%",
                        'Δ': cpu_delta_str
                    })

                    # I/O Wait - CRITICAL
                    if 'cpu_iowait_pct' in server_pred:
                        iowait = server_pred['cpu_iowait_pct']
                        current = iowait.get('current', 0)
                        p50 = iowait.get('p50', [])
                        if p50:
                            future = np.mean(p50[:6]) if len(p50) >= 6 else np.mean(p50)
                            delta = future - current
                            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
                            metric_rows.append({
                                'Metric': 'I/O Wait',
                                'Current': f"{current:.1f}%",
                                'Predicted': f"{future:.1f}%",
                                'Δ': delta_str
                            })

                    # Memory
                    if 'mem_used_pct' in server_pred:
                        mem = server_pred['mem_used_pct']
                        current = mem.get('current', 0)
                        p50 = mem.get('p50', [])
                        if p50:
                            future = np.mean(p50[:6]) if len(p50) >= 6 else np.mean(p50)
                            delta = future - current
                            delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
                            metric_rows.append({
                                'Metric': 'Memory',
                                'Current': f"{current:.1f}%",
                                'Predicted': f"{future:.1f}%",
                                'Δ': delta_str
                            })

                    # Load Average
                    if 'load_average' in server_pred:
                        load = server_pred['load_average']
                        current = load.get('current', 0)
                        p50 = load.get('p50', [])
                        if p50:
                            future = np.mean(p50[:6]) if len(p50) >= 6 else np.mean(p50)
                            delta = future - current
                            delta_str = f"+{delta:.1f}" if delta > 0 else f"{delta:.1f}"
                            metric_rows.append({
                                'Metric': 'Load Avg',
                                'Current': f"{current:.1f}",
                                'Predicted': f"{future:.1f}",
                                'Δ': delta_str
                            })

                    # Display comparison table
                    if metric_rows:
                        df_comparison = pd.DataFrame(metric_rows)
                        st.dataframe(df_comparison, width='stretch', hide_index=True)

                        # Highlight if significant degradation is predicted
                        degrading_metrics = [row for row in metric_rows if '+' in row['Δ']]
                        if len(degrading_metrics) >= 2:
                            st.warning(f"⚠️ {len(degrading_metrics)} metrics predicted to increase")

                # Prediction timeline
                st.markdown("**Prediction Timeline (Next 8 hours)**")

                # Create timeline chart for CPU (calculated from idle)
                if 'cpu_idle_pct' in server_pred:
                    cpu_idle = server_pred['cpu_idle_pct']
                    # Convert idle predictions to CPU used (100 - idle)
                    p10_idle = cpu_idle.get('p10', [])
                    p50_idle = cpu_idle.get('p50', [])
                    p90_idle = cpu_idle.get('p90', [])

                    # Invert: higher idle = lower CPU used, so p10 idle = p90 CPU, p90 idle = p10 CPU
                    p10 = [100 - x for x in p90_idle] if p90_idle else []
                    p50 = [100 - x for x in p50_idle] if p50_idle else []
                    p90 = [100 - x for x in p10_idle] if p10_idle else []

                    if p10 and p50 and p90:
                        # Create time axis (5-min intervals for 8 hours = 96 steps)
                        time_steps = list(range(len(p50)))

                        fig = go.Figure()

                        # Add confidence band
                        fig.add_trace(go.Scatter(
                            x=time_steps + time_steps[::-1],
                            y=p90 + p10[::-1],
                            fill='toself',
                            fillcolor='rgba(0,100,250,0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            name='p10-p90 range',
                            showlegend=True
                        ))

                        # Add p50 line
                        fig.add_trace(go.Scatter(
                            x=time_steps,
                            y=p50,
                            mode='lines',
                            name='p50 (median)',
                            line=dict(color='blue', width=2)
                        ))

                        # Add threshold lines
                        fig.add_hline(y=90, line_dash="dash", line_color="red",
                                     annotation_text="Critical (90%)")
                        fig.add_hline(y=70, line_dash="dash", line_color="orange",
                                     annotation_text="Warning (70%)")

                        fig.update_layout(
                            title="CPU Forecast",
                            xaxis_title="Time Steps (5-min intervals)",
                            yaxis_title="CPU %",
                            height=300,
                            hovermode='x'
                        )

                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"forecast_{server_name}")

    else:
        st.info("Connect to daemon to see top problem servers")
