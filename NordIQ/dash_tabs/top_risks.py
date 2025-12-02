"""
Top Risks Tab - Highest risk servers with detailed metrics
===========================================================

Displays top 5 highest-risk servers with gauge charts and current metrics.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import Dict

# Import data processing utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dash_utils.data_processing import extract_cpu_used, get_risk_color


def render(predictions: Dict, risk_scores: Dict[str, float], server_preds: Dict) -> html.Div:
    """
    Render Top 5 Risks tab.

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores from daemon (optimization!)
        server_preds: Server predictions dict (for metrics extraction)

    Returns:
        html.Div: Tab content
    """
    # Risk scores already calculated in callback - no need to recalculate!

    top_servers = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    if not top_servers or top_servers[0][1] == 0:
        return dbc.Alert("✅ No high-risk servers detected!", color="success")

    # Create gauge charts for top 5
    gauges = []
    for i, (server_name, risk_score) in enumerate(top_servers, 1):
        server_pred = server_preds[server_name]

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{i}. {server_name}"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': get_risk_color(risk_score)},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "lightyellow"},
                    {'range': [80, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=250)

        # Current metrics
        current_cpu = extract_cpu_used(server_pred, 'current')
        current_mem = server_pred.get('mem_used_pct', {}).get('current', 0)
        current_iowait = server_pred.get('cpu_iowait_pct', {}).get('current', 0)

        metrics_card = dbc.Card([
            dbc.CardBody([
                html.H6(f"Rank #{i}: {server_name}", className="card-subtitle mb-2"),
                html.P([
                    html.Strong("CPU: "), f"{current_cpu:.1f}%", html.Br(),
                    html.Strong("Memory: "), f"{current_mem:.1f}%", html.Br(),
                    html.Strong("I/O Wait: "), f"{current_iowait:.1f}%", html.Br(),
                    html.Strong("Risk Score: "), html.Span(f"{risk_score:.0f}",
                        style={'color': get_risk_color(risk_score), 'font-weight': 'bold'})
                ])
            ])
        ])

        row = dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_gauge)], width=6),
            dbc.Col([metrics_card], width=6),
        ], className="mb-3")

        gauges.append(row)

    return html.Div([
        html.H4("⚠️ Top 5 Problem Servers", className="mb-3"),
        *gauges
    ])
