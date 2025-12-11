"""
Heatmap Tab - Risk visualization across server fleet
=====================================================

Provides visual heatmap showing risk scores for top 30 servers.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from typing import Dict


def render(predictions: Dict, risk_scores: Dict[str, float]) -> html.Div:
    """
    Render Heatmap tab.

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores from daemon (optimization!)

    Returns:
        html.Div: Tab content
    """
    server_preds = predictions.get('predictions', {})

    # Risk scores already calculated in callback - no need to recalculate!

    # Create heatmap data
    servers = list(risk_scores.keys())[:30]  # Top 30 for visibility
    risks = [risk_scores[s] for s in servers]

    # Heatmap visualization
    fig = go.Figure(data=go.Heatmap(
        z=[risks],
        x=servers,
        y=['Risk Score'],
        colorscale=[[0, 'green'], [0.5, 'yellow'], [0.8, 'orange'], [1, 'red']],
        zmin=0,
        zmax=100,
        text=[[f"{r:.0f}" for r in risks]],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Risk Score")
    ))

    fig.update_layout(
        title="Server Fleet Risk Heatmap (Top 30 Servers)",
        xaxis_title="Server",
        height=300,
        xaxis={'tickangle': -45}
    )

    # Metric selector (could add dropdown to switch metrics)
    controls = dbc.Alert(
        "📊 Showing: Risk Score (0-100) | Green=Healthy, Red=Critical",
        color="info",
        className="mb-3"
    )

    return html.Div([
        controls,
        dcc.Graph(figure=fig)
    ])
