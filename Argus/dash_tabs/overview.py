"""
Overview Tab - Main dashboard view with KPIs, alerts, and risk distribution
============================================================================

Provides high-level fleet health overview with:
- Fleet health score and cascade risk (from cascade detection)
- Environment status and incident probabilities
- Fleet risk distribution charts (bar + pie)
- Active alerts summary
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from typing import Dict, Optional


def render(predictions: Dict, risk_scores: Dict[str, float], cascade_health: Optional[Dict] = None) -> html.Div:
    """
    Render Overview tab.

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores from daemon (optimization!)
        cascade_health: Optional fleet health from cascade detection

    Returns:
        html.Div: Tab content
    """
    env = predictions.get('environment', {})
    server_preds = predictions.get('predictions', {})

    # Risk scores already calculated in callback - no need to recalculate!

    # Fleet Health Banner (if cascade data available)
    fleet_health_banner = None
    if cascade_health:
        health_score = cascade_health.get('health_score', 0)
        health_status = cascade_health.get('status', 'unknown')
        cascade_risk = cascade_health.get('cascade_risk', 'unknown')
        correlation = cascade_health.get('correlation_score', 0)

        # Determine banner color and icon based on status
        status_config = {
            'healthy': {'color': 'success', 'icon': '✅', 'text': 'Fleet Healthy'},
            'degraded': {'color': 'warning', 'icon': '⚠️', 'text': 'Fleet Degraded'},
            'warning': {'color': 'warning', 'icon': '⚠️', 'text': 'Fleet Warning'},
            'critical': {'color': 'danger', 'icon': '🔴', 'text': 'Fleet Critical'},
        }
        config = status_config.get(health_status, {'color': 'info', 'icon': 'ℹ️', 'text': 'Fleet Status Unknown'})

        fleet_health_banner = dbc.Alert([
            dbc.Row([
                dbc.Col([
                    html.H4([
                        html.Span(config['icon'], className="me-2"),
                        config['text']
                    ], className="mb-0"),
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.Strong("Health Score: "),
                        html.Span(f"{health_score:.1f}", className="fs-4"),
                        html.Span("/100", className="text-muted")
                    ])
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Strong("Cascade Risk: "),
                        dbc.Badge(
                            cascade_risk.upper(),
                            color="danger" if cascade_risk == 'high' else "warning" if cascade_risk == 'medium' else "success"
                        )
                    ])
                ], width=3),
                dbc.Col([
                    html.Div([
                        html.Strong("Correlation: "),
                        html.Span(f"{correlation:.1%}")
                    ])
                ], width=2),
            ], className="align-items-center")
        ], color=config['color'], className="mb-4")

    # KPI cards
    kpis = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Environment Status", className="card-subtitle mb-2 text-muted"),
                    html.H3("🟢 Monitoring", className="card-title")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Incident Risk (30m)", className="card-subtitle mb-2 text-muted"),
                    html.H3(f"{env.get('prob_30m', 0) * 100:.1f}%", className="card-title")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Incident Risk (8h)", className="card-subtitle mb-2 text-muted"),
                    html.H3(f"{env.get('prob_8h', 0) * 100:.1f}%", className="card-title")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Fleet Status", className="card-subtitle mb-2 text-muted"),
                    html.H3(f"{len(server_preds)} Servers", className="card-title")
                ])
            ])
        ], width=3),
    ], className="mb-4")

    # Risk distribution chart
    server_risks = []
    for server_name, risk_score in risk_scores.items():
        status = 'Critical' if risk_score >= 80 else \
                 'Warning' if risk_score >= 60 else \
                 'Degrading' if risk_score >= 50 else 'Healthy'
        server_risks.append({
            'Server': server_name,
            'Risk Score': risk_score,
            'Status': status
        })

    risk_df = pd.DataFrame(server_risks)

    # Bar chart
    fig_bar = px.bar(
        risk_df.sort_values('Risk Score', ascending=False).head(15),
        x='Server',
        y='Risk Score',
        color='Risk Score',
        color_continuous_scale=['green', 'yellow', 'orange', 'red'],
        range_color=[0, 100],
        title="Top 15 Servers by Risk Score",
        height=400
    )
    fig_bar.update_layout(xaxis_tickangle=-45)

    # Pie chart
    status_counts = risk_df['Status'].value_counts()
    fig_pie = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="Server Status Distribution",
        color=status_counts.index,
        color_discrete_map={
            'Healthy': 'green',
            'Degrading': 'gold',
            'Warning': 'orange',
            'Critical': 'red'
        },
        height=400
    )

    # Charts row
    charts = dbc.Row([
        dbc.Col([dcc.Graph(figure=fig_bar)], width=8),
        dbc.Col([dcc.Graph(figure=fig_pie)], width=4),
    ], className="mb-4")

    # Alert count
    alert_count = sum(1 for r in risk_scores.values() if r >= 50)
    alerts_info = dbc.Alert(
        f"⚠️ {alert_count} servers require attention (Risk >= 50)",
        color="warning" if alert_count > 0 else "success"
    )

    # Build content list
    content = []
    if fleet_health_banner:
        content.append(fleet_health_banner)
    content.extend([kpis, charts, alerts_info])

    return html.Div(content)
