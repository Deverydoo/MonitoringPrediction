"""
Cascade Detection Tab - Fleet-wide Cascading Failure Monitoring
================================================================

Provides cross-server correlation analysis and cascading failure detection:
- Fleet health score and cascade risk indicator
- Correlation score visualization
- Cascade event timeline
- Affected servers during cascade events
- Real-time cascade alerts
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional
from datetime import datetime


def render(cascade_status: Optional[Dict], cascade_health: Optional[Dict]) -> html.Div:
    """
    Render Cascade Detection tab.

    Args:
        cascade_status: Full cascade status from /cascade/status endpoint
        cascade_health: Fleet health from /cascade/health endpoint

    Returns:
        html.Div: Tab content
    """
    # Handle case when cascade detection is not available
    if cascade_health is None and cascade_status is None:
        return html.Div([
            dbc.Alert([
                html.H4("Cascade Detection Not Available", className="alert-heading"),
                html.P("The inference daemon may not have cascade detection enabled or is not responding."),
                html.Hr(),
                html.P("Ensure the daemon is running with cascade detection configured.", className="mb-0")
            ], color="warning")
        ])

    # Default values if health endpoint fails
    health = cascade_health or {
        'health_score': 0,
        'status': 'unknown',
        'correlation_score': 0,
        'anomaly_rate': 0,
        'anomalous_servers': 0,
        'total_servers': 0,
        'cascade_risk': 'unknown'
    }

    status = cascade_status or {
        'current_status': {'cascade_detected': False, 'cascades': []},
        'recent_events': [],
        'event_count': 0,
        'thresholds': {}
    }

    # Determine colors based on health status
    status_colors = {
        'healthy': 'success',
        'degraded': 'warning',
        'warning': 'warning',
        'critical': 'danger',
        'unknown': 'secondary'
    }

    risk_colors = {
        'low': 'success',
        'medium': 'warning',
        'high': 'danger',
        'unknown': 'secondary'
    }

    # Fleet Health Header
    health_status = health.get('status', 'unknown')
    health_score = health.get('health_score', 0)
    cascade_risk = health.get('cascade_risk', 'unknown')

    fleet_health_card = dbc.Card([
        dbc.CardHeader([
            html.H5("Fleet Health Score", className="mb-0 d-inline"),
            dbc.Badge(
                health_status.upper(),
                color=status_colors.get(health_status, 'secondary'),
                className="ms-2"
            )
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1(
                            f"{health_score:.1f}",
                            className="display-3 text-center",
                            style={'color': _get_health_color(health_score)}
                        ),
                        html.P("Health Score (0-100)", className="text-center text-muted")
                    ])
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H4("Cascade Risk"),
                        dbc.Badge(
                            cascade_risk.upper(),
                            color=risk_colors.get(cascade_risk, 'secondary'),
                            className="fs-5 px-3 py-2"
                        ),
                        html.Hr(),
                        html.P([
                            html.Strong("Correlation: "),
                            f"{health.get('correlation_score', 0):.1%}"
                        ]),
                        html.P([
                            html.Strong("Anomaly Rate: "),
                            f"{health.get('anomaly_rate', 0):.1%}"
                        ])
                    ])
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H4("Server Status"),
                        html.P([
                            html.Span(
                                f"{health.get('anomalous_servers', 0)}",
                                className="fs-2 text-danger"
                            ),
                            html.Span(" / ", className="fs-4"),
                            html.Span(
                                f"{health.get('total_servers', 0)}",
                                className="fs-2"
                            )
                        ]),
                        html.P("Servers with Anomalies", className="text-muted")
                    ])
                ], width=4),
            ])
        ])
    ], className="mb-4")

    # Active Cascade Alert (if any)
    current = status.get('current_status', {})
    cascade_detected = current.get('cascade_detected', False)

    if cascade_detected:
        cascades = current.get('cascades', [])
        affected_count = current.get('servers_with_anomalies', 0)

        cascade_alert = dbc.Alert([
            html.H4([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "ACTIVE CASCADE DETECTED"
            ], className="alert-heading"),
            html.P(f"{affected_count} servers currently affected by correlated anomalies."),
            html.Hr(),
            html.Div([
                _render_cascade_detail(c) for c in cascades
            ]) if cascades else html.P("Correlation score exceeds threshold.")
        ], color="danger", className="mb-4")
    else:
        cascade_alert = dbc.Alert([
            html.H4([
                html.I(className="fas fa-check-circle me-2"),
                "No Active Cascade"
            ], className="alert-heading"),
            html.P("Servers are operating independently with no significant cross-server correlations.")
        ], color="success", className="mb-4")

    # Correlation Gauge
    correlation_score = health.get('correlation_score', 0)
    correlation_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=correlation_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cross-Server Correlation"},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': _get_correlation_color(correlation_score)},
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 255, 0, 0.2)'},
                {'range': [50, 70], 'color': 'rgba(255, 255, 0, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.2)'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    correlation_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))

    # Anomaly Rate Gauge
    anomaly_rate = health.get('anomaly_rate', 0)
    anomaly_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=anomaly_rate * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fleet Anomaly Rate"},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': 'orange' if anomaly_rate > 0.1 else 'green'},
            'steps': [
                {'range': [0, 10], 'color': 'rgba(0, 255, 0, 0.2)'},
                {'range': [10, 25], 'color': 'rgba(255, 255, 0, 0.2)'},
                {'range': [25, 100], 'color': 'rgba(255, 0, 0, 0.2)'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 25
            }
        }
    ))
    anomaly_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))

    gauges_row = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=correlation_gauge, config={'displayModeBar': False})])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([dcc.Graph(figure=anomaly_gauge, config={'displayModeBar': False})])
            ])
        ], width=6),
    ], className="mb-4")

    # Recent Cascade Events Timeline
    recent_events = status.get('recent_events', [])
    event_count = status.get('event_count', 0)

    if recent_events:
        events_table = dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Timestamp"),
                    html.Th("Correlation"),
                    html.Th("Affected Servers"),
                    html.Th("Cascade Details")
                ])
            ]),
            html.Tbody([
                html.Tr([
                    html.Td(event.get('timestamp', 'N/A')),
                    html.Td([
                        dbc.Badge(
                            f"{event.get('correlation_score', 0):.1%}",
                            color="danger" if event.get('correlation_score', 0) > 0.7 else "warning"
                        )
                    ]),
                    html.Td(str(len(event.get('affected_servers', [])))),
                    html.Td([
                        html.Span(
                            f"{c.get('metric', 'N/A')}: {c.get('server_count', 0)} servers ({c.get('severity', 'N/A')})"
                        ) for c in event.get('cascades', [])
                    ] if event.get('cascades') else "Correlation threshold exceeded")
                ]) for event in recent_events[-5:]  # Show last 5 events
            ])
        ], bordered=True, hover=True, striped=True, className="mb-4")

        events_card = dbc.Card([
            dbc.CardHeader([
                html.H5(f"Recent Cascade Events ({event_count} total)", className="mb-0")
            ]),
            dbc.CardBody([events_table])
        ], className="mb-4")
    else:
        events_card = dbc.Card([
            dbc.CardHeader([
                html.H5("Recent Cascade Events", className="mb-0")
            ]),
            dbc.CardBody([
                html.P("No cascade events recorded.", className="text-muted text-center")
            ])
        ], className="mb-4")

    # Thresholds Info
    thresholds = status.get('thresholds', {})
    thresholds_card = dbc.Card([
        dbc.CardHeader([
            html.H5("Detection Thresholds", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P([
                        html.Strong("Correlation Threshold: "),
                        f"{thresholds.get('correlation', 0.7):.0%}"
                    ])
                ], width=4),
                dbc.Col([
                    html.P([
                        html.Strong("Min Servers for Cascade: "),
                        f"{thresholds.get('cascade_servers', 3)}"
                    ])
                ], width=4),
                dbc.Col([
                    html.P([
                        html.Strong("Anomaly Z-Score: "),
                        f"{thresholds.get('anomaly_z_score', 2.0)}"
                    ])
                ], width=4),
            ])
        ])
    ])

    return html.Div([
        html.H4("Cascading Failure Detection", className="mb-4"),
        fleet_health_card,
        cascade_alert,
        gauges_row,
        events_card,
        thresholds_card
    ])


def _get_health_color(score: float) -> str:
    """Get color based on health score."""
    if score >= 80:
        return '#28a745'  # Green
    elif score >= 60:
        return '#ffc107'  # Yellow
    elif score >= 40:
        return '#fd7e14'  # Orange
    else:
        return '#dc3545'  # Red


def _get_correlation_color(score: float) -> str:
    """Get color based on correlation score (higher = more dangerous)."""
    if score < 0.5:
        return '#28a745'  # Green - low correlation is good
    elif score < 0.7:
        return '#ffc107'  # Yellow
    else:
        return '#dc3545'  # Red - high correlation is bad


def _render_cascade_detail(cascade: Dict) -> html.Div:
    """Render a single cascade detail."""
    severity_colors = {
        'critical': 'danger',
        'high': 'warning',
        'medium': 'info',
        'low': 'secondary'
    }

    return html.Div([
        dbc.Badge(
            cascade.get('severity', 'unknown').upper(),
            color=severity_colors.get(cascade.get('severity', 'low'), 'secondary'),
            className="me-2"
        ),
        html.Strong(cascade.get('metric', 'Unknown metric')),
        html.Span(f": {cascade.get('server_count', 0)} servers affected"),
        html.Br(),
        html.Small(
            f"Servers: {', '.join(cascade.get('affected_servers', [])[:5])}{'...' if len(cascade.get('affected_servers', [])) > 5 else ''}",
            className="text-muted"
        )
    ], className="mb-2")
