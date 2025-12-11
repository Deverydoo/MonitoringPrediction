"""
Drift Monitoring Tab - Model Performance and Drift Detection
=============================================================

Monitors model health and detects concept drift:
- Drift metrics visualization (PER, DSS, FDS, Anomaly Rate)
- Combined drift score gauge
- Feature-level drift analysis
- Auto-retraining status
- Recommendations for model maintenance
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional
from datetime import datetime


def render(drift_status: Optional[Dict], drift_report: Optional[Dict]) -> html.Div:
    """
    Render Drift Monitoring tab.

    Args:
        drift_status: Drift status from /drift/status endpoint
        drift_report: Detailed drift report from /drift/report endpoint

    Returns:
        html.Div: Tab content
    """
    # Handle case when drift monitoring is not available
    if drift_status is None and drift_report is None:
        return html.Div([
            dbc.Alert([
                html.H4("Drift Monitoring Not Available", className="alert-heading"),
                html.P("The inference daemon may not have drift monitoring enabled or is not responding."),
                html.Hr(),
                html.P("Ensure the daemon is running with drift monitoring configured.", className="mb-0")
            ], color="warning")
        ])

    # Use status data, fall back to report data
    status = drift_status or {}
    report = drift_report or {}

    # Extract key metrics
    drift_detected = status.get('drift_detected', report.get('needs_retraining', False))
    auto_retrain_enabled = status.get('auto_retrain_enabled', True)
    last_retrain = status.get('last_retrain', report.get('auto_retrain', {}).get('last_triggered'))

    # Get metrics from status or report
    metrics = status.get('metrics', {})
    if not metrics and report:
        report_metrics = report.get('metrics', {})
        metrics = {k: v.get('value', 0) if isinstance(v, dict) else v for k, v in report_metrics.items()}

    thresholds = status.get('thresholds', {})
    if not thresholds and report:
        report_metrics = report.get('metrics', {})
        thresholds = {k: v.get('threshold', 0) if isinstance(v, dict) else 0 for k, v in report_metrics.items()}

    # Default values
    per = metrics.get('per', 0)
    dss = metrics.get('dss', 0)
    fds = metrics.get('fds', 0)
    anomaly_rate = metrics.get('anomaly_rate', 0)

    per_threshold = thresholds.get('per', 0.10)
    dss_threshold = thresholds.get('dss', 0.20)
    fds_threshold = thresholds.get('fds', 0.15)
    anomaly_threshold = thresholds.get('anomaly_rate', 0.05)

    # Model Health Header
    overall_health = report.get('overall_health', 'good' if not drift_detected else 'warning')
    health_colors = {
        'good': 'success',
        'warning': 'warning',
        'critical': 'danger'
    }

    model_health_card = dbc.Card([
        dbc.CardHeader([
            html.H5("Model Health", className="mb-0 d-inline"),
            dbc.Badge(
                overall_health.upper(),
                color=health_colors.get(overall_health, 'secondary'),
                className="ms-2"
            )
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.I(
                            className=f"fas fa-{'exclamation-triangle text-danger' if drift_detected else 'check-circle text-success'} fa-3x mb-2"
                        ),
                        html.H4(
                            "DRIFT DETECTED" if drift_detected else "Model Healthy",
                            className=f"text-{'danger' if drift_detected else 'success'}"
                        )
                    ], className="text-center")
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H5("Auto-Retrain"),
                        dbc.Badge(
                            "ENABLED" if auto_retrain_enabled else "DISABLED",
                            color="success" if auto_retrain_enabled else "secondary",
                            className="fs-6"
                        ),
                        html.Hr(),
                        html.P([
                            html.Strong("Last Retrain: "),
                            last_retrain if last_retrain else "Never"
                        ], className="mb-0 small")
                    ])
                ], width=4),
                dbc.Col([
                    html.Div([
                        html.H5("Drift Trainings"),
                        html.H3(
                            str(report.get('auto_retrain', {}).get('total_drift_trainings', 0)),
                            className="text-info"
                        ),
                        html.P("Total triggered", className="text-muted small")
                    ])
                ], width=4),
            ])
        ])
    ], className="mb-4")

    # Drift Alert (if detected)
    if drift_detected:
        drift_alert = dbc.Alert([
            html.H4([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "Model Drift Detected"
            ], className="alert-heading"),
            html.P("One or more drift metrics have exceeded their thresholds. "
                   "The model may not be performing optimally."),
            html.Hr(),
            html.P([
                "Action: " + ("Automatic retraining will be triggered." if auto_retrain_enabled
                             else "Manual retraining recommended.")
            ], className="mb-0")
        ], color="danger", className="mb-4")
    else:
        drift_alert = dbc.Alert([
            html.H4([
                html.I(className="fas fa-check-circle me-2"),
                "Model Performing Within Bounds"
            ], className="alert-heading"),
            html.P("All drift metrics are within acceptable thresholds.")
        ], color="success", className="mb-4")

    # Create individual metric gauges
    def create_metric_gauge(value: float, threshold: float, title: str, description: str) -> go.Figure:
        is_exceeded = value > threshold
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 14}},
            number={'suffix': '%', 'font': {'size': 24}},
            delta={'reference': threshold * 100, 'relative': False, 'position': 'bottom'},
            gauge={
                'axis': {'range': [None, max(threshold * 200, 100)], 'tickformat': '.0f'},
                'bar': {'color': 'red' if is_exceeded else 'green'},
                'steps': [
                    {'range': [0, threshold * 100], 'color': 'rgba(0, 255, 0, 0.2)'},
                    {'range': [threshold * 100, threshold * 200], 'color': 'rgba(255, 0, 0, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    per_gauge = create_metric_gauge(per, per_threshold, "PER", "Prediction Error Rate")
    dss_gauge = create_metric_gauge(dss, dss_threshold, "DSS", "Distribution Shift Score")
    fds_gauge = create_metric_gauge(fds, fds_threshold, "FDS", "Feature Drift Score")
    anomaly_gauge = create_metric_gauge(anomaly_rate, anomaly_threshold, "Anomaly Rate", "Anomalous Predictions")

    gauges_row = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=per_gauge, config={'displayModeBar': False}),
                    html.P("Rolling average prediction error", className="text-center text-muted small")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=dss_gauge, config={'displayModeBar': False}),
                    html.P("Input feature distribution change", className="text-center text-muted small")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fds_gauge, config={'displayModeBar': False}),
                    html.P("Individual feature drift", className="text-center text-muted small")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=anomaly_gauge, config={'displayModeBar': False}),
                    html.P("Rate of anomalous predictions", className="text-center text-muted small")
                ])
            ])
        ], width=3),
    ], className="mb-4")

    # Feature-level drift (if available in report)
    feature_drift = report.get('feature_drift', {})
    if feature_drift:
        feature_rows = []
        for feature, data in feature_drift.items():
            if isinstance(data, dict):
                drift_val = data.get('drift', 0)
                drift_status_val = data.get('status', 'stable')
            else:
                drift_val = data
                drift_status_val = 'stable' if drift_val < 0.1 else 'drifting'

            status_badge_color = 'success' if drift_status_val == 'stable' else 'danger'

            feature_rows.append(html.Tr([
                html.Td(feature),
                html.Td(f"{drift_val:.2%}"),
                html.Td([
                    dbc.Badge(drift_status_val.upper(), color=status_badge_color)
                ]),
                html.Td([
                    dbc.Progress(
                        value=min(drift_val * 100, 100),
                        color="danger" if drift_val > 0.1 else "success",
                        style={"height": "10px"}
                    )
                ])
            ]))

        feature_drift_card = dbc.Card([
            dbc.CardHeader([
                html.H5("Feature-Level Drift Analysis", className="mb-0")
            ]),
            dbc.CardBody([
                dbc.Table([
                    html.Thead([
                        html.Tr([
                            html.Th("Feature"),
                            html.Th("Drift Score"),
                            html.Th("Status"),
                            html.Th("Drift Level")
                        ])
                    ]),
                    html.Tbody(feature_rows)
                ], bordered=True, hover=True, striped=True)
            ])
        ], className="mb-4")
    else:
        feature_drift_card = html.Div()

    # Recommendations
    recommendations = report.get('recommendations', [])
    if recommendations:
        rec_items = [html.Li(rec) for rec in recommendations]
        recommendations_card = dbc.Card([
            dbc.CardHeader([
                html.H5("Recommendations", className="mb-0")
            ]),
            dbc.CardBody([
                html.Ul(rec_items)
            ])
        ], className="mb-4")
    else:
        recommendations_card = html.Div()

    # Metrics Explanation
    metrics_explanation = dbc.Card([
        dbc.CardHeader([
            html.H5("Drift Metrics Explained", className="mb-0")
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("PER - Prediction Error Rate"),
                    html.P(
                        "Rolling average of prediction errors. High PER indicates the model's "
                        "predictions are deviating significantly from actual values.",
                        className="text-muted small"
                    )
                ], width=6),
                dbc.Col([
                    html.H6("DSS - Distribution Shift Score"),
                    html.P(
                        "Measures how much the input feature distributions have changed from "
                        "the training data. High DSS indicates data drift.",
                        className="text-muted small"
                    )
                ], width=6),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("FDS - Feature Drift Score"),
                    html.P(
                        "Detects drift in individual features. High FDS for specific features "
                        "can identify which aspects of the data are changing.",
                        className="text-muted small"
                    )
                ], width=6),
                dbc.Col([
                    html.H6("Anomaly Rate"),
                    html.P(
                        "Percentage of predictions flagged as anomalous. High anomaly rate "
                        "suggests the model is encountering unfamiliar patterns.",
                        className="text-muted small"
                    )
                ], width=6),
            ])
        ])
    ])

    return html.Div([
        html.H4("Model Drift Monitoring", className="mb-4"),
        model_health_card,
        drift_alert,
        gauges_row,
        feature_drift_card,
        recommendations_card,
        metrics_explanation
    ])
