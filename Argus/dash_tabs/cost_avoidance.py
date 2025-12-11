"""
Cost Avoidance Tab - Financial impact tracking and ROI analysis
================================================================

Demonstrates business value with:
- Configurable cost assumptions (interactive inputs)
- Projected cost avoidance calculations
- ROI analysis with payback period
- At-risk servers with potential incident costs
- Monthly/Annual projections

This tab uses Dash callbacks for interactivity - all inputs trigger recalculations.

Performance: Target <100ms (mostly calculation + table rendering)
"""

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from typing import Dict

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dash_utils.data_processing import calculate_server_risk_score


def calculate_cost_metrics(server_preds: Dict, cost_per_hour: float,
                           avg_duration: float, prevention_rate: float) -> Dict:
    """
    Calculate all cost avoidance metrics.

    Args:
        server_preds: Server predictions from daemon
        cost_per_hour: Cost per hour of downtime
        avg_duration: Average outage duration in hours
        prevention_rate: Prevention success rate (0-100)

    Returns:
        Dict with calculated metrics
    """
    # Count servers at risk
    high_risk_servers = [
        s for s, p in server_preds.items()
        if calculate_server_risk_score(p) >= 70
    ]
    medium_risk_servers = [
        s for s, p in server_preds.items()
        if 40 <= calculate_server_risk_score(p) < 70
    ]

    # Calculate potential cost avoidance
    critical_incidents_prevented = len(high_risk_servers) * (prevention_rate / 100)
    warning_incidents_prevented = len(medium_risk_servers) * 0.3 * (prevention_rate / 100)

    total_incidents_prevented = critical_incidents_prevented + warning_incidents_prevented
    cost_avoided_per_incident = cost_per_hour * avg_duration
    total_cost_avoided = total_incidents_prevented * cost_avoided_per_incident

    # Projections
    daily_avoidance = total_cost_avoided
    monthly_avoidance = daily_avoidance * 30
    annual_avoidance = daily_avoidance * 365

    return {
        'high_risk_servers': high_risk_servers,
        'medium_risk_servers': medium_risk_servers,
        'total_incidents_prevented': total_incidents_prevented,
        'cost_avoided_per_incident': cost_avoided_per_incident,
        'daily_avoidance': daily_avoidance,
        'monthly_avoidance': monthly_avoidance,
        'annual_avoidance': annual_avoidance
    }


def render(predictions: Dict, risk_scores: Dict[str, float]) -> html.Div:
    """
    Render Cost Avoidance tab.

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores from daemon

    Returns:
        html.Div: Tab content
    """
    server_preds = predictions.get('predictions', {})

    if not server_preds:
        return dbc.Alert([
            html.H4("💰 Cost Avoidance Dashboard"),
            html.P("No predictions available. Start the inference daemon and metrics generator."),
        ], color="warning")

    # Header
    header = html.Div([
        html.H4("💰 Cost Avoidance Dashboard", className="mb-3"),
        dbc.Alert([
            html.Strong("Business Value Tracking: "),
            "Real-time financial impact analysis showing ROI and cost savings from prevented incidents."
        ], color="info")
    ])

    # Cost Assumptions (interactive inputs with default values)
    cost_inputs = dbc.Card([
        dbc.CardHeader(html.H5("💵 Cost Assumptions (Configurable)", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Outage Cost ($/hour)", className="fw-bold"),
                    dcc.Input(
                        id='cost-per-hour',
                        type='number',
                        value=50000,
                        step=5000,
                        className="form-control"
                    ),
                    html.Small("Average cost per hour of downtime for a critical server", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.Label("Avg Outage Duration (hours)", className="fw-bold"),
                    dcc.Input(
                        id='avg-outage-duration',
                        type='number',
                        value=2.5,
                        step=0.5,
                        className="form-control"
                    ),
                    html.Small("Average duration of an incident if not prevented", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.Label("Project Investment ($)", className="fw-bold"),
                    dcc.Input(
                        id='project-cost',
                        type='number',
                        value=250000,
                        step=25000,
                        className="form-control"
                    ),
                    html.Small("Total cost for development, deployment, and first year operations", className="text-muted")
                ], width=3),
                dbc.Col([
                    html.Label("Prevention Success Rate (%)", className="fw-bold"),
                    dcc.Slider(
                        id='prevention-rate',
                        min=50,
                        max=100,
                        value=85,
                        step=5,
                        marks={50: '50%', 65: '65%', 80: '80%', 100: '100%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Small("Percentage of predicted incidents successfully prevented", className="text-muted")
                ], width=3)
            ])
        ])
    ], className="mb-4")

    # Projected Cost Avoidance section (filled by callback)
    cost_avoidance_section = html.Div(id='cost-avoidance-metrics')

    # ROI Analysis section (filled by callback)
    roi_section = html.Div(id='roi-analysis')

    # At-Risk Servers section (filled by callback)
    at_risk_section = html.Div(id='at-risk-servers')

    # Implementation Note
    implementation_note = dbc.Alert([
        html.Strong("💡 POC Implementation Note:"),
        html.Br(), html.Br(),
        "This tab demonstrates financial impact tracking. In production, we would:",
        html.Ul([
            html.Li([html.Strong("Track actual incidents"), " prevented vs predicted (accuracy metrics)"]),
            html.Li([html.Strong("Integrate with ITSM"), " (ServiceNow, JIRA) for actual incident costs"]),
            html.Li([html.Strong("Historical cost avoidance"), " dashboard showing cumulative savings"]),
            html.Li([html.Strong("Per-profile cost models"), " (e.g., ML compute downtime costs more than generic servers)"]),
            html.Li([html.Strong("Executive reporting"), " with monthly/quarterly summaries"])
        ]),
        html.Br(),
        html.Strong("Business Case: "),
        "This system pays for itself in ~3-5 months based on typical financial services downtime costs."
    ], color="light", className="mt-4")

    return html.Div([
        header,
        cost_inputs,
        cost_avoidance_section,
        roi_section,
        at_risk_section,
        implementation_note
    ])


def create_cost_metrics_display(metrics: Dict) -> html.Div:
    """Create the cost avoidance metrics display (KPI cards)."""
    return dbc.Card([
        dbc.CardHeader(html.H5("📊 Projected Cost Avoidance", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Incidents Prevented (Daily)", className="card-subtitle mb-2 text-muted"),
                            html.H3(f"{metrics['total_incidents_prevented']:.1f}", className="card-title text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Daily Cost Avoidance", className="card-subtitle mb-2 text-muted"),
                            html.H3(f"${metrics['daily_avoidance']:,.0f}", className="card-title text-success")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Monthly Projection", className="card-subtitle mb-2 text-muted"),
                            html.H3(f"${metrics['monthly_avoidance']:,.0f}", className="card-title text-info"),
                            html.Small("30-day projection", className="text-muted")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Annual Projection", className="card-subtitle mb-2 text-muted"),
                            html.H3(f"${metrics['annual_avoidance']:,.0f}", className="card-title text-primary"),
                            html.Small("365-day projection", className="text-muted")
                        ])
                    ])
                ], width=3)
            ])
        ])
    ], className="mb-4")


def create_roi_analysis(monthly_avoidance: float, project_cost: float = 250000) -> html.Div:
    """Create ROI analysis display."""
    months_to_roi = (project_cost / monthly_avoidance) if monthly_avoidance > 0 else float('inf')
    annual_avoidance = monthly_avoidance * 12
    first_year_net = annual_avoidance - project_cost

    # ROI badge color
    if months_to_roi < 6:
        roi_color = "success"
        roi_icon = "✅"
        roi_label = "Excellent ROI"
    elif months_to_roi < 12:
        roi_color = "success"
        roi_icon = "✅"
        roi_label = "Strong ROI"
    elif months_to_roi < 24:
        roi_color = "info"
        roi_icon = "📊"
        roi_label = "Good ROI"
    else:
        roi_color = "warning"
        roi_icon = "⚠️"
        roi_label = "Long Payback"

    return dbc.Card([
        dbc.CardHeader(html.H5("🎯 ROI Analysis", className="mb-0")),
        dbc.CardBody([
            dbc.Alert([
                html.H5(f"{roi_icon} {roi_label}", className="mb-2"),
                html.H4(f"Payback: {months_to_roi:.1f} months", className="mb-2"),
                html.P([
                    html.Strong("First Year Net: "),
                    html.Span(
                        f"${first_year_net:,.0f}",
                        style={'color': '#10B981' if first_year_net > 0 else '#EF4444'}
                    )
                ], className="mb-0")
            ], color=roi_color)
        ])
    ], className="mb-4")


def create_at_risk_table(server_preds: Dict, risk_scores: Dict[str, float],
                         cost_per_incident: float) -> html.Div:
    """Create table of at-risk servers with potential costs."""
    high_risk_servers = [
        s for s, risk in risk_scores.items()
        if risk >= 70
    ]
    medium_risk_servers = [
        s for s, risk in risk_scores.items()
        if 40 <= risk < 70
    ]

    if not high_risk_servers and not medium_risk_servers:
        return dbc.Card([
            dbc.CardHeader(html.H5("🎯 Current At-Risk Servers", className="mb-0")),
            dbc.CardBody([
                dbc.Alert([
                    html.H5("✅ No high-risk servers detected", className="mb-0"),
                    html.P("Fleet is healthy!", className="mb-0 mt-2")
                ], color="success")
            ])
        ], className="mb-4")

    risk_breakdown = []

    # Add high-risk servers
    for server in high_risk_servers:
        risk_score = risk_scores[server]
        risk_breakdown.append({
            'Server': server,
            'Risk Level': 'Critical',
            'Risk Score': f"{risk_score:.0f}",
            'Potential Cost if Incident': f"${cost_per_incident:,.0f}",
            'Status': '🔴 Action Required'
        })

    # Add top 5 medium-risk servers
    medium_sorted = sorted(
        [(s, risk_scores[s]) for s in medium_risk_servers],
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for server, risk_score in medium_sorted:
        risk_breakdown.append({
            'Server': server,
            'Risk Level': 'Warning',
            'Risk Score': f"{risk_score:.0f}",
            'Potential Cost if Incident': f"${cost_per_incident * 0.6:,.0f}",
            'Status': '🟠 Monitor'
        })

    df_risks = pd.DataFrame(risk_breakdown)

    # Create Bootstrap table
    table = dbc.Table.from_dataframe(
        df_risks,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True
    )

    return dbc.Card([
        dbc.CardHeader(html.H5("🎯 Current At-Risk Servers", className="mb-0")),
        dbc.CardBody([
            html.P(f"Showing {len(high_risk_servers)} critical and top 5 of {len(medium_risk_servers)} medium-risk servers",
                   className="text-muted"),
            table
        ])
    ], className="mb-4")
