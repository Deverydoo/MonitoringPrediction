"""
Historical Reports Tab - Executive-Friendly Reporting
======================================================

Designed for managers and executives who need:
- Clear summary statistics at a glance
- Time range selection (30m, 1h, 8h, 1d, 1w, 1M)
- Alert history with resolution status
- Environment health trends
- One-click CSV export for reports

No technical jargon - just actionable business information.
"""

from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config for daemon URL
try:
    from dash_config import DAEMON_URL
    API_BASE = DAEMON_URL
except ImportError:
    API_BASE = "http://localhost:8000"


def get_historical_data(endpoint: str, params: Dict = None) -> Dict:
    """Fetch data from historical API endpoints."""
    try:
        url = f"{API_BASE}/historical/{endpoint}"
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return {'success': False, 'error': f'HTTP {response.status_code}'}
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': str(e)}


def render(predictions: Dict, risk_scores: Dict[str, float],
           history: List[Dict] = None, lookback_minutes: int = 30,
           metric_to_plot: str = "Environment Risk (30m)",
           historical_data: Dict = None) -> html.Div:
    """
    Render Historical Reports tab - Executive Edition.

    Args:
        predictions: Current predictions (for context)
        risk_scores: Current risk scores
        history: In-memory history (legacy - now using SQLite)
        lookback_minutes: Legacy param (now using time_range dropdown)
        metric_to_plot: Legacy param
        historical_data: Pre-fetched historical data from API

    Returns:
        html.Div: Executive-friendly reports tab
    """
    # Time range selector
    time_range_selector = dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Select Time Range", className="mb-2"),
                    dbc.RadioItems(
                        id='historical-time-range',
                        options=[
                            {'label': 'Last 30 Minutes', 'value': '30m'},
                            {'label': 'Last Hour', 'value': '1h'},
                            {'label': 'Last 8 Hours', 'value': '8h'},
                            {'label': 'Last 24 Hours', 'value': '1d'},
                            {'label': 'Last Week', 'value': '1w'},
                            {'label': 'Last Month', 'value': '1M'},
                        ],
                        value='1d',
                        inline=True,
                        className="time-range-selector"
                    ),
                ], width=8),
                dbc.Col([
                    html.H5("Export Reports", className="mb-2"),
                    dbc.ButtonGroup([
                        dbc.Button([
                            html.I(className="bi bi-download me-1"),
                            "Alerts CSV"
                        ], id='export-alerts-btn', color="primary", outline=True, size="sm"),
                        dbc.Button([
                            html.I(className="bi bi-download me-1"),
                            "Environment CSV"
                        ], id='export-env-btn', color="success", outline=True, size="sm"),
                    ]),
                    # Hidden components for download
                    dcc.Download(id="download-alerts-csv"),
                    dcc.Download(id="download-env-csv"),
                ], width=4, className="text-end"),
            ])
        ])
    ], className="mb-4")

    # Default time range for initial render
    time_range = '1d'

    # Fetch summary data (will be updated via callback)
    summary = historical_data.get('summary', {}) if historical_data else {}
    alerts = historical_data.get('alerts', []) if historical_data else []
    env_snapshots = historical_data.get('environment', []) if historical_data else []

    # Executive Summary Cards
    summary_cards = create_summary_cards(summary)

    # Alert History Section
    alert_section = create_alert_history_section(alerts)

    # Environment Health Trend Chart
    env_chart = create_environment_chart(env_snapshots)

    # Main layout
    return html.Div([
        html.H4("Historical Reports", className="mb-3"),
        html.P("Review past alerts, incidents, and environment health trends.",
               className="text-muted mb-4"),

        # Time range and export controls
        time_range_selector,

        # Content area (updated via callback)
        html.Div(id='historical-content', children=[
            # Summary cards
            summary_cards,

            # Two-column layout for alerts and chart
            dbc.Row([
                dbc.Col([
                    alert_section
                ], width=6),
                dbc.Col([
                    env_chart
                ], width=6),
            ], className="mb-4"),

            # Server breakdown table
            create_server_breakdown_section(alerts),
        ])
    ])


def create_summary_cards(summary: Dict) -> html.Div:
    """Create executive summary cards with key metrics."""
    if not summary:
        return dbc.Alert("No historical data available yet. Data will accumulate as the system runs.",
                        color="info", className="mb-4")

    total_alerts = summary.get('total_alerts', 0)
    critical_alerts = summary.get('critical_alerts', 0)
    warning_alerts = summary.get('warning_alerts', 0)
    resolved = summary.get('resolved_count', 0)
    unresolved = summary.get('unresolved_count', 0)
    resolution_rate = summary.get('resolution_rate', 0)
    avg_resolution = summary.get('avg_resolution_minutes', 0)
    servers_affected = summary.get('servers_affected', 0)
    incidents = summary.get('incidents_caused', 0)

    # Color coding
    critical_color = "danger" if critical_alerts > 0 else "success"
    resolution_color = "success" if resolution_rate >= 90 else "warning" if resolution_rate >= 70 else "danger"

    cards = dbc.Row([
        # Total Alerts
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Total Alerts", className="text-muted small"),
                        html.H2(f"{total_alerts}", className="mb-0 mt-1"),
                        html.Small(f"{servers_affected} servers affected", className="text-muted")
                    ])
                ])
            ], className="h-100")
        ], width=2),

        # Critical Alerts
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Critical", className="text-muted small"),
                        html.H2(f"{critical_alerts}",
                               className=f"mb-0 mt-1 text-{critical_color}"),
                        html.Small("immediate attention needed" if critical_alerts > 0 else "all clear",
                                  className="text-muted")
                    ])
                ])
            ], className="h-100", color=critical_color if critical_alerts > 0 else None, outline=True)
        ], width=2),

        # Warning Alerts
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Warnings", className="text-muted small"),
                        html.H2(f"{warning_alerts}", className="mb-0 mt-1 text-warning"),
                        html.Small("elevated risk", className="text-muted")
                    ])
                ])
            ], className="h-100")
        ], width=2),

        # Resolution Rate
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Resolution Rate", className="text-muted small"),
                        html.H2(f"{resolution_rate:.0f}%",
                               className=f"mb-0 mt-1 text-{resolution_color}"),
                        html.Small(f"{resolved} resolved, {unresolved} pending", className="text-muted")
                    ])
                ])
            ], className="h-100")
        ], width=2),

        # Avg Resolution Time
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Avg Resolution", className="text-muted small"),
                        html.H2(format_duration(avg_resolution), className="mb-0 mt-1"),
                        html.Small("mean time to resolve", className="text-muted")
                    ])
                ])
            ], className="h-100")
        ], width=2),

        # Incidents
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Incidents", className="text-muted small"),
                        html.H2(f"{incidents}",
                               className=f"mb-0 mt-1 {'text-danger' if incidents > 0 else ''}"),
                        html.Small("confirmed outages", className="text-muted")
                    ])
                ])
            ], className="h-100", color="danger" if incidents > 0 else None, outline=True)
        ], width=2),
    ], className="mb-4 g-3")

    return cards


def create_alert_history_section(alerts: List[Dict]) -> html.Div:
    """Create alert history timeline."""
    if not alerts:
        return dbc.Card([
            dbc.CardHeader([
                html.H5("Recent Alerts", className="mb-0")
            ]),
            dbc.CardBody([
                html.P("No alerts in the selected time range.",
                      className="text-muted text-center my-4"),
                html.P("This is good news - your servers are running smoothly!",
                      className="text-success text-center")
            ])
        ], className="h-100")

    # Create timeline entries
    timeline_items = []
    for alert in alerts[:15]:  # Show last 15 alerts
        timestamp = alert.get('timestamp', '')
        server = alert.get('server_name', 'Unknown')
        event_type = alert.get('event_type', '')
        new_level = alert.get('new_level', '')
        risk_score = alert.get('risk_score', 0)
        resolved = alert.get('resolved_at')
        resolution_time = alert.get('resolution_duration_minutes')

        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M")
            date_str = dt.strftime("%b %d")
        except:
            time_str = "??"
            date_str = ""

        # Icon and color based on level
        if new_level == 'critical':
            icon = "bi-exclamation-triangle-fill"
            color = "danger"
            badge_text = "CRITICAL"
        elif new_level == 'warning':
            icon = "bi-exclamation-circle-fill"
            color = "warning"
            badge_text = "WARNING"
        elif new_level == 'healthy':
            icon = "bi-check-circle-fill"
            color = "success"
            badge_text = "RESOLVED"
        else:
            icon = "bi-info-circle"
            color = "secondary"
            badge_text = new_level.upper() if new_level else "INFO"

        # Resolution status
        status_text = ""
        if resolved:
            if resolution_time:
                status_text = f"Resolved in {format_duration(resolution_time)}"
            else:
                status_text = "Resolved"

        timeline_items.append(
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Small(time_str, className="fw-bold"),
                            html.Br(),
                            html.Small(date_str, className="text-muted")
                        ], className="text-end")
                    ], width=2),
                    dbc.Col([
                        html.I(className=f"bi {icon} text-{color}",
                              style={"fontSize": "1.2rem"})
                    ], width=1, className="text-center"),
                    dbc.Col([
                        html.Div([
                            html.Span(server, className="fw-bold"),
                            dbc.Badge(badge_text, color=color, className="ms-2"),
                            html.Br(),
                            html.Small(f"Risk: {risk_score:.0f}%", className="text-muted"),
                            html.Small(f" | {status_text}" if status_text else "",
                                      className="text-success") if status_text else None
                        ])
                    ], width=9)
                ], className="align-items-center")
            ], className="mb-3 pb-3 border-bottom")
        )

    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    html.H5("Recent Alerts", className="mb-0")
                ]),
                dbc.Col([
                    dbc.Badge(f"{len(alerts)} total", color="secondary")
                ], width="auto")
            ], justify="between", align="center")
        ]),
        dbc.CardBody([
            html.Div(timeline_items, style={"maxHeight": "400px", "overflowY": "auto"})
        ])
    ], className="h-100")


def create_environment_chart(snapshots: List[Dict]) -> html.Div:
    """Create environment health trend chart."""
    if not snapshots or len(snapshots) < 2:
        return dbc.Card([
            dbc.CardHeader([
                html.H5("Environment Health Trend", className="mb-0")
            ]),
            dbc.CardBody([
                html.P("Not enough data for trend chart yet.",
                      className="text-muted text-center my-4"),
                html.P("Snapshots are recorded every 5 minutes.",
                      className="text-muted text-center small")
            ])
        ], className="h-100")

    # Extract data for chart
    timestamps = []
    critical_counts = []
    warning_counts = []
    healthy_counts = []
    avg_risk_scores = []

    for snap in snapshots:
        try:
            ts = datetime.fromisoformat(snap.get('timestamp', ''))
            timestamps.append(ts)
            critical_counts.append(snap.get('critical_count', 0))
            warning_counts.append(snap.get('warning_count', 0))
            healthy_counts.append(snap.get('healthy_count', 0))
            avg_risk_scores.append(snap.get('avg_risk_score', 0))
        except:
            continue

    if not timestamps:
        return dbc.Card([
            dbc.CardHeader(html.H5("Environment Health Trend", className="mb-0")),
            dbc.CardBody(html.P("Unable to parse snapshot data.", className="text-muted"))
        ])

    # Create stacked area chart
    fig = go.Figure()

    # Add traces in reverse order for proper stacking
    fig.add_trace(go.Scatter(
        x=timestamps, y=healthy_counts,
        name='Healthy', fill='tonexty',
        mode='none',
        fillcolor='rgba(40, 167, 69, 0.6)',
        stackgroup='one'
    ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=warning_counts,
        name='Warning', fill='tonexty',
        mode='none',
        fillcolor='rgba(255, 193, 7, 0.6)',
        stackgroup='one'
    ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=critical_counts,
        name='Critical', fill='tonexty',
        mode='none',
        fillcolor='rgba(220, 53, 69, 0.6)',
        stackgroup='one'
    ))

    # Add risk score line
    fig.add_trace(go.Scatter(
        x=timestamps, y=avg_risk_scores,
        name='Avg Risk %',
        mode='lines',
        line=dict(color='#6c757d', width=2, dash='dot'),
        yaxis='y2'
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=40, r=40, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(title=None),
        yaxis=dict(title="Server Count", side='left'),
        yaxis2=dict(title="Risk %", side='right', overlaying='y', range=[0, 100]),
        hovermode='x unified'
    )

    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col([
                    html.H5("Environment Health Trend", className="mb-0")
                ]),
                dbc.Col([
                    html.Small(f"{len(snapshots)} snapshots", className="text-muted")
                ], width="auto")
            ], justify="between", align="center")
        ]),
        dbc.CardBody([
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ])
    ], className="h-100")


def create_server_breakdown_section(alerts: List[Dict]) -> html.Div:
    """Create server-by-server breakdown table."""
    if not alerts:
        return html.Div()

    # Aggregate by server
    server_stats = {}
    for alert in alerts:
        server = alert.get('server_name', 'Unknown')
        if server not in server_stats:
            server_stats[server] = {
                'total': 0,
                'critical': 0,
                'warning': 0,
                'resolved': 0,
                'last_alert': None,
                'last_level': None
            }

        server_stats[server]['total'] += 1

        level = alert.get('new_level', '')
        if level == 'critical':
            server_stats[server]['critical'] += 1
        elif level == 'warning':
            server_stats[server]['warning'] += 1

        if alert.get('resolved_at'):
            server_stats[server]['resolved'] += 1

        # Track most recent alert
        ts = alert.get('timestamp')
        if not server_stats[server]['last_alert'] or ts > server_stats[server]['last_alert']:
            server_stats[server]['last_alert'] = ts
            server_stats[server]['last_level'] = level

    # Sort by total alerts (most problematic first)
    sorted_servers = sorted(server_stats.items(), key=lambda x: x[1]['total'], reverse=True)

    # Create table rows
    table_rows = []
    for server, stats in sorted_servers[:10]:  # Top 10
        resolution_rate = (stats['resolved'] / stats['total'] * 100) if stats['total'] > 0 else 0

        # Last alert time
        last_alert_str = ""
        if stats['last_alert']:
            try:
                dt = datetime.fromisoformat(stats['last_alert'])
                last_alert_str = dt.strftime("%b %d %H:%M")
            except:
                last_alert_str = "Unknown"

        # Status indicator
        if stats['last_level'] == 'critical':
            status = dbc.Badge("CRITICAL", color="danger")
        elif stats['last_level'] == 'warning':
            status = dbc.Badge("WARNING", color="warning")
        else:
            status = dbc.Badge("OK", color="success")

        table_rows.append(
            html.Tr([
                html.Td(server),
                html.Td(status),
                html.Td(stats['total']),
                html.Td(html.Span(stats['critical'], className="text-danger fw-bold") if stats['critical'] > 0 else "0"),
                html.Td(stats['warning'] if stats['warning'] > 0 else "0"),
                html.Td(f"{resolution_rate:.0f}%"),
                html.Td(last_alert_str, className="text-muted small")
            ])
        )

    return dbc.Card([
        dbc.CardHeader([
            html.H5("Server Breakdown", className="mb-0"),
            html.Small("Top 10 servers by alert count", className="text-muted")
        ]),
        dbc.CardBody([
            dbc.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Server"),
                        html.Th("Status"),
                        html.Th("Total"),
                        html.Th("Critical"),
                        html.Th("Warnings"),
                        html.Th("Resolution %"),
                        html.Th("Last Alert")
                    ])
                ]),
                html.Tbody(table_rows)
            ], striped=True, hover=True, responsive=True, size="sm")
        ])
    ]) if table_rows else html.Div()


def format_duration(minutes: float) -> str:
    """Format duration in human-readable form."""
    if minutes is None or minutes == 0:
        return "N/A"

    if minutes < 1:
        return f"{minutes * 60:.0f}s"
    elif minutes < 60:
        return f"{minutes:.0f}m"
    elif minutes < 1440:  # Less than 24 hours
        hours = minutes / 60
        return f"{hours:.1f}h"
    else:
        days = minutes / 1440
        return f"{days:.1f}d"


def get_time_range_label(time_range: str) -> str:
    """Convert time range code to human label."""
    labels = {
        '30m': 'Last 30 Minutes',
        '1h': 'Last Hour',
        '8h': 'Last 8 Hours',
        '1d': 'Last 24 Hours',
        '1w': 'Last Week',
        '1M': 'Last Month'
    }
    return labels.get(time_range, time_range)
