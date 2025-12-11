"""
Top Risks Tab - Executive Dashboard for Problem Servers
========================================================

Shows the top 5 highest-risk servers with:
- Clear risk level indicators (Critical/High/Moderate)
- Current vs predicted metrics
- Trend indicators (getting worse/better)
- Quick access to detailed analysis
- Actionable status for each server
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Tuple

# Import data processing utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dash_utils.data_processing import extract_cpu_used, get_risk_color


def get_risk_level(score: float) -> Tuple[str, str, str, str]:
    """Get risk level details: (level, color, icon, action)."""
    if score >= 80:
        return "Critical", "#DC2626", "🔴", "Immediate action required"
    elif score >= 60:
        return "High", "#F59E0B", "🟠", "Action needed soon"
    elif score >= 40:
        return "Moderate", "#EAB308", "🟡", "Monitor closely"
    else:
        return "Low", "#10B981", "🟢", "No action needed"


def get_trend(current: float, predicted: float) -> Tuple[str, str, str]:
    """Determine trend direction: (icon, text, color)."""
    if predicted is None or current is None:
        return "➡️", "Stable", "#6B7280"

    diff = predicted - current
    pct_change = (diff / max(current, 1)) * 100

    if pct_change > 15:
        return "📈", f"+{pct_change:.0f}%", "#DC2626"  # Getting worse
    elif pct_change > 5:
        return "↗️", f"+{pct_change:.0f}%", "#F59E0B"  # Slightly worse
    elif pct_change < -15:
        return "📉", f"{pct_change:.0f}%", "#10B981"  # Getting better
    elif pct_change < -5:
        return "↘️", f"{pct_change:.0f}%", "#3B82F6"  # Slightly better
    else:
        return "➡️", "Stable", "#6B7280"


def create_mini_sparkline(values: List[float], color: str = "#3B82F6") -> go.Figure:
    """Create a small sparkline chart for trends."""
    if not values or len(values) < 2:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values[:20],  # First 20 predictions
        mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
    ))
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig


def create_risk_gauge(risk_score: float, size: int = 150) -> go.Figure:
    """Create a compact risk gauge."""
    level, color, _, _ = get_risk_level(risk_score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={'suffix': '', 'font': {'size': 28, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#E5E7EB"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E5E7EB",
            'steps': [
                {'range': [0, 40], 'color': '#ECFDF5'},
                {'range': [40, 60], 'color': '#FEF9C3'},
                {'range': [60, 80], 'color': '#FEF3C7'},
                {'range': [80, 100], 'color': '#FEE2E2'}
            ],
        }
    ))
    fig.update_layout(
        height=size,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#374151'}
    )
    return fig


def create_metric_bar(label: str, current: float, predicted: float, max_val: float = 100,
                      warn_threshold: float = 70, crit_threshold: float = 85) -> html.Div:
    """Create a visual metric bar with current and predicted values."""
    # Determine color based on value
    if current >= crit_threshold:
        bar_color = "#DC2626"
    elif current >= warn_threshold:
        bar_color = "#F59E0B"
    else:
        bar_color = "#10B981"

    trend_icon, trend_text, trend_color = get_trend(current, predicted)

    return html.Div([
        html.Div([
            html.Span(label, className="fw-bold", style={'fontSize': '0.85rem'}),
            html.Span([
                html.Span(f"{current:.0f}%", style={'fontWeight': 'bold', 'color': bar_color}),
                html.Span(f" → {predicted:.0f}%" if predicted else "", className="text-muted ms-1",
                         style={'fontSize': '0.8rem'}),
                html.Span(f" {trend_icon}", style={'marginLeft': '4px'})
            ], className="float-end")
        ], className="d-flex justify-content-between mb-1"),
        dbc.Progress(
            value=min(current, max_val),
            max=max_val,
            color="danger" if current >= crit_threshold else "warning" if current >= warn_threshold else "success",
            style={'height': '8px'}
        )
    ], className="mb-2")


def render(predictions: Dict, risk_scores: Dict[str, float], server_preds: Dict) -> html.Div:
    """
    Render enhanced Top 5 Risks tab.

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores from daemon
        server_preds: Server predictions dict

    Returns:
        html.Div: Enhanced tab content
    """
    # Sort by risk and get top 5
    top_servers = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    # Check if all servers are healthy
    if not top_servers or top_servers[0][1] < 30:
        return html.Div([
            html.H4("⚠️ Top Risk Servers", className="mb-4"),
            dbc.Alert([
                html.H4("✅ All Systems Healthy", className="alert-heading"),
                html.P("No servers require immediate attention. All risk scores are below warning thresholds."),
                html.Hr(),
                html.P([
                    "Highest risk server: ",
                    html.Strong(top_servers[0][0] if top_servers else "N/A"),
                    f" at {top_servers[0][1]:.0f}%" if top_servers else ""
                ], className="mb-0")
            ], color="success")
        ])

    # Header with summary
    critical_count = sum(1 for _, score in top_servers if score >= 80)
    high_count = sum(1 for _, score in top_servers if 60 <= score < 80)
    moderate_count = sum(1 for _, score in top_servers if 40 <= score < 60)

    summary_badges = []
    if critical_count > 0:
        summary_badges.append(dbc.Badge(f"{critical_count} Critical", color="danger", className="me-2"))
    if high_count > 0:
        summary_badges.append(dbc.Badge(f"{high_count} High", color="warning", className="me-2"))
    if moderate_count > 0:
        summary_badges.append(dbc.Badge(f"{moderate_count} Moderate", color="info", className="me-2"))

    header = html.Div([
        html.Div([
            html.H4("⚠️ Top Risk Servers", className="mb-0 d-inline"),
            html.Div(summary_badges, className="d-inline ms-3")
        ]),
        html.P("Servers most likely to have issues, ranked by AI risk score",
               className="text-muted mb-4")
    ])

    # Create cards for each server
    server_cards = []
    for rank, (server_name, risk_score) in enumerate(top_servers, 1):
        server_pred = server_preds.get(server_name, {})
        level, color, icon, action = get_risk_level(risk_score)

        # Extract current metrics
        current_cpu = extract_cpu_used(server_pred, 'current')
        current_mem = server_pred.get('mem_used_pct', {}).get('current', 0)
        current_iowait = server_pred.get('cpu_iowait_pct', {}).get('current', 0)
        current_swap = server_pred.get('swap_used_pct', {}).get('current', 0)
        current_disk = server_pred.get('disk_usage_pct', {}).get('current', 0)

        # Extract predicted metrics (30 min ahead - index 6 at 5-sec intervals)
        cpu_p50 = server_pred.get('cpu_idle_pct', {}).get('p50', [])
        pred_cpu = 100 - cpu_p50[6] if len(cpu_p50) > 6 else current_cpu

        mem_p50 = server_pred.get('mem_used_pct', {}).get('p50', [])
        pred_mem = mem_p50[6] if len(mem_p50) > 6 else current_mem

        iowait_p50 = server_pred.get('cpu_iowait_pct', {}).get('p50', [])
        pred_iowait = iowait_p50[6] if len(iowait_p50) > 6 else current_iowait

        # Get server profile
        profile = server_pred.get('profile', 'Unknown')
        alert_info = server_pred.get('alert', {})
        alert_label = alert_info.get('label', f"{icon} {level}")

        # Determine overall trend
        metrics_getting_worse = 0
        if pred_cpu > current_cpu + 5:
            metrics_getting_worse += 1
        if pred_mem > current_mem + 5:
            metrics_getting_worse += 1
        if pred_iowait > current_iowait + 2:
            metrics_getting_worse += 1

        if metrics_getting_worse >= 2:
            trend_status = ("📈 Degrading", "#DC2626")
        elif metrics_getting_worse == 1:
            trend_status = ("↗️ Watch", "#F59E0B")
        else:
            trend_status = ("➡️ Stable", "#6B7280")

        # Create the server card
        card = dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span(f"#{rank}", className="fw-bold me-2",
                                     style={'fontSize': '1.2rem', 'color': color}),
                            html.Span(server_name, className="fw-bold", style={'fontSize': '1.1rem'}),
                            dbc.Badge(profile, color="secondary", className="ms-2")
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            dbc.Badge(alert_label, style={'backgroundColor': color, 'fontSize': '0.9rem'}),
                            html.Span(trend_status[0], className="ms-2",
                                     style={'color': trend_status[1], 'fontWeight': 'bold'})
                        ], className="text-end")
                    ], width=6)
                ])
            ], style={'backgroundColor': f'{color}15', 'borderBottom': f'3px solid {color}'}),

            dbc.CardBody([
                dbc.Row([
                    # Left: Risk gauge
                    dbc.Col([
                        html.Div([
                            dcc.Graph(
                                figure=create_risk_gauge(risk_score),
                                config={'displayModeBar': False},
                                style={'height': '140px'}
                            ),
                            html.P(action, className="text-center text-muted small mt-0 mb-0")
                        ])
                    ], width=4, className="d-flex align-items-center justify-content-center"),

                    # Right: Metrics
                    dbc.Col([
                        html.H6("Current → 30min Forecast", className="text-muted mb-3",
                               style={'fontSize': '0.85rem'}),
                        create_metric_bar("CPU", current_cpu, pred_cpu, warn_threshold=70, crit_threshold=85),
                        create_metric_bar("Memory", current_mem, pred_mem, warn_threshold=80, crit_threshold=90),
                        create_metric_bar("I/O Wait", current_iowait, pred_iowait,
                                         warn_threshold=15, crit_threshold=30),
                        html.Div([
                            html.Div([
                                html.Small("Swap: ", className="text-muted"),
                                html.Small(f"{current_swap:.0f}%",
                                          style={'color': '#DC2626' if current_swap > 5 else '#6B7280',
                                                 'fontWeight': 'bold' if current_swap > 5 else 'normal'})
                            ], className="d-inline me-3"),
                            html.Div([
                                html.Small("Disk: ", className="text-muted"),
                                html.Small(f"{current_disk:.0f}%",
                                          style={'color': '#DC2626' if current_disk > 85 else '#6B7280',
                                                 'fontWeight': 'bold' if current_disk > 85 else 'normal'})
                            ], className="d-inline")
                        ], className="mt-2")
                    ], width=8)
                ])
            ]),

            dbc.CardFooter([
                dbc.Row([
                    dbc.Col([
                        html.Small([
                            "Risk Score: ",
                            html.Strong(f"{risk_score:.0f}/100", style={'color': color})
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.A(
                                "View Analysis →",
                                href="#",
                                className="text-primary small",
                                id={'type': 'insights-link', 'server': server_name},
                                style={'textDecoration': 'none'}
                            )
                        ], className="text-end")
                    ], width=6)
                ])
            ], style={'backgroundColor': '#F9FAFB'})
        ], className="mb-3 shadow-sm")

        server_cards.append(card)

    # Quick reference legend
    legend = dbc.Card([
        dbc.CardBody([
            html.H6("Risk Levels", className="mb-2"),
            html.Div([
                html.Span("🔴 Critical (80+)", className="me-3 small"),
                html.Span("🟠 High (60-79)", className="me-3 small"),
                html.Span("🟡 Moderate (40-59)", className="me-3 small"),
                html.Span("🟢 Low (<40)", className="small")
            ])
        ], className="py-2")
    ], className="mb-4", style={'backgroundColor': '#F9FAFB'})

    return html.Div([
        header,
        legend,
        html.Div(server_cards)
    ])
