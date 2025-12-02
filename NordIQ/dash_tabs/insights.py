"""
Insights Tab - Executive-Friendly AI Analysis
==============================================

Provides clear, actionable insights that executives and managers can understand:
- Executive summary with plain English recommendations
- Visual risk indicators (traffic light style)
- Prioritized action items with business impact
- Technical details hidden in collapsible sections for those who want them

This tab answers: "What's wrong, why, and what should I do about it?"
"""

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import requests
from typing import Dict, List, Optional

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dash_config import DAEMON_URL, get_auth_headers
from dash_utils.data_processing import extract_cpu_used, calculate_server_risk_score


def fetch_explanation(server_name: str, daemon_url: str = DAEMON_URL) -> Optional[Dict]:
    """Fetch XAI explanation for a specific server from the daemon."""
    try:
        response = requests.get(
            f"{daemon_url}/explain/{server_name}",
            headers=get_auth_headers(),
            timeout=10
        )
        if response.ok:
            return response.json()
        return None
    except Exception as e:
        print(f"[ERROR] Error fetching explanation: {str(e)}")
        return None


def get_risk_level(score: float) -> tuple:
    """Get risk level category, color, and icon."""
    if score >= 80:
        return "Critical", "#DC2626", "🔴", "Immediate action required"
    elif score >= 60:
        return "High", "#F59E0B", "🟠", "Action needed soon"
    elif score >= 40:
        return "Moderate", "#EAB308", "🟡", "Monitor closely"
    else:
        return "Low", "#10B981", "🟢", "No action needed"


def get_metric_plain_name(metric: str) -> str:
    """Convert technical metric names to plain English."""
    names = {
        'cpu_user_pct': 'CPU usage',
        'cpu_sys_pct': 'system processes',
        'cpu_iowait_pct': 'disk wait time',
        'cpu_idle_pct': 'idle capacity',
        'java_cpu_pct': 'application load',
        'mem_used_pct': 'memory usage',
        'swap_used_pct': 'emergency memory',
        'disk_usage_pct': 'disk space',
        'net_in_mb_s': 'incoming traffic',
        'net_out_mb_s': 'outgoing traffic',
        'back_close_wait': 'backend connections',
        'front_close_wait': 'user connections',
        'load_average': 'overall load',
        'uptime_days': 'time since restart',
    }
    return names.get(metric, metric.replace('_', ' '))


def generate_executive_summary(shap_data: Dict, risk_score: float, server_name: str) -> html.Div:
    """Generate a plain English executive summary."""
    level, color, icon, urgency = get_risk_level(risk_score)

    # Get top contributing factors
    feature_importance = shap_data.get('feature_importance', [])
    top_factors = []
    for feature, info in feature_importance[:3]:
        plain_name = get_metric_plain_name(feature)
        direction = info.get('direction', '')
        if direction == 'increasing':
            top_factors.append(f"{plain_name} is trending up")
        elif direction == 'decreasing':
            top_factors.append(f"{plain_name} is improving")
        else:
            top_factors.append(f"{plain_name} is elevated")

    # Build the summary sentence
    if risk_score >= 80:
        summary = f"This server needs immediate attention. "
    elif risk_score >= 60:
        summary = f"This server may have issues within the next few hours. "
    elif risk_score >= 40:
        summary = f"This server is showing early warning signs. "
    else:
        summary = f"This server is operating normally. "

    if top_factors:
        summary += f"The main concerns are: {', '.join(top_factors)}."

    return html.Div([
        # Big status indicator
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span(icon, style={'fontSize': '4rem'}),
                        ], className="text-center")
                    ], width=2),
                    dbc.Col([
                        html.H2(f"{level} Risk", style={'color': color, 'marginBottom': '0.5rem'}),
                        html.H4(f"Risk Score: {risk_score:.0f}/100", className="text-muted mb-2"),
                        html.P(urgency, style={'fontSize': '1.1rem', 'fontWeight': '500'})
                    ], width=10)
                ], align="center")
            ])
        ], style={'borderLeft': f'6px solid {color}'}, className="mb-4"),

        # Plain English summary
        dbc.Alert([
            html.H5("📋 Summary", className="alert-heading"),
            html.P(summary, style={'fontSize': '1.1rem', 'marginBottom': '0'})
        ], color="light", className="mb-4")
    ])


def generate_action_cards(counterfactual_data, risk_score: float) -> html.Div:
    """Generate action recommendation cards in plain English."""

    # Handle different data formats
    if isinstance(counterfactual_data, list):
        scenarios = counterfactual_data
    else:
        scenarios = counterfactual_data.get('scenarios', [])

    if not scenarios:
        if risk_score < 40:
            return dbc.Alert([
                html.H5("✅ No Action Required", className="alert-heading"),
                html.P("This server is healthy. Continue monitoring as usual.")
            ], color="success")
        else:
            return dbc.Alert([
                html.H5("⚠️ Analysis Unavailable", className="alert-heading"),
                html.P("Could not generate specific recommendations. Review server metrics manually.")
            ], color="warning")

    action_cards = []

    # Process scenarios into simple action cards
    if isinstance(scenarios, dict):
        scenario_items = list(scenarios.items())
    else:
        scenario_items = [(s.get('scenario', f'Option {i+1}'), s) for i, s in enumerate(scenarios)]

    for i, (name, scenario) in enumerate(scenario_items[:4]):  # Limit to top 4 recommendations
        if isinstance(scenario, dict):
            action = scenario.get('action', 'Review this server')
            change = scenario.get('change', 0)
            effort = scenario.get('effort', 'MEDIUM')
            confidence = scenario.get('confidence', 0.5)
            is_safe = scenario.get('safe', True)
        else:
            action = str(scenario)
            change = 0
            effort = 'MEDIUM'
            confidence = 0.5
            is_safe = True

        # Determine card styling
        if change < -10:
            impact_text = "High Impact"
            impact_color = "#10B981"
            badge_color = "success"
        elif change < 0:
            impact_text = "Moderate Impact"
            impact_color = "#3B82F6"
            badge_color = "primary"
        else:
            impact_text = "Low Impact"
            impact_color = "#6B7280"
            badge_color = "secondary"

        effort_badges = {
            'LOW': ('Easy', 'success'),
            'MEDIUM': ('Moderate Effort', 'warning'),
            'HIGH': ('Complex', 'danger'),
        }
        effort_text, effort_badge = effort_badges.get(effort, ('Unknown', 'secondary'))

        # Simplify action text for executives
        simple_action = action
        if 'restart' in action.lower():
            simple_action = "Restart the server or application"
        elif 'scale' in action.lower():
            simple_action = "Add more resources or servers"
        elif 'memory' in action.lower() or 'cache' in action.lower():
            simple_action = "Free up memory"
        elif 'disk' in action.lower():
            simple_action = "Clear disk space"
        elif 'connection' in action.lower():
            simple_action = "Check network connections"

        card = dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.H5(f"Option {i+1}: {name.replace('_', ' ').title()}", className="mb-0")
                    ], width=8),
                    dbc.Col([
                        dbc.Badge(impact_text, color=badge_color, className="me-2"),
                        dbc.Badge(effort_text, color=effort_badge)
                    ], width=4, className="text-end")
                ])
            ]),
            dbc.CardBody([
                html.P([
                    html.Strong("What to do: "),
                    simple_action
                ], className="mb-2"),
                html.P([
                    html.Strong("Expected result: "),
                    f"{'Improvement' if change < 0 else 'Minimal change'} in server health",
                    f" ({abs(change):.0f}% {'better' if change < 0 else 'impact'})" if change != 0 else ""
                ], className="mb-2 text-muted"),
                dbc.Progress(
                    value=confidence * 100,
                    label=f"{confidence:.0%} confidence",
                    color="success" if confidence > 0.7 else "warning" if confidence > 0.4 else "danger",
                    style={'height': '24px'}
                )
            ])
        ], className="mb-3")

        action_cards.append(card)

    return html.Div([
        html.H5("🎯 Recommended Actions", className="mb-3"),
        html.P("Listed in order of potential impact:", className="text-muted mb-3"),
        html.Div(action_cards)
    ])


def generate_key_factors(shap_data: Dict) -> html.Div:
    """Generate a simple visual of key factors."""
    feature_importance = shap_data.get('feature_importance', [])

    if not feature_importance:
        return html.Div()

    factors = []
    for feature, info in feature_importance[:5]:
        plain_name = get_metric_plain_name(feature)
        impact = info.get('impact', 0) * 100
        direction = info.get('direction', 'neutral')

        if direction == 'increasing':
            icon = "📈"
            status = "Rising"
            color = "#EF4444"  # Red - concerning
        elif direction == 'decreasing':
            icon = "📉"
            status = "Falling"
            color = "#10B981"  # Green - improving
        else:
            icon = "➡️"
            status = "Stable"
            color = "#6B7280"  # Gray

        factors.append(
            dbc.Row([
                dbc.Col([
                    html.Span(icon, style={'fontSize': '1.5rem'})
                ], width=1),
                dbc.Col([
                    html.Strong(plain_name.title()),
                    html.Span(f" - {status}", style={'color': color})
                ], width=5),
                dbc.Col([
                    dbc.Progress(
                        value=min(impact, 100),
                        color="danger" if impact > 50 else "warning" if impact > 25 else "info",
                        style={'height': '20px'}
                    )
                ], width=5),
                dbc.Col([
                    html.Span(f"{impact:.0f}%", className="text-muted")
                ], width=1)
            ], className="mb-2 align-items-center")
        )

    return html.Div([
        html.H5("📊 What's Driving This", className="mb-3"),
        html.P("These factors are most influencing the server's health:", className="text-muted mb-3"),
        html.Div(factors)
    ])


def generate_timeline_summary(attention_data: Dict) -> html.Div:
    """Generate a simple timeline summary."""
    summary = attention_data.get('summary', '')
    important_periods = attention_data.get('important_periods', [])

    if not important_periods:
        return html.Div()

    # Find the most important period
    most_important = max(important_periods, key=lambda x: x.get('attention', 0))
    period_name = most_important.get('period', 'Recent activity')

    return dbc.Alert([
        html.H6("⏰ When It Started", className="mb-2"),
        html.P([
            f"The AI is paying most attention to: ",
            html.Strong(period_name.lower()),
            ". This suggests the issue began or worsened during this timeframe."
        ], className="mb-0")
    ], color="light")


def render_technical_details(shap_data: Dict, attention_data: Dict, counterfactual_data) -> html.Div:
    """Render collapsible technical details for advanced users."""

    # SHAP details
    feature_importance = shap_data.get('feature_importance', [])
    shap_table_data = []
    for feature, info in feature_importance:
        shap_table_data.append({
            'Metric': feature,
            'Impact': f"{info.get('impact', 0) * 100:.1f}%",
            'Direction': info.get('direction', 'neutral'),
            'Importance': info.get('stars', '')
        })

    shap_df = pd.DataFrame(shap_table_data) if shap_table_data else pd.DataFrame()

    # Attention weights chart
    attention_weights = attention_data.get('attention_weights', [])
    attention_chart = html.Div()
    if attention_weights and len(attention_weights) > 10:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(attention_weights))),
            y=attention_weights,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#0EA5E9', width=2)
        ))
        fig.update_layout(
            title="Attention Weights Over Time",
            xaxis_title="Timestep",
            yaxis_title="Attention Weight",
            height=250,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        attention_chart = dcc.Graph(figure=fig)

    return dbc.Accordion([
        dbc.AccordionItem([
            html.P("This table shows the raw SHAP (SHapley Additive exPlanations) values for each metric.",
                   className="text-muted mb-3"),
            dbc.Table.from_dataframe(shap_df, striped=True, bordered=True, hover=True, size='sm')
            if not shap_df.empty else html.P("No SHAP data available")
        ], title="📊 SHAP Feature Importance (Technical)"),

        dbc.AccordionItem([
            html.P("This chart shows which time periods the model weighted most heavily.",
                   className="text-muted mb-3"),
            attention_chart if attention_weights else html.P("No attention data available")
        ], title="⏱️ Attention Weights (Technical)"),

        dbc.AccordionItem([
            html.P("Raw counterfactual scenario data from the model.", className="text-muted mb-3"),
            html.Pre(
                str(counterfactual_data)[:2000] + "..." if len(str(counterfactual_data)) > 2000 else str(counterfactual_data),
                style={'fontSize': '0.8rem', 'whiteSpace': 'pre-wrap', 'backgroundColor': '#f8f9fa', 'padding': '1rem', 'borderRadius': '4px'}
            )
        ], title="🎯 Raw Counterfactual Data (Technical)")
    ], start_collapsed=True, className="mt-4")


def render(predictions: Dict, risk_scores: Dict[str, float],
           selected_server: str = None) -> html.Div:
    """
    Render Insights tab - Executive-friendly version.
    """
    server_preds = predictions.get('predictions', {})

    if not server_preds:
        return dbc.Alert([
            html.H4("🧠 Server Insights"),
            html.P("No servers are being monitored. Start the system to see insights."),
        ], color="warning")

    # Get sorted servers by risk (highest first)
    servers_with_risk = [
        (server, risk_scores.get(server, calculate_server_risk_score(pred)))
        for server, pred in server_preds.items()
    ]
    servers_with_risk.sort(key=lambda x: x[1], reverse=True)
    sorted_servers = [s[0] for s in servers_with_risk]

    # Default to highest risk server
    if not selected_server and sorted_servers:
        selected_server = sorted_servers[0]

    # Create dropdown with simple risk indicators
    server_options = []
    for server, risk in servers_with_risk:
        level, color, icon, _ = get_risk_level(risk)
        server_options.append({
            'label': f"{icon} {server} - {level} ({risk:.0f})",
            'value': server
        })

    # Header
    header = html.Div([
        html.H4("🧠 Server Health Insights", className="mb-2"),
        html.P("Select a server to understand its health and get recommendations.",
               className="text-muted mb-4")
    ])

    # Server selector
    selector = dbc.Row([
        dbc.Col([
            html.Label("Choose a server:", className="fw-bold mb-2"),
            dcc.Dropdown(
                id='insights-server-selector',
                options=server_options,
                value=selected_server,
                clearable=False,
                style={'fontSize': '1.1rem'}
            )
        ], width=8),
        dbc.Col([
            html.Label(" ", className="fw-bold mb-2"),  # Spacer
            dbc.Button(
                "🔄 Refresh",
                id='insights-refresh-button',
                color="primary",
                outline=True,
                className="w-100"
            )
        ], width=2)
    ], className="mb-4")

    # Loading content
    loading_content = dcc.Loading(
        id="insights-loading",
        type="circle",
        children=html.Div(id='insights-content')
    )

    # Help text
    help_text = dbc.Alert([
        html.Strong("💡 Tip: "),
        "Analysis takes 3-5 seconds. Results show why the AI flagged this server and what you can do about it."
    ], color="light", className="mt-3")

    return html.Div([
        header,
        selector,
        loading_content,
        help_text
    ])


# Export the render functions for the callback in dash_app.py
def render_shap_explanation(shap_data: Dict) -> html.Div:
    """Wrapper for backward compatibility with dash_app.py callback."""
    return generate_key_factors(shap_data)


def render_attention_analysis(attention_data: Dict) -> html.Div:
    """Wrapper for backward compatibility with dash_app.py callback."""
    return generate_timeline_summary(attention_data)


def render_counterfactuals(counterfactual_data) -> html.Div:
    """Wrapper for backward compatibility with dash_app.py callback."""
    return generate_action_cards(counterfactual_data, 50)  # Default risk score
