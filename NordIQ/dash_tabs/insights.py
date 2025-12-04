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
    """
    Generate enhanced action recommendation cards.

    Each card shows:
    - Clear action name and what it does
    - Expected improvement (with visual indicator)
    - Effort required and risk level
    - Step-by-step guidance
    - Confidence level
    """
    # Handle different data formats
    if isinstance(counterfactual_data, list):
        scenarios = counterfactual_data
    else:
        scenarios = counterfactual_data.get('scenarios', [])

    if not scenarios:
        if risk_score < 40:
            return dbc.Alert([
                html.H5("✅ No Action Required", className="alert-heading"),
                html.P("This server is healthy. Continue monitoring as usual.", className="mb-0")
            ], color="success")
        else:
            return dbc.Alert([
                html.H5("⚠️ Recommendations Unavailable", className="alert-heading"),
                html.P("Could not generate specific recommendations. Review server metrics manually.", className="mb-0")
            ], color="warning")

    # Process scenarios
    if isinstance(scenarios, dict):
        scenario_items = list(scenarios.items())
    else:
        scenario_items = [(s.get('scenario', f'Option {i+1}'), s) for i, s in enumerate(scenarios)]

    # Filter out "Do nothing" and sort by improvement
    actionable = [(name, s) for name, s in scenario_items
                  if isinstance(s, dict) and s.get('change', 0) < 0]
    actionable.sort(key=lambda x: x[1].get('change', 0))

    if not actionable:
        return dbc.Alert([
            html.H5("📊 Current Status", className="alert-heading"),
            html.P("No interventions would significantly improve this server's health. Continue monitoring.", className="mb-0")
        ], color="info")

    # Build action cards
    action_cards = []

    # Find the best recommendation (highest impact with reasonable effort)
    best_idx = 0
    best_score = float('-inf')
    effort_scores = {'LOW': 3, 'MEDIUM': 2, 'HIGH': 1, 'None': 0}

    for i, (name, scenario) in enumerate(actionable[:4]):
        change = abs(scenario.get('change', 0))
        effort = effort_scores.get(scenario.get('effort', 'MEDIUM'), 2)
        score = change * effort  # Higher is better
        if score > best_score:
            best_score = score
            best_idx = i

    for i, (name, scenario) in enumerate(actionable[:4]):
        is_best = (i == best_idx)

        # Extract scenario data
        action = scenario.get('action', 'Review this server')
        change = scenario.get('change', 0)
        predicted_cpu = scenario.get('predicted_cpu', 0)
        effort = scenario.get('effort', 'MEDIUM')
        risk = scenario.get('risk', 'MEDIUM')
        confidence = scenario.get('confidence', 0.5)
        is_safe = scenario.get('safe', True)
        scenario_name = scenario.get('scenario', name)

        # Calculate improvement percentage
        improvement = abs(change)

        # Friendly scenario names
        friendly_names = {
            'Restart service': ('🔄 Restart the Service', 'Quick reset to clear issues'),
            'Scale horizontally': ('📈 Add More Capacity', 'Distribute the load across more servers'),
            'Stabilize workload': ('⚖️ Stabilize the Workload', 'Prevent further increases'),
            'Reduce memory': ('🧹 Free Up Memory', 'Clear caches and reduce memory pressure'),
            'Optimize disk': ('💾 Optimize Disk Usage', 'Speed up disk operations'),
        }

        # Match scenario to friendly name
        display_name = scenario_name
        description = ""
        for key, (friendly, desc) in friendly_names.items():
            if key.lower() in scenario_name.lower():
                display_name = friendly
                description = desc
                break

        # Effort and risk indicators
        effort_config = {
            'LOW': ('✓ Easy', 'success', 'Can be done quickly with minimal planning'),
            'MEDIUM': ('◐ Moderate', 'warning', 'Requires some planning and coordination'),
            'HIGH': ('✗ Complex', 'danger', 'Needs significant planning and resources'),
        }
        risk_config = {
            'LOW': ('Safe', 'success'),
            'MEDIUM': ('Some Risk', 'warning'),
            'HIGH': ('Risky', 'danger'),
        }

        effort_text, effort_color, effort_desc = effort_config.get(effort, ('Unknown', 'secondary', ''))
        risk_text, risk_color = risk_config.get(risk, ('Unknown', 'secondary'))

        # Simplify technical commands to plain English steps
        steps = generate_action_steps(scenario_name, action)

        # Card border color based on best recommendation
        border_color = '#10B981' if is_best else '#E5E7EB'
        header_bg = '#ECFDF5' if is_best else '#F9FAFB'

        card = dbc.Card([
            dbc.CardHeader([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("⭐ RECOMMENDED" if is_best else "",
                                     className="text-success fw-bold small me-2") if is_best else None,
                            html.H5(display_name, className="mb-0 d-inline"),
                        ]),
                        html.Small(description, className="text-muted") if description else None
                    ], width=8),
                    dbc.Col([
                        html.Div([
                            html.Span(f"−{improvement:.0f}%",
                                     style={'fontSize': '1.5rem', 'fontWeight': 'bold',
                                            'color': '#10B981' if improvement > 10 else '#3B82F6'}),
                            html.Br(),
                            html.Small("improvement", className="text-muted")
                        ], className="text-end")
                    ], width=4)
                ])
            ], style={'backgroundColor': header_bg}),

            dbc.CardBody([
                # Expected outcome
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Strong("Expected Result: ", className="text-muted"),
                            html.Span(
                                f"CPU drops to ~{predicted_cpu:.0f}%" if predicted_cpu else "Improvement expected",
                                style={'color': '#10B981' if is_safe else '#F59E0B'}
                            ),
                            html.Span(" ✓ Safe zone" if is_safe else " ⚠ Still elevated",
                                     className="ms-2 small",
                                     style={'color': '#10B981' if is_safe else '#F59E0B'})
                        ])
                    ], width=12)
                ], className="mb-3"),

                # Steps to take
                html.Div([
                    html.Strong("📋 Steps to Take:", className="mb-2 d-block"),
                    html.Ol([
                        html.Li(step, className="mb-1") for step in steps
                    ], className="mb-0 ps-3", style={'fontSize': '0.95rem'})
                ], className="mb-3 p-2", style={'backgroundColor': '#F8FAFC', 'borderRadius': '6px'}),

                # Effort and Risk indicators
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Strong("Effort: ", className="text-muted"),
                            dbc.Badge(effort_text, color=effort_color, className="me-2"),
                            html.Small(effort_desc, className="text-muted d-none d-md-inline")
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.Strong("Risk: ", className="text-muted"),
                            dbc.Badge(risk_text, color=risk_color)
                        ])
                    ], width=3),
                    dbc.Col([
                        html.Div([
                            html.Strong("Confidence: ", className="text-muted"),
                            html.Span(f"{confidence:.0%}",
                                     style={'fontWeight': 'bold',
                                            'color': '#10B981' if confidence > 0.7 else '#F59E0B' if confidence > 0.5 else '#EF4444'})
                        ])
                    ], width=3)
                ], className="mb-2"),

                # Confidence bar
                dbc.Progress(
                    value=confidence * 100,
                    color="success" if confidence > 0.7 else "warning" if confidence > 0.5 else "danger",
                    style={'height': '6px'},
                    className="mt-2"
                )
            ])
        ], className="mb-3", style={'border': f'2px solid {border_color}'})

        action_cards.append(card)

    return html.Div([
        html.Div([
            html.H5("🎯 Recommended Actions", className="mb-1"),
            html.P("What you can do to improve this server's health, ranked by effectiveness:",
                   className="text-muted mb-3")
        ]),
        html.Div(action_cards)
    ])


def generate_action_steps(scenario_name: str, technical_action: str) -> List[str]:
    """Generate user-friendly step-by-step instructions."""

    scenario_lower = scenario_name.lower()

    if 'restart' in scenario_lower:
        return [
            "Notify affected teams about brief service interruption",
            "Schedule restart during low-traffic period if possible",
            "Execute the service restart (typically 30-60 seconds downtime)",
            "Verify service is back online and responding normally",
            "Monitor for 15 minutes to confirm improvement"
        ]

    elif 'scale' in scenario_lower:
        return [
            "Review current capacity and confirm additional resources are available",
            "Spin up additional server instances (2+ recommended)",
            "Update load balancer to include new instances",
            "Verify traffic is being distributed across all instances",
            "Monitor new instances for health and performance"
        ]

    elif 'stabilize' in scenario_lower or 'workload' in scenario_lower:
        return [
            "Identify the source of increasing load (check logs, metrics)",
            "Enable rate limiting or request throttling if available",
            "Consider redirecting traffic to healthier instances",
            "Communicate with development team about load patterns",
            "Monitor to confirm load has stabilized"
        ]

    elif 'memory' in scenario_lower or 'cache' in scenario_lower:
        return [
            "Identify largest memory consumers (check heap dumps, process list)",
            "Clear application caches if safe to do so",
            "Review and reduce connection pool sizes if oversized",
            "Consider restarting memory-heavy processes",
            "Monitor memory usage after each action"
        ]

    elif 'disk' in scenario_lower:
        return [
            "Identify slow queries or high I/O operations",
            "Add database indexes for frequent queries",
            "Enable query caching if not already active",
            "Consider moving large files to faster storage",
            "Monitor disk I/O metrics after changes"
        ]

    else:
        # Generic steps
        return [
            "Review current server metrics and recent changes",
            "Identify the root cause of the issue",
            "Apply the recommended action",
            "Verify the issue is resolved",
            "Document the fix for future reference"
        ]


def generate_key_factors(shap_data: Dict) -> html.Div:
    """Generate a simple visual of key factors driving the risk."""
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


def render_technical_details(shap_data: Dict, attention_data: Dict, counterfactual_data) -> html.Div:
    """Render collapsible technical details for engineers."""

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

    return dbc.Accordion([
        dbc.AccordionItem([
            html.P("Raw SHAP (SHapley Additive exPlanations) values showing feature contributions.",
                   className="text-muted mb-3"),
            dbc.Table.from_dataframe(shap_df, striped=True, bordered=True, hover=True, size='sm')
            if not shap_df.empty else html.P("No SHAP data available")
        ], title="📊 Feature Importance (Technical)"),

        dbc.AccordionItem([
            html.P("Raw counterfactual scenario data from the model.", className="text-muted mb-3"),
            html.Pre(
                str(counterfactual_data)[:3000] + "..." if len(str(counterfactual_data)) > 3000 else str(counterfactual_data),
                style={'fontSize': '0.8rem', 'whiteSpace': 'pre-wrap', 'backgroundColor': '#f8f9fa', 'padding': '1rem', 'borderRadius': '4px'}
            )
        ], title="🎯 Raw Scenario Data (Technical)")
    ], start_collapsed=True, className="mt-4")


def render(predictions: Dict, risk_scores: Dict[str, float],
           selected_server: str = None) -> html.Div:
    """Render Insights tab - Executive-friendly version."""
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
        html.P("Select a server to understand its health and get actionable recommendations.",
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


# Backward compatibility wrappers
def render_shap_explanation(shap_data: Dict) -> html.Div:
    """Wrapper for backward compatibility."""
    return generate_key_factors(shap_data)


def render_attention_analysis(attention_data: Dict) -> html.Div:
    """Wrapper for backward compatibility - returns empty since we removed this."""
    return html.Div()


def render_counterfactuals(counterfactual_data) -> html.Div:
    """Wrapper for backward compatibility."""
    return generate_action_cards(counterfactual_data, 50)
