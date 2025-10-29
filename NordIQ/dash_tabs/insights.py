"""
Insights (XAI) Tab - Explainable AI Analysis
============================================

Provides deep insights into WHY predictions happen and WHAT TO DO:
- SHAP feature importance (which metrics drove the prediction)
- Attention analysis (which time periods the model focused on)
- Counterfactual scenarios (what-if analysis with actionable recommendations)

This is a key differentiator - shows the AI's reasoning in plain English!

Performance: Target <500ms (complex tab, may use loading states)
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

# Professional metric display names
METRIC_DISPLAY_NAMES = {
    'cpu_user_pct': 'CPU User %',
    'cpu_sys_pct': 'CPU System %',
    'cpu_iowait_pct': 'CPU I/O Wait %',
    'cpu_idle_pct': 'CPU Idle %',
    'java_cpu_pct': 'Java CPU %',
    'mem_used_pct': 'Memory Used %',
    'swap_used_pct': 'Swap Used %',
    'disk_usage_pct': 'Disk Usage %',
    'net_in_mb_s': 'Network In (MB/s)',
    'net_out_mb_s': 'Network Out (MB/s)',
    'back_close_wait': 'Backend Close-Wait Connections',
    'front_close_wait': 'Frontend Close-Wait Connections',
    'load_average': 'System Load Average',
    'uptime_days': 'Server Uptime (days)',
}


def get_metric_display_name(metric_name: str) -> str:
    """Convert internal metric name to user-friendly display name."""
    return METRIC_DISPLAY_NAMES.get(
        metric_name,
        # Fallback: capitalize and replace underscores
        metric_name.replace('_', ' ').replace('pct', '%').title()
    )


def fetch_explanation(server_name: str, daemon_url: str = DAEMON_URL) -> Optional[Dict]:
    """
    Fetch XAI explanation for a specific server from the daemon.

    Args:
        server_name: Server to explain
        daemon_url: URL of the inference daemon

    Returns:
        Dict with SHAP, attention, and counterfactual explanations, or None if error
    """
    try:
        response = requests.get(
            f"{daemon_url}/explain/{server_name}",
            headers=get_auth_headers(),
            timeout=10
        )
        if response.ok:
            return response.json()
        else:
            print(f"[ERROR] Failed to fetch explanation: {response.status_code}")
            if response.status_code == 403:
                print("[ERROR] Authentication failed - check API key")
            return None
    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out - XAI analysis can take a few seconds")
        return None
    except Exception as e:
        print(f"[ERROR] Error fetching explanation: {str(e)}")
        return None


def render_shap_explanation(shap_data: Dict) -> html.Div:
    """
    Render SHAP feature importance analysis.

    Shows which metrics (CPU, memory, disk, network) drove the prediction.
    """
    feature_importance = shap_data.get('feature_importance', [])

    if not feature_importance:
        return dbc.Alert("No feature importance data available", color="info")

    # Display summary
    summary = shap_data.get('summary', 'No summary available')

    # Create bar chart of feature importance
    features = []
    impacts = []
    directions = []
    stars = []

    for feature, info in feature_importance:
        # Use professional display names
        feature_display = get_metric_display_name(feature)
        features.append(feature_display)
        impacts.append(info['impact'] * 100)  # Convert to percentage
        directions.append(info['direction'])
        stars.append(info.get('stars', ''))

    # Create DataFrame for display
    df = pd.DataFrame({
        'Metric': features,
        'Impact': impacts,
        'Direction': directions,
        'Importance': stars
    })

    # Plotly bar chart
    fig = go.Figure()

    # Color by direction
    # Increasing risk (bad) = red, decreasing risk (good) = green, neutral = gray
    colors = [
        '#10B981' if d == 'increasing'  # Green (good - risk going down)
        else '#EF4444' if d == 'decreasing'  # Red (bad - risk going up)
        else '#6B7280'  # Gray (neutral)
        for d in directions
    ]

    fig.add_trace(go.Bar(
        x=impacts,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{imp:.1f}%" for imp in impacts],
        textposition='auto'
    ))

    fig.update_layout(
        title="Feature Impact on Prediction",
        xaxis_title="Impact (%)",
        yaxis_title="Metric",
        height=400,
        showlegend=False
    )

    return html.Div([
        dbc.Alert([
            html.Strong("What This Shows: "),
            "Which metrics are driving the prediction. ",
            "⭐⭐⭐ = Very high impact | ⭐⭐ = Medium impact | ⭐ = Low impact"
        ], color="light", className="mb-3"),
        html.P([html.Strong("Summary: "), summary], className="mb-3"),
        dcc.Graph(figure=fig),
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
            ], title="📋 Detailed Breakdown")
        ], start_collapsed=True)
    ])


def render_attention_analysis(attention_data: Dict) -> html.Div:
    """
    Render attention visualization showing which time periods the model focused on.
    """
    summary = attention_data.get('summary', 'No summary available')
    important_periods = attention_data.get('important_periods', [])

    period_cards = []
    if important_periods:
        for period in important_periods:
            attention_pct = period['attention'] * 100
            importance = period['importance']

            # Color based on importance
            if importance == 'VERY HIGH':
                color = '#EF4444'  # Red
            elif importance == 'HIGH':
                color = '#F59E0B'  # Orange
            elif importance == 'MEDIUM':
                color = '#EAB308'  # Yellow
            else:
                color = '#6B7280'  # Gray

            period_cards.append(
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(period['period'], className="card-subtitle mb-2"),
                            html.H3(f"{attention_pct:.0f}%", style={'color': color}),
                            html.P(f"{importance} importance", className="small text-muted mb-0")
                        ])
                    ], style={'borderLeft': f'4px solid {color}'})
                ], width=12 if len(important_periods) < 3 else 4)
            )

    period_row = dbc.Row(period_cards, className="mb-3") if period_cards else html.Div()

    # Attention weights timeline (if available)
    attention_weights = attention_data.get('attention_weights', [])
    timeline_chart = html.Div()

    if attention_weights and len(attention_weights) > 10:
        # Create line chart
        fig = go.Figure()

        timesteps = list(range(len(attention_weights)))

        # Use Scattergl for GPU-accelerated rendering
        fig.add_trace(go.Scattergl(
            x=timesteps,
            y=attention_weights,
            mode='lines',
            fill='tozeroy',
            line=dict(color='#0EA5E9', width=2),
            name='Attention Weight'
        ))

        fig.update_layout(
            title="Attention Weights Over Time",
            xaxis_title="Timestep (most recent = right)",
            yaxis_title="Attention Weight",
            height=300,
            showlegend=False
        )

        timeline_chart = dbc.Accordion([
            dbc.AccordionItem([
                dcc.Graph(figure=fig)
            ], title="📈 Attention Timeline")
        ], start_collapsed=True, className="mt-3")

    return html.Div([
        dbc.Alert([
            html.Strong("What This Shows: "),
            "Which time periods the model \"paid attention to\" when making the prediction. ",
            "Higher attention = more influence on the prediction."
        ], color="light", className="mb-3"),
        html.P([html.Strong("Summary: "), summary], className="mb-3"),
        period_row,
        timeline_chart
    ])


def render_counterfactuals(counterfactual_data) -> html.Div:
    """
    Render counterfactual scenarios (what-if analysis).

    Args:
        counterfactual_data: Can be a dict with 'scenarios' key, or a list of scenarios directly
    """
    # Handle case where counterfactual_data is a list (legacy format)
    if isinstance(counterfactual_data, list):
        scenarios = counterfactual_data
        summary = 'What-if scenarios showing predicted outcomes for different actions'
    else:
        # Handle dict format
        summary = counterfactual_data.get('summary', 'No summary available')
        scenarios = counterfactual_data.get('scenarios', [])

    if not scenarios:
        return dbc.Alert("No scenario recommendations available", color="info")

    # Scenario icons
    SCENARIO_ICONS = {
        'restart': '🔄',
        'scale': '📈',
        'stabilize': '⚖️',
        'optimize': '⚡',
        'reduce': '🧹',
        'nothing': '⏸️'
    }

    scenario_rows = []

    # Handle both dict and list formats
    if isinstance(scenarios, dict):
        scenario_items = scenarios.items()
    else:
        # If it's a list, create (name, data) pairs
        scenario_items = [(s.get('name', f'Scenario {i+1}'), s) for i, s in enumerate(scenarios)]

    for scenario_name, scenario in scenario_items:
        # Extract scenario data (handle both string keys and dict with 'scenario' key)
        if isinstance(scenario, dict) and 'scenario' in scenario:
            display_name = scenario['scenario']
        else:
            display_name = scenario_name

        icon = SCENARIO_ICONS.get(display_name.lower().split()[0], '💡')
        predicted_cpu = scenario.get('predicted_cpu', 0)
        current_cpu = scenario.get('baseline_cpu', predicted_cpu)
        change = scenario.get('change', predicted_cpu - current_cpu)
        is_safe = scenario.get('safe', True)
        effort = scenario.get('effort', 'UNKNOWN')
        risk = scenario.get('risk', 'MEDIUM')
        action = scenario.get('action', 'No action details available')
        confidence = scenario.get('confidence', 0.5)

        safety_icon = "✅" if is_safe else "⚠️"

        # Color based on change (green=improvement, red=worse)
        change_color = '#10B981' if change < 0 else '#EF4444' if change > 0 else '#6B7280'

        # Effort color (green=low, yellow=medium, red=high)
        effort_color = {
            'LOW': '#10B981',
            'MEDIUM': '#EAB308',
            'HIGH': '#EF4444',
            'None': '#6B7280'
        }.get(effort, '#6B7280')

        # Risk color
        risk_color = {
            'LOW': '#10B981',
            'MEDIUM': '#EAB308',
            'HIGH': '#EF4444'
        }.get(risk, '#6B7280')

        scenario_rows.append(
            dbc.Card([
                dbc.CardBody([
                    # Header row with name and key metrics
                    dbc.Row([
                        dbc.Col([
                            html.H5(f"{safety_icon} {icon} {display_name}", className="mb-2")
                        ], width=6),
                        dbc.Col([
                            html.Div([
                                html.H4(
                                    f"{predicted_cpu:.1f}%",
                                    className="mb-0",
                                    style={'color': '#EF4444' if predicted_cpu > 85 else '#EAB308' if predicted_cpu > 75 else '#10B981'}
                                ),
                                html.Small("Predicted CPU", className="text-muted")
                            ], className="text-center")
                        ], width=2),
                        dbc.Col([
                            html.Div([
                                html.H4(
                                    f"{change:+.1f}%",
                                    className="mb-0",
                                    style={'color': change_color, 'fontWeight': 'bold'}
                                ),
                                html.Small("Change", className="text-muted")
                            ], className="text-center")
                        ], width=2),
                        dbc.Col([
                            html.Div([
                                html.Span(effort, style={'color': effort_color, 'fontWeight': 'bold'}),
                                html.Br(),
                                html.Small("Effort", className="text-muted")
                            ], className="text-center")
                        ], width=1),
                        dbc.Col([
                            html.Div([
                                html.Span(risk, style={'color': risk_color, 'fontWeight': 'bold'}),
                                html.Br(),
                                html.Small("Risk", className="text-muted")
                            ], className="text-center")
                        ], width=1),
                    ], className="mb-2"),

                    # Action row (most important!)
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Strong("📋 How to implement: ", style={'color': '#0EA5E9'}),
                                html.Span(action, style={'fontFamily': 'monospace', 'fontSize': '0.95em'})
                            ], className="p-2", style={'backgroundColor': '#F0F9FF', 'borderRadius': '4px'})
                        ], width=12)
                    ], className="mt-2"),

                    # Confidence indicator
                    dbc.Row([
                        dbc.Col([
                            dbc.Progress(
                                value=confidence * 100,
                                label=f"Confidence: {confidence:.0%}",
                                color="success" if confidence > 0.8 else "warning" if confidence > 0.6 else "danger",
                                className="mt-2",
                                style={'height': '20px'}
                            )
                        ], width=12)
                    ])
                ])
            ], className="mb-3", style={'borderLeft': f'4px solid {change_color}'})
        )

    return html.Div([
        dbc.Alert([
            html.H5("🎯 What-If Scenarios: Actionable Recommendations", className="mb-2"),
            html.P([
                "Each scenario below shows the ", html.Strong("predicted CPU impact"),
                " of taking a specific action, along with ", html.Strong("implementation details"),
                " so you know exactly what to do."
            ], className="mb-2"),
            html.P([
                html.Strong("Key:"),
                " ✅ = Safe (below threshold) | ⚠️ = Risky (above threshold) | ",
                html.Span("Green", style={'color': '#10B981', 'fontWeight': 'bold'}),
                " = Improvement | ",
                html.Span("Red", style={'color': '#EF4444', 'fontWeight': 'bold'}),
                " = Worse"
            ], className="mb-0")
        ], color="info", className="mb-3"),
        html.Div(scenario_rows)
    ])


def render(predictions: Dict, risk_scores: Dict[str, float],
           selected_server: str = None) -> html.Div:
    """
    Render Insights (XAI) tab.

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores
        selected_server: Selected server for analysis (optional)

    Returns:
        html.Div: Tab content
    """
    server_preds = predictions.get('predictions', {})

    if not server_preds:
        return dbc.Alert([
            html.H4("🧠 Explainable AI Insights"),
            html.P("No predictions available. Start the inference daemon and metrics generator."),
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

    # Create server selector dropdown options
    server_options = [
        {
            'label': f"{server} (Risk: {risk:.0f})",
            'value': server
        }
        for server, risk in servers_with_risk
    ]

    # Header and description
    header = html.Div([
        html.H4("🧠 Explainable AI Insights", className="mb-3"),
        dbc.Alert([
            html.P([
                html.Strong("NordIQ AI Advantage: "),
                "Most monitoring tools just show you numbers. ",
                "We show you the ", html.Strong("reasoning"), " behind them."
            ], className="mb-0")
        ], color="info")
    ])

    # Server selector with refresh button
    selector = dbc.Row([
        dbc.Col([
            html.Label("Select server to analyze:", className="fw-bold"),
            dcc.Dropdown(
                id='insights-server-selector',
                options=server_options,
                value=selected_server,
                clearable=False,
                className="mb-3"
            )
        ], width=6),
        dbc.Col([
            html.Label(html.Span(style={'visibility': 'hidden'}), className="fw-bold"),  # Spacer for alignment
            dbc.Button(
                "🔄 Refresh Analysis",
                id='insights-refresh-button',
                color="primary",
                outline=True,
                className="mb-3",
                size="sm"
            )
        ], width=3)
    ])

    # Loading spinner for XAI analysis
    loading_content = dcc.Loading(
        id="insights-loading",
        type="default",
        children=html.Div(id='insights-content')
    )

    # Note about interactivity and manual refresh
    note = dbc.Alert([
        html.Strong("💡 How Insights Works: "),
        html.Br(),
        "• XAI analysis runs when you select a server (3-5 seconds). ",
        html.Br(),
        "• Results are cached - no auto-refresh to prevent aggressive reloading. ",
        html.Br(),
        "• Use the 🔄 Refresh button to manually update analysis with latest data. ",
        html.Br(),
        "• Requires daemon with XAI enabled (/explain endpoint)."
    ], color="light", className="mt-3")

    return html.Div([
        header,
        selector,
        loading_content,
        note
    ])


# This would normally be in dash_app.py as a callback, but showing here for completeness
# @callback(
#     Output('insights-content', 'children'),
#     Input('insights-server-selector', 'value')
# )
# def update_insights_content(selected_server):
#     """Callback to fetch and display XAI explanation for selected server."""
#     if not selected_server:
#         return dbc.Alert("Select a server to analyze", color="info")

#     # Fetch explanation
#     explanation = fetch_explanation(selected_server)

#     if not explanation or 'error' in explanation:
#         return dbc.Alert([
#             html.H5("❌ XAI Analysis Unavailable"),
#             html.P("Could not fetch explanation. Check that daemon has XAI enabled."),
#             html.P("Ensure the /explain endpoint is available on the daemon.", className="mb-0")
#         ], color="danger")

#     # Render XAI components in tabs
#     tabs = dbc.Tabs([
#         dbc.Tab(
#             render_shap_explanation(explanation.get('shap', {})),
#             label="📊 Feature Importance"
#         ),
#         dbc.Tab(
#             render_attention_analysis(explanation.get('attention', {})),
#             label="⏱️ Temporal Focus"
#         ),
#         dbc.Tab(
#             render_counterfactuals(explanation.get('counterfactuals', {})),
#             label="🎯 What-If Scenarios"
#         ),
#     ])

#     return tabs
