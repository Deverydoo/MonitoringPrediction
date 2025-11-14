#!/usr/bin/env python3
"""
Plotly Dash Proof of Concept - ArgusAI Monitoring Dashboard
==============================================================

Performance Target: <500ms page loads (vs 12s in Streamlit)

This PoC demonstrates:
1. 3 tabs (Overview, Heatmap, Top Risks) - copy from Streamlit
2. Real Plotly charts (same code as Streamlit)
3. Connected to existing inference daemon
4. Callback-based updates (only active tab refreshes)

Run: python dash_poc.py
Then open: http://localhost:8050
"""

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache
import sys
from pathlib import Path

# Add src/ to path for imports
NORDIQ_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(NORDIQ_SRC))

# =============================================================================
# CONFIGURATION
# =============================================================================

DAEMON_URL = "http://localhost:8000"
REFRESH_INTERVAL = 5000  # milliseconds (5 seconds)

# API Key for daemon authentication (X-API-Key header)
# Priority: Environment variable > Empty (dev mode)
import os
DAEMON_API_KEY = os.getenv("TFT_API_KEY", "")

if not DAEMON_API_KEY:
    print("[INFO] No API key configured - running in development mode")
    print("[INFO] Dashboard will only work if daemon has no TFT_API_KEY set")
    print("[INFO] For production: set TFT_API_KEY environment variable")

# =============================================================================
# API CLIENT (Reused from Streamlit)
# =============================================================================

def get_auth_headers():
    """Get authentication headers with API key if configured."""
    headers = {"Content-Type": "application/json"}
    if DAEMON_API_KEY:
        headers["X-API-Key"] = DAEMON_API_KEY
    return headers

def fetch_predictions():
    """Fetch current predictions from daemon."""
    try:
        response = requests.get(
            f"{DAEMON_URL}/predictions/current",
            headers=get_auth_headers(),
            timeout=5
        )
        if response.ok:
            return response.json()
        elif response.status_code == 403:
            print(f"Error: Authentication failed - check TFT_API_KEY environment variable")
            return None
    except Exception as e:
        print(f"Error fetching predictions: {e}")
    return None

def check_daemon_health():
    """Check if daemon is connected."""
    try:
        response = requests.get(f"{DAEMON_URL}/health", timeout=2)
        return response.ok
    except:
        return False

# =============================================================================
# UTILITY FUNCTIONS (Copied from Streamlit utils)
# =============================================================================

def extract_cpu_used(server_pred, metric_type='current'):
    """Extract CPU used % (100 - idle)."""
    if 'cpu_idle_pct' in server_pred:
        idle_metric = server_pred['cpu_idle_pct']
        if metric_type == 'current':
            idle = idle_metric.get('current', 0)
            return 100 - idle
        elif metric_type in ['p50', 'p90']:
            percentile = idle_metric.get(metric_type, [])
            if percentile and len(percentile) >= 6:
                avg_idle = np.mean(percentile[:6])
                return 100 - avg_idle
    return 0

@lru_cache(maxsize=500)
def calculate_server_risk_score_cached(server_name, timestamp):
    """Cached wrapper - cache key is server_name + timestamp."""
    # This will be called by the non-cached version
    pass

def calculate_server_risk_score(server_pred):
    """Calculate risk score for a server (0-100)."""
    current_risk = 0.0
    predicted_risk = 0.0

    # CPU Risk
    current_cpu = extract_cpu_used(server_pred, 'current')
    max_cpu_p90 = extract_cpu_used(server_pred, 'p90')

    if current_cpu >= 98:
        current_risk += 60
    elif current_cpu >= 95:
        current_risk += 40
    elif current_cpu >= 90:
        current_risk += 20

    if max_cpu_p90 >= 98:
        predicted_risk += 30
    elif max_cpu_p90 >= 95:
        predicted_risk += 20

    # I/O Wait Risk
    if 'cpu_iowait_pct' in server_pred:
        iowait = server_pred['cpu_iowait_pct']
        current_iowait = iowait.get('current', 0)
        p90 = iowait.get('p90', [])
        max_iowait_p90 = max(p90[:6]) if p90 and len(p90) >= 6 else current_iowait

        if current_iowait >= 30:
            current_risk += 50
        elif current_iowait >= 20:
            current_risk += 30
        elif current_iowait >= 10:
            current_risk += 15

        if max_iowait_p90 >= 30:
            predicted_risk += 25
        elif max_iowait_p90 >= 20:
            predicted_risk += 15

    # Memory Risk
    if 'mem_used_pct' in server_pred:
        mem = server_pred['mem_used_pct']
        current_mem = mem.get('current', 0)
        p90 = mem.get('p90', [])
        max_mem_p90 = max(p90[:6]) if p90 and len(p90) >= 6 else current_mem

        if current_mem >= 98:
            current_risk += 60
        elif current_mem >= 95:
            current_risk += 40
        elif current_mem >= 90:
            current_risk += 20

        if max_mem_p90 >= 98:
            predicted_risk += 30
        elif max_mem_p90 >= 95:
            predicted_risk += 20

    # Weighted final score (70% current, 30% predicted)
    final_risk = (current_risk * 0.7) + (predicted_risk * 0.3)
    return min(final_risk, 100)

def get_risk_color(risk_score):
    """Get color for risk score."""
    if risk_score >= 80:
        return '#ff4444'  # Red
    elif risk_score >= 60:
        return '#ff9900'  # Orange
    elif risk_score >= 50:
        return '#ffcc00'  # Yellow
    else:
        return '#44ff44'  # Green

# =============================================================================
# DASH APP INITIALIZATION
# =============================================================================

# Initialize app with Bootstrap theme (Wells Fargo Red customization)
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Custom CSS for Wells Fargo branding
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>ArgusAI</title>
        {%favicon%}
        {%css%}
        <style>
            .navbar {
                background-color: #D71E28 !important;
                color: white !important;
            }
            .nav-link.active {
                background-color: #B71C1C !important;
            }
            h1, h2, h3 {
                color: #D71E28;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# =============================================================================
# LAYOUT
# =============================================================================

app.layout = dbc.Container([
    # Header with Wells Fargo branding
    dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("🏛️ ArgusAI", className="text-white mb-0"),
                    html.Small("Predictive System Monitoring - Predictive Infrastructure Monitoring",
                              className="text-white-50")
                ])
            ], align="center")
        ]),
        color="#D71E28",
        dark=True,
        className="mb-4"
    ),

    # Connection status
    dbc.Alert(id="connection-status", dismissable=False, className="mb-3"),

    # Performance timer (shows page load time)
    html.Div(id="performance-timer", className="mb-2"),

    # Tabs
    dbc.Tabs([
        dbc.Tab(label="📊 Overview", tab_id="overview"),
        dbc.Tab(label="🔥 Heatmap", tab_id="heatmap"),
        dbc.Tab(label="⚠️ Top 5 Risks", tab_id="risks"),
    ], id="tabs", active_tab="overview", className="mb-3"),

    # Tab content (updated via callback)
    html.Div(id="tab-content"),

    # Hidden div to store predictions data
    dcc.Store(id='predictions-store'),

    # Auto-refresh interval (5 seconds)
    dcc.Interval(
        id='interval-component',
        interval=REFRESH_INTERVAL,
        n_intervals=0
    ),

    # Timer for performance measurement
    html.Div(id='load-start-time', style={'display': 'none'},
             children=str(datetime.now().timestamp()))

], fluid=True, className="px-4")

# =============================================================================
# CALLBACKS
# =============================================================================

@app.callback(
    [Output('predictions-store', 'data'),
     Output('connection-status', 'children'),
     Output('connection-status', 'color')],
    Input('interval-component', 'n_intervals')
)
def update_predictions(n):
    """Fetch predictions from daemon (runs every 5 seconds)."""
    is_connected = check_daemon_health()

    if is_connected:
        predictions = fetch_predictions()
        if predictions:
            return (
                predictions,
                f"✅ Connected to daemon | Last update: {datetime.now().strftime('%H:%M:%S')}",
                "success"
            )

    return (
        None,
        "❌ Daemon not connected - Start daemon: python tft_inference_daemon.py",
        "danger"
    )


@app.callback(
    [Output('tab-content', 'children'),
     Output('performance-timer', 'children')],
    [Input('tabs', 'active_tab'),
     Input('predictions-store', 'data')],
    State('load-start-time', 'children')
)
def render_tab(active_tab, predictions, start_time):
    """
    Render selected tab - ONLY THIS TAB RUNS (not all 11 tabs like Streamlit!)

    This is the key performance win: Dash only executes callbacks for
    what changed. Streamlit reruns the entire script.
    """
    render_start = datetime.now()

    if not predictions:
        return html.Div("⚠️ Waiting for daemon connection..."), ""

    # OPTIMIZATION: Calculate risk scores ONCE for all servers
    # (instead of recalculating in each render function)
    server_preds = predictions.get('predictions', {})
    risk_scores = {}

    # === TIMING CHECKPOINT 1: Extract Pre-Calculated Risk Scores ===
    extract_start = datetime.now()

    # ARCHITECTURAL FIX: Daemon already calculated risk scores!
    # Dashboard job is to DISPLAY, not CALCULATE
    # Extract pre-calculated risk_score from daemon response
    num_servers = len(server_preds)

    for name, pred in server_preds.items():
        # Daemon provides risk_score in the prediction (Phase 3 optimization)
        # If not present (fallback), calculate it client-side
        if 'risk_score' in pred:
            risk_scores[name] = pred['risk_score']
        else:
            # Fallback: calculate client-side (shouldn't happen with Phase 3 daemon)
            print(f"[WARN] Server {name} missing risk_score - calculating client-side")
            risk_scores[name] = calculate_server_risk_score(pred)

    # Apply tab-specific filtering AFTER extraction
    if active_tab == "heatmap":
        # Heatmap only shows top 30
        top_30 = dict(sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:30])
        risk_scores = top_30
    elif active_tab == "risks":
        # Top risks only shows top 5
        top_5 = dict(sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:5])
        risk_scores = top_5

    extract_elapsed = (datetime.now() - extract_start).total_seconds() * 1000
    print(f"[PERF] Risk score extraction: {extract_elapsed:.0f}ms for {num_servers} servers (daemon pre-calculated!)")

    # === TIMING CHECKPOINT 2: Tab Rendering ===
    tab_start = datetime.now()

    # Render only the active tab (key optimization!)
    if active_tab == "overview":
        content = render_overview(predictions, risk_scores)
    elif active_tab == "heatmap":
        content = render_heatmap(predictions, risk_scores)
    elif active_tab == "risks":
        content = render_top_risks(predictions, risk_scores, server_preds)
    else:
        content = html.Div("Unknown tab")

    tab_elapsed = (datetime.now() - tab_start).total_seconds() * 1000
    print(f"[PERF] Tab rendering ({active_tab}): {tab_elapsed:.0f}ms")

    # Calculate actual render time
    elapsed = (datetime.now() - render_start).total_seconds() * 1000
    print(f"[PERF] TOTAL render time: {elapsed:.0f}ms (Target: <500ms)")

    perf_msg = dbc.Badge(
        f"⚡ Render time: {elapsed:.0f}ms (Target: <500ms)",
        color="success" if elapsed < 500 else "warning",
        className="mb-2"
    )

    return content, perf_msg


# =============================================================================
# TAB RENDERERS (Copied from Streamlit tabs with minimal changes)
# =============================================================================

def render_overview(predictions, risk_scores):
    """Render Overview tab - copied from Streamlit overview.py

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores (optimization!)
    """
    env = predictions.get('environment', {})
    server_preds = predictions.get('predictions', {})

    # Risk scores already calculated in callback - no need to recalculate!

    # KPI cards (same as Streamlit)
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

    # Risk distribution chart (SAME Plotly code as Streamlit!)
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

    # Bar chart (copy-paste from Streamlit!)
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

    # Pie chart (copy-paste from Streamlit!)
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

    return html.Div([kpis, charts, alerts_info])


def render_heatmap(predictions, risk_scores):
    """Render Heatmap tab - copied from Streamlit heatmap.py

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores (optimization!)
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


def render_top_risks(predictions, risk_scores, server_preds):
    """Render Top 5 Risks tab - copied from Streamlit top_risks.py

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores (optimization!)
        server_preds: Server predictions dict (for metrics extraction)
    """
    # Risk scores already calculated in callback - no need to recalculate!

    top_servers = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    if not top_servers or top_servers[0][1] == 0:
        return dbc.Alert("✅ No high-risk servers detected!", color="success")

    # Create gauge charts for top 5 (same as Streamlit!)
    gauges = []
    for i, (server_name, risk_score) in enumerate(top_servers, 1):
        server_pred = server_preds[server_name]

        # Risk gauge (copy from Streamlit)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{i}. {server_name}"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': get_risk_color(risk_score)},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "lightyellow"},
                    {'range': [80, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig_gauge.update_layout(height=250)

        # Current metrics
        current_cpu = extract_cpu_used(server_pred, 'current')
        current_mem = server_pred.get('mem_used_pct', {}).get('current', 0)
        current_iowait = server_pred.get('cpu_iowait_pct', {}).get('current', 0)

        metrics_card = dbc.Card([
            dbc.CardBody([
                html.H6(f"Rank #{i}: {server_name}", className="card-subtitle mb-2"),
                html.P([
                    html.Strong("CPU: "), f"{current_cpu:.1f}%", html.Br(),
                    html.Strong("Memory: "), f"{current_mem:.1f}%", html.Br(),
                    html.Strong("I/O Wait: "), f"{current_iowait:.1f}%", html.Br(),
                    html.Strong("Risk Score: "), html.Span(f"{risk_score:.0f}",
                        style={'color': get_risk_color(risk_score), 'font-weight': 'bold'})
                ])
            ])
        ])

        row = dbc.Row([
            dbc.Col([dcc.Graph(figure=fig_gauge)], width=6),
            dbc.Col([metrics_card], width=6),
        ], className="mb-3")

        gauges.append(row)

    return html.Div([
        html.H4("⚠️ Top 5 Problem Servers", className="mb-3"),
        *gauges
    ])


# =============================================================================
# RUN APP
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Plotly Dash Proof of Concept - ArgusAI Dashboard")
    print("="*70)
    print(f"\nStarting Dash app on http://localhost:8050")
    print(f"Daemon URL: {DAEMON_URL}")
    print(f"Auto-refresh: Every {REFRESH_INTERVAL/1000:.0f} seconds")
    print(f"\n⚡ Performance Target: <500ms page loads (vs 12s in Streamlit)")
    print("\nPress Ctrl+C to stop\n")
    print("="*70 + "\n")

    app.run(debug=True, port=8050, host='0.0.0.0')
