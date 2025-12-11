"""
Documentation Tab - Complete user guide for the Dash Dashboard
===============================================================

Comprehensive documentation covering:
- Overview and features (v2.1 with cascade detection and drift monitoring)
- Understanding risk scores
- Alert priority levels
- Contextual intelligence philosophy
- Cascading failure detection
- Model drift monitoring
- Server profiles
- How to interpret alerts
- Best practices
- Quick reference card

Performance: Target <100ms (static content with Bootstrap accordions)
"""

from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
from typing import Dict


def render(predictions: Dict, risk_scores: Dict[str, float]) -> html.Div:
    """
    Render Documentation tab.

    Args:
        predictions: Full predictions dict from daemon (unused, static content)
        risk_scores: PRE-CALCULATED risk scores from daemon (unused, static content)

    Returns:
        html.Div: Tab content
    """
    # Header
    header = html.Div([
        html.H4("📚 Dashboard Documentation", className="mb-3"),
        dbc.Alert([
            html.Strong("Complete Guide: "),
            "Understanding and using the Tachyon Argus Monitoring Dashboard (v2.1)"
        ], color="info"),
        dbc.Alert([
            html.Strong("New in v2.1: "),
            "Cascading failure detection, model drift monitoring, multi-target predictions, and continuous learning!"
        ], color="success")
    ])

    # Create comprehensive documentation in Bootstrap accordions
    documentation_content = dbc.Accordion([

        # Section 1: Overview & Features
        dbc.AccordionItem([
            html.H5("Key Capabilities:", className="mb-3"),
            html.Ul([
                html.Li([html.Strong("Real-time Monitoring: "), "Live metrics from 20+ servers across 7 profiles"]),
                html.Li([html.Strong("Multi-Target Predictions: "), "AI forecasts CPU, Memory, Swap, I/O Wait, and Load Average"]),
                html.Li([html.Strong("8-Hour Horizon: "), "Extended forecasts for capacity planning"]),
                html.Li([html.Strong("Cascading Failure Detection: "), "Fleet-wide correlation analysis to detect spreading issues"]),
                html.Li([html.Strong("Model Drift Monitoring: "), "Automatic detection of model degradation with auto-retraining"]),
                html.Li([html.Strong("Contextual Intelligence: "), "Risk scoring considers server profiles, trends, and multi-metric correlations"]),
                html.Li([html.Strong("Graduated Alerts: "), "7 severity levels from Healthy to Imminent Failure"]),
                html.Li([html.Strong("Early Warning: "), "15-60 minute advance notice before problems become critical"]),
                html.Li([html.Strong("Explainable AI: "), "SHAP values and attention weights explain every prediction"])
            ], className="mb-3"),
            html.H6("Technology Stack:", className="mb-2"),
            html.Ul([
                html.Li("Model: PyTorch Forecasting Temporal Fusion Transformer (TFT)"),
                html.Li("Architecture: Microservices with REST APIs"),
                html.Li("Dashboard: Plotly Dash with real-time updates"),
                html.Li("Training: Streaming training with checkpoints and hot model reload"),
                html.Li("Cascade Detection: 18 fleet-level features for correlation analysis"),
                html.Li("Drift Monitoring: 4 drift metrics (PER, DSS, FDS, Anomaly Rate)")
            ])
        ], title="🎯 Overview & Features"),

        # Section 2: Understanding Risk Scores
        dbc.AccordionItem([
            html.P("Every server receives a Risk Score (0-100) representing overall health and predicted trajectory.", className="mb-3"),
            html.H6("Score Composition:", className="mb-2"),
            html.Pre("Final Risk = (Current State × 70%) + (Predictions × 30%)", className="bg-light p-3 rounded mb-3"),
            html.H6("Why 70/30 Weighting?", className="mb-2"),
            html.Ul([
                html.Li("70% Current State: Executives care about 'what's on fire NOW'"),
                html.Li("30% Predictions: Early warning value without crying wolf")
            ], className="mb-3"),
            html.H6("Risk Components:", className="mb-2"),
            html.Ul([
                html.Li("CPU Usage: Current and predicted utilization"),
                html.Li("Memory Usage: Current and predicted with profile-specific thresholds"),
                html.Li("Latency: Response time degradation"),
                html.Li("Disk Usage: Available space warnings"),
                html.Li("Trend Velocity: Rate of change (climbing vs. steady)"),
                html.Li("Multi-Metric Correlation: Compound stress detection")
            ])
        ], title="📊 Understanding Risk Scores"),

        # Section 3: Alert Priority Levels
        dbc.AccordionItem([
            html.P("The dashboard uses 7 graduated severity levels instead of binary OK/CRITICAL alerts.", className="mb-3"),
            dbc.Table.from_dataframe(
                pd.DataFrame({
                    'Level': ['🔴 Imminent Failure', '🔴 Critical', '🟠 Danger', '🟡 Warning', '🟢 Degrading', '👁️ Watch', '✅ Healthy'],
                    'Risk Score': ['90-100', '80-89', '70-79', '60-69', '50-59', '30-49', '0-29'],
                    'Meaning': [
                        'Server about to crash or failing NOW',
                        'Severe issues requiring immediate attention',
                        'High-priority problems requiring urgent action',
                        'Concerning trends that need monitoring',
                        'Performance declining, investigate soon',
                        'Low concern, background monitoring only',
                        'Normal operations, no concerns'
                    ],
                    'SLA': ['5 minutes', '15 minutes', '30 minutes', '1 hour', '2 hours', 'Best effort', 'N/A']
                }),
                striped=True,
                bordered=True,
                hover=True
            ),
            html.P([
                html.Strong("Key Insight: "),
                "Notice the graduated escalation. You don't go from 'Healthy' to 'Critical' - instead you progress through Watch → Degrading → Warning → Danger, giving teams time to respond proactively."
            ], className="mt-3")
        ], title="🚨 Alert Priority Levels"),

        # Section 4: Contextual Intelligence
        dbc.AccordionItem([
            html.P([html.Strong("Philosophy: "), "'40% CPU may be fine, or may be degrading - depends on context'"], className="mb-3"),
            dbc.Alert([
                html.H6("Traditional Monitoring Problems:", className="mb-2"),
                html.Ul([
                    html.Li("No context: 80% CPU on database = normal, 80% on web server = problem"),
                    html.Li("No trends: 80% steady = fine, 40% → 80% in 10 min = concerning"),
                    html.Li("No prediction: Server at 60% but climbing fast will crash soon"),
                    html.Li("Binary state: Everything is either OK or ON FIRE (no middle ground)"),
                    html.Li("Ignores correlations: High CPU + high memory + high latency = compound risk")
                ])
            ], color="warning", className="mb-3"),
            html.H6("Our Approach: Four Context Factors", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("1. Server Profile", className="mb-2"),
                    html.P([
                        html.Strong("Database (ppdb001): "),
                        "Memory 98% = ✅ Healthy (page cache is normal)"
                    ], className="mb-2"),
                    html.P([
                        html.Strong("ML Compute (ppml0001): "),
                        "Memory 98% = 🔴 Critical (OOM kill imminent)"
                    ])
                ], width=6),
                dbc.Col([
                    html.H6("2. Trend Analysis", className="mb-2"),
                    html.P([
                        html.Strong("Steady: "),
                        "40% CPU for 30 min = Risk 0 (stable)"
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Rapid Climb: "),
                        "20% → 40% → 60% = Risk 56 (will hit 100% soon!)"
                    ])
                ], width=6)
            ], className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("3. Multi-Metric Correlation", className="mb-2"),
                    html.P([
                        html.Strong("Isolated spike: "),
                        "CPU 85%, Mem 35%, Latency 40ms = Risk 28 (just a batch job)"
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Compound stress: "),
                        "CPU 85%, Mem 90%, Latency 350ms = Risk 83 (system under stress)"
                    ])
                ], width=6),
                dbc.Col([
                    html.H6("4. Prediction-Aware", className="mb-2"),
                    html.P([
                        html.Strong("Looks fine now: "),
                        "Current 40%, Predicted 95% = Risk 52 (early warning!)"
                    ], className="mb-2"),
                    html.P([
                        html.Strong("Bad now, improving: "),
                        "Current 85%, Predicted 60% = Risk 38 (resolving itself)"
                    ])
                ], width=6)
            ])
        ], title="🧠 Contextual Intelligence"),

        # Section 5: Cascading Failure Detection
        dbc.AccordionItem([
            html.P([
                html.Strong("Purpose: "),
                "Detect when problems are spreading across multiple servers - a sign of infrastructure-wide issues."
            ], className="mb-3"),
            dbc.Alert([
                html.H6("Why Cascade Detection Matters:", className="mb-2"),
                html.P("Traditional monitoring treats each server independently. But real-world failures often cascade: "
                       "a database slowdown causes web servers to queue up, which causes load balancers to timeout, "
                       "which causes user-facing errors. By the time individual alerts fire, the cascade is in full swing.")
            ], color="warning", className="mb-3"),
            html.H6("Key Metrics:", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.H6("Fleet Health Score (0-100)", className="mb-2"),
                    html.Ul([
                        html.Li("80-100: Healthy - servers operating independently"),
                        html.Li("60-79: Degraded - some correlation detected"),
                        html.Li("40-59: Warning - significant cross-server correlation"),
                        html.Li("0-39: Critical - active cascade in progress")
                    ])
                ], width=6),
                dbc.Col([
                    html.H6("Cascade Risk Levels", className="mb-2"),
                    html.Ul([
                        html.Li([dbc.Badge("LOW", color="success", className="me-2"), "Normal operations"]),
                        html.Li([dbc.Badge("MEDIUM", color="warning", className="me-2"), "Monitor closely"]),
                        html.Li([dbc.Badge("HIGH", color="danger", className="me-2"), "Immediate investigation"])
                    ])
                ], width=6)
            ], className="mb-3"),
            html.H6("Detection Mechanisms:", className="mb-2"),
            html.Ul([
                html.Li([html.Strong("Cross-Server Correlation: "), "Tracks how synchronized metric changes are across servers"]),
                html.Li([html.Strong("Anomaly Rate: "), "Percentage of servers showing anomalous behavior simultaneously"]),
                html.Li([html.Strong("18 Fleet-Level Features: "), "Aggregated metrics used to detect fleet-wide patterns"]),
                html.Li([html.Strong("Cascade Event Timeline: "), "Historical view of detected cascade events"])
            ]),
            dbc.Alert([
                html.Strong("Dashboard Tab: "),
                "View the Cascade Detection tab for real-time fleet health, correlation gauges, and cascade event history."
            ], color="info", className="mt-3")
        ], title="🔗 Cascading Failure Detection"),

        # Section 6: Model Drift Monitoring
        dbc.AccordionItem([
            html.P([
                html.Strong("Purpose: "),
                "Automatically detect when the ML model's accuracy degrades and trigger retraining."
            ], className="mb-3"),
            dbc.Alert([
                html.H6("Why Drift Monitoring Matters:", className="mb-2"),
                html.P("ML models trained on historical data can become stale as infrastructure evolves. "
                       "New servers, workload changes, or seasonal patterns can cause 'concept drift' - "
                       "where the model's predictions no longer match reality. Drift monitoring catches this automatically.")
            ], color="warning", className="mb-3"),
            html.H6("Four Drift Metrics:", className="mb-3"),
            dbc.Table.from_dataframe(
                pd.DataFrame({
                    'Metric': ['PER', 'DSS', 'FDS', 'Anomaly Rate'],
                    'Full Name': ['Prediction Error Rate', 'Distribution Shift Score', 'Feature Drift Score', 'Anomaly Rate'],
                    'What It Measures': [
                        'Rolling average of prediction errors',
                        'How much input distributions changed from training',
                        'Drift in individual features',
                        'Percentage of anomalous predictions'
                    ],
                    'Threshold': ['10%', '20%', '15%', '5%']
                }),
                striped=True,
                bordered=True,
                hover=True
            ),
            html.H6("Auto-Retraining:", className="mb-3 mt-3"),
            html.Ul([
                html.Li([html.Strong("Drift Detection: "), "When any metric exceeds threshold, drift is flagged"]),
                html.Li([html.Strong("Auto-Retrain Trigger: "), "System automatically initiates incremental retraining"]),
                html.Li([html.Strong("24-Hour Cooldown: "), "Prevents over-retraining from transient spikes"]),
                html.Li([html.Strong("Hot Model Reload: "), "New model loaded without daemon restart"])
            ]),
            dbc.Alert([
                html.Strong("Dashboard Tab: "),
                "View the Model Drift tab for drift metric gauges, feature-level analysis, and retraining status."
            ], color="info", className="mt-3")
        ], title="📉 Model Drift Monitoring"),

        # Section 7: Multi-Target Predictions
        dbc.AccordionItem([
            html.P([
                html.Strong("Purpose: "),
                "Predict multiple system metrics simultaneously for comprehensive forecasting."
            ], className="mb-3"),
            html.H6("Predicted Metrics:", className="mb-2"),
            dbc.Table.from_dataframe(
                pd.DataFrame({
                    'Metric': ['CPU User %', 'CPU I/O Wait %', 'Memory Utilization', 'Swap Utilization', 'System Load Average'],
                    'Description': [
                        'User-space CPU usage percentage',
                        'Time spent waiting for I/O operations',
                        'RAM utilization percentage',
                        'Swap space usage (indicates memory pressure)',
                        'System load average (1-minute)'
                    ],
                    'Warning Threshold': ['85%', '10%', '90%', '20%', '0.8 * cores']
                }),
                striped=True,
                bordered=True,
                hover=True
            ),
            html.H6("Benefits of Multi-Target:", className="mb-3 mt-3"),
            html.Ul([
                html.Li([html.Strong("Correlated Predictions: "), "Model learns relationships between metrics"]),
                html.Li([html.Strong("Compound Risk Detection: "), "High CPU + High Memory + High I/O = higher risk than any single metric"]),
                html.Li([html.Strong("Resource Planning: "), "Predict all resources simultaneously for capacity planning"]),
                html.Li([html.Strong("Reduced False Positives: "), "Context from multiple metrics improves accuracy"])
            ])
        ], title="📈 Multi-Target Predictions"),

        # Section 8: Server Profiles
        dbc.AccordionItem([
            html.P("The system automatically detects server profiles and applies profile-specific intelligence.", className="mb-3"),
            dbc.Table.from_dataframe(
                pd.DataFrame({
                    'Profile': ['ML Compute', 'Database', 'Web API', 'Conductor Mgmt', 'Data Ingest', 'Risk Analytics', 'Generic'],
                    'Hostname': ['ppml####', 'ppdb###', 'ppweb###', 'ppcon##', 'ppdi###', 'ppra###', 'ppsrv###'],
                    'Characteristics': [
                        'High CPU/Mem during training',
                        'High memory (page cache)',
                        'Low memory (stateless), Latency-sensitive',
                        'Low CPU/Mem, Management workload',
                        'High disk I/O, Network-intensive',
                        'CPU-intensive analytics',
                        'Balanced workload'
                    ],
                    'Memory Threshold': ['98%', '100%', '85%', '90%', '90%', '95%', '90%']
                }),
                striped=True,
                bordered=True,
                hover=True
            ),
            dbc.Alert([
                html.Strong("Why Profile Awareness Matters: "),
                "A database at 100% memory is healthy (caching), but a web server at 100% memory is about to crash (memory leak). The system adjusts thresholds based on expected behavior patterns."
            ], color="info", className="mt-3")
        ], title="🖥️ Server Profiles"),

        # Section 9: How to Interpret Alerts
        dbc.AccordionItem([
            html.H6("Priority Triage Strategy:", className="mb-3"),
            html.Ol([
                html.Li([html.Strong("Critical+ Servers First: "), "Risk 90+ (Imminent) or 80-89 (Critical) - drop everything"]),
                html.Li([html.Strong("Review Danger/Warning: "), "Risk 70-79 (Danger) or 60-69 (Warning) - team lead investigates"]),
                html.Li([html.Strong("Track Degrading: "), "Risk 50-59 (Degrading) - early warnings, investigate during business hours"]),
                html.Li([html.Strong("Look for Patterns: "), "Multiple servers with same profile degrading? Shared infrastructure issue"])
            ], className="mb-4"),
            html.H6("Understanding Delta (Δ) Values:", className="mb-2"),
            dbc.Alert([
                html.P("Delta values show predicted CHANGE, not absolute values:", className="mb-2"),
                html.Ul([
                    html.Li("CPU Δ: +15.2% → CPU will increase by 15.2% in next 30 minutes"),
                    html.Li("Mem Δ: -5.3% → Memory will decrease by 5.3% (improving)"),
                    html.Li("I/O Wait Δ: +5.1% → I/O wait will increase by 5.1% (degrading)")
                ], className="mb-2"),
                html.P([
                    html.Strong("🚨 Red Flag: "),
                    "All deltas positive (+) = server degrading across all metrics"
                ])
            ], color="light")
        ], title="🔔 How to Interpret Alerts"),

        # Section 10: Best Practices
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([
                    html.H6("👍 Do's", className="mb-3"),
                    html.Ul([
                        html.Li("✅ Check dashboard every 15-30 minutes during business hours"),
                        html.Li("✅ Trust the risk scores - they include context you might miss"),
                        html.Li("✅ Act on Degrading alerts proactively before they become Critical"),
                        html.Li("✅ Look for patterns across multiple servers"),
                        html.Li("✅ Use predictions to plan maintenance windows"),
                        html.Li("✅ Correlate with deployments - did we just push code?"),
                        html.Li("✅ Review Watch servers periodically (Risk 30-49)"),
                        html.Li("✅ Trust profile-specific thresholds (DB at 100% mem = OK)"),
                        html.Li("✅ Check Cascade Detection tab when multiple servers alert simultaneously"),
                        html.Li("✅ Monitor Model Drift tab weekly for model health")
                    ])
                ], width=6),
                dbc.Col([
                    html.H6("👎 Don'ts", className="mb-3"),
                    html.Ul([
                        html.Li("❌ Don't ignore Degrading alerts thinking 'it's only 55% CPU'"),
                        html.Li("❌ Don't panic at single metric spike - look at overall risk score"),
                        html.Li("❌ Don't override profile thresholds without understanding context"),
                        html.Li("❌ Don't dismiss predictions as 'just guesses'"),
                        html.Li("❌ Don't create manual alerts that duplicate dashboard intelligence"),
                        html.Li("❌ Don't compare this to traditional monitoring - it's predictive"),
                        html.Li("❌ Don't ignore improving trends - verify remediation worked"),
                        html.Li("❌ Don't ignore cascade alerts - they indicate fleet-wide issues"),
                        html.Li("❌ Don't disable auto-retrain without understanding drift implications")
                    ])
                ], width=6)
            ])
        ], title="✅ Best Practices"),

        # Section 11: Quick Reference
        dbc.AccordionItem([
            html.Pre("""
╔════════════════════════════════════════════════════════════════╗
║         TACHYON ARGUS v2.1 - QUICK REFERENCE                   ║
╠════════════════════════════════════════════════════════════════╣
║ RISK SCORE FORMULA:                                           ║
║   Final Risk = (Current State × 70%) + (Predictions × 30%)   ║
║                                                                ║
║ PRIORITY LEVELS:                                              ║
║   🔴 Imminent Failure (90+)  → 5-min SLA, CTO escalation     ║
║   🔴 Critical (80-89)        → 15-min SLA, page on-call      ║
║   🟠 Danger (70-79)          → 30-min SLA, team lead         ║
║   🟡 Warning (60-69)         → 1-hour SLA, team awareness    ║
║   🟢 Degrading (50-59)       → 2-hour SLA, email only        ║
║   👁️ Watch (30-49)           → Background monitoring         ║
║   ✅ Healthy (0-29)          → No alerts                      ║
║                                                                ║
║ CASCADE DETECTION:                                            ║
║   Fleet Health 80+ = Healthy    Cascade Risk LOW = Normal    ║
║   Fleet Health 60-79 = Degraded Cascade Risk MEDIUM = Watch  ║
║   Fleet Health <60 = Critical   Cascade Risk HIGH = Action   ║
║                                                                ║
║ DRIFT MONITORING:                                             ║
║   PER > 10%         → Model accuracy degraded                ║
║   DSS > 20%         → Input distribution shifted             ║
║   FDS > 15%         → Feature drift detected                 ║
║   Anomaly Rate > 5% → Too many anomalous predictions         ║
║   → Auto-retrain triggers with 24h cooldown                  ║
║                                                                ║
║ MULTI-TARGET PREDICTIONS:                                     ║
║   CPU User, I/O Wait, Memory, Swap, Load Average             ║
║                                                                ║
║ PROFILE-SPECIFIC THRESHOLDS:                                 ║
║   Database: 100% memory = NORMAL (page cache)                ║
║   ML Compute: 98% memory = CRITICAL (OOM risk)               ║
║   Web API: Latency > 200ms = SEVERE (user impact)           ║
║                                                                ║
║ DASHBOARD TABS:                                               ║
║   Overview → Fleet health, KPIs, risk distribution           ║
║   Cascade Detection → Fleet correlation, cascade events      ║
║   Model Drift → Drift metrics, auto-retrain status           ║
║   Top Risks → Highest risk servers with details              ║
║   Heatmap → Visual grid of server health                     ║
╚════════════════════════════════════════════════════════════════╝
            """, className="bg-light p-3 rounded", style={'fontSize': '11px'})
        ], title="🚀 Quick Reference Card"),

    ], start_collapsed=False, always_open=False)

    # Footer
    footer = dbc.Alert([
        html.Strong("📚 Documentation Complete!"),
        html.Br(), html.Br(),
        "This guide covers the core concepts and operational procedures for the Tachyon Argus Monitoring Dashboard v2.1. ",
        "Key features include cascading failure detection, model drift monitoring, multi-target predictions, and continuous learning. ",
        html.Br(), html.Br(),
        "For future enhancements, see the Roadmap tab. For technical documentation, see the Docs/ folder."
    ], color="success", className="mt-4")

    return html.Div([
        header,
        documentation_content,
        footer
    ])
