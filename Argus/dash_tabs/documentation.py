"""
Documentation Tab - Complete user guide for the Dash Dashboard
===============================================================

Comprehensive documentation covering:
- Overview and features
- Understanding risk scores
- Alert priority levels
- Contextual intelligence philosophy
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
            "Understanding and using the ArgusAI Monitoring Dashboard"
        ], color="info")
    ])

    # Create comprehensive documentation in Bootstrap accordions
    documentation_content = dbc.Accordion([

        # Section 1: Overview & Features
        dbc.AccordionItem([
            html.H5("Key Capabilities:", className="mb-3"),
            html.Ul([
                html.Li([html.Strong("Real-time Monitoring: "), "Live metrics from 20+ servers across 7 profiles"]),
                html.Li([html.Strong("30-Minute Predictions: "), "AI forecasts CPU, Memory, Latency with 85-90% accuracy"]),
                html.Li([html.Strong("8-Hour Horizon: "), "Extended forecasts for capacity planning"]),
                html.Li([html.Strong("Contextual Intelligence: "), "Risk scoring considers server profiles, trends, and multi-metric correlations"]),
                html.Li([html.Strong("Graduated Alerts: "), "7 severity levels from Healthy to Imminent Failure"]),
                html.Li([html.Strong("Early Warning: "), "15-60 minute advance notice before problems become critical"])
            ], className="mb-3"),
            html.H6("Technology Stack:", className="mb-2"),
            html.Ul([
                html.Li("Model: PyTorch Forecasting Temporal Fusion Transformer (TFT)"),
                html.Li("Architecture: Microservices with REST APIs"),
                html.Li("Dashboard: Plotly Dash with real-time updates"),
                html.Li("Training: Transfer learning with profile-specific fine-tuning")
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

        # Section 5: Server Profiles
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

        # Section 6: How to Interpret Alerts
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

        # Section 7: Best Practices
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
                        html.Li("✅ Trust profile-specific thresholds (DB at 100% mem = OK)")
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
                        html.Li("❌ Don't ignore improving trends - verify remediation worked")
                    ])
                ], width=6)
            ])
        ], title="✅ Best Practices"),

        # Section 8: Quick Reference
        dbc.AccordionItem([
            html.Pre("""
╔════════════════════════════════════════════════════════════════╗
║         NORDIQ AI MONITORING - QUICK REFERENCE                 ║
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
║ DELTA INTERPRETATION:                                         ║
║   Positive (+) → Metrics increasing (degrading)               ║
║   Negative (-) → Metrics decreasing (improving)               ║
║                                                                ║
║ PROFILE-SPECIFIC THRESHOLDS:                                 ║
║   Database: 100% memory = NORMAL (page cache)                ║
║   ML Compute: 98% memory = CRITICAL (OOM risk)               ║
║   Web API: Latency > 200ms = SEVERE (user impact)           ║
║                                                                ║
║ RESPONSE PRIORITY:                                            ║
║   1. Imminent Failure → Drop everything                       ║
║   2. Critical → Immediate action                              ║
║   3. Danger → Urgent response                                 ║
║   4. Warning → Monitor closely                                ║
║   5. Degrading → Investigate soon                             ║
╚════════════════════════════════════════════════════════════════╝
            """, className="bg-light p-3 rounded", style={'fontSize': '12px'})
        ], title="🚀 Quick Reference Card"),

    ], start_collapsed=False, always_open=False)

    # Footer
    footer = dbc.Alert([
        html.Strong("📚 Documentation Complete!"),
        html.Br(), html.Br(),
        "This guide covers the core concepts and operational procedures for the ArgusAI Monitoring Dashboard. ",
        "For system diagnostics, see the Advanced tab. For future enhancements, see the Roadmap tab."
    ], color="success", className="mt-4")

    return html.Div([
        header,
        documentation_content,
        footer
    ])
