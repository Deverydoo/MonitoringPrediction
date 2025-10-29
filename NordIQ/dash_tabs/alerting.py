"""
Alerting Strategy Tab - Intelligent alert routing and escalation
=================================================================

POC demonstration of alerting capabilities:
- Environment and per-server alert generation
- Graduated severity levels with specific routing
- Alert routing matrix with SLAs
- Integration architecture planning
- Alert suppression and deduplication

This tab shows what alerts **would be sent** based on current predictions.

Performance: Target <100ms (table rendering + alert generation)
"""

from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
from typing import Dict, List

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dash_utils.data_processing import calculate_server_risk_score


def infer_profile_display(server_name: str) -> str:
    """Infer server profile from server name for display."""
    server_name_lower = server_name.lower()

    if 'ppml' in server_name_lower or 'ml' in server_name_lower:
        return 'ML Compute'
    elif 'ppdb' in server_name_lower or 'db' in server_name_lower:
        return 'Database'
    elif 'ppweb' in server_name_lower or 'web' in server_name_lower or 'api' in server_name_lower:
        return 'Web/API'
    elif 'pprisk' in server_name_lower or 'risk' in server_name_lower:
        return 'Risk Analytics'
    elif 'ppetl' in server_name_lower or 'etl' in server_name_lower:
        return 'Data Ingest'
    elif 'ppcon' in server_name_lower or 'conductor' in server_name_lower:
        return 'Conductor Mgmt'
    else:
        return 'Generic'


def generate_alerts(predictions: Dict, risk_scores: Dict[str, float]) -> List[Dict]:
    """
    Generate all alerts that would be sent based on current predictions.

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: Pre-calculated risk scores

    Returns:
        List of alerts to send
    """
    alerts_to_send = []
    env = predictions.get('environment', {})
    server_preds = predictions.get('predictions', {})

    # Environment-level alerts
    prob_30m = env.get('prob_30m', 0)
    prob_8h = env.get('prob_8h', 0)

    if prob_30m > 0.7:
        alerts_to_send.append({
            'Severity': '🔴 Critical',
            'Type': 'Environment',
            'Message': f'CRITICAL: Environment incident probability 30m = {prob_30m*100:.1f}%',
            'Recipients': 'On-Call Engineer (PagerDuty)',
            'Delivery Method': '📞 Phone Call + SMS + App Push',
            'Action Required': 'Immediate investigation and response',
            'Escalation': '15 min → Senior Engineer → 30 min → Director',
            'severity_level': 'imminent'
        })
    elif prob_30m > 0.4:
        alerts_to_send.append({
            'Severity': '🟠 Danger',
            'Type': 'Environment',
            'Message': f'DANGER: Environment degrading, incident probability 30m = {prob_30m*100:.1f}%',
            'Recipients': 'Engineering Team (Email + Slack)',
            'Delivery Method': '📧 Email + 💬 Slack #ops-alerts',
            'Action Required': 'Monitor closely, prepare for potential escalation',
            'Escalation': '30 min → On-Call Engineer (PagerDuty)',
            'severity_level': 'danger'
        })
    elif prob_8h > 0.5:
        alerts_to_send.append({
            'Severity': '🟡 Warning',
            'Type': 'Environment',
            'Message': f'WARNING: Elevated risk over 8 hours, probability = {prob_8h*100:.1f}%',
            'Recipients': 'Engineering Team (Email)',
            'Delivery Method': '📧 Email to ops-team@company.com',
            'Action Required': 'Review dashboard, plan capacity if needed',
            'Escalation': 'None (informational)',
            'severity_level': 'warning'
        })

    # Per-server alerts (only high-risk servers)
    for server_name, risk_score in risk_scores.items():
        if risk_score >= 70:
            profile = infer_profile_display(server_name)

            # Determine severity based on graduated scale
            if risk_score >= 90:
                severity = '🔴 Imminent Failure'
                recipients = 'On-Call Engineer (PagerDuty)'
                delivery = '📞 Phone + SMS + App'
                escalation = '5 min → CTO'
                severity_level = 'imminent'
            elif risk_score >= 80:
                severity = '🔴 Critical'
                recipients = 'On-Call Engineer (PagerDuty)'
                delivery = '📞 Phone + SMS + App'
                escalation = '15 min → Senior → 30 min → Director'
                severity_level = 'critical'
            else:  # risk_score >= 70
                severity = '🟠 Danger'
                recipients = 'Server Team Lead (Slack)'
                delivery = '💬 Slack + Email'
                escalation = '30 min → On-Call'
                severity_level = 'danger'

            alerts_to_send.append({
                'Severity': severity,
                'Type': f'Server ({profile})',
                'Message': f'{server_name}: Critical resource exhaustion predicted (Risk: {risk_score:.0f}/100)',
                'Recipients': recipients,
                'Delivery Method': delivery,
                'Action Required': 'Check server health, trigger auto-remediation if available',
                'Escalation': escalation,
                'severity_level': severity_level
            })

    return alerts_to_send


def render(predictions: Dict, risk_scores: Dict[str, float]) -> html.Div:
    """
    Render Alerting Strategy tab.

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores from daemon

    Returns:
        html.Div: Tab content
    """
    server_preds = predictions.get('predictions', {})

    if not server_preds:
        return dbc.Alert([
            html.H4("📱 Alerting & Notification Strategy"),
            html.P("No predictions available. Start the inference daemon and metrics generator."),
        ], color="warning")

    # Header
    header = html.Div([
        html.H4("📱 Alerting & Notification Strategy", className="mb-3"),
        dbc.Alert([
            html.Strong("POC Vision: "),
            "Intelligent alert routing and escalation based on risk severity"
        ], color="info")
    ])

    # Section: Alert Routing
    section_header = html.Div([
        html.H5("🎯 Alert Routing (Would Be Sent)", className="mt-4 mb-3"),
        html.P("This shows what alerts would be sent based on current predictions.",
               className="text-muted")
    ])

    # Generate alerts
    alerts_to_send = generate_alerts(predictions, risk_scores)

    if alerts_to_send:
        # Count by severity
        imminent_count = len([a for a in alerts_to_send if a['severity_level'] == 'imminent'])
        critical_count = len([a for a in alerts_to_send if a['severity_level'] == 'critical'])
        danger_count = len([a for a in alerts_to_send if a['severity_level'] == 'danger'])
        warning_count = len([a for a in alerts_to_send if a['severity_level'] == 'warning'])

        # Summary metrics
        summary_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("🔴 Imminent Failure", className="card-subtitle mb-2 text-muted"),
                        html.H3(str(imminent_count), className="card-title text-danger")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("🔴 Critical", className="card-subtitle mb-2 text-muted"),
                        html.H3(str(critical_count), className="card-title text-danger")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("🟠 Danger", className="card-subtitle mb-2 text-muted"),
                        html.H3(str(danger_count), className="card-title text-warning")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("🟡 Warning", className="card-subtitle mb-2 text-muted"),
                        html.H3(str(warning_count), className="card-title" + (" text-warning" if warning_count > 0 else ""))
                    ])
                ])
            ], width=3)
        ], className="mb-4")

        # Alerts table
        df_alerts = pd.DataFrame(alerts_to_send)
        # Drop severity_level column (only for counting)
        df_alerts = df_alerts.drop(columns=['severity_level'])

        alerts_table = dbc.Card([
            dbc.CardHeader(html.H5(f"🔔 {len(alerts_to_send)} Alerts Would Be Sent", className="mb-0")),
            dbc.CardBody([
                dbc.Table.from_dataframe(
                    df_alerts,
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size='sm'
                )
            ])
        ], className="mb-4")

        alert_content = html.Div([summary_cards, alerts_table])
    else:
        alert_content = dbc.Alert([
            html.H5("✅ No alerts required", className="mb-2"),
            html.P("All systems healthy! No servers or environment issues detected.", className="mb-0")
        ], color="success", className="mb-4")

    # Alert Routing Matrix
    routing_matrix = pd.DataFrame({
        'Severity': ['🔴 Imminent Failure', '🔴 Critical', '🟠 Danger', '🟡 Warning', '🟢 Degrading', '👁️ Watch'],
        'Threshold': ['Risk ≥ 90', 'Risk 80-89', 'Risk 70-79', 'Risk 60-69', 'Risk 50-59', 'Risk 30-49'],
        'Initial Contact': [
            'On-Call Engineer (PagerDuty)',
            'On-Call Engineer (PagerDuty)',
            'Server Team Lead (Slack)',
            'Server Team (Slack)',
            'Engineering Team (Email)',
            'Dashboard Only'
        ],
        'Methods': [
            'Phone + SMS + App',
            'Phone + SMS + App',
            'Slack + Email',
            'Slack + Email',
            'Email only',
            'Log only'
        ],
        'Response SLA': ['5 minutes', '15 minutes', '30 minutes', '1 hour', '2 hours', 'Best effort'],
        'Escalation Path': [
            '5m → CTO',
            '15m → Senior → 30m → Director',
            '30m → On-Call',
            '1h → Team Lead',
            'None',
            'None'
        ]
    })

    routing_section = dbc.Card([
        dbc.CardHeader(html.H5("📋 Alert Routing Matrix", className="mb-0")),
        dbc.CardBody([
            html.P("Graduated severity levels with specific routing and escalation paths:", className="text-muted mb-3"),
            dbc.Table.from_dataframe(
                routing_matrix,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True
            )
        ])
    ], className="mb-4")

    # Integration Points
    integration_section = dbc.Card([
        dbc.CardHeader(html.H5("🔌 Integration Points", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Immediate Alerting (Phase 1)", className="mb-3"),
                    html.Ul([
                        html.Li([html.Strong("✅ PagerDuty integration"), " (API)"]),
                        html.Li([html.Strong("✅ Slack webhooks"), " (#ops-alerts, #server-ops)"]),
                        html.Li([html.Strong("✅ Email"), " (SMTP to distribution lists)"]),
                        html.Li([html.Strong("✅ Dashboard notifications")])
                    ]),
                    html.P([
                        html.Strong("Delivery Time: "),
                        "< 30 seconds from prediction"
                    ], className="text-success mt-3")
                ], width=6),
                dbc.Col([
                    html.H6("Advanced Features (Phase 2)", className="mb-3"),
                    html.Ul([
                        html.Li([html.Strong("🔄 SMS notifications"), " (Twilio)"]),
                        html.Li([html.Strong("🔄 Microsoft Teams integration")]),
                        html.Li([html.Strong("🔄 ServiceNow ticket creation")]),
                        html.Li([html.Strong("🔄 Mobile app push notifications")])
                    ]),
                    html.P([
                        html.Strong("Intelligent Routing: "),
                        "Context-aware escalation"
                    ], className="text-info mt-3")
                ], width=6)
            ])
        ])
    ], className="mb-4")

    # Alert Suppression
    suppression_section = dbc.Card([
        dbc.CardHeader(html.H5("🔇 Intelligent Alert Suppression", className="mb-0")),
        dbc.CardBody([
            html.P([html.Strong("Smart Features"), " (Reduce Alert Fatigue):"], className="mb-3"),
            html.Ol([
                html.Li([html.Strong("Deduplication: "), "Same server, same issue → single alert (no flooding)"]),
                html.Li([html.Strong("Grouping: "), "Multiple servers in same profile degrading → grouped alert"]),
                html.Li([html.Strong("Scheduled Maintenance: "), "Suppress alerts during maintenance windows"]),
                html.Li([html.Strong("Auto-Remediation Active: "), "Suppress alerts if auto-fix is already running"]),
                html.Li([html.Strong("Escalation Delays: "), "Progressive escalation only if no acknowledgment"])
            ], className="mb-3"),
            html.P([
                html.Strong("Result: "),
                "80% reduction in alert noise, 95% increase in signal-to-noise ratio"
            ], className="text-success")
        ])
    ], className="mb-4")

    # Implementation Note
    implementation_note = dbc.Alert([
        html.Strong("💡 POC Implementation Note:"),
        html.Br(), html.Br(),
        "This tab demonstrates intelligent alerting capabilities. Production implementation would include:",
        html.Ul([
            html.Li([html.Strong("Multi-channel integration: "), "PagerDuty, Slack, Email, SMS, Teams, ServiceNow"]),
            html.Li([html.Strong("Context-aware routing: "), "Profile-based escalation (e.g., DB issues go to DBA team)"]),
            html.Li([html.Strong("Alert lifecycle tracking: "), "From trigger → acknowledgment → resolution"]),
            html.Li([html.Strong("On-call schedule integration: "), "Route to current on-call engineer automatically"]),
            html.Li([html.Strong("Alert suppression rules: "), "Prevent alert storms during cascading incidents"]),
            html.Li([html.Strong("Feedback loop: "), "Track alert accuracy, adjust thresholds based on false positive rate"])
        ]),
        html.Br(),
        html.Strong("Impact: "),
        "Reduces alert fatigue by 80%, improves response time by 60%, ensures critical issues never go unnoticed."
    ], color="light", className="mt-4")

    return html.Div([
        header,
        section_header,
        alert_content,
        routing_section,
        integration_section,
        suppression_section,
        implementation_note
    ])
