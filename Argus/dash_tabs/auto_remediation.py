"""
Auto-Remediation Tab - Autonomous incident prevention strategy
================================================================

POC demonstration of auto-remediation capabilities:
- Profile-specific remediation actions
- Integration architecture planning
- Approval workflow design
- Rollback strategies
- Real-time remediation plan generation based on risk scores

This tab shows what auto-remediation actions **would be triggered** in production.

Performance: Target <100ms (table rendering + calculations)
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from typing import Dict, List

# Import utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dash_utils.data_processing import calculate_server_risk_score


def infer_profile_from_server_name(server_name: str) -> str:
    """
    Infer server profile from server name.

    In production, this would come from CMDB or daemon metadata.
    """
    server_name_lower = server_name.lower()

    if 'ppml' in server_name_lower or 'ml' in server_name_lower:
        return 'ml_compute'
    elif 'ppdb' in server_name_lower or 'db' in server_name_lower:
        return 'database'
    elif 'ppweb' in server_name_lower or 'web' in server_name_lower or 'api' in server_name_lower:
        return 'web_api'
    elif 'pprisk' in server_name_lower or 'risk' in server_name_lower:
        return 'risk_analytics'
    elif 'ppetl' in server_name_lower or 'etl' in server_name_lower:
        return 'data_ingest'
    elif 'ppcon' in server_name_lower or 'conductor' in server_name_lower:
        return 'conductor_mgmt'
    else:
        return 'generic'


def get_remediation_action(profile: str, predicted_cpu: float) -> Dict[str, str]:
    """
    Determine remediation action based on server profile and predicted metrics.

    Returns:
        Dict with action, integration, and eta
    """
    if profile == 'ml_compute':
        return {
            'action': '🔧 Scale up compute resources (+2 vCPUs)',
            'integration': 'Spectrum Conductor API: POST /resources/scale',
            'eta': '2 minutes',
            'type': 'autonomous'
        }
    elif profile == 'database':
        return {
            'action': '💾 Enable connection pooling, scale read replicas',
            'integration': 'Database Management API',
            'eta': '5 minutes',
            'type': 'autonomous'
        }
    elif profile == 'web_api':
        return {
            'action': '🌐 Scale out (+2 instances), enable rate limiting',
            'integration': 'Load Balancer API + Kubernetes HPA',
            'eta': '3 minutes',
            'type': 'autonomous'
        }
    elif profile == 'risk_analytics':
        return {
            'action': '📊 Queue batch jobs, scale compute resources',
            'integration': 'Job Scheduler API',
            'eta': '4 minutes',
            'type': 'autonomous'
        }
    else:
        return {
            'action': '⚙️ Alert on-call team for manual review',
            'integration': 'PagerDuty API',
            'eta': 'Immediate',
            'type': 'manual'
        }


def generate_remediation_plan(server_preds: Dict, risk_scores: Dict[str, float]) -> List[Dict]:
    """
    Generate remediation plan for all high-risk servers.

    Args:
        server_preds: Server predictions from daemon
        risk_scores: Pre-calculated risk scores

    Returns:
        List of remediation actions
    """
    remediation_plan = []

    for server_name, risk_score in risk_scores.items():
        # Only trigger remediation for high-risk servers (≥70)
        if risk_score >= 70:
            server_pred = server_preds.get(server_name, {})

            # Infer profile
            profile = server_pred.get('profile', infer_profile_from_server_name(server_name))

            # Get predicted CPU (p90)
            cpu = server_pred.get('cpu_percent', {})
            p90_cpu = cpu.get('p90', [])
            max_predicted_cpu = max(p90_cpu[:6]) if p90_cpu and len(p90_cpu) >= 6 else 0

            # Get remediation action
            remediation = get_remediation_action(profile, max_predicted_cpu)

            remediation_plan.append({
                'Server': server_name,
                'Profile': profile.replace('_', ' ').title(),
                'Risk Score': f"{risk_score:.0f}",
                'Predicted CPU (p90)': f"{max_predicted_cpu:.1f}%",
                'Auto-Remediation': remediation['action'],
                'Integration Point': remediation['integration'],
                'ETA to Remediate': remediation['eta'],
                'Status': '🔴 Would Trigger Now',
                'type': remediation['type']  # For counting autonomous vs manual
            })

    return remediation_plan


def render(predictions: Dict, risk_scores: Dict[str, float]) -> html.Div:
    """
    Render Auto-Remediation tab.

    Args:
        predictions: Full predictions dict from daemon
        risk_scores: PRE-CALCULATED risk scores from daemon

    Returns:
        html.Div: Tab content
    """
    server_preds = predictions.get('predictions', {})

    if not server_preds:
        return dbc.Alert([
            html.H4("🤖 Auto-Remediation Strategy"),
            html.P("No predictions available. Start the inference daemon and metrics generator."),
        ], color="warning")

    # Header
    header = html.Div([
        html.H4("🤖 Auto-Remediation Strategy", className="mb-3"),
        dbc.Alert([
            html.Strong("POC Vision: "),
            "Autonomous incident prevention through profile-specific remediation actions"
        ], color="info")
    ])

    # Section: Remediation Actions
    section_header = html.Div([
        html.H5("🎯 Remediation Actions (Would Be Triggered)", className="mt-4 mb-3"),
        html.P("This shows what auto-remediation actions would be triggered in a production environment.",
               className="text-muted")
    ])

    # Generate remediation plan
    remediation_plan = generate_remediation_plan(server_preds, risk_scores)

    if remediation_plan:
        # Summary metrics
        autonomous_actions = len([r for r in remediation_plan if r['type'] == 'autonomous'])
        manual_actions = len([r for r in remediation_plan if r['type'] == 'manual'])

        summary_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Actions Queued", className="card-subtitle mb-2 text-muted"),
                        html.H3(str(len(remediation_plan)), className="card-title text-primary")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Autonomous Actions", className="card-subtitle mb-2 text-muted"),
                        html.H3(str(autonomous_actions), className="card-title text-success")
                    ])
                ])
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6("Manual Review Required", className="card-subtitle mb-2 text-muted"),
                        html.H3(str(manual_actions), className="card-title text-warning")
                    ])
                ])
            ], width=4)
        ], className="mb-4")

        # Remediation table
        df_remediation = pd.DataFrame(remediation_plan)
        # Drop 'type' column (was only for counting)
        df_remediation = df_remediation.drop(columns=['type'])

        remediation_table = dbc.Card([
            dbc.CardHeader(html.H5(f"🚨 {len(remediation_plan)} Auto-Remediations Would Be Triggered", className="mb-0")),
            dbc.CardBody([
                dbc.Table.from_dataframe(
                    df_remediation,
                    striped=True,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    size='sm'
                )
            ])
        ], className="mb-4")

        remediation_content = html.Div([summary_cards, remediation_table])
    else:
        remediation_content = dbc.Alert([
            html.H5("✅ No auto-remediation actions required", className="mb-2"),
            html.P("Fleet is healthy! No servers with risk score ≥ 70.", className="mb-0")
        ], color="success", className="mb-4")

    # Integration Architecture section
    integration_section = dbc.Card([
        dbc.CardHeader(html.H5("🏗️ Integration Architecture", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Phase 1: Auto-Scaling (Weeks 1-2)", className="mb-3"),
                    html.Ul([
                        html.Li([html.Strong("✅ Spectrum Conductor API integration")]),
                        html.Li([html.Strong("✅ Kubernetes HPA triggers")]),
                        html.Li([html.Strong("✅ Database connection pool tuning")]),
                        html.Li([html.Strong("✅ Load balancer configuration")])
                    ]),
                    html.P([
                        html.Strong("Expected Outcome: "),
                        "85% of incidents auto-remediated"
                    ], className="text-success mt-3")
                ], width=6),
                dbc.Col([
                    html.H6("Phase 2: Advanced Actions (Weeks 3-4)", className="mb-3"),
                    html.Ul([
                        html.Li([html.Strong("🔄 Job rescheduling"), " (batch workloads)"]),
                        html.Li([html.Strong("🔄 Traffic rerouting"), " (degraded services)"]),
                        html.Li([html.Strong("🔄 Cache warming"), " (predicted load spikes)"]),
                        html.Li([html.Strong("🔄 Proactive restarts"), " (memory leaks)"])
                    ]),
                    html.P([
                        html.Strong("Expected Outcome: "),
                        "95% incident prevention rate"
                    ], className="text-success mt-3")
                ], width=6)
            ])
        ])
    ], className="mb-4")

    # Approval Workflow section
    approval_section = dbc.Card([
        dbc.CardHeader(html.H5("✅ Approval Workflow (Configurable)", className="mb-0")),
        dbc.CardBody([
            html.P([html.Strong("Safety Controls"), " (Production Implementation):"], className="mb-3"),
            html.Ol([
                html.Li([html.Strong("Low Risk (Score < 50): "), "Auto-approve, log only"]),
                html.Li([html.Strong("Medium Risk (50-70): "), "Auto-approve with 30-second delay, allow manual override"]),
                html.Li([html.Strong("High Risk (70-85): "), "Require senior engineer approval"]),
                html.Li([html.Strong("Critical Risk (>85): "), "Require director-level approval OR auto-approve during off-hours"])
            ], className="mb-4"),
            html.P([html.Strong("Rollback Strategy:")], className="mb-2"),
            html.Ul([
                html.Li("All actions are reversible within 15 minutes"),
                html.Li("Automatic rollback if metrics don't improve within 10 minutes"),
                html.Li("Manual override always available via dashboard or CLI")
            ])
        ])
    ], className="mb-4")

    # Implementation Note
    implementation_note = dbc.Alert([
        html.Strong("💡 POC Implementation Note:"),
        html.Br(), html.Br(),
        "This tab demonstrates autonomous remediation capabilities. The full implementation would include:",
        html.Ul([
            html.Li([html.Strong("API integrations"), " with Spectrum Conductor, Kubernetes, load balancers, databases"]),
            html.Li([html.Strong("Approval workflows"), " with configurable risk thresholds and escalation paths"]),
            html.Li([html.Strong("Audit logging"), " of all automated actions for compliance"]),
            html.Li([html.Strong("Success metrics"), " tracking remediation effectiveness"]),
            html.Li([html.Strong("Rollback mechanisms"), " for automated actions that don't improve metrics"]),
            html.Li([html.Strong("Human override"), " capabilities at any point in the workflow"])
        ]),
        html.Br(),
        html.Strong("Impact: "),
        "Reduces MTTR (Mean Time To Resolution) from hours to minutes, achieving 95%+ incident prevention rate."
    ], color="light", className="mt-4")

    return html.Div([
        header,
        section_header,
        remediation_content,
        integration_section,
        approval_section,
        implementation_note
    ])
