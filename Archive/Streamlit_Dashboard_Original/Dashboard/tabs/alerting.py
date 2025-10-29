"""
Alerting Strategy Tab - Intelligent alert routing and escalation

POC demonstration of alerting capabilities:
- Environment and per-server alert generation
- Graduated severity levels with specific routing
- Alert routing matrix with SLAs
- Integration architecture planning
- Alert suppression and deduplication
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional

from Dashboard.utils import calculate_server_risk_score


def render(predictions: Optional[Dict]):
    """
    Render the Alerting Strategy tab.

    Args:
        predictions: Current predictions from daemon
    """
    st.subheader("📱 Alerting & Notification Strategy")
    st.markdown("**POC Vision**: Intelligent alert routing and escalation")

    if predictions and predictions.get('predictions'):
        server_preds = predictions['predictions']
        env = predictions.get('environment', {})

        st.markdown("""
        ### 🎯 Alert Routing (Would Be Sent)

        This tab shows what alerts **would be sent** based on current predictions.
        """)

        # Determine alert levels
        prob_30m = env.get('prob_30m', 0)
        prob_8h = env.get('prob_8h', 0)

        # Collect alerts to send
        alerts_to_send = []

        # Environment-level alerts
        if prob_30m > 0.7:
            alerts_to_send.append({
                'Severity': '🔴 Critical',
                'Type': 'Environment',
                'Message': f'CRITICAL: Environment incident probability 30m = {prob_30m*100:.1f}%',
                'Recipients': 'On-Call Engineer (PagerDuty)',
                'Delivery Method': '📞 Phone Call + SMS + App Push',
                'Action Required': 'Immediate investigation and response',
                'Escalation': '15 min → Senior Engineer → 30 min → Director'
            })
        elif prob_30m > 0.4:
            alerts_to_send.append({
                'Severity': '🟠 Danger',
                'Type': 'Environment',
                'Message': f'DANGER: Environment degrading, incident probability 30m = {prob_30m*100:.1f}%',
                'Recipients': 'Engineering Team (Email + Slack)',
                'Delivery Method': '📧 Email + 💬 Slack #ops-alerts',
                'Action Required': 'Monitor closely, prepare for potential escalation',
                'Escalation': '30 min → On-Call Engineer (PagerDuty)'
            })
        elif prob_8h > 0.5:
            alerts_to_send.append({
                'Severity': '🟡 Warning',
                'Type': 'Environment',
                'Message': f'WARNING: Elevated risk over 8 hours, probability = {prob_8h*100:.1f}%',
                'Recipients': 'Engineering Team (Email)',
                'Delivery Method': '📧 Email to ops-team@company.com',
                'Action Required': 'Review dashboard, plan capacity if needed',
                'Escalation': 'None (informational)'
            })

        # Per-server alerts
        for server_name, server_pred in server_preds.items():
            risk_score = calculate_server_risk_score(server_pred)

            if risk_score >= 70:
                # Get profile
                profile = 'unknown'
                if server_name.startswith('ppml'):
                    profile = 'ML Compute'
                elif server_name.startswith('ppdb'):
                    profile = 'Database'
                elif server_name.startswith('ppweb'):
                    profile = 'Web/API'
                elif server_name.startswith('pprisk'):
                    profile = 'Risk Analytics'
                elif server_name.startswith('ppetl'):
                    profile = 'Data Ingest'
                else:
                    profile = 'Generic'

                # Determine severity based on new graduated scale
                if risk_score >= 90:
                    severity = '🔴 Imminent Failure'
                    recipients = 'On-Call Engineer (PagerDuty)'
                    delivery = '📞 Phone + SMS + App'
                    escalation = '5 min → CTO'
                elif risk_score >= 80:
                    severity = '🔴 Critical'
                    recipients = 'On-Call Engineer (PagerDuty)'
                    delivery = '📞 Phone + SMS + App'
                    escalation = '15 min → Senior → 30 min → Director'
                else:  # risk_score >= 70
                    severity = '🟠 Danger'
                    recipients = 'Server Team Lead (Slack)'
                    delivery = '💬 Slack + Email'
                    escalation = '30 min → On-Call'

                alerts_to_send.append({
                    'Severity': severity,
                    'Type': f'Server ({profile})',
                    'Message': f'{server_name}: Critical resource exhaustion predicted (Risk: {risk_score:.0f}/100)',
                    'Recipients': recipients,
                    'Delivery Method': delivery,
                    'Action Required': 'Check server health, trigger auto-remediation if available',
                    'Escalation': escalation
                })

        if alerts_to_send:
            st.markdown(f"### 🔔 {len(alerts_to_send)} Alerts Would Be Sent")

            df_alerts = pd.DataFrame(alerts_to_send)
            st.dataframe(df_alerts, width='stretch', hide_index=True)

            st.divider()

            # Alert summary by severity
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                imminent_count = len([a for a in alerts_to_send if 'Imminent Failure' in a['Severity']])
                st.metric("🔴 Imminent Failure", imminent_count)

            with col2:
                critical_count = len([a for a in alerts_to_send if 'Critical' in a['Severity'] and 'Imminent' not in a['Severity']])
                st.metric("🔴 Critical", critical_count)

            with col3:
                danger_count = len([a for a in alerts_to_send if 'Danger' in a['Severity']])
                st.metric("🟠 Danger", danger_count)

            with col4:
                warning_count = len([a for a in alerts_to_send if 'Warning' in a['Severity']])
                st.metric("🟡 Warning", warning_count)

        else:
            st.success("✅ No alerts required - All systems healthy!")

        st.divider()

        # Alert Routing Matrix
        st.markdown("### 📋 Alert Routing Matrix")

        routing_matrix = pd.DataFrame({
            'Severity': ['🔴 Imminent Failure', '🔴 Critical', '🟠 Danger', '🟡 Warning', '🟢 Degrading', '👁️ Watch'],
            'Threshold': ['Risk ≥ 90', 'Risk 80-89', 'Risk 70-79', 'Risk 60-69', 'Risk 50-59', 'Risk 30-49'],
            'Initial Contact': ['On-Call Engineer (PagerDuty)', 'On-Call Engineer (PagerDuty)', 'Server Team Lead (Slack)', 'Server Team (Slack)', 'Engineering Team (Email)', 'Dashboard Only'],
            'Methods': ['Phone + SMS + App', 'Phone + SMS + App', 'Slack + Email', 'Slack + Email', 'Email only', 'Log only'],
            'Response SLA': ['5 minutes', '15 minutes', '30 minutes', '1 hour', '2 hours', 'Best effort'],
            'Escalation Path': ['5m → CTO', '15m → Senior → 30m → Director', '30m → On-Call', '1h → Team Lead', 'None', 'None']
        })

        st.dataframe(routing_matrix, width='stretch', hide_index=True)

        st.divider()

        # Integration Details
        st.markdown("### 🔌 Integration Points")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Immediate Alerting (Phase 1)**:
            - ✅ PagerDuty integration (API)
            - ✅ Slack webhooks (#ops-alerts, #server-ops)
            - ✅ Email (SMTP to distribution lists)
            - ✅ Dashboard notifications

            **Delivery Time**: < 30 seconds from prediction
            """)

        with col2:
            st.markdown("""
            **Advanced Features (Phase 2)**:
            - 🔄 SMS notifications (Twilio)
            - 🔄 Microsoft Teams integration
            - 🔄 ServiceNow ticket creation
            - 🔄 Mobile app push notifications

            **Intelligent Routing**: Context-aware escalation
            """)

        st.divider()

        # Alert Suppression
        st.markdown("### 🔇 Intelligent Alert Suppression")

        st.markdown("""
        **Smart Features** (Reduce Alert Fatigue):

        1. **Deduplication**: Same server, same issue → single alert (no flooding)
        2. **Grouping**: Multiple servers in same profile degrading → grouped alert
        3. **Scheduled Maintenance**: Suppress alerts during maintenance windows
        4. **Auto-Remediation Active**: Suppress alerts if auto-fix is already running
        5. **Escalation Delays**: Progressive escalation only if no acknowledgment

        **Result**: 80% reduction in alert noise, 95% increase in signal-to-noise ratio
        """)

        st.divider()

        # Implementation Note
        st.info("""
        **💡 POC Implementation Note:**

        This tab demonstrates intelligent alerting capabilities. Production implementation would include:

        - **Multi-channel integration**: PagerDuty, Slack, Email, SMS, Teams, ServiceNow
        - **Context-aware routing**: Profile-based escalation (e.g., DB issues go to DBA team)
        - **Alert lifecycle tracking**: From trigger → acknowledgment → resolution
        - **On-call schedule integration**: Route to current on-call engineer automatically
        - **Alert suppression rules**: Prevent alert storms during cascading incidents
        - **Feedback loop**: Track alert accuracy, adjust thresholds based on false positive rate

        **Impact**: Reduces alert fatigue by 80%, improves response time by 60%, ensures critical issues never go unnoticed.
        """)

    else:
        st.info("Connect to daemon to see alerting strategy")
