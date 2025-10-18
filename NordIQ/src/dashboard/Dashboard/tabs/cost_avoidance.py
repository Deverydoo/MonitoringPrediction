"""
Cost Avoidance Tab - Financial impact tracking and ROI analysis

Demonstrates business value with:
- Configurable cost assumptions
- Projected cost avoidance calculations
- ROI analysis
- At-risk servers with potential incident costs
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional

from Dashboard.utils import calculate_server_risk_score


def render(predictions: Optional[Dict]):
    """
    Render the Cost Avoidance tab.

    Args:
        predictions: Current predictions from daemon
    """
    st.subheader("💰 Cost Avoidance Dashboard")
    st.markdown("**POC Vision**: Real-time financial impact tracking")

    if predictions and predictions.get('predictions'):
        server_preds = predictions['predictions']
        env = predictions.get('environment', {})

        # Cost assumptions
        st.markdown("### 💵 Cost Assumptions (Configurable)")
        col1, col2, col3 = st.columns(3)

        with col1:
            cost_per_outage_hour = st.number_input(
                "Outage Cost ($/hour)",
                value=50000,
                step=5000,
                help="Average cost per hour of downtime for a critical server"
            )

        with col2:
            avg_outage_duration = st.number_input(
                "Avg Outage Duration (hours)",
                value=2.5,
                step=0.5,
                help="Average duration of an incident if not prevented"
            )

        with col3:
            prevention_rate = st.slider(
                "Prevention Success Rate (%)",
                min_value=50,
                max_value=100,
                value=85,
                help="Percentage of predicted incidents successfully prevented"
            )

        st.divider()

        # Calculate savings
        st.markdown("### 📊 Projected Cost Avoidance")

        # Count servers at risk
        high_risk_servers = [s for s, p in server_preds.items()
                            if calculate_server_risk_score(p) >= 70]
        medium_risk_servers = [s for s, p in server_preds.items()
                              if 40 <= calculate_server_risk_score(p) < 70]

        # Calculate potential cost avoidance
        critical_incidents_prevented = len(high_risk_servers) * (prevention_rate / 100)
        warning_incidents_prevented = len(medium_risk_servers) * 0.3 * (prevention_rate / 100)

        total_incidents_prevented = critical_incidents_prevented + warning_incidents_prevented
        cost_avoided_per_incident = cost_per_outage_hour * avg_outage_duration
        total_cost_avoided = total_incidents_prevented * cost_avoided_per_incident

        # Monthly and annual projections
        daily_avoidance = total_cost_avoided
        monthly_avoidance = daily_avoidance * 30
        annual_avoidance = daily_avoidance * 365

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Incidents Prevented (Daily)",
                f"{total_incidents_prevented:.1f}",
                delta=None
            )

        with col2:
            st.metric(
                "Daily Cost Avoidance",
                f"${daily_avoidance:,.0f}",
                delta=None
            )

        with col3:
            st.metric(
                "Monthly Projection",
                f"${monthly_avoidance:,.0f}",
                delta=None,
                help="30-day projection"
            )

        with col4:
            st.metric(
                "Annual Projection",
                f"${annual_avoidance:,.0f}",
                delta=None,
                help="365-day projection"
            )

        st.divider()

        # ROI Calculator
        st.markdown("### 🎯 ROI Analysis")

        col1, col2 = st.columns(2)

        with col1:
            project_cost = st.number_input(
                "Project Investment ($)",
                value=250000,
                step=25000,
                help="Total cost for development, deployment, and first year operations"
            )

        with col2:
            months_to_roi = (project_cost / monthly_avoidance) if monthly_avoidance > 0 else float('inf')

            if months_to_roi < 12:
                st.success(f"✅ **ROI in {months_to_roi:.1f} months**")
                st.markdown(f"**Payback**: {months_to_roi:.1f} months")
                st.markdown(f"**First Year Net**: ${(annual_avoidance - project_cost):,.0f}")
            else:
                st.info(f"📊 **ROI in {months_to_roi:.1f} months**")

        st.divider()

        # At-Risk Servers Detail
        st.markdown("### 🎯 Current At-Risk Servers")

        if high_risk_servers or medium_risk_servers:
            risk_breakdown = []

            for server in high_risk_servers:
                risk_breakdown.append({
                    'Server': server,
                    'Risk Level': 'Critical',
                    'Risk Score': calculate_server_risk_score(server_preds[server]),
                    'Potential Cost if Incident': f"${cost_avoided_per_incident:,.0f}",
                    'Status': '🔴 Action Required'
                })

            for server in medium_risk_servers[:5]:  # Top 5 medium risk
                risk_breakdown.append({
                    'Server': server,
                    'Risk Level': 'Warning',
                    'Risk Score': calculate_server_risk_score(server_preds[server]),
                    'Potential Cost if Incident': f"${cost_avoided_per_incident * 0.6:,.0f}",
                    'Status': '🟠 Monitor'
                })

            df_risks = pd.DataFrame(risk_breakdown)
            st.dataframe(df_risks, width='stretch', hide_index=True)
        else:
            st.success("✅ No high-risk servers detected - Fleet is healthy!")

        st.divider()

        # Implementation Note
        st.info("""
        **💡 POC Implementation Note:**

        This tab demonstrates the financial impact tracking capability. In production, we would:

        - **Track actual incidents** prevented vs predicted (accuracy metrics)
        - **Integrate with ITSM** (ServiceNow, JIRA) for actual incident costs
        - **Historical cost avoidance** dashboard showing cumulative savings
        - **Per-profile cost models** (e.g., ML compute downtime costs more than generic servers)
        - **Executive reporting** with monthly/quarterly summaries

        **Business Case**: This system pays for itself in ~3-5 months based on typical financial services downtime costs.
        """)

    else:
        st.info("Connect to daemon to see cost avoidance projections")
