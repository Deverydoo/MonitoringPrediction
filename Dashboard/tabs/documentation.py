"""
Documentation Tab - Complete user guide for the TFT Monitoring Dashboard

Comprehensive documentation covering:
- Overview and features
- Understanding risk scores
- Alert priority levels
- Contextual intelligence philosophy
- Server profiles
- How to interpret alerts
- Environment status
- Trend analysis
- Best practices
- Quick reference card
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional


def render(predictions: Optional[Dict]):
    """
    Render the Documentation tab.

    Args:
        predictions: Current predictions from daemon (unused, static content)
    """
    st.subheader("📚 Dashboard Documentation")
    st.markdown("**Complete guide to understanding and using the TFT Monitoring Dashboard**")

    # Table of Contents
    st.markdown("### 📖 Table of Contents")
    st.markdown("""
    1. [Overview & Features](#overview-features)
    2. [Understanding Risk Scores](#understanding-risk-scores)
    3. [Official Risk Threshold System](#official-risk-threshold-system)
    4. [Alert Priority Levels](#alert-priority-levels)
    5. [Contextual Intelligence](#contextual-intelligence)
    6. [Server Profiles](#server-profiles)
    7. [How to Interpret Alerts](#how-to-interpret-alerts)
    8. [Environment Status](#environment-status)
    9. [Trend Analysis](#trend-analysis)
    """)

    st.divider()

    # Section 1: Overview & Features
    st.markdown("### 🎯 Overview & Features")
    st.markdown("""
    **TFT Monitoring Dashboard** is a predictive monitoring system that uses deep learning (Temporal Fusion Transformer)
    to forecast server health 30 minutes to 8 hours in advance.

    **Key Capabilities**:
    - **Real-time Monitoring**: Live metrics from 20+ servers across 7 profiles
    - **30-Minute Predictions**: AI forecasts CPU, Memory, Latency with 85-90% accuracy
    - **8-Hour Horizon**: Extended forecasts for capacity planning
    - **Contextual Intelligence**: Risk scoring considers server profiles, trends, and multi-metric correlations
    - **Graduated Alerts**: 7 severity levels from Healthy to Imminent Failure
    - **Early Warning**: 15-60 minute advance notice before problems become critical

    **Technology Stack**:
    - **Model**: PyTorch Forecasting Temporal Fusion Transformer (TFT)
    - **Architecture**: Microservices with REST/WebSocket APIs
    - **Dashboard**: Streamlit with real-time updates
    - **Training**: Transfer learning with profile-specific fine-tuning
    """)

    st.divider()

    # Section 2: Understanding Risk Scores
    st.markdown("### 📊 Understanding Risk Scores")
    st.markdown("""
    Every server receives a **Risk Score (0-100)** that represents overall health and predicted trajectory.

    **Score Composition**:
    ```
    Final Risk = (Current State × 70%) + (Predictions × 30%)
    ```

    **Why 70/30 Weighting?**
    - **70% Current State**: Executives care about "what's on fire NOW"
    - **30% Predictions**: Early warning value without crying wolf

    **Risk Components**:
    - **CPU Usage**: Current and predicted utilization
    - **Memory Usage**: Current and predicted with profile-specific thresholds
    - **Latency**: Response time degradation
    - **Disk Usage**: Available space warnings
    - **Trend Velocity**: Rate of change (climbing vs. steady)
    - **Multi-Metric Correlation**: Compound stress detection
    """)

    # Risk Score Examples
    st.markdown("#### 🔢 Risk Score Examples")

    examples_df = pd.DataFrame({
        'Scenario': [
            'Normal Operations',
            'Steady High Load',
            'Degrading Performance',
            'Predicted Spike',
            'Compound Stress',
            'Imminent Failure'
        ],
        'CPU': ['25%', '70%', '40% → 75%', '35% → 95%', '85%', '99%'],
        'Memory': ['35%', '65%', '60% → 80%', '50%', '90%', '99%'],
        'Latency': ['40ms', '80ms', '90ms → 150ms', '60ms', '320ms', '1200ms'],
        'Risk Score': [8, 32, 58, 52, 83, 96],
        'Status': ['Healthy ✅', 'Watch 👁️', 'Degrading 🟢', 'Degrading 🟢', 'Critical 🔴', 'Imminent Failure 🔴']
    })

    st.dataframe(examples_df, hide_index=True, width='stretch')

    st.divider()

    # Section 2.5: Official Risk Threshold System
    st.markdown("### 📏 Official Risk Threshold System")
    st.markdown("""
    This table shows how risk scores map to alert categories across the entire dashboard.
    **All metrics use these consistent thresholds** - Fleet Status, Active Alerts, and visualizations.
    """)

    threshold_system_df = pd.DataFrame({
        'Category': [
            '🔴 Imminent Failure',
            '🔴 Critical',
            '🟠 Danger',
            '🟡 Warning',
            '🟢 Degrading',
            '👁️ Watch',
            '✅ Healthy'
        ],
        'Risk Score': ['90-100', '80-89', '70-79', '60-69', '50-59', '30-49', '0-29'],
        'Shown in Active Alerts?': ['✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes', '✅ Yes', '❌ No', '❌ No'],
        'Counted as Healthy in Fleet Status?': ['❌ No', '❌ No', '❌ No', '❌ No', '❌ No', '✅ Yes', '✅ Yes'],
        'Appears in Pie Chart?': ['Grouped as Critical', '✅ Yes (Red)', 'Grouped as Critical', '✅ Yes (Orange)', '✅ Yes (Gold)', '✅ Yes (Light Blue)', '✅ Yes (Green)']
    })

    st.dataframe(threshold_system_df, hide_index=True, width='stretch')

    st.info("""
    **Key Insight**: The **Active Alerts** table only shows servers with Risk ≥ 50 (Degrading and above).
    Servers below Risk 50 are considered healthy and counted in the **Fleet Status** healthy number.

    **Why this threshold?**
    - Risk < 30: Truly healthy, no concerns
    - Risk 30-49 (Watch): Minor elevation, background monitoring only
    - Risk 50+ (Degrading): Requires investigation, shown as active alert
    """)

    st.divider()

    # Section 3: Alert Priority Levels
    st.markdown("### 🚨 Alert Priority Levels")
    st.markdown("""
    The dashboard uses **7 graduated severity levels** instead of binary OK/CRITICAL alerts.
    This provides nuanced triage and graduated escalation.
    """)

    priority_df = pd.DataFrame({
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
        'Response': [
            'Drop everything, CTO escalation',
            'Page on-call engineer immediately',
            'Team lead engaged, urgent response',
            'Team awareness, monitor closely',
            'Email notification, investigate',
            'Dashboard only, no action needed',
            'No alerts generated'
        ],
        'SLA': ['5 minutes', '15 minutes', '30 minutes', '1 hour', '2 hours', 'Best effort', 'N/A']
    })

    st.dataframe(priority_df, hide_index=True, width='stretch')

    st.markdown("""
    **Key Insight**: Notice how the system provides graduated escalation. You don't go from "Healthy" to "Critical" -
    instead you progress through Watch → Degrading → Warning → Danger, giving teams time to respond proactively.
    """)

    st.divider()

    # Section 4: Contextual Intelligence
    st.markdown("### 🧠 Contextual Intelligence: Beyond Simple Thresholds")
    st.markdown("""
    **Philosophy**: "40% CPU may be fine, or may be degrading - depends on context"

    Traditional monitoring uses **binary thresholds**:
    ```python
    if cpu > 80%:
        alert = "CRITICAL"  # Everything is suddenly on fire!
    else:
        alert = "OK"  # Everything is fine!
    ```

    **Problems**:
    - ❌ No context: 80% CPU on database = normal, 80% on web server = problem
    - ❌ No trends: 80% steady = fine, 40% → 80% in 10 min = concerning
    - ❌ No prediction: Server at 60% but climbing fast will crash soon
    - ❌ Binary state: Everything is either OK or ON FIRE (no middle ground)
    - ❌ Ignores correlations: High CPU + high memory + high latency = compound risk

    **Our Approach**: Contextual intelligence using **fuzzy logic**
    """)

    # Contextual Factors
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Context Factor 1: Server Profile")
        st.markdown("""
        **Same metric, different meaning**:

        **Database Server (ppdb001)**:
        - Memory: 98% = ✅ **Healthy** (page cache is normal)
        - Risk Score: 8

        **ML Compute (ppml0001)**:
        - Memory: 98% = 🔴 **Critical** (OOM kill imminent)
        - Risk Score: 82

        The system understands that databases use 100% memory for
        caching (expected), while compute servers need headroom
        for allocations.
        """)

        st.markdown("#### 📈 Context Factor 2: Trend Analysis")
        st.markdown("""
        **Same current value, different trajectory**:

        **Steady State**:
        - CPU: 40% for last 30 minutes
        - Risk: 0 (stable workload)

        **Rapid Climb**:
        - CPU: 20% → 40% → 60% (climbing 20%/10min)
        - Risk: 56 (will hit 100% in 20 minutes!)

        The system detects velocity and acceleration patterns.
        """)

    with col2:
        st.markdown("#### 🔗 Context Factor 3: Multi-Metric Correlation")
        st.markdown("""
        **Isolated spike vs. compound stress**:

        **Isolated CPU Spike**:
        - CPU: 85% (batch job)
        - Memory: 35%
        - Latency: 40ms
        - Risk: 28 (✅ Healthy - just a batch job)

        **Compound Stress**:
        - CPU: 85% (same value!)
        - Memory: 90%
        - Latency: 350ms
        - Risk: 83 (🔴 Critical - system under stress)
        """)

        st.markdown("#### 🔮 Context Factor 4: Prediction-Aware")
        st.markdown("""
        **Current vs. predicted state**:

        **Looks Fine Now, But...**:
        - Current CPU: 40%
        - Predicted (30m): 95%
        - Risk: 52 (🟢 Degrading - early warning!)

        **Bad Now, Getting Better**:
        - Current CPU: 85%
        - Predicted (30m): 60%
        - Risk: 38 (👁️ Watch - resolving itself)
        """)

    st.markdown("""
    #### 🎯 Result: Intelligent Risk Assessment

    The system combines all four context factors to produce a risk score that reflects **actual operational risk**,
    not just raw metric values. This eliminates false positives while providing earlier detection of real problems.
    """)

    st.divider()

    # Section 5: Server Profiles
    st.markdown("### 🖥️ Server Profiles")
    st.markdown("""
    The system automatically detects server profiles from hostnames and applies **profile-specific intelligence**.
    """)

    profiles_df = pd.DataFrame({
        'Profile': ['ML Compute', 'Database', 'Web API', 'Conductor Mgmt', 'Data Ingest', 'Risk Analytics', 'Generic'],
        'Hostname Pattern': ['ppml####', 'ppdb###', 'ppweb###', 'ppcon##', 'ppdi###', 'ppra###', 'ppsrv###'],
        'Characteristics': [
            'High CPU/Mem during training, Memory-intensive',
            'High memory (page cache), Query CPU spikes',
            'Low memory (stateless), Latency-sensitive',
            'Low CPU/Mem, Management workload',
            'High disk I/O, Network-intensive',
            'CPU-intensive analytics, Batch processing',
            'Balanced workload'
        ],
        'Memory Threshold': ['98%', '100%', '85%', '90%', '90%', '95%', '90%'],
        'Key Metrics': [
            'CPU, Memory allocation',
            'Query latency, Memory cache',
            'Request latency, Error rate',
            'Process health',
            'Disk I/O, Network throughput',
            'CPU usage, GC pauses',
            'Balanced monitoring'
        ]
    })

    st.dataframe(profiles_df, hide_index=True, width='stretch')

    st.info("""
    **Why Profile Awareness Matters**: A database at 100% memory is healthy (caching), but a web server at 100% memory
    is about to crash (memory leak). The system adjusts thresholds and risk calculations based on expected behavior patterns.
    """)

    st.divider()

    # Section 6: How to Interpret Alerts
    st.markdown("### 🔔 How to Interpret Alerts")
    st.markdown("""
    The **Active Alerts** table shows servers requiring attention (Risk ≥ 50). Here's how to read it:
    """)

    st.markdown("#### 📋 Alert Table Columns Explained")

    alert_columns_df = pd.DataFrame({
        'Column': ['Priority', 'Server', 'Profile', 'Risk', 'CPU Now', 'CPU Predicted (30m)', 'CPU Δ', 'Mem Now', 'Mem Predicted (30m)', 'Mem Δ', 'I/O Wait Now', 'I/O Wait Predicted (30m)', 'I/O Wait Δ'],
        'Meaning': [
            'Severity level (Imminent Failure → Critical → Danger → Warning → Degrading)',
            'Server hostname (hover for details)',
            'Detected server workload type',
            'Overall risk score (0-100, higher = more urgent)',
            'Current CPU utilization percentage',
            'AI-predicted CPU in next 30 minutes',
            'Predicted change: + = increasing (degrading), - = decreasing (improving)',
            'Current memory utilization percentage',
            'AI-predicted memory in next 30 minutes',
            'Predicted change: + = increasing, - = decreasing',
            'Current I/O wait percentage (CRITICAL metric)',
            'AI-predicted I/O wait in next 30 minutes',
            'Predicted change: + = increasing I/O contention'
        ]
    })

    st.dataframe(alert_columns_df, hide_index=True, width='stretch')

    st.markdown("#### 🎯 Priority Triage Strategy")

    st.markdown("""
    **Step 1: Check Critical+ Servers First**
    - Risk 90+ (Imminent Failure): Drop everything, server about to crash
    - Risk 80-89 (Critical): Immediate action, page on-call if after hours

    **Step 2: Review Danger/Warning Servers**
    - Risk 70-79 (Danger): High priority, team lead should investigate
    - Risk 60-69 (Warning): Monitor closely, team awareness

    **Step 3: Track Degrading Servers**
    - Risk 50-59 (Degrading): Early warnings, investigate during business hours

    **Step 4: Look for Patterns**
    - Multiple servers with same profile degrading? (Shared infrastructure issue)
    - All servers in datacenter showing latency? (Network problem)
    - Single server with multiple metrics elevated? (Compound stress)
    """)

    st.markdown("#### 📈 Understanding Delta (Δ) Values")

    st.info("""
    **Delta values show predicted CHANGE**, not absolute values:

    - **CPU Δ: +15.2%** → CPU will increase by 15.2% in next 30 minutes
    - **Mem Δ: -5.3%** → Memory will decrease by 5.3% (improving)
    - **I/O Wait Δ: +5.1%** → I/O wait will increase by 5.1% (degrading)

    **🚨 Red Flag Pattern**: All deltas positive (+) = server degrading across all metrics
    **✅ Good Pattern**: All deltas negative (-) = server recovering across all metrics
    **⚠️ Mixed Pattern**: Some + some - = investigate further
    """)

    st.divider()

    # Section 7: Environment Status
    st.markdown("### 🌍 Environment Status")
    st.markdown("""
    The **Environment Status** indicator (top-left of Overview tab) shows fleet-wide health at a glance.
    """)

    env_status_df = pd.DataFrame({
        'Status': ['🔴 Critical', '🟠 Warning', '🟡 Caution', '🟢 Healthy'],
        'Conditions': [
            '>30% of fleet Critical+ (Risk 80+) OR >50% elevated risk',
            '>10% of fleet Critical+ OR >30% elevated risk',
            '>10% of fleet Degrading (Risk 50+)',
            '<10% of fleet has elevated risk'
        ],
        'Interpretation': [
            'MAJOR INCIDENT: Multiple servers failing, immediate executive attention',
            'ELEVATED CONCERN: Significant portion of fleet affected, team mobilization',
            'EARLY WARNING: Some servers showing degradation, proactive investigation',
            'NORMAL OPERATIONS: Fleet healthy, routine monitoring'
        ],
        'Typical Action': [
            'War room, incident commander, all-hands response',
            'Team standup, resource allocation, incident tracking',
            'Email notifications, team awareness, monitoring',
            'No action required, continue normal operations'
        ]
    })

    st.dataframe(env_status_df, hide_index=True, width='stretch')

    st.markdown("""
    **Example Scenario**: You have 20 servers
    - 2 servers at Risk 85 (Critical)
    - 3 servers at Risk 72 (Danger)
    - 15 servers at Risk 20 (Healthy)

    **Calculation**:
    - Critical% = 2/20 = 10%
    - Elevated% = 5/20 = 25%

    **Status**: 🟠 **Warning** (10% critical, 25% elevated)
    **Action**: Team mobilization, incident tracking
    """)

    st.divider()

    # Section 8: Trend Analysis
    st.markdown("### 📊 Trend Analysis")
    st.markdown("""
    Below the alert summary metrics, the **Trend Analysis** section shows movement patterns:
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⬆️ Degrading Trends")
        st.markdown("""
        **Definition**: Servers with positive (+) delta values for CPU or Memory

        **What it means**:
        - Metrics increasing over next 30 minutes
        - Performance declining
        - Requires attention

        **Example**:
        - Alert table shows 12 servers
        - 8 have positive CPU Δ or Mem Δ
        - Display: "⬆️ Degrading: 8/12 (67%)"

        **Interpretation**: Most alerts are degrading situations (not recovering)
        """)

    with col2:
        st.markdown("#### ⬇️ Improving Trends")
        st.markdown("""
        **Definition**: Servers with negative (-) delta values for CPU or Memory

        **What it means**:
        - Metrics decreasing over next 30 minutes
        - Performance improving
        - Problems resolving themselves

        **Example**:
        - Alert table shows 12 servers
        - 4 have negative CPU Δ or Mem Δ
        - Display: "⬇️ Improving: 4/12 (33%)"

        **Interpretation**: Some servers recovering (maybe remediation already applied)
        """)

    st.warning("""
    **Important**: Trend percentages are calculated from **alerts only**, not total fleet.

    - If you have 12 alerts and 8 degrading → "8/12" NOT "8/20"
    - This shows what proportion of your active problems are getting worse vs. better
    """)

    st.divider()

    # Best Practices
    st.markdown("### ✅ Best Practices")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👍 Do's")
        st.markdown("""
        - ✅ **Check dashboard every 15-30 minutes** during business hours
        - ✅ **Trust the risk scores** - they include context you might miss
        - ✅ **Act on Degrading alerts proactively** before they become Critical
        - ✅ **Look for patterns** across multiple servers
        - ✅ **Use predictions** to plan maintenance windows
        - ✅ **Correlate with deployments** - did we just push code?
        - ✅ **Review Watch servers** periodically (Risk 30-49)
        - ✅ **Trust profile-specific thresholds** (DB at 100% mem = OK)
        """)

    with col2:
        st.markdown("#### 👎 Don'ts")
        st.markdown("""
        - ❌ **Don't ignore Degrading alerts** thinking "it's only 55% CPU"
        - ❌ **Don't panic at single metric spike** - look at overall risk score
        - ❌ **Don't override profile thresholds** without understanding context
        - ❌ **Don't dismiss predictions** as "just guesses"
        - ❌ **Don't create manual alerts** that duplicate dashboard intelligence
        - ❌ **Don't compare this to traditional monitoring** - it's predictive
        - ❌ **Don't ignore improving trends** - verify remediation worked
        """)

    st.divider()

    # Quick Reference
    st.markdown("### 🚀 Quick Reference Card")

    st.code("""
╔════════════════════════════════════════════════════════════════╗
║         TFT MONITORING DASHBOARD - QUICK REFERENCE            ║
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
║ ENVIRONMENT STATUS:                                           ║
║   🔴 Critical  → >30% Critical+ OR >50% elevated             ║
║   🟠 Warning   → >10% Critical+ OR >30% elevated             ║
║   🟡 Caution   → >10% Degrading                              ║
║   🟢 Healthy   → <10% elevated risk                          ║
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
    """, language="text")

    st.divider()

    st.success("""
    **📚 Documentation Complete!**

    This guide covers the core concepts and operational procedures for the TFT Monitoring Dashboard.
    For technical implementation details, see the Advanced tab. For future enhancements, see the Roadmap tab.
    """)
