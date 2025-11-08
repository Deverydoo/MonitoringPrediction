"""
Documentation Tab - Complete user guide for the TFT Monitoring Dashboard

complete documentation covering:
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

 # Critical Scope Section - MUST READ
 st.markdown("---")
 st.markdown("## 🎯 What This System Does (And Doesn't Do)")

 st.warning("""
 **IMPORTANT: Read This First - Understanding Our Monitoring Scope**

 This system is **predictive capacity and performance monitoring** - think "Minority Report" for infrastructure.
 We predict problems **before they happen**, not after. This **augments** your existing monitoring, it does **not replace** it.
 """)

 col1, col2 = st.columns(2)

 with col1:
 st.markdown("### ✅ What We DO (Our Scope)")
 st.success("""
 **Predictive Intelligence - Prevent Tomorrow's Incidents Today**

 🔮 **Capacity Exhaustion Prediction**
 - "This server will hit 98% memory in 45 minutes"
 - "CPU trending to 100% within next hour"
 - Early warning: 30-60 minutes before critical

 🔮 **Performance Degradation Detection**
 - Memory leaks (gradual climb pattern)
 - CPU creep (slow resource exhaustion)
 - I/O bottlenecks (storage contention building)

 🔮 **Resource Bottleneck Forecasting**
 - I/O wait spike predictions
 - Swap thrashing risk assessment
 - Network saturation forecasts

 🔮 **Proactive Problem Prevention**
 - Time to scale infrastructure **before** outage
 - Graceful service restarts during maintenance windows
 - Capacity planning with predictive data

 **Value Proposition**: Give operations teams time to respond **before** users are impacted.
 Traditional monitoring says "it's on fire!" - we say "it will be on fire in 30 minutes, here's what to do."
 """)

 with col2:
 st.markdown("### ❌ What We DON'T Do (Standard Monitoring)")
 st.error("""
 **Traditional Monitoring - These Are Handled By Your Existing Tools**

 ❌ **Server Availability Monitoring**
 - Ping/heartbeat checks
 - "Is server reachable?"
 - Network connectivity testing
 - ➡️ **Use**: Nagios, Zabbix, Datadog, CloudWatch

 ❌ **Hardware Failure Detection**
 - Disk failures
 - Network card failures
 - Power supply issues
 - ➡️ **Use**: IPMI, Hardware RAID alerts, datacenter monitoring

 ❌ **Service Health Checks**
 - "Is Apache running?"
 - Port availability checks
 - Process existence verification
 - ➡️ **Use**: Nagios service checks, Kubernetes liveness probes

 ❌ **Application-Level Monitoring**
 - Error rate tracking
 - Request success/failure
 - Business logic failures
 - ➡️ **Use**: APM tools (New Relic, Dynatrace), application logs

 **Why These Aren't Our Focus**: By the time a server is offline or a service has crashed, it's already a **reactive**
 situation. That's traditional monitoring's job - alert when things are broken. Our job is to predict the break
 **before it happens** so you can prevent it entirely.
 """)

 st.divider()

 st.markdown("### 🤝 How This Works WITH Traditional Monitoring")

 st.info("""
 **Think of monitoring as layers of defense:**

 **Layer 1 (This System): Predictive Early Warning** ⏰
 - 30-60 minutes advance notice
 - "ppml0042 will exhaust memory in 45 minutes"
 - Operations team scales infrastructure **proactively**
 - **Outcome**: Incident prevented, users never impacted

 **Layer 2 (Traditional Monitoring): Real-Time Response** 🚨
 - Immediate detection when thresholds crossed
 - "ppweb012 CPU at 100% for 5 minutes"
 - On-call engineer paged **when problem occurs**
 - **Outcome**: Incident mitigated quickly, some user impact

 **Layer 3 (Traditional Monitoring): Catastrophic Failure** 🔥
 - Server completely offline
 - "ppdb003 not responding to ping"
 - Emergency response, full incident management
 - **Outcome**: Major outage, significant user impact

 **The Goal**: Catch everything at **Layer 1** (predictive) so you never hit Layer 2 or 3.
 When predictive monitoring misses something (it's not perfect!), traditional monitoring catches it at Layer 2/3.
 """)

 st.markdown("### 📊 Real-World Example Scenario")

 st.code("""
 SCENARIO: ML training server memory leak

 T-60 min: 🔮 TFT Dashboard detects memory climbing pattern
 - Current: 72% memory
 - Predicted (30m): 94% memory
 - Risk Score: 58 (Degrading)
 - Alert: "ppml0042 memory leak detected, OOM in ~1 hour"

 T-55 min: 👨‍💼 Operations team sees alert, investigates
 - Correlates with deployment 2 hours ago
 - Identifies problematic training job

 T-50 min: 🔧 Team takes proactive action
 - Gracefully stops training job
 - Restarts service during low-usage window
 - Memory drops back to 45%

 T-40 min: ✅ Problem resolved BEFORE critical threshold
 - No user impact
 - No pager alerts
 - No incident report needed

 RESULT: Outage prevented entirely through predictive intelligence

 ---

 ALTERNATE TIMELINE: Without predictive monitoring

 T+0 min: ⚠️ Traditional monitoring: "ppml0042 memory at 95%"
 T+2 min: 🚨 Traditional monitoring: "ppml0042 memory at 98%"
 T+5 min: 🔥 Traditional monitoring: "ppml0042 OOM kill, services down"
 T+5 min: 📟 On-call engineer paged at 2 AM
 T+15 min: 🔧 Engineer investigates, identifies issue, restarts server
 T+20 min: ✅ Services restored

 RESULT: 15-minute outage, customer impact, incident report required,
 engineer woken up at 2 AM, escalation to management
 """, language="text")

 st.success("""
 **Key Takeaway**: Predictive monitoring **prevents incidents**. Traditional monitoring **detects incidents**.
 You need **both** for a complete monitoring strategy. This dashboard is the early warning system that gives you
 time to act before traditional alerts even fire.
 """)

 st.divider()

 # Table of Contents
 st.markdown("### 📖 Table of Contents")
 st.markdown("""
 1. [What This System Does (And Doesn't Do)](#what-this-system-does-and-doesn-t-do) ⬆️ **READ THIS FIRST**
 2. [Overview & Features](#overview-features)
 3. [Understanding Risk Scores](#understanding-risk-scores)
 4. [Official Risk Threshold System](#official-risk-threshold-system)
 5. [Alert Priority Levels](#alert-priority-levels)
 6. [Contextual Intelligence](#contextual-intelligence)
 7. [Server Profiles](#server-profiles)
 8. [How to Interpret Alerts](#how-to-interpret-alerts)
 9. [Environment Status](#environment-status)
 10. [Trend Analysis](#trend-analysis)
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

 examples_df = pd. DataFrame({
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

 threshold_system_df = pd. DataFrame({
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

 priority_df = pd. DataFrame({
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
 alert = "CRITICAL" # Everything is suddenly on fire!
 else:
 alert = "OK" # Everything is fine!
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

 profiles_df = pd. DataFrame({
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

 alert_columns_df = pd. DataFrame({
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

 env_status_df = pd. DataFrame({
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
║ TFT MONITORING DASHBOARD - QUICK REFERENCE ║
╠════════════════════════════════════════════════════════════════╣
║ RISK SCORE FORMULA: ║
║ Final Risk = (Current State × 70%) + (Predictions × 30%) ║
║ ║
║ PRIORITY LEVELS: ║
║ 🔴 Imminent Failure (90+) → 5-min SLA, CTO escalation ║
║ 🔴 Critical (80-89) → 15-min SLA, page on-call ║
║ 🟠 Danger (70-79) → 30-min SLA, team lead ║
║ 🟡 Warning (60-69) → 1-hour SLA, team awareness ║
║ 🟢 Degrading (50-59) → 2-hour SLA, email only ║
║ 👁️ Watch (30-49) → Background monitoring ║
║ ✅ Healthy (0-29) → No alerts ║
║ ║
║ ENVIRONMENT STATUS: ║
║ 🔴 Critical → >30% Critical+ OR >50% elevated ║
║ 🟠 Warning → >10% Critical+ OR >30% elevated ║
║ 🟡 Caution → >10% Degrading ║
║ 🟢 Healthy → <10% elevated risk ║
║ ║
║ DELTA INTERPRETATION: ║
║ Positive (+) → Metrics increasing (degrading) ║
║ Negative (-) → Metrics decreasing (improving) ║
║ ║
║ PROFILE-SPECIFIC THRESHOLDS: ║
║ Database: 100% memory = NORMAL (page cache) ║
║ ML Compute: 98% memory = CRITICAL (OOM risk) ║
║ Web API: Latency > 200ms = SEVERE (user impact) ║
║ ║
║ RESPONSE PRIORITY: ║
║ 1. Imminent Failure → Drop everything ║
║ 2. Critical → Immediate action ║
║ 3. Danger → Urgent response ║
║ 4. Warning → Monitor closely ║
║ 5. Degrading → Investigate soon ║
╚════════════════════════════════════════════════════════════════╝
 """, language="text")

 st.divider()

 # Section 9: Adaptive Retraining System
 st.markdown("### 🔄 Adaptive Retraining System")
 st.markdown("""
 **How the Model Stays Current**: Behind the scenes, the TFT model intelligently retrains itself based on
 **drift detection** rather than a fixed schedule. This ensures predictions remain accurate as your infrastructure evolves.
 """)

 st.info("""
 **Think of it like a self-tuning instrument**:
 - Traditional approach: Retrain every Monday at 2 AM (regardless of need)
 - Adaptive approach: Retrain only when prediction accuracy drops below threshold
 """)

 # Two-column layout for core concepts
 col1, col2 = st.columns(2)

 with col1:
 st.markdown("#### 🎯 What Triggers Retraining?")
 st.markdown("""
 The system monitors **4 drift metrics** continuously:

 **1. Prediction Error Rate (40% weight)**
 - Compares predictions vs. actual outcomes
 - **SLA-Aligned Threshold**: 10% error = 88% accuracy
 - 🟢 Healthy: <8% error (>92% accuracy)
 - 🟡 Warning: 8-10% error (88-92% accuracy)
 - 🔴 Critical: >10% error (<88% accuracy) → **Triggers retraining**

 **2. Distribution Shift (30% weight)**
 - Detects changes in data patterns
 - Example: CPU usage moved from 40% average to 70%
 - Uses statistical distance metrics (KL divergence)

 **3. Feature Drift (20% weight)**
 - Monitors individual metric patterns
 - Example: Memory usage behaving differently
 - Compares recent vs. historical distributions

 **4. Anomaly Rate (10% weight)**
 - Tracks unusual patterns increasing
 - Example: More outliers than normal
 - Indicates new failure modes emerging

 **Drift Score Calculation**:
 ```
 Drift = (PER × 0.40) + (Distribution × 0.30) +
 (Feature × 0.20) + (Anomaly × 0.10)
 ```

 When drift score **>10%** → Retraining triggered
 """)

 with col2:
 st.markdown("#### 🛡️ Safeguards & Constraints")
 st.markdown("""
 The system includes intelligent guardrails:

 **Time-Based Safeguards:**
 - **Minimum**: 6 hours between trainings
 - Prevents thrashing from transient issues
 - Example: Brief deployment spike won't cause retraining

 - **Maximum**: 30 days without training
 - Forces refresh even if drift low
 - Prevents model staleness

 - **Weekly Limit**: Max 3 trainings per week
 - Prevents compute budget overrun
 - Ensures operational stability

 **Intelligent Scheduling:**
 - **Quiet Time Detection**: Learns server load patterns
 - Identifies low-traffic windows (e.g., 2-4 AM)
 - Schedules training when servers least busy
 - Adapts to your infrastructure's rhythm

 - **Impact Awareness**: Won't retrain during:
 - High-traffic periods (market hours)
 - Known maintenance windows
 - Recent deployments (24-hour buffer)

 **Validation & Safety:**
 - New model validated before deployment
 - Rollback if accuracy worsens
 - Gradual rollout (canary deployment)
 """)

 st.markdown("#### 📊 Example Retraining Scenario")

 st.code("""
SCENARIO: New deployment changes CPU patterns

T-0 hours: New microservice deployed, changes load patterns
 - Model still using pre-deployment patterns for predictions
 - Predictions: 88% accurate (at SLA threshold)

T+2 hours: Drift Monitor detects distribution shift
 - CPU distribution: 40% avg → 55% avg (15% shift)
 - Prediction Error Rate: 11.2% (below 88% SLA)
 - Distribution Shift: 18%
 - Drift Score: 11.8% (above 10% threshold)
 - Status: 🔴 Drift detected, retraining recommended

T+2.5 hrs: Retraining Decision Engine evaluates
 - Last training: 3 days ago ✅ (>6 hours)
 - Trainings this week: 1 ✅ (<3)
 - Current time: 10:30 PM ✅ (approaching quiet window)
 - Decision: Schedule retraining at 2:00 AM

T+5.5 hrs: Automatic retraining triggered (2:00 AM)
 - Loads last 30 days of data
 - Trains 1 epoch (incremental learning)
 - Duration: ~90 minutes
 - Validation: New model tested on holdout data

T+7 hrs: New model validated and deployed (3:30 AM)
 - Validation accuracy: 91.5% ✅ (above 88% SLA)
 - Rollout: Gradual deployment over 30 minutes
 - Monitoring: Extra logging enabled for 4 hours

T+8 hrs: Validation complete (4:30 AM)
 - Prediction Error Rate: 8.5% (>88% accuracy) ✅
 - Drift Score: 4.2% (healthy range) ✅
 - Status: 🟢 Model updated successfully

RESULT: Model adapts to new patterns automatically, maintains SLA
 Operations team never paged, happened during quiet hours
 """, language="text")

 st.markdown("#### 🎯 Why This Matters")

 benefits_df = pd. DataFrame({
 'Aspect': ['Accuracy', 'Cost', 'Maintenance', 'Adaptability', 'Reliability'],
 'Fixed Schedule (Old Way)': [
 'Retrains even when not needed',
 '$150/month (daily training)',
 'Manual monitoring required',
 'Slow (waits for next scheduled run)',
 'May miss sudden drift events'
 ],
 'Adaptive (Our Way)': [
 'Retrains only when accuracy drops',
 '$50/month (event-driven)',
 'Fully automated, zero-touch',
 'Fast (responds to drift immediately)',
 'Catches drift as it happens'
 ],
 'Benefit': [
 '↑ Always at/above SLA',
 '↓ 3x cost reduction',
 '↓ Zero engineering time',
 '↑ Faster adaptation',
 '↑ Proactive drift correction'
 ]
 })

 st.dataframe(benefits_df, hide_index=True, width='stretch')

 st.success("""
 **🎯 Key Takeaway**: You don't need to worry about model staleness. The system monitors itself and retrains
 intelligently when needed, during quiet hours, with safety checks. Your job is to use the predictions -
 the model maintenance happens automatically in the background.
 """)

 st.divider()

 st.success("""
 **📚 Documentation Complete!**

 This guide covers the core concepts and operational procedures for the TFT Monitoring Dashboard.
 For technical implementation details, see the Advanced tab. For future enhancements, see the Roadmap tab.
 """)