#!/usr/bin/env python3
"""
tft_dashboard.py - Real-time TFT Monitoring Dashboard
Production-level dashboard for server monitoring with incident simulation
Compatible with Jupyter notebooks and CLI usage
"""

import time
import json
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'TICK_SECONDS': 5,
    'REFRESH_SECONDS': 60,
    'TOTAL_RUNTIME_MIN': 20,
    'ROLLING_WINDOW_MIN': 120,
    'MAX_ENV_POINTS': 500,
    'SEED': 42,
    'SAVE_PLOTS': False,
    'PLOT_DIR': './plots/dashboard/',
    'FLEET_SIZE': 25
}

# =============================================================================
# DEFAULT DATA GENERATOR
# =============================================================================

class FleetDataGenerator:
    """Generate realistic fleet metrics with temporal patterns."""
    
    def __init__(self, fleet_size: int = 25, seed: int = 42):
        np.random.seed(seed)
        self.fleet_size = fleet_size
        self.start_time = pd.Timestamp.utcnow()
        
        # Server profiles
        self.servers = []
        profiles = ['production', 'staging', 'service', 'compute']
        profile_weights = [0.4, 0.25, 0.25, 0.1]
        
        for i in range(fleet_size):
            profile = np.random.choice(profiles, p=profile_weights)
            self.servers.append({
                'server_name': f'{profile}-{i:03d}',
                'profile': profile,
                'base_cpu': np.random.uniform(15, 45),
                'base_mem': np.random.uniform(30, 60),
                'base_latency': np.random.uniform(10, 100),
                'noise_factor': np.random.uniform(0.8, 1.2)
            })
        
        self.tick_count = 0
    
    def get_tick_batch(self) -> pd.DataFrame:
        """Generate one full fleet batch for current tick."""
        self.tick_count += 1
        current_time = self.start_time + timedelta(seconds=self.tick_count * CONFIG['TICK_SECONDS'])
        
        batch_data = []
        
        for server in self.servers:
            # Time-based patterns
            hour_factor = 1 + 0.3 * np.sin(2 * np.pi * current_time.hour / 24)
            minute_cycle = 1 + 0.1 * np.sin(2 * np.pi * current_time.minute / 60)
            
            # Base metrics with temporal variation
            cpu_pct = server['base_cpu'] * hour_factor * minute_cycle * server['noise_factor']
            cpu_pct += np.random.normal(0, 5)
            cpu_pct = max(0, min(100, cpu_pct))
            
            mem_pct = server['base_mem'] * (1 + 0.1 * hour_factor)
            mem_pct += np.random.normal(0, 3)
            mem_pct = max(0, min(100, mem_pct))
            
            # Correlated metrics
            latency_ms = server['base_latency'] * (1 + cpu_pct / 200)
            latency_ms += np.random.exponential(5)
            
            error_rate = max(0, 0.1 * (cpu_pct / 100) ** 2 + np.random.exponential(0.05))
            
            # Network and other metrics
            disk_io_mb_s = np.random.exponential(10) + cpu_pct / 10
            net_in_mb_s = np.random.exponential(5) + np.random.uniform(1, 10)
            net_out_mb_s = net_in_mb_s * np.random.uniform(0.8, 1.2)
            
            gc_pause_ms = np.random.exponential(2) if server['profile'] == 'production' else np.random.exponential(1)
            container_oom = 1 if np.random.random() < 0.001 else 0
            problem_child = 1 if cpu_pct > 80 or error_rate > 0.5 else 0
            
            # Server state logic
            if cpu_pct > 90 or error_rate > 1.0:
                state = 'critical_issue'
            elif cpu_pct > 75 or error_rate > 0.3:
                state = 'warning'
            elif np.random.random() < 0.002:
                state = 'offline'
            else:
                state = 'online'
            
            batch_data.append({
                'timestamp': current_time.isoformat(),
                'server_name': server['server_name'],
                'profile': server['profile'],
                'state': state,
                'cpu_pct': cpu_pct,
                'mem_pct': mem_pct,
                'disk_io_mb_s': disk_io_mb_s,
                'net_in_mb_s': net_in_mb_s,
                'net_out_mb_s': net_out_mb_s,
                'latency_ms': latency_ms,
                'error_rate': error_rate,
                'gc_pause_ms': gc_pause_ms,
                'container_oom': container_oom,
                'problem_child': problem_child
            })
        
        return pd.DataFrame(batch_data)

# =============================================================================
# INCIDENT ORCHESTRATOR
# =============================================================================

class EventOrchestrator:
    """Orchestrate realistic environment incidents."""
    
    def __init__(self, fleet_servers: List[str], seed: int = 42):
        np.random.seed(seed)
        self.fleet_servers = fleet_servers
        self.start_time = pd.Timestamp.utcnow()
        
        # Schedule random incident
        self.start_offset_min = np.random.uniform(5, 40)
        self.duration_min = np.random.uniform(7, 30)
        
        # Incident timing
        self.incident_start = self.start_time + timedelta(minutes=self.start_offset_min)
        self.incident_end = self.incident_start + timedelta(minutes=self.duration_min)
        
        # Phase durations (proportional to total duration)
        total_duration = self.duration_min
        self.lead_in_duration = total_duration * np.random.uniform(0.15, 0.25)
        self.peak_duration = total_duration * np.random.uniform(0.3, 0.5)
        self.recovery_duration = total_duration - self.lead_in_duration - self.peak_duration
        
        self.peak_start = self.incident_start + timedelta(minutes=self.lead_in_duration)
        self.recovery_start = self.peak_start + timedelta(minutes=self.peak_duration)
        
        # Choose incident type
        incident_types = ['db_contention', 'net_partition', 'cache_thundering_herd', 'compute_batch_overrun']
        weights = [0.25, 0.25, 0.25, 0.25]
        self.incident_type = np.random.choice(incident_types, p=weights)
        
        # Select affected servers (15-40% of fleet)
        affected_count = int(len(fleet_servers) * np.random.uniform(0.15, 0.40))
        self.affected_servers = set(np.random.choice(fleet_servers, affected_count, replace=False))
        
        print(f"🎯 Incident Scheduled:")
        print(f"   Type: {self.incident_type}")
        print(f"   Start: +{self.start_offset_min:.1f} min ({self.incident_start.strftime('%H:%M:%S')})")
        print(f"   Duration: {self.duration_min:.1f} min")
        print(f"   Affected: {len(self.affected_servers)}/{len(fleet_servers)} servers")
        print(f"   Phases: Lead-in({self.lead_in_duration:.1f}m) → Peak({self.peak_duration:.1f}m) → Recovery({self.recovery_duration:.1f}m)")
    
    def phase_at(self, timestamp: pd.Timestamp) -> str:
        """Get current incident phase."""
        if timestamp < self.incident_start:
            return 'none'
        elif timestamp < self.peak_start:
            return 'lead_in'
        elif timestamp < self.recovery_start:
            return 'peak'
        elif timestamp < self.incident_end:
            return 'recovery'
        else:
            return 'none'
    
    def time_remaining(self, timestamp: pd.Timestamp) -> Optional[float]:
        """Get minutes remaining in current incident."""
        if self.incident_start <= timestamp <= self.incident_end:
            return (self.incident_end - timestamp).total_seconds() / 60
        return None
    
    def apply_effects(self, df_tick: pd.DataFrame) -> pd.DataFrame:
        """Apply incident effects to the tick batch."""
        current_time = pd.to_datetime(df_tick['timestamp'].iloc[0])
        phase = self.phase_at(current_time)
        
        # Add incident phase column
        df_tick['incident_phase'] = phase
        
        if phase == 'none':
            return df_tick
        
        # Calculate intensity based on phase
        if phase == 'lead_in':
            progress = (current_time - self.incident_start).total_seconds() / 60 / self.lead_in_duration
            intensity = 0.3 + 0.4 * progress  # 0.3 → 0.7
        elif phase == 'peak':
            intensity = 0.8 + 0.2 * np.random.random()  # 0.8 → 1.0
        else:  # recovery
            progress = (current_time - self.recovery_start).total_seconds() / 60 / self.recovery_duration
            intensity = 0.7 * (1 - progress)  # 0.7 → 0.0
        
        # Apply type-specific effects
        affected_mask = df_tick['server_name'].isin(self.affected_servers)
        
        if self.incident_type == 'db_contention':
            self._apply_db_contention(df_tick, affected_mask, intensity)
        elif self.incident_type == 'net_partition':
            self._apply_net_partition(df_tick, affected_mask, intensity)
        elif self.incident_type == 'cache_thundering_herd':
            self._apply_cache_thundering_herd(df_tick, affected_mask, intensity)
        elif self.incident_type == 'compute_batch_overrun':
            self._apply_compute_batch_overrun(df_tick, affected_mask, intensity)
        
        return df_tick
    
    def _apply_db_contention(self, df: pd.DataFrame, mask: pd.Series, intensity: float):
        """Apply database contention effects."""
        df.loc[mask, 'latency_ms'] *= (1 + intensity * 2)
        df.loc[mask, 'error_rate'] *= (1 + intensity * 3)
        df.loc[mask, 'disk_io_mb_s'] *= (1 + intensity * 1.5)
        
        # Some servers may go critical during peak
        if intensity > 0.8:
            critical_mask = mask & (np.random.random(len(df)) < 0.1)
            df.loc[critical_mask, 'state'] = 'critical_issue'
    
    def _apply_net_partition(self, df: pd.DataFrame, mask: pd.Series, intensity: float):
        """Apply network partition effects."""
        df.loc[mask, 'net_in_mb_s'] *= (1 - intensity * 0.6)
        df.loc[mask, 'net_out_mb_s'] *= (1 - intensity * 0.6)
        df.loc[mask, 'latency_ms'] *= (1 + intensity * 4)
        df.loc[mask, 'error_rate'] *= (1 + intensity * 5)
        
        # Some servers go offline
        if intensity > 0.6:
            offline_mask = mask & (np.random.random(len(df)) < 0.05)
            df.loc[offline_mask, 'state'] = 'offline'
    
    def _apply_cache_thundering_herd(self, df: pd.DataFrame, mask: pd.Series, intensity: float):
        """Apply cache thundering herd effects."""
        df.loc[mask, 'cpu_pct'] *= (1 + intensity * 1.5)
        df.loc[mask, 'container_oom'] = np.where(
            mask & (np.random.random(len(df)) < intensity * 0.1), 1, df['container_oom']
        )
        df.loc[mask, 'latency_ms'] *= (1 + intensity * np.random.uniform(0.5, 2.0, mask.sum()))
    
    def _apply_compute_batch_overrun(self, df: pd.DataFrame, mask: pd.Series, intensity: float):
        """Apply compute batch overrun effects."""
        compute_mask = mask & (df['profile'] == 'compute')
        prod_mask = (df['profile'] == 'production')
        
        df.loc[compute_mask, 'cpu_pct'] *= (1 + intensity * 2)
        df.loc[compute_mask, 'mem_pct'] *= (1 + intensity * 1.5)
        
        # Spillover to production
        df.loc[prod_mask, 'latency_ms'] *= (1 + intensity * 0.5)

# =============================================================================
# MODEL ADAPTER
# =============================================================================

class ModelAdapter:
    """Stub model for risk prediction."""
    
    def predict_per_server(self, df_window: pd.DataFrame) -> pd.DataFrame:
        """Predict 30-minute risk per server."""
        if df_window.empty:
            return pd.DataFrame(columns=['server_name', 'risk_30m'])
        
        # Calculate risk based on recent metrics
        recent_window = df_window.tail(20)  # Last 20 ticks
        server_risks = []
        
        for server in df_window['server_name'].unique():
            server_data = recent_window[recent_window['server_name'] == server]
            
            if server_data.empty:
                risk = 0.1
            else:
                # Risk factors
                cpu_risk = (server_data['cpu_pct'].mean() - 40) / 60
                latency_risk = (server_data['latency_ms'].mean() - 50) / 200
                error_risk = server_data['error_rate'].mean() * 2
                state_risk = 0.8 if (server_data['state'] == 'critical_issue').any() else 0
                offline_risk = 0.5 if (server_data['state'] == 'offline').any() else 0
                incident_risk = 0.3 if (server_data['incident_phase'] != 'none').any() else 0
                
                risk = max(0, min(1, cpu_risk + latency_risk + error_risk + state_risk + offline_risk + incident_risk))
            
            server_risks.append({'server_name': server, 'risk_30m': risk})
        
        return pd.DataFrame(server_risks)
    
    def predict_environment(self, df_window: pd.DataFrame) -> Dict[str, float]:
        """Predict environment incident probabilities."""
        if df_window.empty:
            return {'prob_30m': 0.1, 'prob_8h': 0.2}
        
        recent = df_window.tail(50)  # Last 50 ticks
        
        # Environment risk indicators
        p95_latency = recent['latency_ms'].quantile(0.95)
        mean_error_rate = recent['error_rate'].mean()
        critical_fraction = (recent['state'] == 'critical_issue').mean()
        offline_fraction = (recent['state'] == 'offline').mean()
        incident_active = (recent['incident_phase'] != 'none').any()
        
        # Calculate probabilities
        latency_factor = min(0.4, p95_latency / 500)
        error_factor = min(0.3, mean_error_rate * 10)
        critical_factor = critical_fraction * 0.5
        offline_factor = offline_fraction * 0.6
        incident_factor = 0.7 if incident_active else 0
        
        prob_30m = min(0.95, latency_factor + error_factor + critical_factor + offline_factor + incident_factor)
        prob_8h = min(0.90, prob_30m * 0.8 + 0.2)  # 8h is generally higher baseline
        
        return {'prob_30m': prob_30m, 'prob_8h': prob_8h}

# =============================================================================
# DASHBOARD
# =============================================================================

class LiveDashboard:
    """Real-time monitoring dashboard."""
    
    def __init__(self, data_generator=None, model_adapter=None, config=None):
        self.config = config or CONFIG
        self.data_generator = data_generator or FleetDataGenerator(self.config['FLEET_SIZE'], self.config['SEED'])
        self.model_adapter = model_adapter or ModelAdapter()
        
        # Initialize with first batch to get server list
        initial_batch = self.data_generator.get_tick_batch()
        server_list = initial_batch['server_name'].tolist()
        
        self.event_orchestrator = EventOrchestrator(server_list, self.config['SEED'])
        self.df_window = pd.DataFrame()
        self.env_prob_history = []
        self.server_order = sorted(server_list)  # Stable ordering for plots
        
        # Dashboard state
        self.start_time = pd.Timestamp.utcnow()
        self.last_refresh = self.start_time
        self.tick_count = 0
        
        # Setup matplotlib
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.size'] = 10
        
        if self.config['SAVE_PLOTS']:
            Path(self.config['PLOT_DIR']).mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Main dashboard loop."""
        print(f"🚀 Starting TFT Monitoring Dashboard")
        print(f"   Runtime: {self.config['TOTAL_RUNTIME_MIN']} minutes")
        print(f"   Refresh: every {self.config['REFRESH_SECONDS']} seconds")
        print(f"   Fleet size: {self.config['FLEET_SIZE']} servers")
        print("=" * 60)
        
        end_time = self.start_time + timedelta(minutes=self.config['TOTAL_RUNTIME_MIN'])
        
        try:
            while pd.Timestamp.utcnow() < end_time:
                # Get new tick
                tick_batch = self.data_generator.get_tick_batch()
                tick_batch = self.event_orchestrator.apply_effects(tick_batch)
                
                # Update window
                self._update_window(tick_batch)
                self.tick_count += 1
                
                # Log incident phase if active
                current_phase = self.event_orchestrator.phase_at(pd.Timestamp.utcnow())
                if current_phase != 'none':
                    remaining = self.event_orchestrator.time_remaining(pd.Timestamp.utcnow())
                    print(f"[{pd.Timestamp.utcnow().strftime('%H:%M:%S')}] Incident: {current_phase} ({remaining:.1f}min remaining)")
                
                # Refresh dashboard if needed
                if (pd.Timestamp.utcnow() - self.last_refresh).total_seconds() >= self.config['REFRESH_SECONDS']:
                    self._refresh_dashboard()
                    self.last_refresh = pd.Timestamp.utcnow()
                
                time.sleep(self.config['TICK_SECONDS'])
                
        except KeyboardInterrupt:
            print("\n⏹️  Dashboard stopped by user")
        
        self._print_summary()
    
    def _update_window(self, tick_batch: pd.DataFrame):
        """Update rolling window."""
        self.df_window = pd.concat([self.df_window, tick_batch], ignore_index=True)
        
        # Remove old data
        cutoff_time = pd.Timestamp.utcnow() - timedelta(minutes=self.config['ROLLING_WINDOW_MIN'])
        self.df_window = self.df_window[pd.to_datetime(self.df_window['timestamp']) > cutoff_time]
    
    def _refresh_dashboard(self):
        """Refresh all dashboard figures."""
        if self.df_window.empty:
            return
        
        # Get predictions
        server_risks = self.model_adapter.predict_per_server(self.df_window)
        env_probs = self.model_adapter.predict_environment(self.df_window)
        
        # Store environment probability history
        self.env_prob_history.append({
            'timestamp': pd.Timestamp.utcnow(),
            'prob_30m': env_probs['prob_30m'],
            'prob_8h': env_probs['prob_8h']
        })
        
        # Limit history size
        if len(self.env_prob_history) > self.config['MAX_ENV_POINTS']:
            self.env_prob_history = self.env_prob_history[-self.config['MAX_ENV_POINTS']:]
        
        # Clear and refresh display
        clear_output(wait=True)
        
        # Create all figures
        self._create_kpi_figure(env_probs)
        self._create_problem_servers_figure(server_risks)
        self._create_env_probability_figure()
        self._create_fleet_risk_strip(server_risks)
        self._create_rolling_metrics_figure()
        
        if self.config['SAVE_PLOTS']:
            self._save_all_plots()
    
    def _create_kpi_figure(self, env_probs: Dict[str, float]):
        """Figure 1: Header KPIs (text-only)."""
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        
        # Environment health status
        prob_30m = env_probs['prob_30m']
        if prob_30m < 0.2:
            health_status = "GOOD"
            health_color = 'green'
        elif prob_30m < 0.5:
            health_status = "WARNING"
            health_color = 'orange'
        else:
            health_status = "CRITICAL"
            health_color = 'red'
        
        # Fleet stats
        active_servers = (self.df_window['state'] != 'offline').sum()
        fleet_size = len(self.df_window['server_name'].unique())
        
        # Incident info
        current_phase = self.event_orchestrator.phase_at(pd.Timestamp.utcnow())
        remaining = self.event_orchestrator.time_remaining(pd.Timestamp.utcnow())
        
        # Layout KPIs
        kpi_text = f"""
ENVIRONMENT HEALTH: {health_status}                    INCIDENT PROBABILITY
Incident Type: {self.event_orchestrator.incident_type}                    30 min: {prob_30m:.1%}
Fleet: {active_servers}/{fleet_size} servers active                    8 hour: {env_probs['prob_8h']:.1%}
Current Time: {pd.Timestamp.utcnow().strftime('%H:%M:%S UTC')}
Refresh: every {self.config['REFRESH_SECONDS']}s                    Ticks: {self.tick_count}
"""
        
        if current_phase != 'none':
            kpi_text += f"\nACTIVE INCIDENT: {current_phase.upper()} ({remaining:.1f} min remaining)"
        
        ax.text(0.05, 0.5, kpi_text, fontsize=14, fontfamily='monospace', 
                verticalalignment='center', transform=ax.transAxes)
        
        # Health status highlight
        ax.text(0.25, 0.8, health_status, fontsize=24, fontweight='bold', 
                color=health_color, transform=ax.transAxes)
        
        plt.title('TFT Monitoring Dashboard - System Status', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
    
    def _create_problem_servers_figure(self, server_risks: pd.DataFrame):
        """Figure 2: Top-5 Problem Servers."""
        if server_risks.empty:
            return
        
        top_5 = server_risks.nlargest(5, 'risk_30m')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.barh(range(len(top_5)), top_5['risk_30m'], 
                      color=['red' if x > 0.7 else 'orange' if x > 0.4 else 'yellow' for x in top_5['risk_30m']])
        
        # Add server info labels
        labels = []
        for _, row in top_5.iterrows():
            server_info = self.df_window[self.df_window['server_name'] == row['server_name']].iloc[-1]
            labels.append(f"{row['server_name']} ({server_info['profile']} | {server_info['state']})")
        
        ax.set_yticks(range(len(top_5)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('30-Minute Risk Score')
        ax.set_title('Top 5 Problem Servers', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add value annotations
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _create_env_probability_figure(self):
        """Figure 3: Environment Incident Probability Over Time."""
        if len(self.env_prob_history) < 2:
            return
        
        df_probs = pd.DataFrame(self.env_prob_history)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df_probs['timestamp'], df_probs['prob_30m'], 
               label='30-min probability', linewidth=2, color='red')
        ax.plot(df_probs['timestamp'], df_probs['prob_8h'], 
               label='8-hour probability', linewidth=2, color='blue')
        
        ax.set_ylabel('Incident Probability')
        ax.set_title('Environment Incident Probability Trend', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Format x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _create_fleet_risk_strip(self, server_risks: pd.DataFrame):
        """Figure 4: Fleet Risk Strip."""
        if server_risks.empty:
            return
        
        # Create risk array in stable server order
        risk_array = []
        for server in self.server_order:
            risk_row = server_risks[server_risks['server_name'] == server]
            risk_value = risk_row['risk_30m'].iloc[0] if not risk_row.empty else 0.0
            risk_array.append(risk_value)
        
        risk_array = np.array(risk_array).reshape(1, -1)
        
        fig, ax = plt.subplots(figsize=(15, 3))
        
        im = ax.imshow(risk_array, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        # Server labels (every 5th server to avoid clutter)
        server_ticks = list(range(0, len(self.server_order), 5))
        server_labels = [self.server_order[i].split('-')[0] for i in server_ticks]
        
        ax.set_xticks(server_ticks)
        ax.set_xticklabels(server_labels, rotation=45)
        ax.set_yticks([])
        ax.set_title('Fleet Risk Heat Map (0=Low Risk, 1=High Risk)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.1)
        cbar.set_label('Risk Level')
        
        plt.tight_layout()
        plt.show()
    
    def _create_rolling_metrics_figure(self):
        """Figure 5: Rolling Fleet Metrics."""
        if len(self.df_window) < 10:
            return
        
        # Downsample for plotting (every minute)
        df_metrics = self.df_window.copy()
        df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'])
        df_metrics = df_metrics.set_index('timestamp').resample('1min').agg({
            'cpu_pct': 'median',
            'latency_ms': lambda x: x.quantile(0.95),
            'error_rate': 'mean'
        }).dropna()
        
        if len(df_metrics) < 2:
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # CPU Usage
        axes[0].plot(df_metrics.index, df_metrics['cpu_pct'], color='blue', linewidth=2)
        axes[0].set_ylabel('CPU %')
        axes[0].set_title('Fleet Rolling Metrics (60-min window)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 100)
        
        # Latency P95
        axes[1].plot(df_metrics.index, df_metrics['latency_ms'], color='orange', linewidth=2)
        axes[1].set_ylabel('Latency P95 (ms)')
        axes[1].grid(True, alpha=0.3)
        
        # Error Rate
        axes[2].plot(df_metrics.index, df_metrics['error_rate'], color='red', linewidth=2)
        axes[2].set_ylabel('Error Rate')
        axes[2].set_xlabel('Time')
        axes[2].grid(True, alpha=0.3)
        
        # Format x-axis
        import matplotlib.dates as mdates
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def _save_all_plots(self):
        """Save current plots as PNG files."""
        timestamp = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
        save_dir = Path(self.config['PLOT_DIR'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # This would save the current figures - implementation depends on requirements
        pass
    
    def _print_summary(self):
        """Print final dashboard summary."""
        runtime = pd.Timestamp.utcnow() - self.start_time
        
        print("\n" + "="*60)
        print("📊 DASHBOARD SUMMARY")
        print("="*60)
        print(f"Runtime: {runtime}")
        print(f"Ticks processed: {self.tick_count}")
        print(f"Incident type: {self.event_orchestrator.incident_type}")
        print(f"Final window size: {len(self.df_window)} records")
        
        if not self.df_window.empty:
            final_env_probs = self.model_adapter.predict_environment(self.df_window)
            server_risks = self.model_adapter.predict_per_server(self.df_window)
            
            print(f"Final environment probabilities:")
            print(f"  30-min: {final_env_probs['prob_30m']:.1%}")
            print(f"  8-hour: {final_env_probs['prob_8h']:.1%}")
            
            if not server_risks.empty:
                top_3 = server_risks.nlargest(3, 'risk_30m')
                print(f"Top 3 risk servers:")
                for _, row in top_3.iterrows():
                    print(f"  {row['server_name']}: {row['risk_30m']:.2f}")

# =============================================================================
# MODULE INTERFACES
# =============================================================================

def run_dashboard(runtime_min: int = 20, refresh_sec: int = 60, fleet_size: int = 25, save_plots: bool = False):
    """Run dashboard with custom parameters - for Jupyter usage."""
    config = CONFIG.copy()
    config.update({
        'TOTAL_RUNTIME_MIN': runtime_min,
        'REFRESH_SECONDS': refresh_sec,
        'FLEET_SIZE': fleet_size,
        'SAVE_PLOTS': save_plots
    })
    
    dashboard = LiveDashboard(config=config)
    dashboard.run()

def create_dashboard_components():
    """Create dashboard components for external use."""
    generator = FleetDataGenerator()
    model = ModelAdapter()
    return generator, model

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="TFT Real-time Monitoring Dashboard")
    parser.add_argument("--runtime", type=int, default=20, help="Runtime in minutes")
    parser.add_argument("--refresh", type=int, default=60, help="Refresh interval in seconds")
    parser.add_argument("--fleet-size", type=int, default=25, help="Number of servers in fleet")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    config = CONFIG.copy()
    config.update({
        'TOTAL_RUNTIME_MIN': args.runtime,
        'REFRESH_SECONDS': args.refresh,
        'FLEET_SIZE': args.fleet_size,
        'SAVE_PLOTS': args.save_plots,
        'SEED': args.seed
    })
    
    try:
        dashboard = LiveDashboard(config=config)
        dashboard.run()
        return 0
    except Exception as e:
        print(f"❌ Dashboard failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

# =============================================================================
# JUPYTER CELL EXAMPLE
# =============================================================================

"""
# JUPYTER USAGE EXAMPLE:
# Run this cell to start the dashboard

# Import and run with default settings
from tft_dashboard import run_dashboard
run_dashboard(runtime_min=10, refresh_sec=30, fleet_size=20)

# Or create custom dashboard
from tft_dashboard import LiveDashboard, CONFIG
config = CONFIG.copy()
config['TOTAL_RUNTIME_MIN'] = 15
config['REFRESH_SECONDS'] = 45
config['FLEET_SIZE'] = 30
config['SAVE_PLOTS'] = True

dashboard = LiveDashboard(config=config)
dashboard.run()
"""