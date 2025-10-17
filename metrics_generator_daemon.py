#!/usr/bin/env python3
"""
Metrics Generator Daemon - Stream Mode with REST API

Runs as a service that:
1. Continuously generates realistic metrics using metrics_generator.py logic
2. Streams data to inference daemon every 5 seconds
3. Accepts scenario changes via REST API (healthy/degrading/critical)
4. Dashboard can change scenarios on-the-fly without restart

Usage:
    python metrics_generator_daemon.py --stream --servers 20 --port 8001
"""

import asyncio
import time
import os
from datetime import datetime
from typing import Dict, List, Any

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our awesome metrics generator logic (which imports from config/)
from metrics_generator import (
    Config, make_server_fleet, ServerState, ServerProfile,
    PROFILE_BASELINES, STATE_MULTIPLIERS,
    get_state_transition_probs, diurnal_multiplier
)

# Import API configuration (SINGLE SOURCE OF TRUTH)
from config.api_config import API_CONFIG

# =============================================================================
# CONFIGURATION
# =============================================================================

GENERATOR_PORT = API_CONFIG['metrics_generator_port']  # 8001
INFERENCE_URL = API_CONFIG['daemon_url']                # http://localhost:8000
TICK_INTERVAL = API_CONFIG['streaming']['tick_interval']  # 5 seconds

# =============================================================================
# DATA MODELS
# =============================================================================

class ScenarioChangeRequest(BaseModel):
    """Request to change scenario mode."""
    scenario: str  # "healthy", "degrading", or "critical"

# =============================================================================
# METRICS GENERATOR DAEMON
# =============================================================================

class MetricsGeneratorDaemon:
    """
    Daemon that continuously generates and streams metrics.

    Features:
    - Uses metrics_generator.py logic (ONE source of truth!)
    - Streams to inference daemon every 5 seconds
    - Accepts scenario changes via REST API
    - Smooth transitions between scenarios
    """

    def __init__(self, config: Config, inference_url: str = INFERENCE_URL):
        self.config = config
        self.inference_url = inference_url
        self.tick_count = 0
        self.start_time = datetime.now()

        # Load API key from environment
        self.api_key = os.getenv("TFT_API_KEY")
        if not self.api_key:
            print("[WARNING] TFT_API_KEY not set - authentication may fail")
        else:
            print(f"[OK] API key loaded: {self.api_key[:8]}...")

        # Current scenario mode
        self.scenario = "healthy"
        self.scenario_start_tick = 0  # Track when scenario started for gradual trending

        # Scenario-based gradual degradation (realistic trending patterns)
        self.scenario_config = {
            'healthy': {
                'use_trending': False,  # Natural state transitions
                'affected_pct': 0.0,    # No servers affected
                'description': 'Normal operations - natural state transitions'
            },
            'degrading': {
                'use_trending': True,   # Gradual climb over time
                'affected_pct': 0.20,   # 20% of fleet (3-4 servers out of 20)
                'trending': {
                    'duration_ticks': 60,  # Ramp up over 60 ticks (5 minutes)
                    'cpu_target': 0.75,    # Climb to 75% CPU (not 100%)
                    'mem_target': 0.80,    # Climb to 80% memory
                    'iowait_target': 0.18, # DB servers: I/O wait climbs to 18%
                    'description': 'Gradual degradation - metrics climb over 5 minutes to concerning levels'
                },
                'description': '3-4 servers showing gradual resource climb (5min ramp) - early warning'
            },
            'critical': {
                'use_trending': True,   # Fast climb to critical
                'escalate_from_degrading': True,  # Check if coming from degrading scenario
                'affected_pct_fresh': 0.25,  # 25% (~5 servers) if jumping from healthy
                'trending_fresh': {
                    'duration_ticks': 24,  # 2 minutes for fresh critical
                    'cpu_target': 0.90,    # Climb to 90% CPU
                    'mem_target': 0.93,    # Climb to 93% memory
                    'iowait_target': 0.35, # I/O bottleneck
                    'description': 'Fresh critical - 5 servers rapidly degrading to critical levels'
                },
                'trending_escalation': {
                    'duration_ticks': 12,  # 1 minute rapid escalation for already-degrading servers
                    'cpu_target': 0.92,    # Push higher to 92% CPU
                    'mem_target': 0.95,    # Push to 95% memory (near OOM)
                    'iowait_target': 0.40, # Severe I/O bottleneck
                    'description': 'Escalation - degrading servers rapidly worsen to critical'
                },
                'description': 'Critical incident - servers rapidly reach critical resource levels'
            }
        }

        # Create fleet (matches training data exactly)
        print(f"[INIT] Creating server fleet...")
        self.fleet = make_server_fleet(config)
        print(f"[OK] Fleet created: {len(self.fleet)} servers")

        # Initialize server states
        self.server_states = {name: ServerState.HEALTHY for name in self.fleet['server_name']}

        # Track affected servers for degrading/critical scenarios
        self.affected_servers = set()
        self.update_affected_servers()

        # Streaming control
        self.running = False

    def update_affected_servers(self, previous_scenario: str = None):
        """Update which servers are affected by current scenario."""
        scenario_conf = self.scenario_config[self.scenario]

        # Special handling for CRITICAL scenario
        if self.scenario == 'critical' and scenario_conf.get('escalate_from_degrading'):
            if previous_scenario == 'degrading' and len(self.affected_servers) > 0:
                # ESCALATION PATH: Keep existing degrading servers, they get worse
                self._came_from_degrading = True
                print(f"[SCENARIO] {self.scenario.upper()}: ESCALATION from degrading")
                print(f"           {len(self.affected_servers)} servers already degrading will rapidly worsen")
                trending = scenario_conf['trending_escalation']
            else:
                # FRESH PATH: Jumping from healthy → critical, pick new servers
                self._came_from_degrading = False
                affected_pct = scenario_conf['affected_pct_fresh']
                num_affected = max(1, int(len(self.fleet) * affected_pct))
                self.affected_servers = set(np.random.choice(
                    self.fleet['server_name'],
                    num_affected,
                    replace=False
                ))
                print(f"[SCENARIO] {self.scenario.upper()}: FRESH critical incident")
                print(f"           {num_affected} servers randomly selected for rapid degradation")
                trending = scenario_conf['trending_fresh']

            # Show trending info
            duration_sec = trending['duration_ticks'] * TICK_INTERVAL
            print(f"           Rapid trending over {duration_sec:.0f} seconds")
            print(f"           CPU→{trending['cpu_target']*100:.0f}%, MEM→{trending['mem_target']*100:.0f}%")
            print(f"           Description: {trending['description']}")
            return

        # Standard scenario handling (healthy, degrading)
        affected_pct = scenario_conf.get('affected_pct', 0.0)

        if affected_pct == 0.0:
            self.affected_servers = set()
            print(f"[SCENARIO] {self.scenario.upper()}: All servers healthy, natural transitions")
        else:
            # Select affected servers based on scenario config
            num_affected = max(1, int(len(self.fleet) * affected_pct))
            self.affected_servers = set(np.random.choice(
                self.fleet['server_name'],
                num_affected,
                replace=False
            ))

            print(f"[SCENARIO] {self.scenario.upper()}: {num_affected} servers affected")

            # Show trending info if applicable
            if scenario_conf.get('use_trending'):
                trending = scenario_conf['trending']
                duration_sec = trending['duration_ticks'] * TICK_INTERVAL
                print(f"           Gradual trending over {duration_sec:.0f} seconds")
                print(f"           CPU: {trending['cpu_target']*100:.0f}%, MEM: {trending['mem_target']*100:.0f}%")

            print(f"           Description: {scenario_conf['description']}")

    def set_scenario(self, scenario: str) -> Dict[str, Any]:
        """
        Change the current scenario mode.

        Args:
            scenario: "healthy", "degrading", or "critical"

        Returns:
            Status dict
        """
        if scenario not in ['healthy', 'degrading', 'critical']:
            return {"status": "error", "message": f"Invalid scenario: {scenario}"}

        old_scenario = self.scenario
        self.scenario = scenario
        self.scenario_start_tick = self.tick_count  # Track when scenario started for trending
        self.update_affected_servers(previous_scenario=old_scenario)

        print(f"\n[SCENARIO CHANGE] {old_scenario.upper()} → {scenario.upper()}")
        if self.affected_servers:
            print(f"   Affected servers: {', '.join(list(self.affected_servers)[:5])}{'...' if len(self.affected_servers) > 5 else ''}")
        else:
            print(f"   All servers healthy")

        # Show trending info if applicable
        scenario_conf = self.scenario_config[scenario]
        if scenario_conf.get('use_trending'):
            trending = scenario_conf['trending']
            duration_sec = trending['duration_ticks'] * TICK_INTERVAL
            print(f"   Trending: Metrics will climb over {duration_sec:.0f} seconds")
            print(f"   Targets: CPU→{trending['cpu_target']*100:.0f}%, MEM→{trending['mem_target']*100:.0f}%")
        print()

        return {
            "status": "success",
            "old_scenario": old_scenario,
            "new_scenario": scenario,
            "affected_servers": len(self.affected_servers),
            "tick_count": self.tick_count
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status."""
        return {
            "running": self.running,
            "scenario": self.scenario,
            "tick_count": self.tick_count,
            "fleet_size": len(self.fleet),
            "affected_servers": len(self.affected_servers),
            "inference_url": self.inference_url
        }

    def generate_tick(self) -> List[Dict]:
        """
        Generate one tick of data for entire fleet.
        Uses gradual trending multipliers for realistic degradation patterns.
        """
        self.tick_count += 1
        current_time = datetime.now()
        hour = current_time.hour

        scenario_conf = self.scenario_config[self.scenario]
        batch = []

        # Calculate trending progress (0.0 to 1.0) based on ticks since scenario started
        trending_progress = 0.0
        active_trending = None
        if scenario_conf.get('use_trending'):
            # Determine which trending config to use (critical has two paths)
            if self.scenario == 'critical':
                # Check if we have degrading servers (escalation path)
                if hasattr(self, '_came_from_degrading') and self._came_from_degrading:
                    active_trending = scenario_conf['trending_escalation']
                else:
                    active_trending = scenario_conf['trending_fresh']
            else:
                active_trending = scenario_conf.get('trending')

            if active_trending:
                ticks_elapsed = self.tick_count - self.scenario_start_tick
                duration_ticks = active_trending['duration_ticks']
                trending_progress = min(1.0, ticks_elapsed / duration_ticks)  # Clamp to 1.0

        for _, server in self.fleet.iterrows():
            server_name = server['server_name']
            profile = server['profile']
            is_problem_child = server['problem_child']
            is_affected = server_name in self.affected_servers

            # Natural state transitions (no more forced states)
            current_state = self.server_states[server_name]
            probs = get_state_transition_probs(current_state, hour, is_problem_child)
            states = list(probs.keys())
            probabilities = list(probs.values())
            next_state = np.random.choice(states, p=probabilities)

            self.server_states[server_name] = next_state

            # Skip offline servers (sparse mode for streaming)
            # Offline = no metrics sent (standard monitoring handles this)
            if next_state == ServerState.OFFLINE:
                continue

            # Generate ALL 14 LINBORG metrics using profile baselines + state multipliers + diurnal
            profile_enum = ServerProfile(profile)
            baselines = PROFILE_BASELINES[profile_enum]

            # LINBORG metrics - use exact names AS THEY APPEAR in PROFILE_BASELINES (no _pct suffix yet)
            linborg_metrics = [
                'cpu_user', 'cpu_sys', 'cpu_iowait', 'cpu_idle', 'java_cpu',
                'mem_used', 'swap_used', 'disk_usage',
                'net_in_mb_s', 'net_out_mb_s',
                'back_close_wait', 'front_close_wait',
                'load_average', 'uptime_days'
            ]

            metrics = {}
            for metric in linborg_metrics:
                if metric in baselines:
                    mean, std = baselines[metric]
                    value = np.random.normal(mean, std)

                    # Apply state multiplier (NATURAL)
                    multiplier = STATE_MULTIPLIERS[next_state].get(metric, 1.0)
                    value *= multiplier

                    # Apply diurnal pattern
                    diurnal_mult = diurnal_multiplier(hour, profile_enum, next_state)
                    value *= diurnal_mult

                    # Apply GRADUAL TRENDING for affected servers (realistic degradation)
                    if is_affected and active_trending:
                        # Calculate trending multiplier for this metric
                        if metric == 'cpu_user':
                            # Gradually climb CPU toward target
                            target = active_trending['cpu_target']
                            current_baseline = mean * multiplier * diurnal_mult
                            # Interpolate from current to target over trending_progress
                            target_value = target
                            trending_value = current_baseline + (target_value - current_baseline) * trending_progress
                            value = trending_value

                        elif metric == 'mem_used':
                            # Gradually climb memory toward target
                            target = active_trending['mem_target']
                            current_baseline = mean * multiplier * diurnal_mult
                            target_value = target
                            trending_value = current_baseline + (target_value - current_baseline) * trending_progress
                            value = trending_value

                        elif metric == 'cpu_iowait' and profile == 'database':
                            # DB servers: I/O wait climbs (causes cascading issues)
                            target = active_trending['iowait_target']
                            current_baseline = mean * multiplier * diurnal_mult
                            target_value = target
                            trending_value = current_baseline + (target_value - current_baseline) * trending_progress
                            value = trending_value

                    # Apply bounds based on metric type
                    # NOTE: PROFILE_BASELINES uses fractional values (0-1) for cpu/mem
                    # We need to multiply by 100 and add _pct suffix for output
                    if metric in ['cpu_user', 'cpu_sys', 'cpu_iowait', 'cpu_idle', 'java_cpu',
                                 'mem_used', 'swap_used', 'disk_usage']:
                        # Fractional → Percentage: multiply by 100, add _pct suffix
                        value = np.clip(value * 100, 0, 100)
                        metrics[f'{metric}_pct'] = round(value, 2)
                    elif metric in ['back_close_wait', 'front_close_wait']:
                        # Connection counts: integers, non-negative
                        metrics[metric] = max(0, int(value))
                    elif metric == 'uptime_days':
                        # Uptime: 0-30 days, integer
                        metrics[metric] = int(np.clip(value, 0, 30))
                    elif metric in ['net_in_mb_s', 'net_out_mb_s']:
                        # Network: MB/s, non-negative floats (baselines already in MB/s)
                        metrics[metric] = round(max(0, value), 2)
                    elif metric == 'load_average':
                        # Load average: non-negative float (baselines already correct scale)
                        metrics[metric] = round(max(0, value), 2)

            # Build record
            record = {
                'timestamp': current_time.isoformat(),
                'server_name': server_name,
                'profile': profile,
                'state': next_state.value,
                'problem_child': bool(is_problem_child),
                **metrics,
                # OOM based on LINBORG mem_used_pct
                'container_oom': int(np.random.random() < 0.01 if metrics.get('mem_used_pct', 0) > 85 else 0),
                'notes': ''
            }

            batch.append(record)

        return batch

    async def stream_loop(self):
        """
        Main streaming loop - generates and sends data every tick.
        """
        self.running = True

        print(f"\n{'='*60}")
        print(f"🚀 STREAMING STARTED")
        print(f"{'='*60}")
        print(f"   Scenario: {self.scenario.upper()}")
        print(f"   Fleet: {len(self.fleet)} servers")
        print(f"   Target: {self.inference_url}")
        print(f"   Interval: {TICK_INTERVAL} seconds")
        print(f"{'='*60}\n")

        while self.running:
            tick_start = time.time()

            # Generate batch
            batch = self.generate_tick()

            # Send to inference daemon with API key
            try:
                headers = {}
                if self.api_key:
                    headers["X-API-Key"] = self.api_key

                response = requests.post(
                    f"{self.inference_url}/feed/data",
                    json={"records": batch},
                    headers=headers,
                    timeout=2
                )

                if response.ok:
                    status_icon = {
                        'healthy': '🟢',
                        'degrading': '🟡',
                        'critical': '🔴'
                    }[self.scenario]

                    # Calculate active vs offline
                    fleet_size = len(self.fleet)
                    active_count = len(batch)
                    offline_count = fleet_size - active_count

                    elapsed = (datetime.now() - self.start_time).total_seconds()

                    # Show breakdown: "19 active, 1 offline" or just "20 active" if all online
                    if offline_count > 0:
                        server_status = f"{active_count:2d} active, {offline_count} offline"
                    else:
                        server_status = f"{active_count:2d} active"

                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tick {self.tick_count:4d} | {status_icon} {self.scenario.upper():9s} | {server_status} | Elapsed: {elapsed:.0f}s")
                else:
                    print(f"⚠️  Inference daemon error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                print(f"⚠️  Cannot connect to inference daemon at {self.inference_url}")
                print(f"   Waiting for daemon to start...")
            except Exception as e:
                print(f"❌ Error: {e}")

            # Wait for next tick (compensate for processing time)
            elapsed = time.time() - tick_start
            sleep_time = max(0, TICK_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self):
        """Stop the streaming loop."""
        self.running = False
        print(f"\n⏹️  Streaming stopped after {self.tick_count} ticks")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

# Global daemon instance
daemon = None

# Create FastAPI app
app = FastAPI(
    title="Metrics Generator Daemon",
    description="Streams realistic metrics to inference daemon with scenario control",
    version="2.0"
)

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """Initialize daemon and start streaming."""
    global daemon

    # This will be set by main() before starting uvicorn
    # daemon is created with config passed from CLI args

    # Start streaming loop in background
    asyncio.create_task(daemon.stream_loop())

@app.post("/scenario/set")
async def set_scenario(request: ScenarioChangeRequest):
    """
    Change the current scenario mode.

    Called by dashboard when user clicks scenario buttons.

    Example:
        POST /scenario/set
        {"scenario": "degrading"}
    """
    return daemon.set_scenario(request.scenario)

@app.get("/scenario/status")
async def get_scenario_status():
    """Get current scenario status."""
    return {
        "scenario": daemon.scenario,
        "affected_servers": list(daemon.affected_servers)[:10],  # First 10 for brevity
        "total_affected": len(daemon.affected_servers),
        "fleet_size": len(daemon.fleet),
        "tick_count": daemon.tick_count
    }

@app.get("/status")
async def get_status():
    """Get daemon health status."""
    return daemon.get_status()

@app.get("/")
async def root():
    """Root endpoint with info."""
    return {
        "service": "Metrics Generator Daemon",
        "version": "2.0",
        "status": "streaming" if daemon.running else "stopped",
        "scenario": daemon.scenario,
        "fleet_size": len(daemon.fleet),
        "endpoints": {
            "set_scenario": "POST /scenario/set",
            "scenario_status": "GET /scenario/status",
            "status": "GET /status"
        }
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Start the daemon."""
    import argparse

    parser = argparse.ArgumentParser(description="Metrics Generator Daemon - Stream Mode")

    # Mode flag (for compatibility)
    parser.add_argument("--stream", action="store_true", help="Run in streaming mode (always true for daemon)")

    # Fleet configuration
    parser.add_argument("--servers", type=int, default=20, help="Total number of servers to generate")

    # Daemon configuration
    parser.add_argument("--port", type=int, default=GENERATOR_PORT, help=f"Port for REST API (default: {GENERATOR_PORT})")
    parser.add_argument("--inference-url", default=INFERENCE_URL, help=f"Inference daemon URL (default: {INFERENCE_URL})")
    parser.add_argument("--scenario", choices=["healthy", "degrading", "critical"], default="healthy",
                       help="Initial scenario (default: healthy)")

    args = parser.parse_args()

    # Create config (uses total_servers to match training)
    config = Config(
        total_servers=args.servers,
        tick_seconds=TICK_INTERVAL,
        problem_child_pct=0.10,
        seed=42
    )

    # Create daemon instance
    global daemon
    daemon = MetricsGeneratorDaemon(config, inference_url=args.inference_url)
    daemon.scenario = args.scenario
    daemon.update_affected_servers()

    print(f"\n{'='*60}")
    print(f"METRICS GENERATOR DAEMON")
    print(f"{'='*60}")
    print(f"   REST API Port: {args.port}")
    print(f"   Initial Scenario: {args.scenario.upper()}")
    print(f"   Fleet Size: {args.servers} servers")
    print(f"   Inference Target: {args.inference_url}")
    print(f"{'='*60}\n")

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",  # Reduce noise, our custom logging is cleaner
        access_log=False
    )

if __name__ == "__main__":
    main()
