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
from datetime import datetime
from typing import Dict, List, Any

import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our awesome metrics generator logic
from metrics_generator import (
    Config, make_server_fleet, ServerState, ServerProfile,
    PROFILE_BASELINES, STATE_MULTIPLIERS,
    get_state_transition_probs, diurnal_multiplier
)

# =============================================================================
# CONFIGURATION
# =============================================================================

GENERATOR_PORT = 8001
INFERENCE_URL = "http://localhost:8000"
TICK_INTERVAL = 5  # seconds

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

        # Current scenario mode
        self.scenario = "healthy"
        self.scenario_multipliers = {
            'healthy': 1.0,      # Normal operations
            'degrading': 1.15,   # Subtle increase (15%) - early warning signs
            'critical': 1.6      # Significant issues (60%) - clear problems
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

    def update_affected_servers(self):
        """Update which servers are affected by current scenario."""
        if self.scenario == 'healthy':
            self.affected_servers = set()
        else:
            # Affect 25% of fleet for degrading/critical
            num_affected = max(1, int(len(self.fleet) * 0.25))
            self.affected_servers = set(np.random.choice(
                self.fleet['server_name'],
                num_affected,
                replace=False
            ))
            print(f"[SCENARIO] {self.scenario.upper()}: {num_affected} servers affected")

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
        self.update_affected_servers()

        print(f"\n[SCENARIO CHANGE] {old_scenario.upper()} → {scenario.upper()}")
        if self.affected_servers:
            print(f"   Affected servers: {', '.join(list(self.affected_servers)[:5])}{'...' if len(self.affected_servers) > 5 else ''}")
        else:
            print(f"   All servers healthy")
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
        Uses the SAME awesome logic from metrics_generator.py!
        """
        self.tick_count += 1
        current_time = datetime.now()
        hour = current_time.hour

        scenario_mult = self.scenario_multipliers[self.scenario]
        batch = []

        for _, server in self.fleet.iterrows():
            server_name = server['server_name']
            profile = server['profile']
            is_problem_child = server['problem_child']
            is_affected = server_name in self.affected_servers

            # State transitions (using metrics_generator logic)
            current_state = self.server_states[server_name]
            probs = get_state_transition_probs(current_state, hour, is_problem_child)
            states = list(probs.keys())
            probabilities = list(probs.values())
            next_state = np.random.choice(states, p=probabilities)
            self.server_states[server_name] = next_state

            # Skip offline servers (sparse mode for streaming)
            if next_state == ServerState.OFFLINE:
                continue

            # Generate metrics using profile baselines
            profile_enum = ServerProfile(profile)
            baselines = PROFILE_BASELINES[profile_enum]

            metrics = {}
            for metric in ['cpu', 'mem', 'disk_io_mb_s', 'net_in_mb_s', 'net_out_mb_s',
                          'latency_ms', 'error_rate', 'gc_pause_ms']:
                if metric in baselines:
                    mean, std = baselines[metric]
                    value = np.random.normal(mean, std)

                    # Apply state multiplier
                    multiplier = STATE_MULTIPLIERS[next_state].get(metric, 1.0)
                    value *= multiplier

                    # Apply diurnal pattern
                    diurnal_mult = diurnal_multiplier(hour, profile_enum, next_state)
                    value *= diurnal_mult

                    # Apply scenario degradation for affected servers
                    if is_affected and self.scenario != 'healthy':
                        if metric in ['cpu', 'mem', 'latency_ms', 'error_rate', 'gc_pause_ms']:
                            value *= scenario_mult

                    # Apply bounds
                    if metric in ['cpu', 'mem']:
                        value = np.clip(value * 100, 0, 100)
                        metrics[f'{metric}_pct'] = round(value, 2)
                    else:
                        value = max(0, value)
                        metrics[metric] = round(value, 2)

            # Build record
            record = {
                'timestamp': current_time.isoformat(),
                'server_name': server_name,
                'profile': profile,
                'state': next_state.value,
                'problem_child': bool(is_problem_child),
                **metrics,
                'container_oom': int(np.random.random() < 0.01 if metrics.get('mem_pct', 0) > 85 else 0),
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

            # Send to inference daemon
            try:
                response = requests.post(
                    f"{self.inference_url}/feed/data",
                    json={"records": batch},
                    timeout=2
                )

                if response.ok:
                    status_icon = {
                        'healthy': '🟢',
                        'degrading': '🟡',
                        'critical': '🔴'
                    }[self.scenario]

                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Tick {self.tick_count:4d} | {status_icon} {self.scenario.upper():9s} | {len(batch):2d} servers | Elapsed: {elapsed:.0f}s")
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
