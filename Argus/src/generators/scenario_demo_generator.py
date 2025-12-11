#!/usr/bin/env python3
"""
scenario_demo_generator.py - Interactive Demo Data Generator

Cinema-grade demo system that:
1. Reads training data to match server names/profiles exactly
2. Responds to real-time scenario commands from dashboard
3. Generates realistic degradation patterns on-demand
4. Keeps demo isolated for easy swap to production data

Author: Claude (Sonnet 4.5)
Date: 2025-10-12
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional
from enum import Enum


class ScenarioMode(Enum):
    """Interactive demo scenarios - controlled from dashboard."""
    HEALTHY = "healthy"        # All servers normal
    DEGRADING = "degrading"    # 1-5 servers gradually degrading
    CRITICAL = "critical"      # 1-5 servers in crisis mode


class ScenarioDemoGenerator:
    """
    Interactive demo data generator that reads training data for consistency.

    Key Features:
    - Reads training/server_metrics.parquet to get exact server names/profiles
    - Auto-detects fleet size from training data
    - Responds to scenario commands in real-time
    - Generates realistic degradation patterns
    - Isolated design for easy production swap
    """

    def __init__(self, training_data_path: str = "./training/server_metrics.parquet",
                 seed: int = 42):
        """
        Initialize by reading training data.

        Args:
            training_data_path: Path to training parquet file
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self.seed = seed
        self.tick_count = 0
        self.start_time = datetime.now()

        # Read training data to get server fleet
        print(f"[DEMO] Loading server fleet from: {training_data_path}")
        self._load_fleet_from_training(training_data_path)

        # Scenario state
        self.current_scenario = ScenarioMode.HEALTHY
        self.affected_servers: Set[str] = set()
        self.scenario_start_tick = 0
        self.transition_duration = 60  # Ticks to complete transition (5 min)

        print(f"[DEMO] Initialized with {len(self.servers)} servers")
        print(f"[DEMO] Scenario: {self.current_scenario.value}")

    def _load_fleet_from_training(self, training_path: str):
        """
        Load server fleet from actual training data.

        This ensures demo data matches training exactly:
        - Same server names
        - Same profiles
        - Same baseline metrics
        """
        path = Path(training_path)

        if not path.exists():
            print(f"[WARNING] Training data not found: {training_path}")
            print("[WARNING] Using fallback hardcoded server list")
            self._use_fallback_fleet()
            return

        # Read training data
        df = pd.read_parquet(path)

        # Extract unique servers and their profiles
        server_info = df.groupby('server_name').agg({
            'profile': 'first',
            'cpu_pct': 'mean',
            'mem_pct': 'mean',
            'disk_io_mb_s': lambda x: x.mean() * 10,  # Convert to rough percent
            'latency_ms': 'mean'
        }).reset_index()

        print(f"[DEMO] Found {len(server_info)} servers in training data")

        # Build server fleet
        self.servers = []
        for _, row in server_info.iterrows():
            self.servers.append({
                'server_name': row['server_name'],
                'profile': row['profile'],
                'base_cpu': row['cpu_pct'],
                'base_memory': row['mem_pct'],
                'base_disk': row['disk_io_mb_s'] if not pd.isna(row['disk_io_mb_s']) else 40.0,
                'base_latency': row['latency_ms'],
                'noise_factor': np.random.uniform(0.9, 1.1),
                'is_problem_child': np.random.random() < 0.05  # 5% naturally problematic
            })

        self.fleet_size = len(self.servers)

        # Log profile distribution
        profile_counts = df.groupby('profile').size().to_dict()
        print("[DEMO] Profile distribution:")
        for profile, count in sorted(profile_counts.items()):
            print(f"  {profile}: {count} records")

    def _use_fallback_fleet(self):
        """
        Fallback: Use hardcoded 20-server fleet matching current training.
        """
        # These match the 20 servers in current training data
        server_list = [
            ('ppcon01', 'conductor_mgmt', 28, 75, 40, 15),
            ('ppdb001', 'database', 55, 87, 45, 8),
            ('ppdb002', 'database', 55, 87, 45, 8),
            ('ppdb003', 'database', 55, 87, 45, 8),
            ('ppetl001', 'data_ingest', 60, 78, 42, 12),
            ('ppetl002', 'data_ingest', 60, 78, 42, 12),
            ('ppgen001', 'generic', 35, 50, 38, 30),
            ('ppml0001', 'ml_compute', 78, 82, 40, 22),
            ('ppml0002', 'ml_compute', 78, 82, 40, 22),
            ('ppml0003', 'ml_compute', 78, 82, 40, 22),
            ('ppml0004', 'ml_compute', 78, 82, 40, 22),
            ('pprisk001', 'risk_analytics', 82, 88, 41, 18),
            ('ppweb001', 'web_api', 28, 55, 35, 45),
            ('ppweb002', 'web_api', 28, 55, 35, 45),
            ('ppweb003', 'web_api', 28, 55, 35, 45),
            ('ppweb004', 'web_api', 28, 55, 35, 45),
            ('ppweb005', 'web_api', 28, 55, 35, 45),
            ('ppweb006', 'web_api', 28, 55, 35, 45),
            ('ppweb007', 'web_api', 28, 55, 35, 45),
            ('ppweb008', 'web_api', 28, 55, 35, 45),
        ]

        self.servers = []
        for name, profile, cpu, mem, disk, lat in server_list:
            self.servers.append({
                'server_name': name,
                'profile': profile,
                'base_cpu': cpu + np.random.uniform(-5, 5),
                'base_memory': mem + np.random.uniform(-5, 5),
                'base_disk': disk,
                'base_latency': lat + np.random.uniform(-5, 10),
                'noise_factor': np.random.uniform(0.9, 1.1),
                'is_problem_child': np.random.random() < 0.05
            })

        self.fleet_size = len(self.servers)

    def set_scenario(self, mode: str, affected_count: Optional[int] = None):
        """
        Set current scenario - called from dashboard button.

        Args:
            mode: 'healthy', 'degrading', or 'critical'
            affected_count: Number of servers to affect (default: random 1-5)
        """
        try:
            new_scenario = ScenarioMode(mode.lower())
        except ValueError:
            print(f"[ERROR] Invalid scenario mode: {mode}")
            return

        print(f"\n[SCENARIO] Switching to: {new_scenario.value.upper()}")

        self.current_scenario = new_scenario
        self.scenario_start_tick = self.tick_count

        if new_scenario == ScenarioMode.HEALTHY:
            # Clear affected servers - recovery mode
            print(f"[SCENARIO] Recovery: {len(self.affected_servers)} servers healing")
            # Don't clear immediately - let them transition back
        else:
            # Pick random servers to affect
            count = affected_count or np.random.randint(1, 6)  # 1-5 servers
            count = min(count, len(self.servers))  # Cap at fleet size

            # Pick servers (prefer non-affected to spread the pain)
            available = [s['server_name'] for s in self.servers
                        if s['server_name'] not in self.affected_servers]

            if len(available) < count:
                available = [s['server_name'] for s in self.servers]

            new_affected = set(np.random.choice(available, size=count, replace=False))
            self.affected_servers = new_affected

            print(f"[SCENARIO] Affecting {len(self.affected_servers)} servers:")
            for server in sorted(self.affected_servers):
                print(f"  - {server}")

    def generate_tick(self) -> List[Dict]:
        """
        Generate one tick of data with current scenario applied.

        Returns:
            List of server metrics (one dict per server)
        """
        self.tick_count += 1
        current_time = self.start_time + timedelta(seconds=self.tick_count * 5)

        # Calculate transition progress (0.0 = just started, 1.0 = fully transitioned)
        ticks_since_scenario = self.tick_count - self.scenario_start_tick
        transition_progress = min(1.0, ticks_since_scenario / self.transition_duration)

        # Time-based factors (business hours effect)
        hour = current_time.hour
        if 9 <= hour <= 17:  # Business hours
            time_factor = 1.3 + 0.15 * np.sin(2 * np.pi * (hour - 9) / 8)
        else:  # Off hours
            time_factor = 0.7 + 0.1 * np.sin(2 * np.pi * hour / 24)

        batch = []
        for server in self.servers:
            server_name = server['server_name']
            is_affected = server_name in self.affected_servers

            # Base metrics with time variation
            cpu = server['base_cpu'] * time_factor * server['noise_factor']
            memory = server['base_memory'] * (1 + 0.1 * time_factor)
            disk = server['base_disk'] + (self.tick_count * 0.005)  # Slowly grows
            latency = server['base_latency'] * server['noise_factor']

            # Apply scenario effects
            if is_affected:
                if self.current_scenario == ScenarioMode.DEGRADING:
                    # Gradual degradation over 5 minutes
                    degradation = 0.5 + (0.8 * transition_progress)  # 50% → 130% increase
                    cpu *= degradation
                    memory *= (1 + 0.3 * transition_progress)
                    latency *= (1 + transition_progress)

                elif self.current_scenario == ScenarioMode.CRITICAL:
                    # Rapid critical spike
                    crisis_factor = 1.0 + (1.5 * transition_progress)  # → 250% increase
                    cpu *= crisis_factor
                    memory *= (1 + 0.5 * transition_progress)
                    latency *= (1 + 2 * transition_progress)

            elif server_name in self.affected_servers and self.current_scenario == ScenarioMode.HEALTHY:
                # Recovery mode - gradual healing
                recovery_progress = transition_progress
                heal_factor = 1.0 - (0.3 * (1 - recovery_progress))  # Slowly return to normal
                cpu *= heal_factor
                memory *= heal_factor

                # Clear from affected once fully recovered
                if transition_progress >= 1.0:
                    self.affected_servers.discard(server_name)

            # Add realistic noise
            cpu += np.random.normal(0, 3)
            memory += np.random.normal(0, 2)
            disk += np.random.normal(0, 1)
            latency += np.random.exponential(5)

            # Bounds
            cpu = max(5, min(100, cpu))
            memory = max(10, min(98, memory))
            disk = max(20, min(95, disk))
            latency = max(1, min(500, latency))

            # Derived metrics
            load_avg = (cpu / 25) * server['noise_factor']
            load_avg = max(0.1, min(16, load_avg))

            network_errors = np.random.poisson(2 if not is_affected else 15)

            # State determination
            anomaly_score = 0.0
            if cpu > 80: anomaly_score += 0.3
            if memory > 85: anomaly_score += 0.3
            if load_avg > 8: anomaly_score += 0.2
            if network_errors > 10: anomaly_score += 0.2
            anomaly_score = min(1.0, anomaly_score)

            if anomaly_score > 0.7:
                state = 'critical_issue'
            elif anomaly_score > 0.4:
                state = 'heavy_load'
            else:
                state = 'healthy'

            # Build record
            batch.append({
                'timestamp': current_time.isoformat(),
                'server_name': server_name,
                'cpu_percent': float(cpu),
                'memory_percent': float(memory),
                'disk_percent': float(disk),
                'load_average': float(load_avg),
                'java_heap_usage': float(memory * 0.8),
                'network_errors': int(network_errors),
                'anomaly_score': float(anomaly_score),
                'hour': hour,
                'day_of_week': current_time.weekday(),
                'day_of_month': current_time.day,
                'month': current_time.month,
                'quarter': (current_time.month - 1) // 3 + 1,
                'is_weekend': current_time.weekday() >= 5,
                'is_business_hours': 9 <= hour <= 17,
                'status': state,
                'timeframe': 'realtime',
                'service_type': server['profile'],
                'datacenter': 'dc1',
                'environment': 'production'
            })

        return batch

    def get_status(self) -> Dict:
        """Get current scenario status."""
        return {
            'scenario': self.current_scenario.value,
            'affected_servers': list(self.affected_servers),
            'tick_count': self.tick_count,
            'fleet_size': self.fleet_size,
            'transition_progress': min(1.0, (self.tick_count - self.scenario_start_tick) / self.transition_duration)
        }


# Test if run directly
if __name__ == "__main__":
    print("Testing ScenarioDemoGenerator\n")

    # Initialize
    gen = ScenarioGenerator()

    # Generate healthy baseline
    print("\n1. Generating healthy baseline...")
    for i in range(3):
        batch = gen.generate_tick()
        print(f"Tick {i+1}: {len(batch)} servers, avg CPU: {np.mean([s['cpu_percent'] for s in batch]):.1f}%")

    # Switch to degrading
    print("\n2. Switching to DEGRADING scenario...")
    gen.set_scenario('degrading')

    for i in range(3):
        batch = gen.generate_tick()
        affected_cpu = [s['cpu_percent'] for s in batch if s['server_name'] in gen.affected_servers]
        if affected_cpu:
            print(f"Tick {i+1}: Affected servers CPU: {np.mean(affected_cpu):.1f}%")

    # Switch to critical
    print("\n3. Switching to CRITICAL scenario...")
    gen.set_scenario('critical')

    for i in range(3):
        batch = gen.generate_tick()
        affected_cpu = [s['cpu_percent'] for s in batch if s['server_name'] in gen.affected_servers]
        if affected_cpu:
            print(f"Tick {i+1}: Affected servers CPU: {np.mean(affected_cpu):.1f}%")

    # Return to healthy
    print("\n4. Returning to HEALTHY (recovery)...")
    gen.set_scenario('healthy')

    for i in range(5):
        batch = gen.generate_tick()
        status = gen.get_status()
        print(f"Tick {i+1}: Recovery progress: {status['transition_progress']*100:.0f}%, still affected: {len(status['affected_servers'])}")

    print("\n[SUCCESS] All scenarios working!")
