"""
Real-time streaming demo data generator for TFT monitoring dashboard.
Generates continuous tick-by-tick data with realistic degradation patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Literal
import random


class DemoStreamGenerator:
    """Generate realistic streaming monitoring data for demo scenarios."""

    def __init__(self, num_servers: int = 20, seed: int = None):
        """
        Initialize demo stream generator.

        Args:
            num_servers: Number of servers to simulate
            seed: Random seed for reproducibility (None for random)
        """
        self.num_servers = num_servers
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Use EXACT server names from training data (SOURCE OF TRUTH)
        # Financial ML Platform - Profile-based servers (90 total)
        training_servers = [
            # ML Compute - Spectrum Conductor workers (20 servers)
            'ppml0001', 'ppml0002', 'ppml0003', 'ppml0004', 'ppml0005',
            'ppml0006', 'ppml0007', 'ppml0008', 'ppml0009', 'ppml0010',
            'ppml0011', 'ppml0012', 'ppml0013', 'ppml0014', 'ppml0015',
            'ppml0016', 'ppml0017', 'ppml0018', 'ppml0019', 'ppml0020',
            # Database - Oracle, PostgreSQL, MongoDB (15 servers)
            'ppdb001', 'ppdb002', 'ppdb003', 'ppdb004', 'ppdb005',
            'ppdb006', 'ppdb007', 'ppdb008', 'ppdb009', 'ppdb010',
            'ppdb011', 'ppdb012', 'ppdb013', 'ppdb014', 'ppdb015',
            # Web/API - REST endpoints, gateways (25 servers)
            'ppweb001', 'ppweb002', 'ppweb003', 'ppweb004', 'ppweb005',
            'ppweb006', 'ppweb007', 'ppweb008', 'ppweb009', 'ppweb010',
            'ppweb011', 'ppweb012', 'ppweb013', 'ppweb014', 'ppweb015',
            'ppweb016', 'ppweb017', 'ppweb018', 'ppweb019', 'ppweb020',
            'ppweb021', 'ppweb022', 'ppweb023', 'ppweb024', 'ppweb025',
            # Conductor Management - Job scheduling (5 servers)
            'ppcon01', 'ppcon02', 'ppcon03', 'ppcon04', 'ppcon05',
            # Data Ingest - Kafka, Spark, ETL (10 servers)
            'ppetl001', 'ppetl002', 'ppetl003', 'ppetl004', 'ppetl005',
            'ppetl006', 'ppetl007', 'ppetl008', 'ppetl009', 'ppetl010',
            # Risk Analytics - VaR, Monte Carlo (8 servers)
            'pprisk001', 'pprisk002', 'pprisk003', 'pprisk004',
            'pprisk005', 'pprisk006', 'pprisk007', 'pprisk008',
            # Generic - Utility servers (7 servers)
            'ppgen001', 'ppgen002', 'ppgen003', 'ppgen004',
            'ppgen005', 'ppgen006', 'ppgen007',
        ]

        # Use subset if num_servers < 90, otherwise use all 90
        self.server_names = training_servers[:min(num_servers, 90)]

        # Track current state for each server
        self.server_states = {
            server: {
                'cpu_pct': np.random.uniform(20, 40),
                'mem_pct': np.random.uniform(30, 50),
                'disk_io_mb_s': np.random.uniform(10, 30),
                'net_in_mb_s': np.random.uniform(5, 15),
                'net_out_mb_s': np.random.uniform(5, 15),
                'latency_ms': np.random.uniform(10, 30),
                'error_rate': np.random.uniform(0, 0.5),
                'gc_pause_ms': np.random.uniform(5, 15),
                'container_oom': 0,
                'baseline_cpu': np.random.uniform(20, 40),
                'baseline_mem': np.random.uniform(30, 50),
                'baseline_latency': np.random.uniform(10, 30),
                'degradation_rate': 0.0,
                'target_degradation': 0.0,
                'is_degrading': False,
                'degradation_step': 0,
                'max_degradation_steps': 60
            }
            for server in self.server_names
        }

        self.current_time = datetime.now()
        self.tick_count = 0
        self.scenario_mode: Literal['stable', 'degrading', 'critical'] = 'stable'
        self.degrading_servers = []

    def set_scenario(self, mode: Literal['stable', 'degrading', 'critical']):
        """
        Set the scenario mode and select servers for degradation.

        Args:
            mode: 'stable', 'degrading', or 'critical'
        """
        self.scenario_mode = mode
        self.tick_count = 0

        if mode == 'stable':
            # Reset all servers to stable state
            self.degrading_servers = []
            for server, state in self.server_states.items():
                state['is_degrading'] = False
                state['degradation_rate'] = 0.0
                state['target_degradation'] = 0.0
                state['degradation_step'] = 0

        elif mode == 'degrading':
            # Select 20-30% of servers to degrade slowly
            num_degrading = max(2, int(self.num_servers * np.random.uniform(0.2, 0.3)))
            self.degrading_servers = random.sample(self.server_names, num_degrading)

            for server in self.degrading_servers:
                state = self.server_states[server]
                state['is_degrading'] = True
                # Gradual degradation over ~60 ticks (5 minutes at 5-second intervals)
                state['max_degradation_steps'] = 60
                state['target_degradation'] = np.random.uniform(0.6, 0.8)
                state['degradation_rate'] = state['target_degradation'] / state['max_degradation_steps']
                state['degradation_step'] = 0

        elif mode == 'critical':
            # Select 10-20% of servers for critical degradation
            num_critical = max(2, int(self.num_servers * np.random.uniform(0.1, 0.2)))
            self.degrading_servers = random.sample(self.server_names, num_critical)

            for server in self.degrading_servers:
                state = self.server_states[server]
                state['is_degrading'] = True
                # Faster degradation to critical levels over ~40 ticks (3-4 minutes)
                state['max_degradation_steps'] = 40
                state['target_degradation'] = np.random.uniform(0.85, 0.95)
                state['degradation_rate'] = state['target_degradation'] / state['max_degradation_steps']
                state['degradation_step'] = 0

    def _apply_noise(self, value: float, noise_pct: float = 0.05) -> float:
        """Add realistic noise to a value."""
        noise = np.random.normal(0, value * noise_pct)
        return max(0, value + noise)

    def _smooth_transition(self, current: float, target: float, speed: float = 0.1) -> float:
        """Smoothly transition from current to target value."""
        return current * (1 - speed) + target * speed

    def _update_server_state(self, server: str):
        """Update a single server's state for the next tick."""
        state = self.server_states[server]

        if state['is_degrading'] and state['degradation_step'] < state['max_degradation_steps']:
            # Calculate current degradation intensity (0.0 to 1.0)
            progress = state['degradation_step'] / state['max_degradation_steps']
            intensity = min(1.0, progress)

            # Apply degradation with smooth transitions
            # CPU increases most dramatically
            target_cpu = state['baseline_cpu'] + (state['target_degradation'] * 50 * intensity)
            state['cpu_pct'] = self._smooth_transition(
                state['cpu_pct'],
                self._apply_noise(target_cpu, noise_pct=0.08),
                speed=0.15
            )

            # Memory increases moderately
            target_mem = state['baseline_mem'] + (state['target_degradation'] * 35 * intensity)
            state['mem_pct'] = self._smooth_transition(
                state['mem_pct'],
                self._apply_noise(target_mem, noise_pct=0.03),
                speed=0.08
            )

            # Disk I/O increases with load
            target_disk = 15 + (state['target_degradation'] * 70 * intensity)
            state['disk_io_mb_s'] = self._smooth_transition(
                state['disk_io_mb_s'],
                self._apply_noise(target_disk, noise_pct=0.10),
                speed=0.12
            )

            # Latency increases significantly under load
            target_latency = state['baseline_latency'] + (state['target_degradation'] * 180 * intensity)
            state['latency_ms'] = self._smooth_transition(
                state['latency_ms'],
                self._apply_noise(target_latency, noise_pct=0.12),
                speed=0.10
            )

            # Error rate increases
            target_error = state['target_degradation'] * 4.0 * intensity
            state['error_rate'] = self._smooth_transition(
                state['error_rate'],
                self._apply_noise(target_error, noise_pct=0.15),
                speed=0.10
            )

            # GC pauses increase with memory pressure
            target_gc = 10 + (state['target_degradation'] * 140 * intensity)
            state['gc_pause_ms'] = self._smooth_transition(
                state['gc_pause_ms'],
                self._apply_noise(target_gc, noise_pct=0.10),
                speed=0.12
            )

            # Network traffic increases
            target_net = 8 + (state['target_degradation'] * 45 * intensity)
            state['net_in_mb_s'] = self._smooth_transition(
                state['net_in_mb_s'],
                self._apply_noise(target_net, noise_pct=0.10),
                speed=0.10
            )
            state['net_out_mb_s'] = self._smooth_transition(
                state['net_out_mb_s'],
                self._apply_noise(target_net * 0.9, noise_pct=0.10),
                speed=0.10
            )

            # Container OOM events in critical scenarios at high intensity
            if self.scenario_mode == 'critical' and intensity > 0.7:
                state['container_oom'] = 1 if np.random.random() < 0.08 else 0
            else:
                state['container_oom'] = 0

            state['degradation_step'] += 1

        else:
            # Stable server with normal fluctuations
            # Apply gentle random walk around baseline
            state['cpu_pct'] = self._smooth_transition(
                state['cpu_pct'],
                self._apply_noise(state['baseline_cpu'], noise_pct=0.12),
                speed=0.10
            )
            state['mem_pct'] = self._smooth_transition(
                state['mem_pct'],
                self._apply_noise(state['baseline_mem'], noise_pct=0.06),
                speed=0.05
            )
            state['disk_io_mb_s'] = self._smooth_transition(
                state['disk_io_mb_s'],
                self._apply_noise(np.random.uniform(10, 30), noise_pct=0.15),
                speed=0.15
            )
            state['latency_ms'] = self._smooth_transition(
                state['latency_ms'],
                self._apply_noise(state['baseline_latency'], noise_pct=0.15),
                speed=0.12
            )
            state['error_rate'] = self._smooth_transition(
                state['error_rate'],
                self._apply_noise(np.random.uniform(0, 0.5), noise_pct=0.20),
                speed=0.15
            )
            state['gc_pause_ms'] = self._smooth_transition(
                state['gc_pause_ms'],
                self._apply_noise(np.random.uniform(5, 15), noise_pct=0.15),
                speed=0.12
            )
            state['net_in_mb_s'] = self._smooth_transition(
                state['net_in_mb_s'],
                self._apply_noise(np.random.uniform(5, 15), noise_pct=0.15),
                speed=0.12
            )
            state['net_out_mb_s'] = self._smooth_transition(
                state['net_out_mb_s'],
                self._apply_noise(np.random.uniform(5, 15), noise_pct=0.15),
                speed=0.12
            )
            state['container_oom'] = 0

        # Apply bounds
        state['cpu_pct'] = min(100, max(0, state['cpu_pct']))
        state['mem_pct'] = min(100, max(0, state['mem_pct']))
        state['disk_io_mb_s'] = max(0, state['disk_io_mb_s'])
        state['latency_ms'] = max(0, state['latency_ms'])
        state['error_rate'] = max(0, state['error_rate'])
        state['gc_pause_ms'] = max(0, state['gc_pause_ms'])
        state['net_in_mb_s'] = max(0, state['net_in_mb_s'])
        state['net_out_mb_s'] = max(0, state['net_out_mb_s'])

    def generate_tick(self) -> pd.DataFrame:
        """
        Generate data for one time tick (5 seconds).

        Returns:
            DataFrame with one row per server for the current timestamp
        """
        # Update all server states
        for server in self.server_names:
            self._update_server_state(server)

        # Create records
        records = []
        for server in self.server_names:
            state = self.server_states[server]

            # Determine status based on metrics
            if state['cpu_pct'] > 85 or state['mem_pct'] > 85 or state['latency_ms'] > 150:
                status = 'critical'
            elif state['cpu_pct'] > 70 or state['mem_pct'] > 70 or state['latency_ms'] > 100:
                status = 'warning'
            else:
                status = 'healthy'

            # Generate notes for degrading servers
            notes = ''
            if server in self.degrading_servers:
                progress_pct = int((state['degradation_step'] / state['max_degradation_steps']) * 100)
                if self.scenario_mode == 'degrading':
                    notes = f'Gradual degradation: {progress_pct}% complete'
                elif self.scenario_mode == 'critical':
                    notes = f'Critical degradation: {progress_pct}% complete'

            record = {
                'timestamp': self.current_time,
                'server_id': server,
                'cpu_pct': round(state['cpu_pct'], 2),
                'mem_pct': round(state['mem_pct'], 2),
                'disk_io_mb_s': round(state['disk_io_mb_s'], 2),
                'net_in_mb_s': round(state['net_in_mb_s'], 2),
                'net_out_mb_s': round(state['net_out_mb_s'], 2),
                'latency_ms': round(state['latency_ms'], 2),
                'error_rate': round(state['error_rate'], 3),
                'gc_pause_ms': round(state['gc_pause_ms'], 2),
                'container_oom': state['container_oom'],
                'state': status,
                'notes': notes
            }
            records.append(record)

        # Advance time by 5 seconds
        self.current_time += timedelta(seconds=5)
        self.tick_count += 1

        return pd.DataFrame(records)

    def get_scenario_info(self) -> Dict:
        """Get information about the current scenario."""
        return {
            'mode': self.scenario_mode,
            'tick_count': self.tick_count,
            'total_servers': self.num_servers,
            'degrading_servers': len(self.degrading_servers),
            'degrading_server_names': self.degrading_servers
        }


# Standalone test
if __name__ == "__main__":
    print("Testing Demo Stream Generator")
    print("=" * 60)

    gen = DemoStreamGenerator(num_servers=10, seed=42)

    # Test stable mode
    print("\n1. STABLE MODE - Testing 5 ticks")
    print("-" * 60)
    gen.set_scenario('stable')
    for i in range(5):
        data = gen.generate_tick()
        print(f"\nTick {i+1}:")
        print(f"  Servers: {len(data)}")
        print(f"  CPU range: {data['cpu_pct'].min():.1f}% - {data['cpu_pct'].max():.1f}%")
        print(f"  Mem range: {data['mem_pct'].min():.1f}% - {data['mem_pct'].max():.1f}%")
        print(f"  Latency range: {data['latency_ms'].min():.1f}ms - {data['latency_ms'].max():.1f}ms")
        print(f"  Status: {dict(data['state'].value_counts())}")

    # Test degrading mode
    print("\n\n2. DEGRADING MODE - Testing 10 ticks")
    print("-" * 60)
    gen.set_scenario('degrading')
    info = gen.get_scenario_info()
    print(f"Degrading servers ({info['degrading_servers']}): {info['degrading_server_names']}")

    for i in range(10):
        data = gen.generate_tick()
        print(f"\nTick {i+1}:")
        print(f"  CPU range: {data['cpu_pct'].min():.1f}% - {data['cpu_pct'].max():.1f}%")
        print(f"  Mem range: {data['mem_pct'].min():.1f}% - {data['mem_pct'].max():.1f}%")
        print(f"  Latency range: {data['latency_ms'].min():.1f}ms - {data['latency_ms'].max():.1f}ms")
        print(f"  Status: {dict(data['state'].value_counts())}")

        # Show degrading server details
        degrading_data = data[data['server_id'].isin(info['degrading_server_names'])]
        if len(degrading_data) > 0:
            print(f"  Degrading servers CPU: {degrading_data['cpu_pct'].mean():.1f}% avg")

    # Test critical mode
    print("\n\n3. CRITICAL MODE - Testing 10 ticks")
    print("-" * 60)
    gen.set_scenario('critical')
    info = gen.get_scenario_info()
    print(f"Critical servers ({info['degrading_servers']}): {info['degrading_server_names']}")

    for i in range(10):
        data = gen.generate_tick()
        print(f"\nTick {i+1}:")
        print(f"  CPU range: {data['cpu_pct'].min():.1f}% - {data['cpu_pct'].max():.1f}%")
        print(f"  Mem range: {data['mem_pct'].min():.1f}% - {data['mem_pct'].max():.1f}%")
        print(f"  Latency range: {data['latency_ms'].min():.1f}ms - {data['latency_ms'].max():.1f}ms")
        print(f"  Error rate range: {data['error_rate'].min():.2f} - {data['error_rate'].max():.2f}")
        print(f"  Status: {dict(data['state'].value_counts())}")
        print(f"  OOM events: {data['container_oom'].sum()}")

        # Show critical server details
        critical_data = data[data['server_id'].isin(info['degrading_server_names'])]
        if len(critical_data) > 0:
            print(f"  Critical servers CPU: {critical_data['cpu_pct'].mean():.1f}% avg")
            print(f"  Critical servers Latency: {critical_data['latency_ms'].mean():.1f}ms avg")

    print("\n" + "=" * 60)
    print("Test complete!")
