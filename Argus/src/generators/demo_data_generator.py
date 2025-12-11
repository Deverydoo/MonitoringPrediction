#!/usr/bin/env python3
n# Setup Python path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
"""
demo_data_generator.py - Reproducible Demo Dataset Generator
Generates demo datasets with configurable system health scenarios.

Scenarios:
- HEALTHY: 100% healthy system, no issues, stable baselines
- DEGRADING: System starts healthy and gradually degrades (CPU, RAM, IOWait increase)
- CRITICAL: System starts healthy, then shows acute failure signs with spikes

Default pattern (DEGRADING):
- 0:00-1:30 : Stable baseline (healthy)
- 1:30-2:30 : Gradual escalation (warning signs)
- 2:30-3:30 : Incident peak (critical)
- 3:30-5:00 : Recovery to stable (gradual normalization)
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class DemoDataGenerator:
    """Generate reproducible demo data with configurable health scenarios."""

    def __init__(self,
                 duration_minutes: int = 5,
                 tick_seconds: int = 5,
                 fleet_size: int = 10,
                 seed: int = 42,
                 scenario: str = 'degrading'):
        """
        Initialize demo data generator.

        Args:
            duration_minutes: Total duration of demo data
            tick_seconds: Interval between data points
            fleet_size: Number of servers in demo fleet
            seed: Random seed for reproducibility
            scenario: Health scenario - 'healthy', 'degrading', or 'critical'
        """
        self.duration_minutes = duration_minutes
        self.tick_seconds = tick_seconds
        self.fleet_size = fleet_size
        self.seed = seed
        self.scenario = scenario.lower()

        # Validate scenario
        if self.scenario not in ['healthy', 'degrading', 'critical']:
            raise ValueError(f"Invalid scenario '{scenario}'. Must be 'healthy', 'degrading', or 'critical'")

        # Fixed seed for reproducibility
        np.random.seed(seed)

        # Calculate number of ticks
        self.total_ticks = (duration_minutes * 60) // tick_seconds

        # Define incident timeline (in seconds) - used for degrading and critical
        self.phase_boundaries = {
            'stable': (0, 90),           # 0:00 - 1:30
            'escalation': (90, 150),      # 1:30 - 2:30
            'peak': (150, 210),           # 2:30 - 3:30
            'recovery': (210, 300)        # 3:30 - 5:00
        }

        # Server profiles with realistic baselines
        self.servers = self._create_fleet()

    def _create_fleet(self) -> List[Dict]:
        """
        Create demo fleet matching IBM Spectrum Conductor infrastructure.

        Profiles match production dataset generator:
        - production: Main production nodes (pprva00a##)
        - staging: Staging/test nodes (psrva00a##)
        - compute: Compute/worker nodes (cppr##)
        - service: Service/master nodes (csrva##)
        - container: Container/notebook nodes (crva##)
        """
        fleet = []

        # IBM Spectrum Conductor infrastructure profiles
        # Baselines derived from production metrics_generator.py
        profiles = [
            {
                'type': 'production',
                'count': max(2, self.fleet_size // 3),  # ~33% production nodes
                'prefix': 'pprva00a',
                'cpu_base': 32,    # 32% average
                'mem_base': 47,    # 47% average
                'disk_base': 12,   # 12 MB/s
                'latency_base': 28,  # 28ms
                'error_base': 0.003
            },
            {
                'type': 'compute',
                'count': max(2, self.fleet_size // 3),  # ~33% compute nodes
                'prefix': 'cppr',
                'cpu_base': 28,    # Compute workloads
                'mem_base': 35,
                'disk_base': 25,   # Higher disk I/O for compute
                'latency_base': 32,
                'error_base': 0.002
            },
            {
                'type': 'service',
                'count': max(1, self.fleet_size // 5),  # ~20% service/master nodes
                'prefix': 'csrva',
                'cpu_base': 18,    # Lighter load
                'mem_base': 32,
                'disk_base': 5,
                'latency_base': 24,
                'error_base': 0.002
            },
            {
                'type': 'container',
                'count': max(1, self.fleet_size // 10),  # ~10% container/notebook nodes
                'prefix': 'crva',
                'cpu_base': 25,
                'mem_base': 36,
                'disk_base': 15,
                'latency_base': 30,
                'error_base': 0.004
            }
        ]

        # Adjust counts to match requested fleet size
        total_count = sum(p['count'] for p in profiles)
        if total_count < self.fleet_size:
            # Add remaining to production
            profiles[0]['count'] += (self.fleet_size - total_count)

        for profile in profiles:
            for i in range(1, profile['count'] + 1):
                fleet.append({
                    'server_name': f"{profile['prefix']}{i:02d}",
                    'profile': profile['type'],
                    'cpu_base': profile['cpu_base'] + np.random.uniform(-5, 5),
                    'mem_base': profile['mem_base'] + np.random.uniform(-3, 3),
                    'latency_base': profile['latency_base'] + np.random.uniform(-5, 5),
                    'disk_base': profile['disk_base'] + np.random.uniform(-2, 2),
                    'error_base': profile['error_base'],
                    'affected': i == 1  # First server of each type is affected by incident
                })

        return fleet[:self.fleet_size]  # Ensure exact fleet size

    def _get_phase(self, elapsed_seconds: float) -> str:
        """Determine current phase based on elapsed time."""
        # Healthy scenario has no phases
        if self.scenario == 'healthy':
            return 'stable'

        for phase, (start, end) in self.phase_boundaries.items():
            if start <= elapsed_seconds < end:
                return phase
        return 'recovery'

    def _get_phase_intensity(self, elapsed_seconds: float) -> float:
        """
        Calculate intensity multiplier based on phase and scenario.
        Returns value between 0.0 (stable) and 1.0 (peak).
        """
        # Healthy scenario: always 0.0
        if self.scenario == 'healthy':
            return 0.0

        phase = self._get_phase(elapsed_seconds)

        # DEGRADING scenario: gradual increase over time
        if self.scenario == 'degrading':
            if phase == 'stable':
                # Stable period: minimal variation
                return 0.0

            elif phase == 'escalation':
                # Gradual increase from 0.0 to 0.7
                phase_progress = (elapsed_seconds - self.phase_boundaries['escalation'][0]) / 60
                return 0.7 * phase_progress

            elif phase == 'peak':
                # High intensity with some fluctuation
                return 0.8 + 0.2 * np.sin(elapsed_seconds * 0.5)

            elif phase == 'recovery':
                # Gradual decrease from 0.7 to 0.0
                phase_progress = (elapsed_seconds - self.phase_boundaries['recovery'][0]) / 90
                return 0.7 * (1 - phase_progress)

        # CRITICAL scenario: acute spikes and failures
        elif self.scenario == 'critical':
            if phase == 'stable':
                # Mostly stable with occasional micro-spikes
                return 0.1 * np.random.random()

            elif phase == 'escalation':
                # Rapid increase with spikes
                phase_progress = (elapsed_seconds - self.phase_boundaries['escalation'][0]) / 60
                base_intensity = 0.9 * phase_progress
                # Add random spikes
                spike = 0.3 * np.random.random() if np.random.random() > 0.7 else 0
                return min(1.0, base_intensity + spike)

            elif phase == 'peak':
                # Critical failures with severe spikes
                base = 0.9
                spike = 0.4 * np.random.random() if np.random.random() > 0.5 else 0
                return min(1.0, base + spike)

            elif phase == 'recovery':
                # Slower recovery with residual spikes
                phase_progress = (elapsed_seconds - self.phase_boundaries['recovery'][0]) / 90
                base_intensity = 0.8 * (1 - phase_progress)
                spike = 0.2 * np.random.random() if np.random.random() > 0.8 else 0
                return max(0.0, base_intensity + spike)

        return 0.0

    def _generate_server_metrics(self,
                                  server: Dict,
                                  timestamp: datetime,
                                  elapsed_seconds: float) -> Dict:
        """Generate metrics for a single server at a specific time."""

        phase = self._get_phase(elapsed_seconds)
        intensity = self._get_phase_intensity(elapsed_seconds)

        # Base metrics with natural variation
        cpu_pct = server['cpu_base'] + np.random.normal(0, 2)
        mem_pct = server['mem_base'] + np.random.normal(0, 1.5)
        latency_ms = server['latency_base'] + np.random.exponential(2)
        error_rate = server['error_base'] + np.random.exponential(server['error_base'])

        # Scenario-specific behavior
        if self.scenario == 'healthy':
            # Keep everything at baseline with minimal variation
            pass  # Already set to baseline above

        elif self.scenario == 'degrading':
            # Gradual resource exhaustion for affected servers
            if server['affected']:
                # CPU gradually increases
                cpu_pct += intensity * 40

                # Memory fills up slowly
                mem_pct += intensity * 25

                # Latency increases due to resource contention
                latency_ms += intensity * 200

                # Error rate increases as system struggles
                error_rate += intensity * 0.5

        elif self.scenario == 'critical':
            # Acute failures with spikes for affected servers
            if server['affected']:
                # Severe CPU spikes
                cpu_pct += intensity * 50 + (np.random.random() * 20 if np.random.random() > 0.6 else 0)

                # Memory leaks and OOM pressure
                mem_pct += intensity * 35 + (np.random.random() * 15 if np.random.random() > 0.7 else 0)

                # Severe latency spikes (IOWait, thrashing)
                latency_ms += intensity * 300 + (np.random.random() * 200 if np.random.random() > 0.5 else 0)

                # High error rates during failures
                error_rate += intensity * 0.7 + (np.random.random() * 0.3 if np.random.random() > 0.5 else 0)

        # Clamp values to realistic ranges
        cpu_pct = max(0, min(100, cpu_pct))
        mem_pct = max(0, min(100, mem_pct))
        latency_ms = max(0, latency_ms)
        error_rate = max(0, min(1.0, error_rate))

        # Derived metrics
        # Disk I/O increases with intensity (simulating IOWait issues)
        disk_io_mb_s = server['disk_base'] * (1 + intensity * 0.5) + np.random.exponential(5)
        if self.scenario == 'critical' and server['affected']:
            # Add IOWait spikes in critical scenario
            disk_io_mb_s += intensity * 100 + (np.random.random() * 50 if np.random.random() > 0.6 else 0)

        net_in_mb_s = 5 + intensity * 10 + np.random.exponential(2)
        net_out_mb_s = net_in_mb_s * np.random.uniform(0.8, 1.2)
        gc_pause_ms = np.random.exponential(3) * (1 + intensity * 2)

        # Determine state based on metrics (matching production states)
        # States: healthy, heavy_load, critical_issue, recovery, maintenance, offline
        if cpu_pct > 90 or mem_pct > 90 or error_rate > 0.3:
            state = 'critical_issue'
        elif cpu_pct > 75 or mem_pct > 80 or error_rate > 0.1:
            state = 'heavy_load'
        elif phase == 'recovery' and intensity > 0.3:
            state = 'recovery'
        else:
            state = 'healthy'

        return {
            'timestamp': timestamp.isoformat(),
            'server_name': server['server_name'],
            'profile': server['profile'],
            'state': state,
            'cpu_pct': round(cpu_pct, 2),
            'mem_pct': round(mem_pct, 2),
            'disk_io_mb_s': round(disk_io_mb_s, 2),
            'net_in_mb_s': round(net_in_mb_s, 2),
            'net_out_mb_s': round(net_out_mb_s, 2),
            'latency_ms': round(latency_ms, 2),
            'error_rate': round(error_rate, 4),
            'gc_pause_ms': round(gc_pause_ms, 2),
            'container_oom': 0,
            'problem_child': 1 if server['affected'] else 0,
            'incident_phase': phase,
            'scenario': self.scenario
        }

    def generate(self, start_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate complete demo dataset.

        Args:
            start_time: Start timestamp (defaults to duration_minutes ago, ending at now)

        Returns:
            DataFrame with all server metrics across the demo timeline
        """
        if start_time is None:
            # End at current time, start duration_minutes in the past
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=self.duration_minutes)

        all_records = []

        for tick in range(self.total_ticks):
            elapsed_seconds = tick * self.tick_seconds
            timestamp = start_time + timedelta(seconds=elapsed_seconds)

            # Generate metrics for all servers at this tick
            for server in self.servers:
                metrics = self._generate_server_metrics(server, timestamp, elapsed_seconds)
                all_records.append(metrics)

        df = pd.DataFrame(all_records)
        return df.sort_values(['timestamp', 'server_name']).reset_index(drop=True)

    def save_demo_dataset(self,
                         output_dir: str = "./demo_data/",
                         filename: str = "demo_dataset",
                         save_csv: bool = False,
                         save_json: bool = False) -> Dict[str, Path]:
        """
        Generate and save demo dataset (Parquet by default).

        Args:
            output_dir: Output directory
            filename: Base filename (without extension)
            save_csv: Also save CSV format (slower)
            save_json: Also save JSON format (slowest, legacy)

        Returns:
            Dict with paths to created files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate data
        print(f"🎬 Generating demo dataset...")
        print(f"   Duration: {self.duration_minutes} minutes")
        print(f"   Tick interval: {self.tick_seconds} seconds")
        print(f"   Fleet size: {self.fleet_size} servers")
        print(f"   Total data points: {self.total_ticks * self.fleet_size}")

        df = self.generate()

        # Show time range
        print(f"   Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   (Data ends at current time, starts {self.duration_minutes} minutes ago)")

        created_files = {}

        # Save as Parquet (default, fastest)
        try:
            parquet_path = output_path / f"{filename}.parquet"
            df.to_parquet(parquet_path, compression='snappy', index=False)
            created_files['parquet'] = parquet_path
            print(f"✅ Parquet saved: {parquet_path}")
        except ImportError:
            print("⚠️  PyArrow not available, falling back to CSV")
            save_csv = True

        # Optionally save as CSV
        if save_csv:
            csv_path = output_path / f"{filename}.csv"
            df.to_csv(csv_path, index=False)
            created_files['csv'] = csv_path
            print(f"✅ CSV saved: {csv_path}")

        # Optionally save as JSON
        if save_json:
            json_path = output_path / f"{filename}.json"
            records = df.to_dict('records')
            metadata = {
                'generated_at': datetime.utcnow().isoformat(),
                'generator': 'demo_data_generator',
                'total_records': len(records)
            }
            with open(json_path, 'w') as f:
                json.dump({'records': records, 'metadata': metadata}, f, indent=2, default=str)
            created_files['json'] = json_path
            print(f"✅ JSON saved: {json_path}")

        # Save metadata
        scenario_descriptions = {
            'healthy': '100% healthy system with stable baselines, no incidents',
            'degrading': 'System starts healthy and gradually degrades (CPU, RAM, IOWait increase)',
            'critical': 'System starts healthy then shows acute failure signs with severe spikes'
        }

        metadata = {
            'generated_at': datetime.utcnow().isoformat(),
            'generator': 'demo_data_generator',
            'version': '2.0',
            'scenario': self.scenario,
            'scenario_description': scenario_descriptions.get(self.scenario, 'Unknown'),
            'duration_minutes': self.duration_minutes,
            'tick_seconds': self.tick_seconds,
            'fleet_size': self.fleet_size,
            'total_ticks': self.total_ticks,
            'total_records': len(df),
            'seed': self.seed,
            'phase_boundaries': self.phase_boundaries,
            'incident_pattern': {
                'description': f'Scenario: {self.scenario.upper()}',
                'stable_duration': '0:00-1:30',
                'escalation_duration': '1:30-2:30',
                'peak_duration': '2:30-3:30',
                'recovery_duration': '3:30-5:00'
            } if self.scenario != 'healthy' else {
                'description': 'Healthy system - no incident phases',
                'duration': f'0:00-{self.duration_minutes}:00'
            },
            'affected_servers': [s['server_name'] for s in self.servers if s['affected']],
            'phase_distribution': df['incident_phase'].value_counts().to_dict()
        }

        metadata_path = output_path / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        created_files['metadata'] = metadata_path
        print(f"✅ Metadata saved: {metadata_path}")

        # Print summary
        print(f"\n📊 Demo Dataset Summary:")
        print(f"   Infrastructure: IBM Spectrum Conductor")
        print(f"   Scenario: {self.scenario.upper()}")

        print(f"\n   Profile distribution:")
        for profile, count in df['profile'].value_counts().items():
            print(f"      {profile}: {count} records ({count/len(df)*100:.1f}%)")

        print(f"\n   Phase distribution:")
        for phase, count in df['incident_phase'].value_counts().items():
            print(f"      {phase}: {count} records ({count/len(df)*100:.1f}%)")

        print(f"\n   Affected servers: {len([s for s in self.servers if s['affected']])}")
        for server in self.servers:
            if server['affected']:
                print(f"      - {server['server_name']} ({server['profile']})")

        print(f"\n   State distribution:")
        for state, count in df['state'].value_counts().items():
            print(f"      {state}: {count} records ({count/len(df)*100:.1f}%)")

        return created_files


def generate_demo_dataset(output_dir: str = "./demo_data/",
                         duration_minutes: int = 5,
                         fleet_size: int = 10,
                         seed: int = 42,
                         scenario: str = 'degrading') -> bool:
    """
    Module interface for generating demo dataset.

    Args:
        output_dir: Output directory
        duration_minutes: Duration of demo data
        fleet_size: Number of servers
        seed: Random seed for reproducibility
        scenario: Health scenario - 'healthy', 'degrading', or 'critical'

    Returns:
        True if successful
    """
    try:
        generator = DemoDataGenerator(
            duration_minutes=duration_minutes,
            fleet_size=fleet_size,
            seed=seed,
            scenario=scenario
        )

        generator.save_demo_dataset(output_dir=output_dir)
        print(f"\n✅ Demo dataset generated successfully!")
        return True

    except Exception as e:
        print(f"❌ Failed to generate demo dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Generate reproducible demo dataset with configurable scenarios")
    parser.add_argument("--output-dir", default="./demo_data/",
                       help="Output directory")
    parser.add_argument("--duration", type=int, default=5,
                       help="Duration in minutes")
    parser.add_argument("--fleet-size", type=int, default=10,
                       help="Number of servers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--scenario", type=str, default='degrading',
                       choices=['healthy', 'degrading', 'critical'],
                       help="Health scenario: healthy (no issues), degrading (gradual), critical (acute spikes)")
    parser.add_argument("--filename", default="demo_dataset",
                       help="Base filename (without extension)")
    parser.add_argument("--csv", action="store_true",
                       help="Also save CSV format (slower)")
    parser.add_argument("--json", action="store_true",
                       help="Also save JSON format (slowest, legacy)")

    args = parser.parse_args()

    generator = DemoDataGenerator(
        duration_minutes=args.duration,
        fleet_size=args.fleet_size,
        seed=args.seed,
        scenario=args.scenario
    )

    created_files = generator.save_demo_dataset(
        output_dir=args.output_dir,
        filename=args.filename,
        save_csv=args.csv,
        save_json=args.json
    )

    print(f"\n🎉 Demo dataset generation complete!")
    print(f"   Created files:")
    for file_type, path in created_files.items():
        print(f"      {file_type}: {path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
