#!/usr/bin/env python3
"""
Enhanced Server Metrics Generator - Production Level
Generates realistic time-based multi-server telemetry for TFT training

Features:
- Fleet-wide 5-second polling simulation
- Realistic server profiles and operational states
- Time-based patterns (diurnal, weekly)
- Problem children modeling
- Both CSV and Parquet output
- Streaming-friendly sequential format
"""

import argparse
import json
import warnings
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ServerProfile(Enum):
    """Server profile types with different operational characteristics."""
    PRODUCTION = "production"
    STAGING = "staging" 
    COMPUTE = "compute"
    SERVICE = "service"
    CONTAINER = "container"

class ServerState(Enum):
    """Operational states with realistic transitions."""
    IDLE = "idle"
    HEALTHY = "healthy"
    MORNING_SPIKE = "morning_spike"
    HEAVY_LOAD = "heavy_load"
    CRITICAL_ISSUE = "critical_issue"
    MAINTENANCE = "maintenance"
    RECOVERY = "recovery"
    OFFLINE = "offline"

@dataclass
class Config:
    """Configuration for fleet generation."""
    # Time settings
    start_time: Optional[str] = None  # Auto-calculated if None
    hours: int = 24
    tick_seconds: int = 5
    timezone: str = "UTC"
    
    # Fleet composition
    num_prod: int = 40
    num_stage: int = 10
    num_compute: int = 20
    num_service: int = 20
    num_container: int = 10
    
    # Behavior settings
    problem_child_pct: float = 0.10
    offline_fill: str = "nan"  # "nan" or "zeros"
    seed: Optional[int] = 42
    
    # Output settings
    out_dir: str = "./training"
    output_format: str = "parquet"  # "csv", "parquet", or "both" (default: parquet only)
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.08 <= self.problem_child_pct <= 0.12:
            raise ValueError("problem_child_pct must be between 8% and 12%")
        if self.offline_fill not in ["nan", "zeros"]:
            raise ValueError("offline_fill must be 'nan' or 'zeros'")

# Profile baselines: (mean, std) for each metric
PROFILE_BASELINES = {
    ServerProfile.PRODUCTION: {
        "cpu": (0.32, 0.08), "mem": (0.47, 0.06), "disk_io_mb_s": (12.0, 3.0),
        "net_in_mb_s": (3.5, 1.0), "net_out_mb_s": (2.8, 0.8), 
        "latency_ms": (28.0, 6.0), "error_rate": (0.003, 0.001), "gc_pause_ms": (8.0, 3.0)
    },
    ServerProfile.STAGING: {
        "cpu": (0.22, 0.07), "mem": (0.38, 0.08), "disk_io_mb_s": (8.0, 2.5),
        "net_in_mb_s": (2.2, 0.7), "net_out_mb_s": (1.8, 0.5),
        "latency_ms": (35.0, 8.0), "error_rate": (0.005, 0.002), "gc_pause_ms": (6.0, 2.5)
    },
    ServerProfile.COMPUTE: {
        "cpu": (0.28, 0.15), "mem": (0.35, 0.12), "disk_io_mb_s": (25.0, 8.0),
        "net_in_mb_s": (1.8, 0.6), "net_out_mb_s": (1.5, 0.4),
        "latency_ms": (32.0, 7.0), "error_rate": (0.002, 0.001), "gc_pause_ms": (12.0, 5.0)
    },
    ServerProfile.SERVICE: {
        "cpu": (0.18, 0.04), "mem": (0.32, 0.05), "disk_io_mb_s": (5.0, 1.5),
        "net_in_mb_s": (4.2, 1.2), "net_out_mb_s": (3.8, 1.0),
        "latency_ms": (24.0, 5.0), "error_rate": (0.002, 0.001), "gc_pause_ms": (4.0, 1.5)
    },
    ServerProfile.CONTAINER: {
        "cpu": (0.25, 0.18), "mem": (0.36, 0.15), "disk_io_mb_s": (15.0, 6.0),
        "net_in_mb_s": (2.8, 1.2), "net_out_mb_s": (2.2, 1.0),
        "latency_ms": (30.0, 9.0), "error_rate": (0.004, 0.002), "gc_pause_ms": (7.0, 4.0)
    }
}

# State multipliers for baseline adjustment
STATE_MULTIPLIERS = {
    ServerState.IDLE: {
        "cpu": 0.6, "mem": 0.9, "disk_io_mb_s": 0.4, "net_in_mb_s": 0.3, "net_out_mb_s": 0.3,
        "latency_ms": 0.8, "error_rate": 0.5, "gc_pause_ms": 0.2
    },
    ServerState.HEALTHY: {
        "cpu": 1.0, "mem": 1.0, "disk_io_mb_s": 1.0, "net_in_mb_s": 1.0, "net_out_mb_s": 1.0,
        "latency_ms": 1.0, "error_rate": 1.0, "gc_pause_ms": 1.0
    },
    ServerState.MORNING_SPIKE: {
        "cpu": 1.7, "mem": 1.3, "disk_io_mb_s": 1.8, "net_in_mb_s": 2.2, "net_out_mb_s": 1.9,
        "latency_ms": 1.5, "error_rate": 1.8, "gc_pause_ms": 2.1
    },
    ServerState.HEAVY_LOAD: {
        "cpu": 1.9, "mem": 1.4, "disk_io_mb_s": 1.6, "net_in_mb_s": 1.8, "net_out_mb_s": 1.7,
        "latency_ms": 1.6, "error_rate": 1.6, "gc_pause_ms": 1.8
    },
    ServerState.CRITICAL_ISSUE: {
        "cpu": 2.4, "mem": 1.7, "disk_io_mb_s": 0.8, "net_in_mb_s": 0.6, "net_out_mb_s": 0.4,
        "latency_ms": 2.8, "error_rate": 4.5, "gc_pause_ms": 3.2
    },
    ServerState.MAINTENANCE: {
        "cpu": 0.4, "mem": 0.8, "disk_io_mb_s": 1.5, "net_in_mb_s": 0.2, "net_out_mb_s": 0.2,
        "latency_ms": 0.9, "error_rate": 0.3, "gc_pause_ms": 0.1
    },
    ServerState.RECOVERY: {
        "cpu": 0.9, "mem": 1.0, "disk_io_mb_s": 0.7, "net_in_mb_s": 0.8, "net_out_mb_s": 0.7,
        "latency_ms": 1.3, "error_rate": 1.2, "gc_pause_ms": 0.8
    },
    ServerState.OFFLINE: {
        "cpu": 0.0, "mem": 0.0, "disk_io_mb_s": 0.0, "net_in_mb_s": 0.0, "net_out_mb_s": 0.0,
        "latency_ms": 0.0, "error_rate": 0.0, "gc_pause_ms": 0.0
    }
}

def generate_server_names(profile: ServerProfile, count: int) -> List[str]:
    """Generate server names following naming conventions."""
    if profile == ServerProfile.PRODUCTION:
        return [f"pprva00a{i:02d}" for i in range(1, count + 1)]
    elif profile == ServerProfile.STAGING:
        return [f"psrva00a{i:02d}" for i in range(1, count + 1)]
    elif profile == ServerProfile.COMPUTE:
        return [f"cppr{i:02d}" for i in range(1, count + 1)]
    elif profile == ServerProfile.SERVICE:
        return [f"csrva{i:02d}" for i in range(1, count + 1)]
    elif profile == ServerProfile.CONTAINER:
        return [f"crva{i:02d}" for i in range(1, count + 1)]
    else:
        raise ValueError(f"Unknown profile: {profile}")

def make_server_fleet(config: Config) -> pd.DataFrame:
    """Create server fleet with profiles and problem child assignments."""
    fleet = []
    
    # Generate servers by profile
    profiles_counts = [
        (ServerProfile.PRODUCTION, config.num_prod),
        (ServerProfile.STAGING, config.num_stage),
        (ServerProfile.COMPUTE, config.num_compute),
        (ServerProfile.SERVICE, config.num_service),
        (ServerProfile.CONTAINER, config.num_container)
    ]
    
    for profile, count in profiles_counts:
        if count <= 0:
            continue
            
        server_names = generate_server_names(profile, count)
        for name in server_names:
            fleet.append({
                'server_name': name,
                'profile': profile.value,
                'problem_child': False  # Will be set later
            })
    
    fleet_df = pd.DataFrame(fleet)
    
    # Assign problem children randomly
    total_servers = len(fleet_df)
    num_problems = int(total_servers * config.problem_child_pct)
    
    if config.seed is not None:
        np.random.seed(config.seed)
    
    problem_indices = np.random.choice(total_servers, num_problems, replace=False)
    fleet_df.loc[problem_indices, 'problem_child'] = True
    
    return fleet_df.reset_index(drop=True)

def generate_schedule(config: Config) -> pd.DatetimeIndex:
    """
    Generate 5-second interval timestamp schedule.

    By default, creates timestamps ending at current time and starting X hours in the past,
    where X is config.hours. This ensures data appears recent and realistic.
    """
    if config.start_time is None:
        # Auto-calculate: end at current time, start X hours in the past
        end_time = datetime.now(timezone.utc)
        start = end_time - timedelta(hours=config.hours)
    else:
        # Use explicit start time if provided
        start = pd.Timestamp(config.start_time).tz_convert('UTC')
        end_time = start + pd.Timedelta(hours=config.hours)

    return pd.date_range(start=start, end=end_time, freq=f'{config.tick_seconds}S', inclusive='left')

def diurnal_multiplier(hour: int, profile: ServerProfile, state: ServerState) -> float:
    """Calculate time-of-day multiplier for metrics."""
    # Base diurnal curve (business hours peak)
    if 7 <= hour <= 9:  # Morning spike
        base = 1.4
    elif 10 <= hour <= 17:  # Business hours
        base = 1.2
    elif 18 <= hour <= 23:  # Evening (compute heavy)
        base = 1.1 if profile == ServerProfile.COMPUTE else 0.9
    else:  # Night/early morning
        base = 0.7
    
    # Weekend reduction (simplified - assume 30% reduction)
    # In real implementation, would check day of week
    
    # Profile-specific adjustments
    if profile == ServerProfile.COMPUTE and 18 <= hour <= 23:
        base *= 1.3  # Evening batch jobs
    elif profile == ServerProfile.SERVICE:
        base *= 0.95  # More steady throughout day
    
    return max(0.3, min(2.0, base))

def ar1_series(base: np.ndarray, phi: float = 0.85, sigma: float = 0.03, seed: Optional[int] = None) -> np.ndarray:
    """Generate AR(1) autocorrelated series for smoothing."""
    if seed is not None:
        np.random.seed(seed)
    
    n = len(base)
    result = np.zeros(n)
    result[0] = base[0]
    
    noise = np.random.normal(0, sigma, n)
    
    for i in range(1, n):
        result[i] = phi * result[i-1] + (1 - phi) * base[i] + noise[i]
    
    return result

def get_state_transition_probs(current_state: ServerState, hour: int, is_problem_child: bool) -> Dict[ServerState, float]:
    """Get state transition probabilities based on current state, time, and server type."""
    
    # Base transition matrix
    transitions = {
        ServerState.IDLE: {
            ServerState.IDLE: 0.7,
            ServerState.HEALTHY: 0.25,
            ServerState.MAINTENANCE: 0.05
        },
        ServerState.HEALTHY: {
            ServerState.HEALTHY: 0.85,
            ServerState.MORNING_SPIKE: 0.08 if 7 <= hour <= 9 else 0.02,
            ServerState.HEAVY_LOAD: 0.06 if 10 <= hour <= 17 else 0.02,
            ServerState.CRITICAL_ISSUE: 0.02,
            ServerState.IDLE: 0.03 if hour < 6 or hour > 22 else 0.01
        },
        ServerState.MORNING_SPIKE: {
            ServerState.MORNING_SPIKE: 0.6,
            ServerState.HEALTHY: 0.25,
            ServerState.HEAVY_LOAD: 0.1,
            ServerState.CRITICAL_ISSUE: 0.05
        },
        ServerState.HEAVY_LOAD: {
            ServerState.HEAVY_LOAD: 0.7,
            ServerState.HEALTHY: 0.2,
            ServerState.CRITICAL_ISSUE: 0.08,
            ServerState.IDLE: 0.02
        },
        ServerState.CRITICAL_ISSUE: {
            ServerState.CRITICAL_ISSUE: 0.3,
            ServerState.OFFLINE: 0.4,
            ServerState.RECOVERY: 0.3
        },
        ServerState.MAINTENANCE: {
            ServerState.MAINTENANCE: 0.8,
            ServerState.OFFLINE: 0.1,
            ServerState.HEALTHY: 0.1
        },
        ServerState.RECOVERY: {
            ServerState.RECOVERY: 0.6,
            ServerState.HEALTHY: 0.35,
            ServerState.IDLE: 0.05
        },
        ServerState.OFFLINE: {
            ServerState.OFFLINE: 0.7,
            ServerState.RECOVERY: 0.3
        }
    }
    
    probs = transitions.get(current_state, {ServerState.HEALTHY: 1.0})
    
    # Increase problem rates for problem children
    if is_problem_child:
        if ServerState.CRITICAL_ISSUE in probs:
            probs[ServerState.CRITICAL_ISSUE] *= 2.0
        if ServerState.OFFLINE in probs:
            probs[ServerState.OFFLINE] *= 1.5
    
    # Normalize probabilities
    total = sum(probs.values())
    return {state: prob/total for state, prob in probs.items()}

def simulate_states(fleet: pd.DataFrame, schedule: pd.DatetimeIndex, config: Config) -> pd.DataFrame:
    """Simulate server states using Markov chain."""
    if config.seed is not None:
        np.random.seed(config.seed)
    
    results = []
    server_states = {name: ServerState.HEALTHY for name in fleet['server_name']}  # Initial states
    
    for timestamp in schedule:
        hour = timestamp.hour
        
        for _, server in fleet.iterrows():
            server_name = server['server_name'] 
            profile = server['profile']
            is_problem_child = server['problem_child']
            
            current_state = server_states[server_name]
            
            # Get transition probabilities and sample next state
            probs = get_state_transition_probs(current_state, hour, is_problem_child)
            states = list(probs.keys())
            probabilities = list(probs.values())
            
            next_state = np.random.choice(states, p=probabilities)
            server_states[server_name] = next_state
            
            results.append({
                'timestamp': timestamp,
                'server_name': server_name,
                'profile': profile,
                'state': next_state.value,
                'problem_child': is_problem_child
            })
    
    return pd.DataFrame(results)

def simulate_metrics(state_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Add realistic metrics based on states and profiles."""
    if config.seed is not None:
        np.random.seed(config.seed + 1)  # Different seed for metrics
    
    df = state_df.copy()
    
    # Initialize metric columns
    metric_columns = ['cpu_pct', 'mem_pct', 'disk_io_mb_s', 'net_in_mb_s', 
                     'net_out_mb_s', 'latency_ms', 'error_rate', 'gc_pause_ms', 
                     'container_oom', 'notes']
    
    for col in metric_columns:
        df[col] = 0.0
    
    # Process by server for temporal continuity
    for server_name in df['server_name'].unique():
        server_mask = df['server_name'] == server_name
        server_data = df[server_mask].copy()
        
        profile_str = server_data.iloc[0]['profile']
        profile = ServerProfile(profile_str)
        
        # Get baseline parameters
        baselines = PROFILE_BASELINES[profile]
        
        n_points = len(server_data)
        
        # Generate base metrics for each column
        for metric in ['cpu', 'mem', 'disk_io_mb_s', 'net_in_mb_s', 'net_out_mb_s', 
                      'latency_ms', 'error_rate', 'gc_pause_ms']:
            
            if metric in baselines:
                mean, std = baselines[metric]
                
                # Generate base series
                base_values = np.random.normal(mean, std, n_points)
                
                # Apply state multipliers
                for i, (_, row) in enumerate(server_data.iterrows()):
                    state = ServerState(row['state'])
                    multiplier = STATE_MULTIPLIERS[state].get(metric, 1.0)
                    
                    # Add diurnal effect
                    hour = row['timestamp'].hour
                    diurnal_mult = diurnal_multiplier(hour, profile, state)
                    
                    base_values[i] *= multiplier * diurnal_mult
                
                # Apply AR(1) smoothing for temporal continuity
                smoothed_values = ar1_series(base_values, phi=0.85, sigma=std*0.1, seed=config.seed)
                
                # Handle offline state
                offline_mask = server_data['state'] == 'offline'
                if config.offline_fill == "nan":
                    smoothed_values[offline_mask] = np.nan
                else:  # zeros
                    smoothed_values[offline_mask] = 0.0
                
                # Apply bounds and store
                if metric in ['cpu', 'mem']:
                    # Percentage metrics
                    smoothed_values = np.clip(smoothed_values * 100, 0, 100)
                    df.loc[server_mask, f'{metric}_pct'] = smoothed_values
                else:
                    # Other metrics
                    smoothed_values = np.maximum(smoothed_values, 0)  # Non-negative
                    df.loc[server_mask, metric] = smoothed_values
        
        # Special handling for container OOM events
        if profile == ServerProfile.CONTAINER:
            # Higher chance of OOM during high memory usage
            mem_values = df.loc[server_mask, 'mem_pct']
            oom_prob = np.where(mem_values > 85, 0.05, 0.001)  # 5% chance when mem > 85%
            df.loc[server_mask, 'container_oom'] = np.random.binomial(1, oom_prob)
        
        # Add notes for interesting states
        notes = []
        for _, row in server_data.iterrows():
            state = row['state']
            if state == 'morning_spike':
                notes.append("auth surge" if np.random.random() < 0.3 else "batch warmup")
            elif state == 'critical_issue':
                issues = ["high cpu", "memory leak", "network timeout", "disk full"]
                notes.append(np.random.choice(issues))
            elif state == 'maintenance':
                notes.append("scheduled maintenance")
            elif state == 'recovery':
                notes.append("service restart")
            else:
                notes.append("")
        
        df.loc[server_mask, 'notes'] = notes
    
    return df

def write_outputs(df: pd.DataFrame, config: Config) -> None:
    """Write outputs in Parquet (default), CSV, or both formats."""
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort by timestamp and server name for proper ordering
    df = df.sort_values(['timestamp', 'server_name']).reset_index(drop=True)

    # Show time range
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    duration = end_time - start_time
    print(f"\n⏰ Time Range:")
    print(f"   Start: {start_time}")
    print(f"   End:   {end_time}")
    print(f"   Duration: {duration}")
    print(f"   (Data ends at current time, starts {config.hours} hours ago)\n")

    if config.output_format in ["csv", "both"]:
        # Write CSV (single file)
        csv_path = out_dir / "server_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"📄 CSV written: {csv_path} ({len(df):,} rows)")

    if config.output_format in ["parquet", "both"]:
        # Write Parquet file (single consolidated file for fast loading)
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            parquet_path = out_dir / "server_metrics.parquet"
            df.to_parquet(parquet_path, compression='snappy', index=False)

            # Get file size for reporting
            size_mb = parquet_path.stat().st_size / (1024 * 1024)
            print(f"📊 Parquet written: {parquet_path} ({len(df):,} rows, {size_mb:.1f} MB)")

        except ImportError:
            print("⚠️  PyArrow not available - install with: pip install pyarrow")
            print("   Falling back to CSV output...")
            # Fallback to CSV if parquet fails
            if config.output_format == "parquet":
                csv_path = out_dir / "server_metrics.csv"
                df.to_csv(csv_path, index=False)
                print(f"📄 CSV written: {csv_path} ({len(df):,} rows)")
    
    # Create metadata file for backward compatibility
    metadata = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'total_samples': len(df),
        'time_span_hours': config.hours,
        'start_time': df['timestamp'].min().isoformat(),
        'end_time': df['timestamp'].max().isoformat(),
        'servers_count': len(df['server_name'].unique()),
        'problem_children_pct': config.problem_child_pct,
        'tick_interval_seconds': config.tick_seconds,
        'profiles': df['profile'].value_counts().to_dict(),
        'format_version': "3.0_fleet"
    }
    
    with open(out_dir / 'metrics_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def validate_output(df: pd.DataFrame, config: Config) -> bool:
    """Validate generated data meets requirements."""
    print("\n🔍 Validating generated data...")
    
    try:
        # Check timestamp intervals
        timestamps = df['timestamp'].unique()
        intervals = pd.Series(timestamps[1:]) - pd.Series(timestamps[:-1])
        expected_interval = pd.Timedelta(seconds=config.tick_seconds)
        
        assert all(intervals == expected_interval), "Timestamp intervals not consistent"
        
        # Check server coverage
        servers_per_timestamp = df.groupby('timestamp')['server_name'].nunique()
        expected_servers = len(df['server_name'].unique())
        
        assert all(servers_per_timestamp == expected_servers), "Missing servers for some timestamps"
        
        # Check problem children percentage
        total_servers = len(df['server_name'].unique()) 
        problem_children = df['problem_child'].sum() / len(df) * total_servers
        problem_pct = problem_children / total_servers
        
        assert 0.08 <= problem_pct <= 0.12, f"Problem children % {problem_pct:.3f} outside [8%, 12%]"
        
        # Check profile baselines (sample production servers)
        prod_data = df[df['profile'] == 'production']
        if not prod_data.empty:
            mean_cpu = prod_data['cpu_pct'].mean()
            mean_mem = prod_data['mem_pct'].mean()
            assert 25 <= mean_cpu <= 40, f"Production CPU mean {mean_cpu:.1f}% outside [25%, 40%]"
            assert 40 <= mean_mem <= 50, f"Production mem mean {mean_mem:.1f}% outside [40%, 50%]"
        
        print("✅ All validation checks passed")
        return True
        
    except AssertionError as e:
        print(f"❌ Validation failed: {e}")
        return False

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Generate realistic server fleet telemetry")
    
    # Time settings
    parser.add_argument("--start", default=None, 
                       help="Start timestamp (ISO format). If not specified, auto-calculated as current time minus duration")
    parser.add_argument("--hours", type=int, default=24, 
                       help="Duration in hours")
    parser.add_argument("--tick_seconds", type=int, default=5, 
                       help="Polling interval in seconds")
    
    # Fleet composition
    parser.add_argument("--num_prod", type=int, default=20, 
                       help="Number of production servers")
    parser.add_argument("--num_stage", type=int, default=5, 
                       help="Number of staging servers")
    parser.add_argument("--num_compute", type=int, default=10, 
                       help="Number of compute servers")
    parser.add_argument("--num_service", type=int, default=10, 
                       help="Number of service servers")
    parser.add_argument("--num_container", type=int, default=5, 
                       help="Number of container servers")
    
    # Behavior
    parser.add_argument("--problem_child_pct", type=float, default=0.10, 
                       help="Fraction of problem child servers")
    parser.add_argument("--offline_fill", choices=["nan", "zeros"], default="nan",
                       help="How to fill metrics for offline servers")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    
    # Output
    parser.add_argument("--out_dir", default="./training",
                       help="Output directory")
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="parquet",
                       help="Output format (default: parquet for speed)")
    parser.add_argument("--csv", action="store_true",
                       help="Also output CSV format (slower)")
    parser.add_argument("--json", action="store_true",
                       help="Also output JSON format (slowest, legacy)")
    
    args = parser.parse_args()

    # Determine output format based on flags
    output_format = args.format
    if args.csv and args.json:
        output_format = "both"  # Legacy: output all formats
    elif args.csv:
        output_format = "both"  # Parquet + CSV
    # args.json is handled separately below

    # Create configuration
    config = Config(
        start_time=args.start,
        hours=args.hours,
        tick_seconds=args.tick_seconds,
        num_prod=args.num_prod,
        num_stage=args.num_stage,
        num_compute=args.num_compute,
        num_service=args.num_service,
        num_container=args.num_container,
        problem_child_pct=args.problem_child_pct,
        offline_fill=args.offline_fill,
        seed=args.seed,
        out_dir=args.out_dir,
        output_format=output_format
    )
    
    print("🚀 Enhanced Fleet Telemetry Generator")
    print("=" * 50)
    
    # Generate fleet
    print("📡 Creating server fleet...")
    fleet = make_server_fleet(config)
    
    # Generate schedule
    print("⏰ Generating timestamp schedule...")
    schedule = generate_schedule(config)
    
    # Simulate states
    print("🔄 Simulating operational states...")
    state_df = simulate_states(fleet, schedule, config)
    
    # Simulate metrics
    print("📊 Generating realistic metrics...")
    final_df = simulate_metrics(state_df, config)
    
    # Write outputs
    print("💾 Writing output files...")
    write_outputs(final_df, config)

    # Optional JSON output (if --json flag used)
    if args.json:
        print("📝 Writing JSON output (legacy format)...")
        json_path = Path(config.out_dir) / 'metrics_dataset.json'
        records = final_df.to_dict('records')
        metadata = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_samples': len(records),
            'time_span_hours': config.hours,
            'servers_count': len(fleet),
            'problem_children_pct': config.problem_child_pct,
            'format_version': "3.0_fleet"
        }
        with open(json_path, 'w') as f:
            json.dump({'records': records, 'metadata': metadata}, f, indent=2, default=str)
        print(f"📊 JSON written: {json_path} ({len(records):,} records)")

    # Validate
    validate_output(final_df, config)
    
    # Summary
    print(f"\n📋 Generation Summary:")
    print(f"   Time range: {schedule[0]} to {schedule[-1]}")
    print(f"   Total duration: {config.hours} hours")
    print(f"   Sampling interval: {config.tick_seconds} seconds") 
    print(f"   Total timestamps: {len(schedule):,}")
    print(f"   Total rows: {len(final_df):,}")
    print(f"   Total servers: {len(fleet):,}")
    print(f"   Problem children: {fleet['problem_child'].sum()} ({fleet['problem_child'].mean():.1%})")
    
    print(f"\n🏗️  Fleet Composition:")
    for profile in ServerProfile:
        count = (fleet['profile'] == profile.value).sum()
        if count > 0:
            print(f"   {profile.value.title()}: {count}")
    
    print(f"\n👀 Sample Data (first 5 rows):")
    display_cols = ['timestamp', 'server_name', 'profile', 'state', 
                   'cpu_pct', 'mem_pct', 'latency_ms', 'error_rate']
    sample_df = final_df[display_cols].head()
    
    for _, row in sample_df.iterrows():
        ts = row['timestamp'].strftime('%H:%M:%S')
        print(f"   {ts} | {row['server_name']} | {row['profile'][:4]} | "
              f"{row['state'][:8]:8s} | CPU:{row['cpu_pct']:5.1f}% | "
              f"Mem:{row['mem_pct']:5.1f}% | Lat:{row['latency_ms']:5.1f}ms")
    
    print(f"\n✅ Fleet telemetry generation complete!")
    return 0

# Module interface for notebook usage
def generate_dataset(hours: int = 24, output_file: Optional[str] = None) -> bool:
    """Generate dataset - module interface for backward compatibility."""
    config = Config(
        hours=hours,
        out_dir=str(Path(output_file).parent) if output_file else "./training",
        num_prod=10,  # Smaller default for compatibility
        num_stage=3,
        num_compute=5,
        num_service=5,
        num_container=2
    )
    
    try:
        fleet = make_server_fleet(config)
        schedule = generate_schedule(config)  
        state_df = simulate_states(fleet, schedule, config)
        final_df = simulate_metrics(state_df, config)
        write_outputs(final_df, config)
        
        # Create backward-compatible JSON format
        if output_file and output_file.endswith('.json'):
            records = final_df.to_dict('records')
            metadata = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'total_samples': len(records),
                'time_span_hours': hours,
                'servers_count': len(fleet),
                'format_version': "3.0_enhanced"
            }
            
            with open(output_file, 'w') as f:
                json.dump({'records': records, 'metadata': metadata}, f, 
                         indent=2, default=str)
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset generation failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    sys.exit(main())