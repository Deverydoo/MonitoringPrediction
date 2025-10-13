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
    """
    Server profile types for financial ML platform (Spectrum Conductor).

    Profiles define expected behavior patterns, resource usage baselines,
    and incident characteristics. Used for transfer learning when new
    servers come online.
    """
    ML_COMPUTE = "ml_compute"           # ML training nodes (Spectrum Conductor workers)
    DATABASE = "database"                # Oracle, PostgreSQL, MongoDB
    WEB_API = "web_api"                 # Web servers, API gateways, REST endpoints
    CONDUCTOR_MGMT = "conductor_mgmt"   # Spectrum Conductor management/scheduler nodes
    DATA_INGEST = "data_ingest"         # ETL, Kafka, Spark streaming
    RISK_ANALYTICS = "risk_analytics"   # Risk calculation, VaR, Monte Carlo simulations
    GENERIC = "generic"                  # Fallback for unknown/unclassified servers

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

    # Fleet composition - Use EITHER total_servers OR individual counts
    total_servers: Optional[int] = None  # Auto-distributes across profiles (recommended)
    num_ml_compute: Optional[int] = None       # ML training nodes (CPU/GPU intensive)
    num_database: Optional[int] = None          # Database servers (I/O + memory intensive)
    num_web_api: Optional[int] = None           # Web/API servers (network intensive)
    num_conductor_mgmt: Optional[int] = None     # Conductor management nodes
    num_data_ingest: Optional[int] = None       # ETL/streaming servers
    num_risk_analytics: Optional[int] = None     # Risk calculation servers
    num_generic: Optional[int] = None            # Generic/utility servers

    # Behavior settings
    problem_child_pct: float = 0.10
    offline_mode: str = "dense"  # "dense" (all rows) or "sparse" (no offline rows) - DEFAULT: "dense" for stability
    offline_fill: str = "zeros"  # "nan" or "zeros" (only used in dense mode) - DEFAULT: "zeros"
    seed: Optional[int] = 42

    # Output settings
    out_dir: str = "./training"
    output_format: str = "parquet"  # "csv", "parquet", or "both" (default: parquet only)

    def __post_init__(self):
        """Validate configuration and auto-distribute servers if needed."""
        if not 0.08 <= self.problem_child_pct <= 0.12:
            raise ValueError("problem_child_pct must be between 8% and 12%")
        if self.offline_mode not in ["dense", "sparse"]:
            raise ValueError("offline_mode must be 'dense' or 'sparse'")
        if self.offline_fill not in ["nan", "zeros"]:
            raise ValueError("offline_fill must be 'nan' or 'zeros'")

        # Smart fleet distribution
        if self.total_servers is not None:
            self._distribute_fleet()
        else:
            # Check if any counts specified
            counts_specified = [self.num_ml_compute, self.num_database, self.num_web_api,
                              self.num_conductor_mgmt, self.num_data_ingest,
                              self.num_risk_analytics, self.num_generic]
            if all(c is None for c in counts_specified):
                # No fleet specified - use default 90
                self.total_servers = 90
                self._distribute_fleet()
            else:
                # Some counts specified - fill in None with defaults
                if self.num_ml_compute is None: self.num_ml_compute = 20
                if self.num_database is None: self.num_database = 15
                if self.num_web_api is None: self.num_web_api = 25
                if self.num_conductor_mgmt is None: self.num_conductor_mgmt = 5
                if self.num_data_ingest is None: self.num_data_ingest = 10
                if self.num_risk_analytics is None: self.num_risk_analytics = 8
                if self.num_generic is None: self.num_generic = 7

    def _distribute_fleet(self):
        """
        Intelligently distribute servers across profiles.

        Distribution Strategy (Financial ML Platform):
        - Web/API: 28% (highest - user-facing services)
        - ML Compute: 22% (training workloads)
        - Database: 17% (critical infrastructure)
        - Data Ingest: 11% (ETL pipelines)
        - Risk Analytics: 9% (EOD calculations)
        - Generic: 7% (utility, capped at 10 max)
        - Conductor Mgmt: 6% (orchestration)

        Guarantees:
        - Minimum 1 server per profile (7 total minimum)
        - Generic capped at 10 servers max
        - Exact total matches request
        """
        n = max(self.total_servers, 7)  # Minimum 7 (one per profile)

        if n < 7:
            print(f"[FLEET] Enforcing minimum: 7 servers (1 per profile)")

        # Distribution percentages (sum = 100%)
        pct = {
            'web_api': 0.28,         # 28% - User-facing
            'ml_compute': 0.22,      # 22% - Training
            'database': 0.17,        # 17% - Data layer
            'data_ingest': 0.11,     # 11% - ETL
            'risk_analytics': 0.09,  # 9% - Risk calcs
            'generic': 0.07,         # 7% - Utility (max 10)
            'conductor_mgmt': 0.06   # 6% - Orchestration
        }

        # Calculate raw distribution
        dist = {k: max(1, int(n * v)) for k, v in pct.items()}

        # Cap generic at 10
        if dist['generic'] > 10:
            excess = dist['generic'] - 10
            dist['generic'] = 10
            dist['ml_compute'] += excess  # Give excess to ml_compute

        # Adjust to match exact total
        diff = n - sum(dist.values())
        if diff != 0:
            # Add/subtract from largest flexible profile
            target = 'web_api' if diff > 0 else max(dist, key=dist.get)
            dist[target] += diff

        # Apply to config
        self.num_web_api = dist['web_api']
        self.num_ml_compute = dist['ml_compute']
        self.num_database = dist['database']
        self.num_data_ingest = dist['data_ingest']
        self.num_risk_analytics = dist['risk_analytics']
        self.num_generic = dist['generic']
        self.num_conductor_mgmt = dist['conductor_mgmt']

        # Validate
        actual_total = sum(dist.values())
        assert actual_total == n, f"Distribution error: {actual_total} != {n}"
        assert all(v >= 1 for v in dist.values()), "All profiles must have ≥1 server"
        assert dist['generic'] <= 10, "Generic capped at 10"

# Profile baselines: (mean, std) for each metric
# Realistic values for financial ML platform with Spectrum Conductor
PROFILE_BASELINES = {
    ServerProfile.ML_COMPUTE: {
        # High CPU/Memory during training, moderate I/O, low network
        "cpu": (0.78, 0.12), "mem": (0.82, 0.10), "disk_io_mb_s": (45.0, 15.0),
        "net_in_mb_s": (8.5, 3.0), "net_out_mb_s": (5.2, 2.0),
        "latency_ms": (22.0, 5.0), "error_rate": (0.002, 0.001), "gc_pause_ms": (15.0, 8.0)
    },
    ServerProfile.DATABASE: {
        # Moderate CPU, high memory (caching), very high I/O, moderate network
        "cpu": (0.55, 0.15), "mem": (0.87, 0.08), "disk_io_mb_s": (180.0, 45.0),
        "net_in_mb_s": (35.0, 12.0), "net_out_mb_s": (28.0, 10.0),
        "latency_ms": (8.5, 3.0), "error_rate": (0.001, 0.0005), "gc_pause_ms": (5.0, 2.0)
    },
    ServerProfile.WEB_API: {
        # Low CPU, moderate memory, low I/O, high network (user traffic)
        "cpu": (0.28, 0.08), "mem": (0.55, 0.12), "disk_io_mb_s": (12.0, 4.0),
        "net_in_mb_s": (85.0, 25.0), "net_out_mb_s": (120.0, 35.0),
        "latency_ms": (45.0, 15.0), "error_rate": (0.004, 0.002), "gc_pause_ms": (8.0, 3.0)
    },
    ServerProfile.CONDUCTOR_MGMT: {
        # Low CPU (scheduling), high memory (job queue), low I/O, moderate network
        "cpu": (0.28, 0.08), "mem": (0.75, 0.10), "disk_io_mb_s": (18.0, 6.0),
        "net_in_mb_s": (22.0, 8.0), "net_out_mb_s": (18.0, 6.0),
        "latency_ms": (15.0, 4.0), "error_rate": (0.001, 0.0005), "gc_pause_ms": (10.0, 4.0)
    },
    ServerProfile.DATA_INGEST: {
        # Moderate CPU (transforms), high memory (buffering), very high I/O, high network
        "cpu": (0.60, 0.15), "mem": (0.78, 0.12), "disk_io_mb_s": (220.0, 60.0),
        "net_in_mb_s": (150.0, 45.0), "net_out_mb_s": (95.0, 30.0),
        "latency_ms": (12.0, 5.0), "error_rate": (0.003, 0.001), "gc_pause_ms": (18.0, 8.0)
    },
    ServerProfile.RISK_ANALYTICS: {
        # High CPU (Monte Carlo), high memory (matrices), moderate I/O, low network
        "cpu": (0.82, 0.10), "mem": (0.88, 0.08), "disk_io_mb_s": (38.0, 12.0),
        "net_in_mb_s": (12.0, 4.0), "net_out_mb_s": (8.0, 3.0),
        "latency_ms": (18.0, 6.0), "error_rate": (0.002, 0.001), "gc_pause_ms": (25.0, 12.0)
    },
    ServerProfile.GENERIC: {
        # Balanced baseline for unknown servers
        "cpu": (0.35, 0.10), "mem": (0.50, 0.12), "disk_io_mb_s": (25.0, 10.0),
        "net_in_mb_s": (15.0, 8.0), "net_out_mb_s": (12.0, 6.0),
        "latency_ms": (30.0, 10.0), "error_rate": (0.003, 0.001), "gc_pause_ms": (10.0, 5.0)
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
    """
    Generate server names following financial institution naming conventions.

    Naming patterns enable automatic profile inference in production.
    """
    if profile == ServerProfile.ML_COMPUTE:
        return [f"ppml{i:04d}" for i in range(1, count + 1)]
    elif profile == ServerProfile.DATABASE:
        return [f"ppdb{i:03d}" for i in range(1, count + 1)]
    elif profile == ServerProfile.WEB_API:
        return [f"ppweb{i:03d}" for i in range(1, count + 1)]
    elif profile == ServerProfile.CONDUCTOR_MGMT:
        return [f"ppcon{i:02d}" for i in range(1, count + 1)]
    elif profile == ServerProfile.DATA_INGEST:
        return [f"ppetl{i:03d}" for i in range(1, count + 1)]
    elif profile == ServerProfile.RISK_ANALYTICS:
        return [f"pprisk{i:03d}" for i in range(1, count + 1)]
    elif profile == ServerProfile.GENERIC:
        return [f"ppgen{i:03d}" for i in range(1, count + 1)]
    else:
        raise ValueError(f"Unknown profile: {profile}")


def infer_profile_from_name(server_name: str) -> ServerProfile:
    """
    Infer server profile from naming convention.

    Uses regex patterns to match common prefixes. This enables transfer
    learning when new servers come online - model can predict based on
    profile patterns learned from existing servers.

    Examples:
        ppml0015 -> ML_COMPUTE
        ppdb042 -> DATABASE
        ppweb123 -> WEB_API
        unknown_server -> GENERIC (fallback)
    """
    import re

    # Profile detection patterns (order matters - more specific first)
    patterns = [
        (r'^ppml\d+', ServerProfile.ML_COMPUTE),
        (r'^ppgpu\d+', ServerProfile.ML_COMPUTE),
        (r'^cptrain\d+', ServerProfile.ML_COMPUTE),
        (r'^ppdb\d+', ServerProfile.DATABASE),
        (r'^psdb\d+', ServerProfile.DATABASE),
        (r'^oracle\d+', ServerProfile.DATABASE),
        (r'^mongo\d+', ServerProfile.DATABASE),
        (r'^postgres\d+', ServerProfile.DATABASE),
        (r'^ppweb\d+', ServerProfile.WEB_API),
        (r'^ppapi\d+', ServerProfile.WEB_API),
        (r'^nginx\d+', ServerProfile.WEB_API),
        (r'^tomcat\d+', ServerProfile.WEB_API),
        (r'^ppcon\d+', ServerProfile.CONDUCTOR_MGMT),
        (r'^conductor\d+', ServerProfile.CONDUCTOR_MGMT),
        (r'^egomgmt\d+', ServerProfile.CONDUCTOR_MGMT),
        (r'^ppetl\d+', ServerProfile.DATA_INGEST),
        (r'^ppkafka\d+', ServerProfile.DATA_INGEST),
        (r'^stream\d+', ServerProfile.DATA_INGEST),
        (r'^spark\d+', ServerProfile.DATA_INGEST),
        (r'^pprisk\d+', ServerProfile.RISK_ANALYTICS),
        (r'^varrisk\d+', ServerProfile.RISK_ANALYTICS),
        (r'^credit\d+', ServerProfile.RISK_ANALYTICS),
    ]

    server_lower = server_name.lower()

    for pattern, profile in patterns:
        if re.match(pattern, server_lower):
            return profile

    # Fallback to generic if no match
    return ServerProfile.GENERIC

def make_server_fleet(config: Config) -> pd.DataFrame:
    """
    Create server fleet with realistic profiles for financial ML platform.

    Generates diverse fleet matching production infrastructure:
    - ML training nodes (Spectrum Conductor workers)
    - Database servers (Oracle, PostgreSQL, MongoDB)
    - Web/API servers (REST endpoints, gateways)
    - Conductor management nodes (scheduling, job queue)
    - Data ingestion servers (ETL, Kafka, Spark)
    - Risk analytics servers (VaR, Monte Carlo)
    - Generic utility servers
    """
    fleet = []

    # Generate servers by profile (Financial ML Platform composition)
    profiles_counts = [
        (ServerProfile.ML_COMPUTE, config.num_ml_compute),
        (ServerProfile.DATABASE, config.num_database),
        (ServerProfile.WEB_API, config.num_web_api),
        (ServerProfile.CONDUCTOR_MGMT, config.num_conductor_mgmt),
        (ServerProfile.DATA_INGEST, config.num_data_ingest),
        (ServerProfile.RISK_ANALYTICS, config.num_risk_analytics),
        (ServerProfile.GENERIC, config.num_generic)
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
    """
    Calculate time-of-day multiplier for metrics.

    Financial institutions have distinct patterns:
    - Market hours: 9:30am-4pm EST (peak trading)
    - Pre-market: 7am-9:30am (analytics, prep)
    - After-hours: 4pm-8pm (EOD processing, risk calculations)
    - Overnight: 8pm-7am (batch jobs, ML training)
    """
    # Base diurnal curve (simplified - Eastern Time assumed)
    if 7 <= hour <= 9:  # Pre-market preparation
        base = 1.3
    elif 9 <= hour <= 16:  # Market hours (peak)
        base = 1.5
    elif 16 <= hour <= 19:  # After-hours (EOD processing)
        base = 1.6  # Often busiest time
    elif 19 <= hour <= 23:  # Evening batch/ML training
        base = 1.2
    else:  # Overnight (0-7am)
        base = 0.6

    # Profile-specific adjustments (financial platform patterns)
    if profile == ServerProfile.ML_COMPUTE:
        # ML training runs overnight and evenings
        if 19 <= hour <= 6:
            base *= 1.4
    elif profile == ServerProfile.DATABASE:
        # Databases busy during market hours, EOD reports
        if 9 <= hour <= 16:
            base *= 1.2
        if 16 <= hour <= 19:
            base *= 1.4  # EOD queries spike
    elif profile == ServerProfile.WEB_API:
        # User traffic follows market hours
        if 9 <= hour <= 16:
            base *= 1.3
        elif hour < 7 or hour > 20:
            base *= 0.5  # Low usage overnight
    elif profile == ServerProfile.RISK_ANALYTICS:
        # Risk calculations at market close
        if 16 <= hour <= 19:
            base *= 2.0  # EOD risk window is critical
    elif profile == ServerProfile.DATA_INGEST:
        # Streaming data heaviest during market hours
        if 9 <= hour <= 16:
            base *= 1.5

    return max(0.3, min(2.5, base))

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
    """
    Simulate server states using Markov chain.

    Supports two modes:
    - Dense: All servers appear at all timestamps (offline servers have state='offline')
    - Sparse: Offline servers don't appear (realistic production behavior)
    """
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

            # SPARSE MODE: Skip offline servers (they don't send data)
            if config.offline_mode == "sparse" and next_state == ServerState.OFFLINE:
                continue  # Don't create row for offline server

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

                # Handle offline state (only in dense mode - sparse mode already filtered)
                if config.offline_mode == "dense":
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
        
        # Special handling for profile-specific incidents
        if profile == ServerProfile.ML_COMPUTE:
            # Higher chance of OOM during high memory usage (ML models)
            mem_values = df.loc[server_mask, 'mem_pct']
            oom_prob = np.where(mem_values > 85, 0.08, 0.001)  # 8% chance when mem > 85%
            df.loc[server_mask, 'container_oom'] = np.random.binomial(1, oom_prob)
        elif profile == ServerProfile.RISK_ANALYTICS:
            # OOM risk from large matrix operations
            mem_values = df.loc[server_mask, 'mem_pct']
            oom_prob = np.where(mem_values > 88, 0.06, 0.0005)
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
        'offline_mode': config.offline_mode,
        'offline_fill': config.offline_fill if config.offline_mode == "dense" else None,
        'profiles': df['profile'].value_counts().to_dict(),
        'format_version': "3.1_sparse_support"
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
        
        # Check server coverage (only for dense mode)
        if config.offline_mode == "dense":
            servers_per_timestamp = df.groupby('timestamp')['server_name'].nunique()
            expected_servers = len(df['server_name'].unique())
            assert all(servers_per_timestamp == expected_servers), "Missing servers for some timestamps"
        else:
            # Sparse mode: variable server count per timestamp is expected
            servers_per_timestamp = df.groupby('timestamp')['server_name'].nunique()
            print(f"   Sparse mode: avg {servers_per_timestamp.mean():.1f} servers per timestamp (variable)")
        
        # Check problem children percentage
        total_servers = len(df['server_name'].unique()) 
        problem_children = df['problem_child'].sum() / len(df) * total_servers
        problem_pct = problem_children / total_servers
        
        assert 0.08 <= problem_pct <= 0.12, f"Problem children % {problem_pct:.3f} outside [8%, 12%]"
        
        # Check profile baselines (sample ML compute servers - highest resource usage)
        ml_data = df[df['profile'] == 'ml_compute']
        if not ml_data.empty:
            mean_cpu = ml_data['cpu_pct'].mean()
            mean_mem = ml_data['mem_pct'].mean()
            assert 60 <= mean_cpu <= 90, f"ML Compute CPU mean {mean_cpu:.1f}% outside [60%, 90%]"
            assert 70 <= mean_mem <= 95, f"ML Compute mem mean {mean_mem:.1f}% outside [70%, 95%]"
        
        print("✅ All validation checks passed")
        return True
        
    except AssertionError as e:
        print(f"❌ Validation failed: {e}")
        return False

def stream_to_daemon(config: Config, daemon_url: str = "http://localhost:8000", scenario: str = "healthy"):
    """
    Stream mode: Generate and feed data to inference daemon in real-time.

    Perfect for demos - uses the same awesome metrics_generator logic that creates training data!

    Args:
        config: Configuration (uses total_servers or individual counts to match training)
        daemon_url: Inference daemon URL
        scenario: 'healthy', 'degrading', or 'critical'
    """
    import requests
    import time

    print(f"\n🎬 STREAMING MODE - Live Demo Feed")
    print("=" * 60)
    print(f"   Daemon: {daemon_url}")
    print(f"   Scenario: {scenario.upper()}")
    print(f"   Fleet: {config.num_ml_compute + config.num_database + config.num_web_api + config.num_conductor_mgmt + config.num_data_ingest + config.num_risk_analytics + config.num_generic} servers")
    print(f"   Tick: Every {config.tick_seconds} seconds")
    print("=" * 60)
    print()

    # Create fleet (matches training data exactly)
    fleet = make_server_fleet(config)
    print(f"✅ Fleet created: {len(fleet)} servers")

    # Initialize server states
    server_states = {name: ServerState.HEALTHY for name in fleet['server_name']}

    # Scenario multipliers for degradation
    scenario_multipliers = {
        'healthy': 1.0,
        'degrading': 1.4,
        'critical': 2.2
    }
    scenario_mult = scenario_multipliers.get(scenario, 1.0)

    # Determine affected servers for degrading/critical scenarios
    if scenario != 'healthy':
        num_affected = max(1, int(len(fleet) * 0.25))  # 25% of fleet
        affected_servers = set(np.random.choice(fleet['server_name'], num_affected, replace=False))
        print(f"⚠️  {num_affected} servers will degrade: {', '.join(list(affected_servers)[:5])}...")
    else:
        affected_servers = set()

    tick_count = 0
    start_time = datetime.now()

    print(f"\n🚀 Starting stream at {start_time.strftime('%H:%M:%S')}")
    print("   Press Ctrl+C to stop\n")

    try:
        while True:
            tick_count += 1
            current_time = datetime.now()

            # Generate one tick of data for entire fleet
            batch = []

            for _, server in fleet.iterrows():
                server_name = server['server_name']
                profile = server['profile']
                is_problem_child = server['problem_child']
                is_affected = server_name in affected_servers

                # State transitions
                current_state = server_states[server_name]
                hour = current_time.hour
                probs = get_state_transition_probs(current_state, hour, is_problem_child)
                states = list(probs.keys())
                probabilities = list(probs.values())
                next_state = np.random.choice(states, p=probabilities)
                server_states[server_name] = next_state

                # Skip offline servers (sparse mode for streaming)
                if next_state == ServerState.OFFLINE:
                    continue

                # Generate metrics
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
                        if is_affected and scenario != 'healthy':
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

            # Send to daemon
            try:
                response = requests.post(
                    f"{daemon_url}/feed/data",
                    json={"records": batch},
                    timeout=2
                )

                if response.ok:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    status = "🟢 HEALTHY" if scenario == 'healthy' else ("🟡 DEGRADING" if scenario == 'degrading' else "🔴 CRITICAL")
                    print(f"[{current_time.strftime('%H:%M:%S')}] Tick {tick_count:4d} | {status} | {len(batch)} servers | Elapsed: {elapsed:.0f}s")
                else:
                    print(f"❌ Daemon error: {response.status_code}")

            except requests.exceptions.ConnectionError:
                print(f"⚠️  Cannot connect to daemon at {daemon_url}")
                print("   Make sure daemon is running: python tft_inference.py --daemon")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

            # Wait for next tick
            time.sleep(config.tick_seconds)

    except KeyboardInterrupt:
        print(f"\n\n⏹️  Stream stopped after {tick_count} ticks")
        print(f"   Duration: {(datetime.now() - start_time).total_seconds():.0f} seconds")
        print("   ✅ Clean shutdown")

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Generate realistic server fleet telemetry")

    # Mode selection
    parser.add_argument("--stream", action="store_true",
                       help="Stream mode: continuously feed data to inference daemon (perfect for demos!)")
    parser.add_argument("--scenario", choices=["healthy", "degrading", "critical"], default="healthy",
                       help="Scenario for stream mode (default: healthy)")
    parser.add_argument("--daemon-url", default="http://localhost:8000",
                       help="Inference daemon URL for stream mode")

    # Time settings
    parser.add_argument("--start", default=None,
                       help="Start timestamp (ISO format). If not specified, auto-calculated as current time minus duration")
    parser.add_argument("--hours", type=int, default=24,
                       help="Duration in hours (batch mode only)")
    parser.add_argument("--tick_seconds", type=int, default=5,
                       help="Polling interval in seconds")
    
    # Fleet composition (Financial ML Platform)
    parser.add_argument("--num_ml_compute", type=int, default=20,
                       help="Number of ML training servers (Spectrum Conductor workers)")
    parser.add_argument("--num_database", type=int, default=15,
                       help="Number of database servers (Oracle, PostgreSQL, MongoDB)")
    parser.add_argument("--num_web_api", type=int, default=25,
                       help="Number of web/API servers (REST endpoints, gateways)")
    parser.add_argument("--num_conductor_mgmt", type=int, default=5,
                       help="Number of Conductor management nodes (scheduling)")
    parser.add_argument("--num_data_ingest", type=int, default=10,
                       help="Number of data ingestion servers (ETL, Kafka, Spark)")
    parser.add_argument("--num_risk_analytics", type=int, default=8,
                       help="Number of risk analytics servers (VaR, Monte Carlo)")
    parser.add_argument("--num_generic", type=int, default=7,
                       help="Number of generic/utility servers")
    
    # Behavior
    parser.add_argument("--problem_child_pct", type=float, default=0.10,
                       help="Fraction of problem child servers")
    parser.add_argument("--offline_mode", choices=["dense", "sparse"], default="dense",
                       help="Dense: all servers at all times (zeros when offline). Sparse: no rows for offline servers (realistic)")
    parser.add_argument("--offline_fill", choices=["nan", "zeros"], default="zeros",
                       help="How to fill metrics for offline servers in dense mode (default: zeros for training compatibility)")
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
        num_ml_compute=args.num_ml_compute,
        num_database=args.num_database,
        num_web_api=args.num_web_api,
        num_conductor_mgmt=args.num_conductor_mgmt,
        num_data_ingest=args.num_data_ingest,
        num_risk_analytics=args.num_risk_analytics,
        num_generic=args.num_generic,
        problem_child_pct=args.problem_child_pct,
        offline_mode=args.offline_mode,
        offline_fill=args.offline_fill,
        seed=args.seed,
        out_dir=args.out_dir,
        output_format=output_format
    )

    # STREAMING MODE - Feed daemon in real-time
    if args.stream:
        stream_to_daemon(config, daemon_url=args.daemon_url, scenario=args.scenario)
        return 0

    # BATCH MODE - Generate training dataset
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
def generate_dataset(hours: int = 24, total_servers: int = 90, output_file: Optional[str] = None) -> bool:
    """
    Generate dataset with smart profile distribution.

    Args:
        hours: Duration in hours (default: 24)
        total_servers: Total fleet size - auto-distributes across profiles (default: 90)
        output_file: Optional output file path

    Returns:
        True if successful, False otherwise

    Examples:
        >>> generate_dataset(hours=24, total_servers=90)  # Full fleet
        >>> generate_dataset(hours=24, total_servers=10)  # Small test (still gets all 7 profiles)
        >>> generate_dataset(hours=720, total_servers=45)  # 1 month, medium fleet
    """
    config = Config(
        hours=hours,
        total_servers=total_servers,  # Smart distribution handles the rest
        out_dir=str(Path(output_file).parent) if output_file else "./training"
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