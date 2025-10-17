"""
Metrics Configuration - LINBORG Baselines and State Multipliers

SINGLE SOURCE OF TRUTH for all metric generation parameters:
- Profile baselines (7 server profiles × 14 LINBORG metrics)
- State multipliers (8 operational states × 14 metrics)
- Diurnal patterns (time-of-day effects)
- Server naming conventions
- State transition probabilities

ALL metrics_generator.py logic uses these values.
"""

from enum import Enum
from typing import Dict, Tuple

# =============================================================================
# Server Profiles and States (Enums)
# =============================================================================

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


# =============================================================================
# LINBORG Metrics Configuration
# =============================================================================

# 14 LINBORG Metrics (matches production monitoring)
LINBORG_METRICS = [
    # CPU metrics (5)
    'cpu_user_pct',
    'cpu_sys_pct',
    'cpu_iowait_pct',      # CRITICAL - "System troubleshooting 101"
    'cpu_idle_pct',
    'java_cpu_pct',

    # Memory metrics (2)
    'mem_used_pct',
    'swap_used_pct',

    # Disk metrics (1)
    'disk_usage_pct',

    # Network metrics (2)
    'net_in_mb_s',
    'net_out_mb_s',

    # Connection metrics (2)
    'back_close_wait',
    'front_close_wait',

    # System metrics (2)
    'load_average',
    'uptime_days'
]

# =============================================================================
# Profile Baselines - (mean, std) for each metric
# =============================================================================
# ** LINBORG-COMPATIBLE METRICS ** - Matches actual production monitoring
# All CPU metrics are 0-1 scale (will be converted to 0-100% in output)

PROFILE_BASELINES = {
    ServerProfile.ML_COMPUTE: {
        # ML training nodes - High compute during jobs, Spark/Java heavy
        "cpu_user": (0.45, 0.12),      # User space (Spark workers)
        "cpu_sys": (0.08, 0.03),       # System/kernel
        "cpu_iowait": (0.02, 0.01),    # I/O wait (should be low)
        "cpu_idle": (0.45, 0.15),      # Idle (inverse of busy)
        "java_cpu": (0.50, 0.15),      # Java/Spark CPU usage
        "mem_used": (0.72, 0.10),      # High memory for models
        "swap_used": (0.05, 0.03),     # Minimal swap (bad if high)
        "disk_usage": (0.55, 0.08),    # Checkpoints, logs
        "net_in_mb_s": (8.5, 3.0),     # Network ingress
        "net_out_mb_s": (5.2, 2.0),    # Network egress
        "back_close_wait": (2, 1),     # TCP back connections
        "front_close_wait": (2, 1),    # TCP front connections
        "load_average": (6.5, 2.0),    # System load
        "uptime_days": (25, 2)         # ~monthly maintenance
    },
    ServerProfile.DATABASE: {
        # Database servers - I/O intensive, high iowait
        "cpu_user": (0.25, 0.08),
        "cpu_sys": (0.12, 0.04),       # Higher system (I/O operations)
        "cpu_iowait": (0.15, 0.05),    # ** HIGH - Critical for DBs **
        "cpu_idle": (0.48, 0.12),
        "java_cpu": (0.10, 0.05),      # Minimal Java
        "mem_used": (0.68, 0.10),      # Buffer pools
        "swap_used": (0.03, 0.02),
        "disk_usage": (0.70, 0.10),    # Databases fill disks
        "net_in_mb_s": (35.0, 12.0),   # High network (queries)
        "net_out_mb_s": (28.0, 10.0),
        "back_close_wait": (8, 3),     # Many connections
        "front_close_wait": (6, 2),
        "load_average": (4.2, 1.5),
        "uptime_days": (25, 2)
    },
    ServerProfile.WEB_API: {
        # Web/API servers - Network heavy, lower compute
        "cpu_user": (0.18, 0.06),
        "cpu_sys": (0.05, 0.02),
        "cpu_iowait": (0.03, 0.02),
        "cpu_idle": (0.74, 0.10),      # Mostly idle
        "java_cpu": (0.25, 0.08),      # Tomcat/Spring
        "mem_used": (0.45, 0.10),      # Connection pools
        "swap_used": (0.02, 0.01),
        "disk_usage": (0.35, 0.08),    # Low disk (stateless)
        "net_in_mb_s": (85.0, 25.0),   # ** HIGH network **
        "net_out_mb_s": (120.0, 35.0),
        "back_close_wait": (15, 5),    # Lots of API connections
        "front_close_wait": (12, 4),
        "load_average": (2.8, 1.0),
        "uptime_days": (25, 2)
    },
    ServerProfile.CONDUCTOR_MGMT: {
        # Spectrum Conductor - Orchestration, job scheduling
        "cpu_user": (0.15, 0.06),
        "cpu_sys": (0.06, 0.02),
        "cpu_iowait": (0.02, 0.01),
        "cpu_idle": (0.77, 0.08),
        "java_cpu": (0.20, 0.06),      # EGO/Conductor Java
        "mem_used": (0.42, 0.08),      # Job metadata
        "swap_used": (0.02, 0.01),
        "disk_usage": (0.40, 0.08),    # Logs
        "net_in_mb_s": (22.0, 8.0),
        "net_out_mb_s": (18.0, 6.0),
        "back_close_wait": (5, 2),
        "front_close_wait": (4, 2),
        "load_average": (3.2, 1.2),
        "uptime_days": (25, 2)
    },
    ServerProfile.DATA_INGEST: {
        # ETL/Kafka/Spark streaming - High I/O and network
        "cpu_user": (0.35, 0.10),
        "cpu_sys": (0.10, 0.03),
        "cpu_iowait": (0.08, 0.03),    # Moderate I/O wait
        "cpu_idle": (0.47, 0.12),
        "java_cpu": (0.45, 0.12),      # Kafka/Spark Java
        "mem_used": (0.62, 0.12),      # Stream buffers
        "swap_used": (0.04, 0.02),
        "disk_usage": (0.65, 0.12),    # Writes data
        "net_in_mb_s": (150.0, 45.0),  # ** VERY HIGH ingress **
        "net_out_mb_s": (95.0, 30.0),
        "back_close_wait": (10, 4),
        "front_close_wait": (8, 3),
        "load_average": (5.5, 2.0),
        "uptime_days": (25, 2)
    },
    ServerProfile.RISK_ANALYTICS: {
        # Risk calculations - CPU intensive, EOD spikes
        "cpu_user": (0.55, 0.15),      # ** HIGH - Monte Carlo **
        "cpu_sys": (0.08, 0.03),
        "cpu_iowait": (0.02, 0.01),
        "cpu_idle": (0.35, 0.15),
        "java_cpu": (0.60, 0.15),      # Java risk calcs
        "mem_used": (0.70, 0.10),      # Simulation data
        "swap_used": (0.05, 0.03),
        "disk_usage": (0.50, 0.10),    # Result writes
        "net_in_mb_s": (12.0, 4.0),
        "net_out_mb_s": (8.0, 3.0),
        "back_close_wait": (3, 1),
        "front_close_wait": (2, 1),
        "load_average": (8.2, 2.5),    # High load during calcs
        "uptime_days": (25, 2)
    },
    ServerProfile.GENERIC: {
        # Utility/monitoring servers - Low everything
        "cpu_user": (0.12, 0.05),
        "cpu_sys": (0.04, 0.02),
        "cpu_iowait": (0.01, 0.01),
        "cpu_idle": (0.83, 0.08),      # Mostly idle
        "java_cpu": (0.08, 0.03),
        "mem_used": (0.35, 0.08),
        "swap_used": (0.01, 0.01),
        "disk_usage": (0.30, 0.08),
        "net_in_mb_s": (15.0, 8.0),
        "net_out_mb_s": (12.0, 6.0),
        "back_close_wait": (2, 1),
        "front_close_wait": (2, 1),
        "load_average": (1.5, 0.8),
        "uptime_days": (25, 2)
    }
}

# =============================================================================
# State Multipliers - Adjust baselines based on operational state
# =============================================================================
# IMPORTANT: These are multiplied with baselines AND diurnal patterns
# Keep moderate to avoid 100% CPU/memory in healthy scenarios

STATE_MULTIPLIERS = {
    ServerState.IDLE: {
        "cpu_user": 0.3, "cpu_sys": 0.5, "cpu_iowait": 0.2, "cpu_idle": 1.5,
        "java_cpu": 0.2, "mem_used": 0.8, "swap_used": 0.5, "disk_usage": 1.0,
        "net_in_mb_s": 0.2, "net_out_mb_s": 0.2, "back_close_wait": 0.3, "front_close_wait": 0.3,
        "load_average": 0.4, "uptime_days": 1.0
    },
    ServerState.HEALTHY: {
        "cpu_user": 1.0, "cpu_sys": 1.0, "cpu_iowait": 1.0, "cpu_idle": 1.0,
        "java_cpu": 1.0, "mem_used": 1.0, "swap_used": 1.0, "disk_usage": 1.0,
        "net_in_mb_s": 1.0, "net_out_mb_s": 1.0, "back_close_wait": 1.0, "front_close_wait": 1.0,
        "load_average": 1.0, "uptime_days": 1.0
    },
    ServerState.MORNING_SPIKE: {
        # Moderate spike - morning load (pre-market activity)
        "cpu_user": 1.3, "cpu_sys": 1.2, "cpu_iowait": 1.4, "cpu_idle": 0.7,
        "java_cpu": 1.4, "mem_used": 1.1, "swap_used": 1.2, "disk_usage": 1.0,
        "net_in_mb_s": 1.5, "net_out_mb_s": 1.4, "back_close_wait": 1.3, "front_close_wait": 1.3,
        "load_average": 1.3, "uptime_days": 1.0
    },
    ServerState.HEAVY_LOAD: {
        # Heavy but not critical - market hours activity
        "cpu_user": 1.4, "cpu_sys": 1.3, "cpu_iowait": 1.5, "cpu_idle": 0.6,
        "java_cpu": 1.5, "mem_used": 1.2, "swap_used": 1.3, "disk_usage": 1.0,
        "net_in_mb_s": 1.5, "net_out_mb_s": 1.4, "back_close_wait": 1.4, "front_close_wait": 1.4,
        "load_average": 1.4, "uptime_days": 1.0
    },
    ServerState.CRITICAL_ISSUE: {
        # Actual incident - high CPU, I/O wait spikes, memory pressure
        "cpu_user": 2.5, "cpu_sys": 2.0, "cpu_iowait": 3.5, "cpu_idle": 0.3,
        "java_cpu": 2.8, "mem_used": 1.4, "swap_used": 3.0, "disk_usage": 1.0,
        "net_in_mb_s": 0.4, "net_out_mb_s": 0.3, "back_close_wait": 2.5, "front_close_wait": 2.5,
        "load_average": 2.5, "uptime_days": 1.0
    },
    ServerState.MAINTENANCE: {
        # Planned maintenance - low activity, possible disk operations
        "cpu_user": 0.3, "cpu_sys": 0.5, "cpu_iowait": 0.8, "cpu_idle": 1.4,
        "java_cpu": 0.2, "mem_used": 0.8, "swap_used": 0.5, "disk_usage": 1.0,
        "net_in_mb_s": 0.2, "net_out_mb_s": 0.2, "back_close_wait": 0.2, "front_close_wait": 0.2,
        "load_average": 0.5, "uptime_days": 1.0
    },
    ServerState.RECOVERY: {
        # Post-incident recovery - ramping back up
        "cpu_user": 0.8, "cpu_sys": 0.9, "cpu_iowait": 1.2, "cpu_idle": 1.1,
        "java_cpu": 0.7, "mem_used": 1.0, "swap_used": 1.1, "disk_usage": 1.0,
        "net_in_mb_s": 0.8, "net_out_mb_s": 0.7, "back_close_wait": 0.9, "front_close_wait": 0.9,
        "load_average": 0.9, "uptime_days": 1.0
    },
    ServerState.OFFLINE: {
        "cpu_user": 0.0, "cpu_sys": 0.0, "cpu_iowait": 0.0, "cpu_idle": 0.0,
        "java_cpu": 0.0, "mem_used": 0.0, "swap_used": 0.0, "disk_usage": 0.0,
        "net_in_mb_s": 0.0, "net_out_mb_s": 0.0, "back_close_wait": 0, "front_close_wait": 0,
        "load_average": 0.0, "uptime_days": 1.0
    }
}

# =============================================================================
# Diurnal Patterns Configuration
# =============================================================================

# Financial institutions have distinct patterns:
# - Market hours: 9:30am-4pm EST (peak trading)
# - Pre-market: 7am-9:30am (analytics, prep)
# - After-hours: 4pm-8pm (EOD processing, risk calculations)
# - Overnight: 8pm-7am (batch jobs, ML training)

DIURNAL_CONFIG = {
    'pre_market': {
        'hours': (7, 9),
        'base_multiplier': 1.0
    },
    'market_hours': {
        'hours': (9, 16),
        'base_multiplier': 1.1  # Peak trading
    },
    'after_hours': {
        'hours': (16, 19),
        'base_multiplier': 1.2  # EOD processing
    },
    'evening': {
        'hours': (19, 23),
        'base_multiplier': 1.0  # Batch/ML training
    },
    'overnight': {
        'hours': (0, 7),
        'base_multiplier': 0.8  # Low activity
    }
}

# Profile-specific diurnal adjustments (multiply with base)
PROFILE_DIURNAL_ADJUSTMENTS = {
    ServerProfile.ML_COMPUTE: {
        'evening_overnight': 1.15  # ML training runs at night (19-6)
    },
    ServerProfile.DATABASE: {
        'market_hours': 1.1,       # Busy during market hours
        'after_hours': 1.2         # EOD reports
    },
    ServerProfile.WEB_API: {
        'market_hours': 1.15,      # User traffic follows market
        'overnight': 0.6           # Very low at night
    },
    ServerProfile.RISK_ANALYTICS: {
        'after_hours': 1.3         # Risk calculations at market close
    },
    ServerProfile.DATA_INGEST: {
        'market_hours': 1.2        # Streaming data heaviest during market
    }
}

# =============================================================================
# Server Naming Conventions
# =============================================================================

SERVER_NAMING_PATTERNS = {
    ServerProfile.ML_COMPUTE: {'prefix': 'ppml', 'format': 'ppml{i:04d}'},
    ServerProfile.DATABASE: {'prefix': 'ppdb', 'format': 'ppdb{i:03d}'},
    ServerProfile.WEB_API: {'prefix': 'ppweb', 'format': 'ppweb{i:03d}'},
    ServerProfile.CONDUCTOR_MGMT: {'prefix': 'ppcon', 'format': 'ppcon{i:02d}'},
    ServerProfile.DATA_INGEST: {'prefix': 'ppetl', 'format': 'ppetl{i:03d}'},
    ServerProfile.RISK_ANALYTICS: {'prefix': 'pprisk', 'format': 'pprisk{i:03d}'},
    ServerProfile.GENERIC: {'prefix': 'ppgen', 'format': 'ppgen{i:03d}'}
}

# Profile inference patterns (regex) - order matters (more specific first)
PROFILE_INFERENCE_PATTERNS = [
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

# =============================================================================
# State Transition Probabilities
# =============================================================================

# Base transition matrix (will be adjusted by time of day and server type)
STATE_TRANSITION_MATRIX = {
    ServerState.IDLE: {
        ServerState.IDLE: 0.7,
        ServerState.HEALTHY: 0.25,
        ServerState.MAINTENANCE: 0.05
    },
    ServerState.HEALTHY: {
        ServerState.HEALTHY: 0.96,         # Stay healthy most of the time
        ServerState.MORNING_SPIKE: 0.02,   # Rare spikes (time-of-day adjusted)
        ServerState.HEAVY_LOAD: 0.01,      # Rare load (0.5% per tick = ~1 server in 20)
        ServerState.CRITICAL_ISSUE: 0.005, # Very rare critical (0.5%)
        ServerState.IDLE: 0.005            # Very rare idle during business hours
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

# Problem child multipliers (increase failure probability)
PROBLEM_CHILD_MULTIPLIERS = {
    ServerState.CRITICAL_ISSUE: 2.0,  # 2x more likely to go critical
    ServerState.OFFLINE: 1.5           # 1.5x more likely to go offline
}

# =============================================================================
# Fleet Distribution Configuration
# =============================================================================

# Default fleet distribution for financial ML platform
FLEET_DISTRIBUTION = {
    'web_api': 0.28,         # 28% - User-facing services
    'ml_compute': 0.22,      # 22% - Training workloads
    'database': 0.17,        # 17% - Data layer
    'data_ingest': 0.11,     # 11% - ETL pipelines
    'risk_analytics': 0.09,  # 9% - Risk calculations
    'generic': 0.07,         # 7% - Utility (max 10 servers)
    'conductor_mgmt': 0.06   # 6% - Orchestration
}

# Fleet generation settings
FLEET_CONFIG = {
    'min_total_servers': 7,           # Minimum: 1 per profile
    'max_generic_servers': 10,        # Cap generic servers
    'problem_child_pct_min': 0.08,    # 8% minimum problem children
    'problem_child_pct_max': 0.12,    # 12% maximum problem children
    'problem_child_pct_default': 0.10 # 10% default
}

# =============================================================================
# Time Series Generation Configuration
# =============================================================================

TIMESERIES_CONFIG = {
    'poll_interval_seconds': 5,       # Data collection interval (LINBORG standard)
    'ar1_phi': 0.85,                  # AR(1) autocorrelation parameter
    'ar1_sigma_multiplier': 0.1,      # AR(1) noise as fraction of std
    'uptime_max_days': 30,            # Maximum uptime before restart
    'uptime_normal_days': 25,         # Normal uptime target
}

# Metric bounds and formatting
METRIC_BOUNDS = {
    # Percentage metrics (0-100%)
    'cpu_user_pct': (0, 100),
    'cpu_sys_pct': (0, 100),
    'cpu_iowait_pct': (0, 100),
    'cpu_idle_pct': (0, 100),
    'java_cpu_pct': (0, 100),
    'mem_used_pct': (0, 100),
    'swap_used_pct': (0, 100),
    'disk_usage_pct': (0, 100),

    # Network metrics (MB/s, non-negative)
    'net_in_mb_s': (0, None),
    'net_out_mb_s': (0, None),

    # Connection counts (integers, non-negative)
    'back_close_wait': (0, None),
    'front_close_wait': (0, None),

    # Load average (non-negative)
    'load_average': (0, None),

    # Uptime (0-30 days)
    'uptime_days': (0, 30)
}

# =============================================================================
# Validation and Export
# =============================================================================

# Package everything for easy import
METRICS_CONFIG = {
    'linborg_metrics': LINBORG_METRICS,
    'profile_baselines': PROFILE_BASELINES,
    'state_multipliers': STATE_MULTIPLIERS,
    'diurnal_config': DIURNAL_CONFIG,
    'profile_diurnal_adjustments': PROFILE_DIURNAL_ADJUSTMENTS,
    'server_naming_patterns': SERVER_NAMING_PATTERNS,
    'profile_inference_patterns': PROFILE_INFERENCE_PATTERNS,
    'state_transition_matrix': STATE_TRANSITION_MATRIX,
    'problem_child_multipliers': PROBLEM_CHILD_MULTIPLIERS,
    'fleet_distribution': FLEET_DISTRIBUTION,
    'fleet_config': FLEET_CONFIG,
    'timeseries_config': TIMESERIES_CONFIG,
    'metric_bounds': METRIC_BOUNDS
}

# Export enums
__all__ = [
    'ServerProfile',
    'ServerState',
    'METRICS_CONFIG',
    'LINBORG_METRICS',
    'PROFILE_BASELINES',
    'STATE_MULTIPLIERS'
]
