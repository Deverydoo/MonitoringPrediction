#!/usr/bin/env python3
"""
Central Constants - Single Source of Truth
All valid states, profiles, and fleet configuration defined here.
Import from this file to avoid hardcoding and inconsistencies.

Version: 1.0.0
Conforms to: DATA_CONTRACT.md v1.0.0
"""

from enum import Enum
from typing import List, Dict

# =============================================================================
# DATA CONTRACT v1.0.0 - VALID STATES
# =============================================================================
# These are the ONLY valid states throughout the system.
# DO NOT modify without updating DATA_CONTRACT.md and retraining models.

VALID_STATES: List[str] = [
    'critical_issue',
    'healthy',
    'heavy_load',
    'idle',
    'maintenance',
    'morning_spike',
    'offline',
    'recovery'
]

# State count for validation
NUM_STATES = len(VALID_STATES)  # Should always be 8

# =============================================================================
# SERVER PROFILES - Financial ML Platform
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


# Profile list for iteration
VALID_PROFILES: List[str] = [p.value for p in ServerProfile]

# Profile count for validation
NUM_PROFILES = len(VALID_PROFILES)  # Should always be 7

# =============================================================================
# FLEET COMPOSITION - Default 90-Server Fleet
# =============================================================================

DEFAULT_FLEET_COMPOSITION: Dict[str, int] = {
    'ml_compute': 20,        # ML training nodes (CPU/GPU intensive)
    'database': 15,          # Database servers (I/O + memory intensive)
    'web_api': 25,           # Web/API servers (network intensive)
    'conductor_mgmt': 5,     # Conductor management nodes
    'data_ingest': 10,       # ETL/streaming servers
    'risk_analytics': 8,     # Risk calculation servers
    'generic': 7             # Generic/utility servers
}

# Total fleet size
DEFAULT_FLEET_SIZE = sum(DEFAULT_FLEET_COMPOSITION.values())  # Should be 90

# =============================================================================
# SERVER NAMING CONVENTIONS
# =============================================================================
# These patterns enable automatic profile inference from server names

SERVER_NAME_PATTERNS: Dict[ServerProfile, str] = {
    ServerProfile.ML_COMPUTE: "ppml{:04d}",           # ppml0001-ppml0020
    ServerProfile.DATABASE: "ppdb{:03d}",             # ppdb001-ppdb015
    ServerProfile.WEB_API: "ppweb{:03d}",             # ppweb001-ppweb025
    ServerProfile.CONDUCTOR_MGMT: "ppcon{:02d}",      # ppcon01-ppcon05
    ServerProfile.DATA_INGEST: "ppetl{:03d}",         # ppetl001-ppetl010
    ServerProfile.RISK_ANALYTICS: "pprisk{:03d}",     # pprisk001-pprisk008
    ServerProfile.GENERIC: "ppgen{:03d}"              # ppgen001-ppgen007
}

# =============================================================================
# PROFILE INFERENCE REGEX PATTERNS
# =============================================================================
# For inferring profile from server name in production

PROFILE_INFERENCE_PATTERNS: List[tuple] = [
    # (regex_pattern, ServerProfile)
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
# HELPER FUNCTIONS
# =============================================================================

def generate_server_names(profile: ServerProfile, count: int) -> List[str]:
    """
    Generate server names following naming conventions.

    Args:
        profile: Server profile type
        count: Number of servers to generate

    Returns:
        List of server names in correct format

    Example:
        >>> generate_server_names(ServerProfile.ML_COMPUTE, 3)
        ['ppml0001', 'ppml0002', 'ppml0003']
    """
    pattern = SERVER_NAME_PATTERNS[profile]
    return [pattern.format(i) for i in range(1, count + 1)]


def infer_profile_from_name(server_name: str) -> ServerProfile:
    """
    Infer server profile from naming convention.

    Uses regex patterns to match common prefixes. This enables transfer
    learning when new servers come online - model can predict based on
    profile patterns learned from existing servers.

    Args:
        server_name: Server hostname

    Returns:
        Inferred ServerProfile (falls back to GENERIC if no match)

    Examples:
        >>> infer_profile_from_name('ppml0015')
        <ServerProfile.ML_COMPUTE: 'ml_compute'>

        >>> infer_profile_from_name('unknown_server')
        <ServerProfile.GENERIC: 'generic'>
    """
    import re

    server_lower = server_name.lower()

    for pattern, profile in PROFILE_INFERENCE_PATTERNS:
        if re.match(pattern, server_lower):
            return profile

    # Fallback to generic if no match
    return ServerProfile.GENERIC


def get_full_server_list() -> List[str]:
    """
    Generate full 90-server fleet with correct naming.

    Returns:
        List of all 90 server names in default fleet

    Example:
        >>> servers = get_full_server_list()
        >>> len(servers)
        90
        >>> servers[0]
        'ppml0001'
    """
    all_servers = []

    for profile in ServerProfile:
        count = DEFAULT_FLEET_COMPOSITION.get(profile.value, 0)
        if count > 0:
            all_servers.extend(generate_server_names(profile, count))

    return all_servers


def validate_constants() -> bool:
    """
    Validate that constants are consistent.

    Returns:
        True if all validations pass

    Raises:
        AssertionError: If any validation fails
    """
    # Validate state count
    assert NUM_STATES == 8, f"Expected 8 states, got {NUM_STATES}"
    assert len(VALID_STATES) == NUM_STATES, "VALID_STATES length mismatch"

    # Validate profile count
    assert NUM_PROFILES == 7, f"Expected 7 profiles, got {NUM_PROFILES}"
    assert len(VALID_PROFILES) == NUM_PROFILES, "VALID_PROFILES length mismatch"
    assert len(ServerProfile) == NUM_PROFILES, "ServerProfile enum count mismatch"

    # Validate fleet composition
    assert DEFAULT_FLEET_SIZE == 90, f"Expected 90 servers, got {DEFAULT_FLEET_SIZE}"
    assert sum(DEFAULT_FLEET_COMPOSITION.values()) == DEFAULT_FLEET_SIZE, "Fleet composition sum mismatch"

    # Validate all profiles have composition
    for profile in ServerProfile:
        assert profile.value in DEFAULT_FLEET_COMPOSITION, f"Missing composition for {profile.value}"

    # Validate all profiles have naming pattern
    for profile in ServerProfile:
        assert profile in SERVER_NAME_PATTERNS, f"Missing naming pattern for {profile.value}"

    # Validate server name generation
    full_fleet = get_full_server_list()
    assert len(full_fleet) == DEFAULT_FLEET_SIZE, f"Generated fleet size mismatch: {len(full_fleet)} vs {DEFAULT_FLEET_SIZE}"

    return True


# Run validation on import
try:
    validate_constants()
except AssertionError as e:
    import warnings
    warnings.warn(f"Constants validation failed: {e}")

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # States
    'VALID_STATES',
    'NUM_STATES',
    # Profiles
    'ServerProfile',
    'VALID_PROFILES',
    'NUM_PROFILES',
    # Fleet
    'DEFAULT_FLEET_COMPOSITION',
    'DEFAULT_FLEET_SIZE',
    # Naming
    'SERVER_NAME_PATTERNS',
    'PROFILE_INFERENCE_PATTERNS',
    # Functions
    'generate_server_names',
    'infer_profile_from_name',
    'get_full_server_list',
    'validate_constants'
]
