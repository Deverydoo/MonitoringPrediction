#!/usr/bin/env python3
"""
Server Profile Detection - Shared Module
Single source of truth for server workload classification

This module provides consistent server profile detection across all components
based on naming conventions. Used for transfer learning - when new servers
come online, the model can predict based on profile patterns learned from
existing servers.

Version: 1.0.0
"""

import re
from enum import Enum


class ServerProfile(Enum):
    """
    Server workload profiles for transfer learning.

    Each profile has characteristic resource usage patterns that the model
    learns. When a new server matches a profile pattern, predictions are
    based on that profile's learned behavior.
    """
    ML_COMPUTE = "ml_compute"           # ML training, GPU workloads
    DATABASE = "database"               # RDBMS, data storage
    WEB_API = "web_api"                 # HTTP services, REST APIs
    CONDUCTOR_MGMT = "conductor_mgmt"   # Orchestration, job scheduling
    DATA_INGEST = "data_ingest"         # ETL, streaming, data pipelines
    RISK_ANALYTICS = "risk_analytics"   # Financial risk calculation
    GENERIC = "generic"                 # Default fallback


# =============================================================================
# PROFILE DETECTION PATTERNS
# =============================================================================

# Pattern registry for server name → profile mapping
# Order matters: more specific patterns should come first
PROFILE_PATTERNS = [
    # ML Compute / GPU Training
    (r'^ppml\d+', ServerProfile.ML_COMPUTE),
    (r'^ppgpu\d+', ServerProfile.ML_COMPUTE),
    (r'^cptrain\d+', ServerProfile.ML_COMPUTE),
    (r'^ml-node-\d+', ServerProfile.ML_COMPUTE),

    # Database Servers
    (r'^ppdb\d+', ServerProfile.DATABASE),
    (r'^psdb\d+', ServerProfile.DATABASE),
    (r'^oracle\d+', ServerProfile.DATABASE),
    (r'^mongo\d+', ServerProfile.DATABASE),
    (r'^postgres\d+', ServerProfile.DATABASE),
    (r'^mysql\d+', ServerProfile.DATABASE),

    # Web API / HTTP Services
    (r'^ppweb\d+', ServerProfile.WEB_API),
    (r'^ppapi\d+', ServerProfile.WEB_API),
    (r'^nginx\d+', ServerProfile.WEB_API),
    (r'^tomcat\d+', ServerProfile.WEB_API),
    (r'^api-server-\d+', ServerProfile.WEB_API),

    # Conductor / Orchestration
    (r'^ppcon\d+', ServerProfile.CONDUCTOR_MGMT),
    (r'^conductor\d+', ServerProfile.CONDUCTOR_MGMT),
    (r'^egomgmt\d+', ServerProfile.CONDUCTOR_MGMT),
    (r'^maestro\d+', ServerProfile.CONDUCTOR_MGMT),

    # Data Ingest / ETL / Streaming
    (r'^ppetl\d+', ServerProfile.DATA_INGEST),
    (r'^ppkafka\d+', ServerProfile.DATA_INGEST),
    (r'^stream\d+', ServerProfile.DATA_INGEST),
    (r'^spark\d+', ServerProfile.DATA_INGEST),
    (r'^flink\d+', ServerProfile.DATA_INGEST),

    # Risk Analytics / Financial
    (r'^pprisk\d+', ServerProfile.RISK_ANALYTICS),
    (r'^varrisk\d+', ServerProfile.RISK_ANALYTICS),
    (r'^credit\d+', ServerProfile.RISK_ANALYTICS),
    (r'^quant\d+', ServerProfile.RISK_ANALYTICS),
]


def infer_profile_from_name(server_name: str) -> ServerProfile:
    """
    Infer server profile from naming convention.

    Uses regex patterns to match common naming prefixes. This enables transfer
    learning when new servers come online - the model can predict based on
    profile patterns learned from existing servers.

    Args:
        server_name: Server hostname (e.g., 'ppml0015', 'ppdb042')

    Returns:
        ServerProfile enum value (e.g., ServerProfile.ML_COMPUTE)

    Examples:
        >>> infer_profile_from_name('ppml0015')
        ServerProfile.ML_COMPUTE

        >>> infer_profile_from_name('ppdb042')
        ServerProfile.DATABASE

        >>> infer_profile_from_name('ppweb123')
        ServerProfile.WEB_API

        >>> infer_profile_from_name('unknown_server')
        ServerProfile.GENERIC
    """
    server_lower = server_name.lower()

    for pattern, profile in PROFILE_PATTERNS:
        if re.match(pattern, server_lower):
            return profile

    # Fallback to generic if no match
    return ServerProfile.GENERIC


def get_profile_display_name(profile: ServerProfile) -> str:
    """
    Get human-friendly display name for a profile.

    Args:
        profile: ServerProfile enum value

    Returns:
        Display-friendly string

    Examples:
        >>> get_profile_display_name(ServerProfile.ML_COMPUTE)
        'ML Compute'

        >>> get_profile_display_name(ServerProfile.DATABASE)
        'Database'
    """
    display_names = {
        ServerProfile.ML_COMPUTE: 'ML Compute',
        ServerProfile.DATABASE: 'Database',
        ServerProfile.WEB_API: 'Web API',
        ServerProfile.CONDUCTOR_MGMT: 'Conductor',
        ServerProfile.DATA_INGEST: 'Data Ingest',
        ServerProfile.RISK_ANALYTICS: 'Risk Analytics',
        ServerProfile.GENERIC: 'Generic',
    }
    return display_names.get(profile, profile.value.replace('_', ' ').title())


def add_custom_pattern(pattern: str, profile: ServerProfile) -> None:
    """
    Add a custom naming pattern to the profile detection registry.

    Useful for organization-specific naming conventions not covered by defaults.

    Args:
        pattern: Regex pattern (e.g., r'^mycompany-ml\d+')
        profile: ServerProfile to map to

    Examples:
        >>> add_custom_pattern(r'^mycompany-ml\d+', ServerProfile.ML_COMPUTE)
        >>> infer_profile_from_name('mycompany-ml001')
        ServerProfile.ML_COMPUTE
    """
    # Add at beginning (more specific patterns first)
    PROFILE_PATTERNS.insert(0, (pattern, profile))


# =============================================================================
# USAGE EXAMPLES & TESTING
# =============================================================================

if __name__ == "__main__":
    print("Server Profile Detection - Test Cases")
    print("=" * 70)

    test_cases = [
        ('ppml0015', ServerProfile.ML_COMPUTE),
        ('ppgpu042', ServerProfile.ML_COMPUTE),
        ('ppdb001', ServerProfile.DATABASE),
        ('postgres123', ServerProfile.DATABASE),
        ('ppweb456', ServerProfile.WEB_API),
        ('nginx789', ServerProfile.WEB_API),
        ('ppcon012', ServerProfile.CONDUCTOR_MGMT),
        ('ppetl345', ServerProfile.DATA_INGEST),
        ('spark678', ServerProfile.DATA_INGEST),
        ('pprisk901', ServerProfile.RISK_ANALYTICS),
        ('unknown_server', ServerProfile.GENERIC),
    ]

    print("\nTest Results:")
    print("-" * 70)
    passed = 0
    failed = 0

    for server_name, expected_profile in test_cases:
        detected_profile = infer_profile_from_name(server_name)
        display_name = get_profile_display_name(detected_profile)
        status = "✓" if detected_profile == expected_profile else "✗"

        if detected_profile == expected_profile:
            passed += 1
        else:
            failed += 1

        print(f"{status} {server_name:20s} → {display_name:20s} "
              f"(Expected: {expected_profile.value})")

    print("-" * 70)
    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ {failed} test(s) failed")

    # Show all registered patterns
    print("\n" + "=" * 70)
    print("REGISTERED PATTERNS:")
    print("=" * 70)
    for pattern, profile in PROFILE_PATTERNS:
        print(f"{pattern:30s} → {profile.value}")

    print(f"\nTotal patterns: {len(PROFILE_PATTERNS)}")
