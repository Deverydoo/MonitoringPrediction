"""
API Configuration - URLs, Ports, Endpoints, Timeouts

SINGLE SOURCE OF TRUTH for all API-related configuration:
- Service URLs and ports
- API endpoints
- Timeout settings
- WebSocket configuration
- HTTP settings
"""

# =============================================================================
# Service URLs and Ports
# =============================================================================

API_CONFIG = {
    # Inference Daemon (TFT model predictions)
    'daemon_url': 'http://localhost:8000',
    'daemon_port': 8000,
    'daemon_host': 'localhost',

    # Metrics Generator Daemon (simulated metrics)
    'metrics_generator_url': 'http://localhost:8001',
    'metrics_generator_port': 8001,
    'metrics_generator_host': 'localhost',

    # Dashboard (Dash web interface)
    'dashboard_url': 'http://localhost:8050',
    'dashboard_port': 8050,
    'dashboard_host': 'localhost',

    # =============================================================================
    # API Endpoints
    # =============================================================================

    'endpoints': {
        # Inference Daemon Endpoints
        'daemon_health': '/health',
        'daemon_status': '/status',
        'daemon_predictions': '/predictions',
        'daemon_alerts': '/alerts',
        'daemon_feed_data': '/feed/data',
        'daemon_websocket': '/ws',

        # Metrics Generator Endpoints
        'generator_health': '/health',
        'generator_scenario_status': '/scenario/status',
        'generator_scenario_set': '/scenario/set',
        'generator_metrics': '/metrics',
    },

    # =============================================================================
    # Timeout Settings (seconds)
    # =============================================================================

    'timeouts': {
        'health_check': 2,          # Fast health check
        'prediction': 5,            # Model inference
        'feed_data': 3,             # Data ingestion
        'scenario_change': 2,       # Scenario switching
        'default': 10,              # Default timeout
        'long_running': 30          # Long operations
    },

    # =============================================================================
    # WebSocket Configuration
    # =============================================================================

    'websocket': {
        'reconnect_delay': 5,       # Seconds between reconnect attempts
        'max_reconnect_attempts': 10,  # Max reconnection attempts
        'ping_interval': 30,        # Keep-alive ping interval
        'ping_timeout': 10,         # Ping response timeout
        'buffer_size': 65536        # WebSocket buffer size (bytes)
    },

    # =============================================================================
    # HTTP Configuration
    # =============================================================================

    'http': {
        'max_retries': 3,           # Max retry attempts for failed requests
        'retry_delay': 1,           # Seconds between retries
        'user_agent': 'TFT-Monitor/1.0',  # User agent string
        'verify_ssl': True,         # SSL certificate verification
        'allow_redirects': True     # Follow redirects
    },

    # =============================================================================
    # Data Streaming Configuration
    # =============================================================================

    'streaming': {
        'tick_interval': 5,         # Seconds between data ticks
        'batch_size': 20,           # Records per batch
        'max_batch_age': 10,        # Max seconds to hold batch
        'buffer_size': 1000         # Max records in buffer
    },

    # =============================================================================
    # Dashboard Refresh Configuration
    # =============================================================================

    'dashboard': {
        'default_refresh_interval': 60,  # Seconds (1 minute default)
        'min_refresh_interval': 5,       # Minimum allowed
        'max_refresh_interval': 300      # Maximum allowed (5 minutes)
    },

    # =============================================================================
    # Production Configuration (Template for future use)
    # =============================================================================

    'production': {
        # Production URLs (to be configured per environment)
        'daemon_url': 'https://tft-daemon.prod.company.com',
        'metrics_forwarder_url': 'https://metrics.prod.company.com',
        'dashboard_url': 'https://tft-dashboard.prod.company.com',

        # Production timeouts (more conservative)
        'timeouts': {
            'health_check': 5,
            'prediction': 15,
            'feed_data': 10,
            'default': 30
        },

        # Production HTTP settings
        'http': {
            'max_retries': 5,
            'retry_delay': 2,
            'verify_ssl': True,
            'allow_redirects': False  # Strict in production
        }
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_full_url(service: str, endpoint: str, use_production: bool = False) -> str:
    """
    Construct full URL for a service endpoint.

    Args:
        service: Service name ('daemon', 'generator', 'dashboard')
        endpoint: Endpoint name (e.g., 'predictions', 'health')
        use_production: Use production URLs instead of local

    Returns:
        Full URL string

    Examples:
        >>> get_full_url('daemon', 'predictions')
        'http://localhost:8000/predictions'

        >>> get_full_url('generator', 'scenario_status')
        'http://localhost:8001/scenario/status'
    """
    config = API_CONFIG['production'] if use_production else API_CONFIG

    # Get base URL
    base_url_map = {
        'daemon': config.get('daemon_url', API_CONFIG['daemon_url']),
        'generator': config.get('metrics_generator_url', API_CONFIG['metrics_generator_url']),
        'dashboard': config.get('dashboard_url', API_CONFIG['dashboard_url'])
    }

    base_url = base_url_map.get(service)
    if not base_url:
        raise ValueError(f"Unknown service: {service}")

    # Get endpoint path
    endpoint_key = f"{service}_{endpoint}"
    endpoint_path = API_CONFIG['endpoints'].get(endpoint_key, f"/{endpoint}")

    return f"{base_url}{endpoint_path}"


def get_timeout(operation: str = 'default') -> int:
    """
    Get timeout value for an operation.

    Args:
        operation: Operation type ('health_check', 'prediction', etc.)

    Returns:
        Timeout in seconds
    """
    return API_CONFIG['timeouts'].get(operation, API_CONFIG['timeouts']['default'])


def get_websocket_url(use_production: bool = False) -> str:
    """
    Get WebSocket URL for daemon streaming.

    Args:
        use_production: Use production URL instead of local

    Returns:
        WebSocket URL (ws:// or wss://)
    """
    if use_production:
        base_url = API_CONFIG['production']['daemon_url']
        # Convert https:// to wss://
        ws_url = base_url.replace('https://', 'wss://')
    else:
        base_url = API_CONFIG['daemon_url']
        # Convert http:// to ws://
        ws_url = base_url.replace('http://', 'ws://')

    return f"{ws_url}{API_CONFIG['endpoints']['daemon_websocket']}"


__all__ = [
    'API_CONFIG',
    'get_full_url',
    'get_timeout',
    'get_websocket_url'
]
