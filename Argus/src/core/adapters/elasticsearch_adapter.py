#!/usr/bin/env python3
"""
Elasticsearch Production Adapter
Fetches server metrics from Elasticsearch and forwards to TFT Inference Daemon

Usage:
    # Continuous streaming mode (production)
    python elasticsearch_adapter.py --daemon --interval 5

    # One-time fetch (testing)
    python elasticsearch_adapter.py --once

    # Custom configuration
    python elasticsearch_adapter.py --config config.json --daemon

Configuration:
    Create config.json with Elasticsearch connection details:
    {
        "elasticsearch": {
            "hosts": ["localhost:9200"],
            "index_pattern": "linborg-metrics-*",
            "username": "tft_readonly",
            "password": "changeme",
            "use_ssl": false,
            "verify_certs": false,
            "ca_certs": "/path/to/ca.pem"
        },
        "tft_daemon": {
            "url": "http://localhost:8000",
            "api_key": "your-api-key-here"
        }
    }

NOTE: Elasticsearch licensing - This adapter uses the official elasticsearch-py client.
      Ensure compliance with Elastic License 2.0 or your organization's license.
      This is a data CLIENT only (read-only operations).
"""

import argparse
import json
import time
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import ConnectionError, TransportError
except ImportError:
    print("❌ Error: elasticsearch not installed")
    print("   Install with: pip install elasticsearch")
    print("   Note: Check Elastic licensing requirements for your use case")
    sys.exit(1)

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_api_key_from_env() -> Optional[str]:
    """
    Load TFT API key from .env file or environment variable.

    Priority:
    1. Read from .env file in project root (../.env from adapters/)
    2. Fall back to TFT_API_KEY environment variable

    Returns:
        API key string or None if not found
    """
    # Try to read from .env file first
    env_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),  # Project root
        os.path.join(os.getcwd(), '.env'),  # Current directory
        '.env'  # Relative path
    ]

    for env_path in env_paths:
        if os.path.exists(env_path):
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('TFT_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            logger.debug(f"✅ Loaded API key from {env_path}")
                            return api_key
            except Exception as e:
                logger.warning(f"⚠️ Error reading .env file {env_path}: {e}")
                continue

    # Fall back to environment variable
    api_key = os.getenv('TFT_API_KEY')
    if api_key:
        logger.debug("✅ Loaded API key from environment variable")
        return api_key

    logger.warning("⚠️ No API key found in .env file or environment")
    return None


class ElasticsearchAdapter:
    """
    Elasticsearch adapter for TFT Monitoring System.
    Fetches server metrics from Elasticsearch and forwards to inference daemon.
    """

    def __init__(self, config: Dict):
        """
        Initialize Elasticsearch adapter.

        Args:
            config: Configuration dictionary with elasticsearch and tft_daemon settings
        """
        self.config = config
        self.es_config = config.get('elasticsearch', {})
        self.tft_config = config.get('tft_daemon', {})

        # Elasticsearch connection
        self.es = None

        # State tracking
        self.last_fetch_time = None
        self.total_records_forwarded = 0
        self.errors = 0

        # Connect to Elasticsearch
        self._connect_elasticsearch()

    def _connect_elasticsearch(self):
        """Establish Elasticsearch connection with authentication."""
        try:
            hosts = self.es_config.get('hosts', ['localhost:9200'])
            username = self.es_config.get('username')
            password = self.es_config.get('password')
            use_ssl = self.es_config.get('use_ssl', False)
            verify_certs = self.es_config.get('verify_certs', False)
            ca_certs = self.es_config.get('ca_certs')

            # Build connection parameters
            es_params = {
                'hosts': hosts,
                'timeout': 30,
                'max_retries': 3,
                'retry_on_timeout': True
            }

            # Add authentication if provided
            if username and password:
                es_params['http_auth'] = (username, password)

            # Add SSL/TLS if configured
            if use_ssl:
                es_params['use_ssl'] = True
                es_params['verify_certs'] = verify_certs
                if ca_certs:
                    es_params['ca_certs'] = ca_certs

            logger.info(f"Connecting to Elasticsearch: {hosts}")

            self.es = Elasticsearch(**es_params)

            # Test connection
            if not self.es.ping():
                raise ConnectionError("Elasticsearch ping failed")

            # Get cluster info
            info = self.es.info()

            logger.info(f"✅ Connected to Elasticsearch")
            logger.info(f"   Cluster: {info['cluster_name']}")
            logger.info(f"   Version: {info['version']['number']}")
            logger.info(f"   Index: {self.es_config.get('index_pattern', 'linborg-metrics-*')}")

        except ConnectionError as e:
            logger.error(f"❌ Elasticsearch connection failed: {e}")
            raise
        except TransportError as e:
            logger.error(f"❌ Elasticsearch transport error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Elasticsearch setup error: {e}")
            raise

    def fetch_recent_metrics(self, since: Optional[datetime] = None,
                            size: int = 1000) -> List[Dict]:
        """
        Fetch recent server metrics from Elasticsearch.

        Args:
            since: Fetch metrics after this timestamp (default: last 5 minutes)
            size: Maximum number of documents to fetch

        Returns:
            List of metric documents
        """
        try:
            # Default: last 5 minutes
            if since is None:
                since = datetime.utcnow() - timedelta(minutes=5)

            # Build Elasticsearch query
            index_pattern = self.es_config.get('index_pattern', 'linborg-metrics-*')

            query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "range": {
                                    "@timestamp": {
                                        "gte": since.isoformat()
                                    }
                                }
                            }
                        ]
                    }
                },
                "sort": [
                    {"@timestamp": {"order": "asc"}}
                ],
                "size": size
            }

            # Execute search
            response = self.es.search(
                index=index_pattern,
                body=query
            )

            # Extract hits
            hits = response['hits']['hits']
            metrics = [hit['_source'] for hit in hits]

            logger.info(f"📊 Fetched {len(metrics)} metrics since {since.isoformat()}")

            return metrics

        except Exception as e:
            logger.error(f"❌ Error fetching metrics: {e}")
            self.errors += 1
            return []

    def transform_to_tft_format(self, es_docs: List[Dict]) -> List[Dict]:
        """
        Transform Elasticsearch documents to TFT daemon format.

        Args:
            es_docs: List of Elasticsearch documents

        Returns:
            List of TFT-compatible metric records
        """
        tft_records = []

        for doc in es_docs:
            try:
                # Extract timestamp (Elasticsearch commonly uses @timestamp)
                timestamp = doc.get('@timestamp', doc.get('timestamp'))
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.isoformat()
                elif isinstance(timestamp, str):
                    pass  # Already string
                else:
                    timestamp = datetime.utcnow().isoformat()

                # Handle nested fields (common in Elasticsearch)
                # Example: {"host": {"name": "server1"}, "system": {"cpu": {"user": {"pct": 0.5}}}}
                host = doc.get('host', {})
                system = doc.get('system', {})
                metrics = doc.get('metrics', {})

                # Build TFT record with NordIQ Metrics Framework metrics
                # Adapt field paths based on your Elasticsearch schema
                record = {
                    'timestamp': timestamp,
                    'server_name': (
                        doc.get('server_name') or
                        doc.get('hostname') or
                        host.get('name') or
                        host.get('hostname') or
                        'unknown'
                    ),
                    'profile': doc.get('profile', doc.get('server_type', 'generic')),

                    # CPU metrics - handle nested paths
                    'cpu_user_pct': float(
                        doc.get('cpu_user_pct') or
                        doc.get('cpu.user.pct') or
                        metrics.get('cpu', {}).get('user_pct') or
                        system.get('cpu', {}).get('user', {}).get('pct', 0)
                    ) * 100 if 'pct' in str(doc) else float(doc.get('cpu_user_pct', 0)),

                    'cpu_sys_pct': float(
                        doc.get('cpu_sys_pct') or
                        doc.get('cpu.sys.pct') or
                        metrics.get('cpu', {}).get('sys_pct') or
                        system.get('cpu', {}).get('system', {}).get('pct', 0)
                    ) * 100 if 'pct' in str(doc) else float(doc.get('cpu_sys_pct', 0)),

                    'cpu_iowait_pct': float(
                        doc.get('cpu_iowait_pct') or
                        doc.get('cpu.iowait.pct') or
                        metrics.get('cpu', {}).get('iowait_pct') or
                        system.get('cpu', {}).get('iowait', {}).get('pct', 0)
                    ) * 100 if 'pct' in str(doc) else float(doc.get('cpu_iowait_pct', 0)),

                    'cpu_idle_pct': float(
                        doc.get('cpu_idle_pct') or
                        doc.get('cpu.idle.pct') or
                        metrics.get('cpu', {}).get('idle_pct') or
                        system.get('cpu', {}).get('idle', {}).get('pct', 100)
                    ) * 100 if 'pct' in str(doc) else float(doc.get('cpu_idle_pct', 100)),

                    'java_cpu_pct': float(doc.get('java_cpu_pct', doc.get('java_cpu', 0))),

                    # Memory metrics
                    'mem_used_pct': float(
                        doc.get('mem_used_pct') or
                        doc.get('memory.used.pct') or
                        metrics.get('memory', {}).get('used_pct') or
                        system.get('memory', {}).get('used', {}).get('pct', 0)
                    ) * 100 if 'pct' in str(doc) else float(doc.get('mem_used_pct', 0)),

                    'swap_used_pct': float(
                        doc.get('swap_used_pct') or
                        doc.get('swap.used.pct') or
                        metrics.get('swap', {}).get('used_pct') or
                        system.get('swap', {}).get('used', {}).get('pct', 0)
                    ) * 100 if 'pct' in str(doc) else float(doc.get('swap_used_pct', 0)),

                    # Disk metrics
                    'disk_usage_pct': float(
                        doc.get('disk_usage_pct') or
                        doc.get('disk.usage.pct') or
                        metrics.get('disk', {}).get('usage_pct') or
                        system.get('filesystem', {}).get('used', {}).get('pct', 0)
                    ) * 100 if 'pct' in str(doc) else float(doc.get('disk_usage_pct', 0)),

                    # Network metrics
                    'net_in_mb_s': float(
                        doc.get('net_in_mb_s') or
                        doc.get('network.in.mb_s') or
                        metrics.get('network', {}).get('in_mb_s', 0)
                    ),

                    'net_out_mb_s': float(
                        doc.get('net_out_mb_s') or
                        doc.get('network.out.mb_s') or
                        metrics.get('network', {}).get('out_mb_s', 0)
                    ),

                    # Connection metrics
                    'back_close_wait': int(doc.get('back_close_wait', 0)),
                    'front_close_wait': int(doc.get('front_close_wait', 0)),

                    # System metrics
                    'load_average': float(
                        doc.get('load_average') or
                        doc.get('system.load.1') or
                        system.get('load', {}).get('1', 0)
                    ),

                    'uptime_days': float(doc.get('uptime_days', doc.get('uptime', 0)))
                }

                tft_records.append(record)

            except Exception as e:
                logger.warning(f"⚠️ Error transforming document: {e}")
                logger.debug(f"   Document: {json.dumps(doc, indent=2)[:200]}")
                continue

        logger.info(f"✅ Transformed {len(tft_records)} records to TFT format")
        return tft_records

    def forward_to_tft_daemon(self, records: List[Dict]) -> bool:
        """
        Forward metrics to TFT inference daemon.

        Args:
            records: List of TFT-formatted metric records

        Returns:
            True if successful, False otherwise
        """
        if not records:
            logger.debug("No records to forward")
            return True

        try:
            daemon_url = self.tft_config.get('url', 'http://localhost:8000')

            # Priority: config file > .env file > environment variable
            api_key = self.tft_config.get('api_key')
            if not api_key:
                api_key = load_api_key_from_env()

            headers = {
                'Content-Type': 'application/json'
            }

            if api_key:
                headers['X-API-Key'] = api_key
            else:
                logger.warning("⚠️ No API key configured - daemon may reject request")

            # Forward to TFT daemon /feed endpoint
            response = requests.post(
                f"{daemon_url}/feed",
                json=records,
                headers=headers,
                timeout=10
            )

            if response.ok:
                logger.info(f"✅ Forwarded {len(records)} records to TFT daemon")
                self.total_records_forwarded += len(records)
                return True
            else:
                logger.error(f"❌ TFT daemon error: {response.status_code} - {response.text}")
                self.errors += 1
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error forwarding to TFT daemon: {e}")
            self.errors += 1
            return False

    def run_once(self) -> bool:
        """
        Fetch and forward metrics once (for testing).

        Returns:
            True if successful, False otherwise
        """
        logger.info("🔄 Running one-time fetch...")

        # Fetch recent metrics
        metrics = self.fetch_recent_metrics()

        if not metrics:
            logger.warning("⚠️ No metrics found")
            return False

        # Transform to TFT format
        records = self.transform_to_tft_format(metrics)

        if not records:
            logger.error("❌ No records after transformation - check field mappings")
            return False

        # Forward to TFT daemon
        success = self.forward_to_tft_daemon(records)

        return success

    def run_daemon(self, interval: int = 5):
        """
        Run continuous streaming daemon.

        Args:
            interval: Fetch interval in seconds (default: 5)
        """
        logger.info("🚀 Starting Elasticsearch adapter daemon")
        logger.info(f"   Fetch interval: {interval} seconds")
        logger.info(f"   TFT daemon: {self.tft_config.get('url')}")
        logger.info("   Press Ctrl+C to stop")

        self.last_fetch_time = datetime.utcnow() - timedelta(seconds=interval)

        try:
            while True:
                cycle_start = time.time()

                # Fetch metrics since last fetch
                metrics = self.fetch_recent_metrics(since=self.last_fetch_time)

                if metrics:
                    # Transform and forward
                    records = self.transform_to_tft_format(metrics)
                    if records:
                        self.forward_to_tft_daemon(records)

                        # Update last fetch time to latest metric timestamp
                        latest = max(records, key=lambda r: r['timestamp'])
                        self.last_fetch_time = datetime.fromisoformat(
                            latest['timestamp'].replace('Z', '+00:00')
                        )
                else:
                    logger.debug("No new metrics found")

                # Stats
                if self.total_records_forwarded > 0 and self.total_records_forwarded % 100 == 0:
                    logger.info(f"📊 Stats: {self.total_records_forwarded} records forwarded, "
                              f"{self.errors} errors")

                # Sleep until next interval
                cycle_time = time.time() - cycle_start
                sleep_time = max(0, interval - cycle_time)

                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("\n🛑 Shutting down Elasticsearch adapter")
            logger.info(f"   Total records forwarded: {self.total_records_forwarded}")
            logger.info(f"   Total errors: {self.errors}")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            raise
        finally:
            if self.es:
                self.es.close()
                logger.info("✅ Elasticsearch connection closed")


def load_config(config_path: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"✅ Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in configuration file: {e}")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Elasticsearch Production Adapter for TFT Monitoring System'
    )
    parser.add_argument(
        '--config',
        default='elasticsearch_adapter_config.json',
        help='Configuration file path (default: elasticsearch_adapter_config.json)'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run in continuous streaming mode'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Fetch and forward once (for testing)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Fetch interval in seconds (daemon mode, default: 5)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args.config)

    # Create adapter
    adapter = ElasticsearchAdapter(config)

    # Run mode
    if args.once:
        success = adapter.run_once()
        sys.exit(0 if success else 1)
    elif args.daemon:
        adapter.run_daemon(interval=args.interval)
    else:
        # Default: show help
        parser.print_help()
        print("\n💡 Tip: Use --once for testing or --daemon for production")


if __name__ == '__main__':
    main()
