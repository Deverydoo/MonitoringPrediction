#!/usr/bin/env python3
"""
MongoDB Production Adapter
Fetches server metrics from MongoDB and forwards to TFT Inference Daemon

Usage:
    # Continuous streaming mode (production)
    python mongodb_adapter.py --daemon --interval 5

    # One-time fetch (testing)
    python mongodb_adapter.py --once

    # Custom configuration
    python mongodb_adapter.py --config config.json --daemon

Configuration:
    Create config.json with MongoDB connection details:
    {
        "mongodb": {
            "uri": "mongodb://localhost:27017",
            "database": "linborg",
            "collection": "server_metrics",
            "username": "tft_readonly",
            "password": "changeme"
        },
        "tft_daemon": {
            "url": "http://localhost:8000",
            "api_key": "your-api-key-here"
        }
    }
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
    from pymongo import MongoClient, ASCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
except ImportError:
    print("❌ Error: pymongo not installed")
    print("   Install with: pip install pymongo")
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


class MongoDBAdapter:
    """
    MongoDB adapter for TFT Monitoring System.
    Fetches server metrics from MongoDB and forwards to inference daemon.
    """

    def __init__(self, config: Dict):
        """
        Initialize MongoDB adapter.

        Args:
            config: Configuration dictionary with mongodb and tft_daemon settings
        """
        self.config = config
        self.mongo_config = config.get('mongodb', {})
        self.tft_config = config.get('tft_daemon', {})

        # MongoDB connection
        self.client = None
        self.db = None
        self.collection = None

        # State tracking
        self.last_fetch_time = None
        self.total_records_forwarded = 0
        self.errors = 0

        # Connect to MongoDB
        self._connect_mongodb()

    def _connect_mongodb(self):
        """Establish MongoDB connection with authentication."""
        try:
            uri = self.mongo_config.get('uri', 'mongodb://localhost:27017')
            username = self.mongo_config.get('username')
            password = self.mongo_config.get('password')

            # Build connection URI with auth if provided
            if username and password:
                # Extract host:port from URI
                if '://' in uri:
                    protocol, rest = uri.split('://', 1)
                    uri = f"{protocol}://{username}:{password}@{rest}"
                else:
                    uri = f"mongodb://{username}:{password}@{uri}"

            logger.info(f"Connecting to MongoDB: {uri.split('@')[-1]}")  # Hide credentials

            self.client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )

            # Test connection
            self.client.server_info()

            # Select database and collection
            db_name = self.mongo_config.get('database', 'linborg')
            collection_name = self.mongo_config.get('collection', 'server_metrics')

            self.db = self.client[db_name]
            self.collection = self.db[collection_name]

            # Create index on timestamp for efficient queries
            self.collection.create_index([('timestamp', ASCENDING)])

            # Get collection stats
            count = self.collection.count_documents({})

            logger.info(f"✅ Connected to MongoDB")
            logger.info(f"   Database: {db_name}")
            logger.info(f"   Collection: {collection_name}")
            logger.info(f"   Documents: {count:,}")

        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
        except ServerSelectionTimeoutError as e:
            logger.error(f"❌ MongoDB server timeout: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ MongoDB setup error: {e}")
            raise

    def fetch_recent_metrics(self, since: Optional[datetime] = None,
                            limit: int = 1000) -> List[Dict]:
        """
        Fetch recent server metrics from MongoDB.

        Args:
            since: Fetch metrics after this timestamp (default: last 5 minutes)
            limit: Maximum number of documents to fetch

        Returns:
            List of metric documents
        """
        try:
            # Default: last 5 minutes
            if since is None:
                since = datetime.utcnow() - timedelta(minutes=5)

            # Query for recent metrics
            query = {
                'timestamp': {'$gte': since}
            }

            # Fetch documents sorted by timestamp
            cursor = self.collection.find(query).sort('timestamp', ASCENDING).limit(limit)
            metrics = list(cursor)

            logger.info(f"📊 Fetched {len(metrics)} metrics since {since.isoformat()}")

            return metrics

        except Exception as e:
            logger.error(f"❌ Error fetching metrics: {e}")
            self.errors += 1
            return []

    def transform_to_tft_format(self, mongo_docs: List[Dict]) -> List[Dict]:
        """
        Transform MongoDB documents to TFT daemon format.

        Args:
            mongo_docs: List of MongoDB documents

        Returns:
            List of TFT-compatible metric records
        """
        tft_records = []

        for doc in mongo_docs:
            try:
                # Extract timestamp
                timestamp = doc.get('timestamp')
                if isinstance(timestamp, datetime):
                    timestamp = timestamp.isoformat()
                elif isinstance(timestamp, str):
                    pass  # Already string
                else:
                    timestamp = datetime.utcnow().isoformat()

                # Build TFT record with NordIQ Metrics Framework metrics
                record = {
                    'timestamp': timestamp,
                    'server_name': doc.get('server_name', doc.get('hostname', 'unknown')),
                    'profile': doc.get('profile', 'generic'),

                    # CPU metrics
                    'cpu_user_pct': float(doc.get('cpu_user_pct', doc.get('cpu_user', 0))),
                    'cpu_sys_pct': float(doc.get('cpu_sys_pct', doc.get('cpu_sys', 0))),
                    'cpu_iowait_pct': float(doc.get('cpu_iowait_pct', doc.get('cpu_iowait', 0))),
                    'cpu_idle_pct': float(doc.get('cpu_idle_pct', doc.get('cpu_idle', 0))),
                    'java_cpu_pct': float(doc.get('java_cpu_pct', doc.get('java_cpu', 0))),

                    # Memory metrics
                    'mem_used_pct': float(doc.get('mem_used_pct', doc.get('memory_used', 0))),
                    'swap_used_pct': float(doc.get('swap_used_pct', doc.get('swap_used', 0))),

                    # Disk metrics
                    'disk_usage_pct': float(doc.get('disk_usage_pct', doc.get('disk_used', 0))),

                    # Network metrics
                    'net_in_mb_s': float(doc.get('net_in_mb_s', doc.get('network_in', 0))),
                    'net_out_mb_s': float(doc.get('net_out_mb_s', doc.get('network_out', 0))),

                    # Connection metrics
                    'back_close_wait': int(doc.get('back_close_wait', 0)),
                    'front_close_wait': int(doc.get('front_close_wait', 0)),

                    # System metrics
                    'load_average': float(doc.get('load_average', doc.get('load', 0))),
                    'uptime_days': float(doc.get('uptime_days', doc.get('uptime', 0)))
                }

                tft_records.append(record)

            except Exception as e:
                logger.warning(f"⚠️ Error transforming document {doc.get('_id')}: {e}")
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

        # Forward to TFT daemon
        success = self.forward_to_tft_daemon(records)

        return success

    def run_daemon(self, interval: int = 5):
        """
        Run continuous streaming daemon.

        Args:
            interval: Fetch interval in seconds (default: 5)
        """
        logger.info("🚀 Starting MongoDB adapter daemon")
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
                    self.forward_to_tft_daemon(records)

                    # Update last fetch time to latest metric timestamp
                    if records:
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
            logger.info("\n🛑 Shutting down MongoDB adapter")
            logger.info(f"   Total records forwarded: {self.total_records_forwarded}")
            logger.info(f"   Total errors: {self.errors}")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            raise
        finally:
            if self.client:
                self.client.close()
                logger.info("✅ MongoDB connection closed")


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
        description='MongoDB Production Adapter for TFT Monitoring System'
    )
    parser.add_argument(
        '--config',
        default='mongodb_adapter_config.json',
        help='Configuration file path (default: mongodb_adapter_config.json)'
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
    adapter = MongoDBAdapter(config)

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
