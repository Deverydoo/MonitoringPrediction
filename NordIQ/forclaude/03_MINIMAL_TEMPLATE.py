#!/usr/bin/env python3
"""
Generic Data Adapter for NordIQ Inference Engine
Polls data source and feeds to inference daemon.

CUSTOMIZE the 3 functions marked with "# CUSTOMIZE THIS":
1. poll_data_source() - Connect to your data source
2. transform_record() - Map your field names to NordIQ format
3. match_profile() - (OPTIONAL) Custom profile matching, or skip for auto-detection

That's it! Everything else is ready to use.
"""

import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# ============================================================================
# CONFIGURATION - Adjust these for your environment
# ============================================================================

# NordIQ Inference Daemon
INFERENCE_DAEMON_URL = "http://localhost:8000"
API_KEY_FILE = ".nordiq_key"  # Relative to NordIQ root directory

# Polling interval (seconds)
POLL_INTERVAL_SECONDS = 5

# Data source configuration - CUSTOMIZE THESE
DATA_SOURCE_URL = "http://your-monitoring-system/api/metrics"
DATA_SOURCE_AUTH = None  # Or {"username": "user", "password": "pass"}
DATA_SOURCE_TIMEOUT = 10  # seconds

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS - Already implemented, no changes needed
# ============================================================================

def load_api_key(key_file: str = API_KEY_FILE) -> str:
    """Load NordIQ API key from file."""
    try:
        with open(key_file, 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.error(f"API key file not found: {key_file}")
        logger.error("Run: python bin/generate_api_key.py")
        raise

def send_to_inference_daemon(records: List[Dict[str, Any]], api_key: str) -> bool:
    """
    Send batch of records to NordIQ inference daemon.

    Already implemented - no changes needed!
    """
    if not records:
        return True

    try:
        response = requests.post(
            f"{INFERENCE_DAEMON_URL}/feed/data",
            json={"records": records},
            headers={"X-API-Key": api_key},
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        logger.info(
            f"✓ Sent {result.get('records_received', 0)} records, "
            f"updated {len(result.get('servers_updated', []))} servers"
        )

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"✗ Failed to send data to inference daemon: {e}")
        return False

# ============================================================================
# CUSTOMIZE THESE 3 FUNCTIONS FOR YOUR DATA SOURCE
# ============================================================================

def match_profile(server_name: str) -> str:
    """
    OPTIONAL: Match server name to profile.

    OPTION 1 (RECOMMENDED): Let inference daemon handle it
    --------------------------------------------------
    Just return 'generic' - inference daemon will auto-detect based on
    server name prefixes (ppdb* → database, ppweb* → web_api, etc.)

    OPTION 2: Implement custom logic
    --------------------------------------------------
    Use CMDB lookups, service tags, or custom patterns.

    Valid profiles:
    - 'ml_compute': ML/AI workloads, GPU servers
    - 'database': Database servers (SQL, NoSQL)
    - 'web_api': Web servers, API gateways
    - 'conductor_mgmt': Orchestration, management
    - 'data_ingest': ETL, Kafka, data pipelines
    - 'risk_analytics': Risk calculation, analytics
    - 'generic': Unknown/other (default)
    """
    # OPTION 1: Let inference daemon auto-detect (recommended)
    return 'generic'  # Inference daemon will auto-detect from server name

    # OPTION 2: Custom matching (uncomment and customize)
    # name_lower = server_name.lower()
    #
    # if 'db' in name_lower or 'sql' in name_lower:
    #     return 'database'
    # if 'web' in name_lower or 'api' in name_lower:
    #     return 'web_api'
    # if 'ml' in name_lower or 'gpu' in name_lower:
    #     return 'ml_compute'
    # if 'etl' in name_lower or 'kafka' in name_lower:
    #     return 'data_ingest'
    # if 'risk' in name_lower:
    #     return 'risk_analytics'
    #
    # return 'generic'

def transform_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    CUSTOMIZE THIS: Transform your data source format to NordIQ format.

    Your data source has different field names. Map them to NordIQ's 9 required fields:

    Required fields (you MUST provide these):
    - timestamp: ISO 8601 string (e.g., "2025-10-30T14:35:00")
    - server_name: Unique identifier
    - cpu_pct: CPU usage percentage (0-100)
    - memory_pct: Memory usage percentage (0-100)
    - disk_pct: Disk usage percentage (0-100)
    - network_in_mbps: Network inbound (Mbps)
    - network_out_mbps: Network outbound (Mbps)
    - disk_read_mbps: Disk read throughput (MB/s)
    - disk_write_mbps: Disk write throughput (MB/s)

    Optional fields:
    - status: "healthy", "degraded", "warning", "critical" (default: "healthy")
    - profile: Server profile (default: auto-detected)

    Example mapping (CUSTOMIZE field names for your data source):
    """
    return {
        # REQUIRED FIELDS - Map your field names here
        'timestamp': raw_record.get('collected_at', datetime.now().isoformat()),
        'server_name': raw_record.get('hostname', 'unknown'),
        'cpu_pct': raw_record.get('cpu_usage', 0.0),
        'memory_pct': raw_record.get('mem_usage', 0.0),
        'disk_pct': raw_record.get('disk_usage', 0.0),
        'network_in_mbps': raw_record.get('net_in', 0.0),
        'network_out_mbps': raw_record.get('net_out', 0.0),
        'disk_read_mbps': raw_record.get('disk_read', 0.0),
        'disk_write_mbps': raw_record.get('disk_write', 0.0),

        # OPTIONAL FIELDS - Uncomment if you want to include them
        # 'status': raw_record.get('health_status', 'healthy'),
        # 'profile': match_profile(raw_record.get('hostname', 'unknown'))
    }

def poll_data_source() -> List[Dict[str, Any]]:
    """
    CUSTOMIZE THIS: Poll your data source and return raw records.

    This is where you connect to:
    - Wells Fargo Linborg API
    - Elasticsearch
    - MongoDB
    - InfluxDB
    - Custom REST API
    - Or any other monitoring system

    Return: List of raw records (any format - will be transformed next)
    """
    try:
        # EXAMPLE: REST API query (CUSTOMIZE THIS)
        response = requests.get(
            DATA_SOURCE_URL,
            auth=DATA_SOURCE_AUTH,
            timeout=DATA_SOURCE_TIMEOUT
        )
        response.raise_for_status()

        data = response.json()

        # CUSTOMIZE: Adjust based on your API response format
        # Your API might return:
        # - {"servers": [...]}
        # - {"metrics": [...]}
        # - {"data": [...]}
        # - [...]  (direct array)

        return data.get('servers', [])  # CUSTOMIZE THIS LINE

    except Exception as e:
        logger.error(f"Failed to poll data source: {e}")
        return []

    # OTHER EXAMPLES (uncomment and customize for your source):

    # ELASTICSEARCH:
    # from elasticsearch import Elasticsearch
    # es = Elasticsearch(['http://localhost:9200'])
    # result = es.search(index="metrics-*", body={
    #     "query": {"range": {"@timestamp": {"gte": "now-5s"}}}
    # })
    # return [hit['_source'] for hit in result['hits']['hits']]

    # MONGODB:
    # from pymongo import MongoClient
    # client = MongoClient('mongodb://localhost:27017')
    # db = client.monitoring
    # return list(db.metrics.find({'timestamp': {'$gt': last_poll_time}}))

    # INFLUXDB:
    # from influxdb_client import InfluxDBClient
    # client = InfluxDBClient(url="http://localhost:8086", token="...")
    # query_api = client.query_api()
    # result = query_api.query('from(bucket:"metrics") |> range(start: -5s)')
    # return [record.values for table in result for record in table.records]

# ============================================================================
# MAIN LOOP - Already implemented, no changes needed
# ============================================================================

def run_adapter():
    """Main adapter loop. Already implemented - no changes needed!"""
    logger.info("=" * 60)
    logger.info("NordIQ Data Adapter Starting")
    logger.info("=" * 60)

    # Load API key
    try:
        api_key = load_api_key()
        logger.info("✓ API key loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load API key: {e}")
        return

    # Verify connection to inference daemon
    try:
        response = requests.get(f"{INFERENCE_DAEMON_URL}/health", timeout=5)
        response.raise_for_status()
        logger.info(f"✓ Connected to inference daemon at {INFERENCE_DAEMON_URL}")
    except Exception as e:
        logger.error(f"✗ Cannot reach inference daemon: {e}")
        logger.error(f"  URL: {INFERENCE_DAEMON_URL}")
        logger.error(f"  Make sure NordIQ is running: ./start_all.sh")
        return

    logger.info(f"✓ Polling every {POLL_INTERVAL_SECONDS} seconds")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Main polling loop
    iteration = 0
    while True:
        try:
            iteration += 1
            logger.info(f"[Iteration {iteration}] Polling data source...")

            # Step 1: Poll data source
            raw_records = poll_data_source()

            if not raw_records:
                logger.debug("No records from data source")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            logger.info(f"  → Polled {len(raw_records)} raw records")

            # Step 2: Transform to NordIQ format
            nordiq_records = []
            for i, raw in enumerate(raw_records):
                try:
                    transformed = transform_record(raw)
                    nordiq_records.append(transformed)
                except Exception as e:
                    logger.warning(f"  ⚠ Failed to transform record {i+1}: {e}")
                    continue

            if not nordiq_records:
                logger.warning("  ⚠ No valid records after transformation")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            logger.info(f"  → Transformed {len(nordiq_records)} records")

            # Step 3: Send to inference daemon
            success = send_to_inference_daemon(nordiq_records, api_key)

            if not success:
                logger.warning("  ⚠ Failed to send batch, will retry next iteration")

            # Step 4: Wait for next poll
            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("\n" + "=" * 60)
            logger.info("Shutting down gracefully...")
            logger.info("=" * 60)
            break

        except Exception as e:
            logger.error(f"✗ Unexpected error in main loop: {e}", exc_info=True)
            logger.error(f"  Will retry in {POLL_INTERVAL_SECONDS} seconds...")
            time.sleep(POLL_INTERVAL_SECONDS)

    logger.info("=" * 60)
    logger.info("NordIQ Data Adapter Stopped")
    logger.info("=" * 60)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_adapter()
