#!/usr/bin/env python3
"""
Production Metrics Forwarder - TEMPLATE
========================================

This is a template script for forwarding production metrics to the TFT Inference Daemon.

INSTRUCTIONS:
1. Copy this file to your production repository
2. Implement the `collect_metrics_from_your_system()` function
3. Update the configuration section
4. Test with sample data
5. Deploy and monitor

The script includes:
- Retry logic with exponential backoff
- Health checks and monitoring
- Graceful error handling
- Comprehensive logging
- Production-ready patterns

Replace the placeholder metric collection with your actual monitoring system integration.
"""

import requests
import time
import logging
import sys
from datetime import datetime
from typing import List, Dict, Optional
import json

# =============================================================================
# CONFIGURATION - CUSTOMIZE THIS SECTION
# =============================================================================

# Inference daemon settings
INFERENCE_URL = "http://localhost:8000/feed/data"
INFERENCE_HEALTH_URL = "http://localhost:8000/health"
SEND_INTERVAL = 5  # seconds between batches
REQUEST_TIMEOUT = 5  # seconds
MAX_RETRIES = 3

# Your server list (replace with actual server names)
SERVER_LIST = [
    'ppdb001', 'ppdb002', 'ppdb003',
    'ppml0001', 'ppml0002', 'ppml0003', 'ppml0004',
    'ppweb001', 'ppweb002', 'ppweb003'
]

# Logging configuration
LOG_FILE = 'metrics_forwarder.log'
LOG_LEVEL = logging.INFO

# Alert thresholds
MAX_CONSECUTIVE_FAILURES = 10
ALERT_ON_CONSECUTIVE_FAILURES = 5

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# =============================================================================
# METRIC COLLECTION - IMPLEMENT THIS SECTION
# =============================================================================

def collect_metrics_from_your_system() -> List[Dict]:
    """
    REPLACE THIS FUNCTION with your actual metrics collection logic.

    This should query your monitoring system (Prometheus, InfluxDB, CloudWatch,
    DataDog, etc.) and return metrics in the inference daemon format.

    Returns:
        List of metric records in the format:
        [
            {
                "timestamp": "2025-10-13T16:00:00Z",
                "server_name": "ppdb001",
                "cpu_pct": 45.2,
                "mem_pct": 67.8,
                "disk_io_mb_s": 123.4,
                "latency_ms": 12.5,
                "state": "healthy"
            },
            ...
        ]
    """
    records = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    # ===========================================================================
    # EXAMPLE 1: Prometheus Integration (Uncomment and modify)
    # ===========================================================================
    # from prometheus_api_client import PrometheusConnect
    # prom = PrometheusConnect(url="http://prometheus:9090")
    #
    # for server in SERVER_LIST:
    #     cpu_query = f'avg(rate(node_cpu_seconds_total{{instance=~"{server}.*",mode!="idle"}}[1m])) * 100'
    #     mem_query = f'(1 - node_memory_MemAvailable_bytes{{instance=~"{server}.*"}} / node_memory_MemTotal_bytes{{instance=~"{server}.*"}}) * 100'
    #
    #     cpu_result = prom.custom_query(cpu_query)
    #     mem_result = prom.custom_query(mem_query)
    #
    #     cpu_pct = float(cpu_result[0]['value'][1]) if cpu_result else 0.0
    #     mem_pct = float(mem_result[0]['value'][1]) if mem_result else 0.0
    #
    #     records.append({
    #         "timestamp": timestamp,
    #         "server_name": server,
    #         "cpu_pct": cpu_pct,
    #         "mem_pct": mem_pct,
    #         "disk_io_mb_s": 0.0,  # Add disk I/O query
    #         "latency_ms": 0.0,    # Add latency query
    #         "state": derive_state(cpu_pct, mem_pct)
    #     })

    # ===========================================================================
    # EXAMPLE 2: InfluxDB Integration (Uncomment and modify)
    # ===========================================================================
    # from influxdb_client import InfluxDBClient
    #
    # client = InfluxDBClient(url="http://influxdb:8086", token="your-token", org="your-org")
    # query_api = client.query_api()
    #
    # for server in SERVER_LIST:
    #     query = f'''
    #     from(bucket: "server-metrics")
    #       |> range(start: -5s)
    #       |> filter(fn: (r) => r["host"] == "{server}")
    #       |> last()
    #     '''
    #     tables = query_api.query(query)
    #
    #     metrics = {}
    #     for table in tables:
    #         for record in table.records:
    #             metrics[record.get_field()] = record.get_value()
    #
    #     records.append({
    #         "timestamp": timestamp,
    #         "server_name": server,
    #         "cpu_pct": metrics.get('cpu', 0.0),
    #         "mem_pct": metrics.get('memory', 0.0),
    #         "disk_io_mb_s": metrics.get('disk_io', 0.0),
    #         "latency_ms": metrics.get('latency', 0.0),
    #         "state": metrics.get('state', 'healthy')
    #     })

    # ===========================================================================
    # EXAMPLE 3: Log File Parsing (Uncomment and modify)
    # ===========================================================================
    # import json
    #
    # log_file = "/var/log/server-metrics.log"
    # with open(log_file, 'r') as f:
    #     # Read last N lines
    #     lines = f.readlines()[-len(SERVER_LIST):]
    #
    #     for line in lines:
    #         try:
    #             data = json.loads(line)
    #             records.append({
    #                 "timestamp": data.get('timestamp', timestamp),
    #                 "server_name": data['hostname'],
    #                 "cpu_pct": data['cpu'],
    #                 "mem_pct": data['memory'],
    #                 "disk_io_mb_s": data['disk_io'],
    #                 "latency_ms": data['latency'],
    #                 "state": data.get('state', 'healthy')
    #             })
    #         except (json.JSONDecodeError, KeyError) as e:
    #             logger.error(f"Failed to parse log line: {e}")

    # ===========================================================================
    # PLACEHOLDER: Remove this when you implement real metric collection
    # ===========================================================================
    logger.warning("Using placeholder metrics - REPLACE with actual implementation!")

    for server in SERVER_LIST:
        records.append({
            "timestamp": timestamp,
            "server_name": server,
            "cpu_pct": 45.0,  # TODO: Get from your monitoring system
            "mem_pct": 60.0,  # TODO: Get from your monitoring system
            "disk_io_mb_s": 100.0,  # TODO: Get from your monitoring system
            "latency_ms": 10.0,  # TODO: Get from your monitoring system
            "state": "healthy"  # TODO: Derive from actual metrics
        })

    return records


def derive_state(cpu_pct: float, mem_pct: float, latency_ms: float = 0.0) -> str:
    """
    Derive operational state from current metrics.

    This is a simple heuristic - customize based on your operational definitions.

    Args:
        cpu_pct: CPU utilization percentage
        mem_pct: Memory utilization percentage
        latency_ms: Request latency in milliseconds

    Returns:
        One of: healthy, heavy_load, critical_issue, idle
    """
    if cpu_pct > 95 or mem_pct > 95 or latency_ms > 200:
        return "critical_issue"
    elif cpu_pct > 80 or mem_pct > 80 or latency_ms > 100:
        return "heavy_load"
    elif cpu_pct < 10 and mem_pct < 30:
        return "idle"
    else:
        return "healthy"

# =============================================================================
# INFERENCE DAEMON INTEGRATION
# =============================================================================

def check_daemon_health() -> bool:
    """Check if inference daemon is running and healthy."""
    try:
        response = requests.get(INFERENCE_HEALTH_URL, timeout=2)
        if response.status_code == 200:
            data = response.json()
            return data.get('status') == 'healthy'
        return False
    except requests.exceptions.RequestException:
        return False


def send_to_inference(records: List[Dict]) -> Optional[Dict]:
    """
    Send metrics to inference daemon with retry logic.

    Args:
        records: List of metric records

    Returns:
        Response dict if successful, None if failed
    """
    if not records:
        logger.warning("No records to send")
        return None

    payload = {"records": records}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                INFERENCE_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()

                # Log success with warmup status
                warmup_status = "✓ READY" if result.get('warmup_complete') else \
                    f"{result.get('servers_ready', 0)}/{result.get('servers_tracked', 0)} servers"

                logger.info(
                    f"✓ Sent {len(records)} records | "
                    f"Tick: {result.get('tick', 0)} | "
                    f"Warmup: {warmup_status}"
                )

                return result

            else:
                logger.error(f"HTTP {response.status_code}: {response.text}")

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt}/{MAX_RETRIES}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection failed on attempt {attempt}/{MAX_RETRIES}")
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt}/{MAX_RETRIES}: {e}")

        # Wait before retry (exponential backoff)
        if attempt < MAX_RETRIES:
            wait_time = 2 ** attempt  # 2s, 4s, 8s
            time.sleep(wait_time)

    logger.error(f"❌ Failed to send after {MAX_RETRIES} attempts")
    return None


def get_warmup_status() -> Dict:
    """Get current warmup status from daemon."""
    try:
        response = requests.get("http://localhost:8000/status", timeout=2)
        if response.status_code == 200:
            return response.json().get('warmup', {})
    except:
        pass
    return {}

# =============================================================================
# MONITORING AND ALERTS
# =============================================================================

def alert_consecutive_failures(count: int):
    """
    Alert on consecutive failures.

    CUSTOMIZE THIS: Integrate with your alerting system (PagerDuty, Slack, etc.)
    """
    logger.critical(f"🚨 ALERT: {count} consecutive failures sending to inference daemon!")

    # Example: Send to Slack (uncomment and configure)
    # import requests
    # slack_webhook = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
    # requests.post(slack_webhook, json={
    #     "text": f"⚠️ Metrics forwarder: {count} consecutive failures!"
    # })

    # Example: Send email (uncomment and configure)
    # import smtplib
    # from email.mime.text import MIMEText
    # msg = MIMEText(f"Metrics forwarder has failed {count} times in a row")
    # msg['Subject'] = 'TFT Inference Forwarder Alert'
    # msg['From'] = 'alerts@company.com'
    # msg['To'] = 'oncall@company.com'
    # s = smtplib.SMTP('localhost')
    # s.send_message(msg)
    # s.quit()


def print_startup_banner():
    """Print startup information."""
    logger.info("=" * 70)
    logger.info("TFT INFERENCE METRICS FORWARDER")
    logger.info("=" * 70)
    logger.info(f"Inference URL: {INFERENCE_URL}")
    logger.info(f"Send interval: {SEND_INTERVAL}s")
    logger.info(f"Servers tracked: {len(SERVER_LIST)}")
    logger.info(f"Request timeout: {REQUEST_TIMEOUT}s")
    logger.info(f"Max retries: {MAX_RETRIES}")
    logger.info("=" * 70)


def print_statistics(stats: Dict):
    """Print periodic statistics."""
    logger.info("-" * 70)
    logger.info("STATISTICS")
    logger.info("-" * 70)
    logger.info(f"Total batches sent: {stats['total_batches']}")
    logger.info(f"Total records sent: {stats['total_records']}")
    logger.info(f"Success rate: {stats['success_rate']:.1f}%")
    logger.info(f"Consecutive failures: {stats['consecutive_failures']}")
    logger.info(f"Uptime: {stats['uptime']}")
    logger.info("-" * 70)

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    """Main event loop."""
    print_startup_banner()

    # Check daemon health before starting
    logger.info("Checking inference daemon health...")
    if not check_daemon_health():
        logger.error("❌ Inference daemon is not responding!")
        logger.error("Start it with: python tft_inference_daemon.py")
        sys.exit(1)

    logger.info("✓ Inference daemon is healthy")

    # Wait for warmup if needed
    warmup = get_warmup_status()
    if not warmup.get('is_warmed_up', False):
        logger.info(f"⏳ Daemon is warming up: {warmup.get('message', 'Starting...')}")
        logger.info(f"   This will take ~{SEND_INTERVAL * 150 / 60:.1f} minutes")

    # Statistics
    stats = {
        'total_batches': 0,
        'total_records': 0,
        'successful_batches': 0,
        'consecutive_failures': 0,
        'start_time': time.time()
    }

    try:
        while True:
            start_time = time.time()

            # Collect metrics from your system
            try:
                records = collect_metrics_from_your_system()
                stats['total_records'] += len(records)
            except Exception as e:
                logger.error(f"Failed to collect metrics: {e}", exc_info=True)
                records = []

            # Send to inference daemon
            if records:
                result = send_to_inference(records)
                stats['total_batches'] += 1

                if result:
                    stats['successful_batches'] += 1
                    stats['consecutive_failures'] = 0
                else:
                    stats['consecutive_failures'] += 1

                    # Alert on consecutive failures
                    if stats['consecutive_failures'] == ALERT_ON_CONSECUTIVE_FAILURES:
                        alert_consecutive_failures(stats['consecutive_failures'])

                    # Exit on max failures
                    if stats['consecutive_failures'] >= MAX_CONSECUTIVE_FAILURES:
                        logger.critical(
                            f"❌ Exceeded {MAX_CONSECUTIVE_FAILURES} consecutive failures. Exiting."
                        )
                        sys.exit(1)

            # Print statistics every 100 batches
            if stats['total_batches'] % 100 == 0 and stats['total_batches'] > 0:
                uptime_seconds = time.time() - stats['start_time']
                uptime_hours = uptime_seconds / 3600
                stats['success_rate'] = (stats['successful_batches'] / stats['total_batches']) * 100
                stats['uptime'] = f"{uptime_hours:.1f}h"
                print_statistics(stats)

            # Wait for next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, SEND_INTERVAL - elapsed)

            if elapsed > SEND_INTERVAL:
                logger.warning(f"Metric collection took {elapsed:.2f}s (longer than {SEND_INTERVAL}s interval)")

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down gracefully...")
        logger.info(f"Final stats: {stats['successful_batches']}/{stats['total_batches']} batches successful")
    except Exception as e:
        logger.critical(f"💥 Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
