# Data Adapter Development Guide

**Purpose**: Guide for building custom data adapter daemons that feed production data to NordIQ inference engine.

**Target Audience**: Claude 3.7 (or any developer) building adapters for unknown data sources.

**Use Case**: Wells Fargo Linborg system, or any custom monitoring infrastructure.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Inference Daemon API Contract](#inference-daemon-api-contract)
3. [Profile Matching Logic](#profile-matching-logic)
4. [Required Metrics Fields](#required-metrics-fields)
5. [Complete Template Code](#complete-template-code)
6. [Implementation Checklist](#implementation-checklist)
7. [Testing & Validation](#testing--validation)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ YOUR DATA SOURCE (Unknown)                                      │
├─────────────────────────────────────────────────────────────────┤
│ • Wells Fargo Linborg                                           │
│ • Direct service monitoring                                     │
│ • Elasticsearch/MongoDB                                         │
│ • InfluxDB/Prometheus                                           │
│ • Any monitoring system                                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                ┌───────────────────────┐
                │  YOUR ADAPTER DAEMON  │
                │  (You build this)     │
                ├───────────────────────┤
                │ 1. Poll/Subscribe     │
                │ 2. Transform          │
                │ 3. Profile Match      │
                │ 4. HTTP POST          │
                └───────────────────────┘
                            ↓
                      HTTP POST
                  /feed/data endpoint
                            ↓
        ┌───────────────────────────────────────┐
        │ NORDIQ INFERENCE DAEMON (Existing)    │
        ├───────────────────────────────────────┤
        │ ✓ Accepts multi-server data           │
        │ ✓ Profile matching (if not provided)  │
        │ ✓ Rolling window management           │
        │ ✓ TFT predictions                     │
        │ ✓ Risk scoring                        │
        │ ✓ Alert generation                    │
        └───────────────────────────────────────┘
                            ↓
                      Dashboard
```

### Key Responsibilities

**Your Adapter Daemon (What you build):**
- ✅ Connect to data source (Linborg, etc.)
- ✅ Poll or subscribe to metrics
- ✅ Transform to required format (see below)
- ⚠️ Optional: Profile matching (inference daemon can do this)
- ✅ Batch multiple servers in single HTTP POST
- ✅ Handle retries and errors
- ✅ Run as daemon process

**NordIQ Inference Daemon (Already exists):**
- ✅ Accept data via `/feed/data` endpoint
- ✅ Auto-detect profiles from server names (if not provided)
- ✅ Manage rolling window warmup
- ✅ Make TFT predictions
- ✅ Calculate risk scores
- ✅ Generate alerts
- ✅ Serve dashboard

---

## Inference Daemon API Contract

### Endpoint Details

**URL**: `http://localhost:8000/feed/data`
**Method**: `POST`
**Authentication**: `X-API-Key` header
**Rate Limit**: 60 requests/minute (1 per second)
**Content-Type**: `application/json`

### Request Format

```json
{
  "records": [
    {
      "timestamp": "2025-10-30T14:35:00",
      "server_name": "ppdb001",
      "cpu_pct": 45.2,
      "memory_pct": 78.5,
      "disk_pct": 62.3,
      "network_in_mbps": 125.4,
      "network_out_mbps": 89.2,
      "disk_read_mbps": 42.1,
      "disk_write_mbps": 18.3,
      "status": "healthy",
      "profile": "database"
    },
    {
      "timestamp": "2025-10-30T14:35:00",
      "server_name": "ppweb003",
      "cpu_pct": 32.1,
      "memory_pct": 45.6,
      "disk_pct": 38.9,
      "network_in_mbps": 89.3,
      "network_out_mbps": 156.7,
      "disk_read_mbps": 5.2,
      "disk_write_mbps": 2.1,
      "status": "healthy",
      "profile": "web_api"
    }
  ]
}
```

### Response Format

**Success (200 OK)**:
```json
{
  "success": true,
  "records_received": 2,
  "servers_updated": ["ppdb001", "ppweb003"],
  "warmup_status": {
    "ready": 2,
    "total": 2,
    "is_ready": true
  }
}
```

**Error (400/500)**:
```json
{
  "detail": "Error message here"
}
```

---

## Profile Matching Logic

### Option 1: Let Inference Daemon Handle It (Easiest)

**Don't include `profile` field in your records.**

The inference daemon will automatically detect profiles from server names:

```python
# Inference daemon code (tft_inference_daemon.py:579-589)
def get_profile(server_name):
    if server_name.startswith('ppml'): return 'ml_compute'
    if server_name.startswith('ppdb'): return 'database'
    if server_name.startswith('ppweb'): return 'web_api'
    if server_name.startswith('ppcon'): return 'conductor_mgmt'
    if server_name.startswith('ppetl'): return 'data_ingest'
    if server_name.startswith('pprisk'): return 'risk_analytics'
    return 'generic'
```

**Pro**: Simplest for you, zero work needed.
**Con**: Limited to prefix-based matching.

### Option 2: Your Adapter Does Profile Matching (More Flexible)

**Include `profile` field in your records.**

Your adapter can use any logic:
- Database lookups (CMDB)
- Regex patterns
- Service tags
- Custom business logic
- Wells Fargo-specific mappings

**Pro**: Full control, sophisticated matching.
**Con**: You have to implement it.

### Available Profiles

```python
# All valid profile values:
profiles = [
    'ml_compute',       # ML/AI workloads
    'database',         # Database servers
    'web_api',          # Web/API servers
    'conductor_mgmt',   # Orchestration/mgmt
    'data_ingest',      # ETL/data pipelines
    'risk_analytics',   # Risk calculation
    'generic'           # Unknown/other
]
```

**Risk Scoring**: Different profiles have different thresholds. For example:
- `database`: High memory OK (page cache expected)
- `ml_compute`: High CPU OK (training expected)
- `web_api`: High network OK (traffic expected)

### Example: Custom Profile Matching

```python
def match_profile(server_name, service_tags=None, cmdb_info=None):
    """
    Custom profile matching for Wells Fargo environment.

    Args:
        server_name: Server hostname
        service_tags: Optional service metadata
        cmdb_info: Optional CMDB lookup result

    Returns:
        profile: One of the 7 valid profiles
    """
    # Method 1: CMDB lookup
    if cmdb_info and 'role' in cmdb_info:
        role = cmdb_info['role'].lower()
        if 'database' in role or 'sql' in role or 'oracle' in role:
            return 'database'
        if 'web' in role or 'api' in role or 'nginx' in role:
            return 'web_api'
        if 'ml' in role or 'gpu' in role or 'cuda' in role:
            return 'ml_compute'
        if 'etl' in role or 'kafka' in role or 'spark' in role:
            return 'data_ingest'
        if 'risk' in role or 'quant' in role:
            return 'risk_analytics'

    # Method 2: Service tags (if provided)
    if service_tags:
        for tag in service_tags:
            if tag in ['postgres', 'mysql', 'mssql', 'oracle']:
                return 'database'
            if tag in ['nginx', 'apache', 'api-gateway']:
                return 'web_api'
            if tag in ['tensorflow', 'pytorch', 'ml-training']:
                return 'ml_compute'

    # Method 3: Hostname patterns (Wells Fargo specific)
    name_lower = server_name.lower()

    # Wells Fargo naming conventions (examples - adjust to your actual naming)
    if 'db' in name_lower or 'sql' in name_lower:
        return 'database'
    if 'web' in name_lower or 'api' in name_lower:
        return 'web_api'
    if 'ml' in name_lower or 'gpu' in name_lower:
        return 'ml_compute'
    if 'etl' in name_lower or 'kafka' in name_lower:
        return 'data_ingest'
    if 'risk' in name_lower or 'quant' in name_lower:
        return 'risk_analytics'
    if 'conductor' in name_lower or 'mgmt' in name_lower:
        return 'conductor_mgmt'

    # Default: generic profile
    return 'generic'
```

---

## Required Metrics Fields

### Mandatory Fields

These fields **MUST** be present in every record:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `timestamp` | string (ISO 8601) | Metric collection time | `"2025-10-30T14:35:00"` |
| `server_name` | string | Unique server identifier | `"ppdb001"` |
| `cpu_pct` | float | CPU usage percentage | `45.2` |
| `memory_pct` | float | Memory usage percentage | `78.5` |
| `disk_pct` | float | Disk usage percentage | `62.3` |
| `network_in_mbps` | float | Network inbound (Mbps) | `125.4` |
| `network_out_mbps` | float | Network outbound (Mbps) | `89.2` |
| `disk_read_mbps` | float | Disk read throughput (MB/s) | `42.1` |
| `disk_write_mbps` | float | Disk write throughput (MB/s) | `18.3` |

### Optional Fields

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `status` | string | Server health status | `"healthy"` |
| `profile` | string | Server profile type | Auto-detected |

### Valid Values

**status**: One of `"healthy"`, `"degraded"`, `"warning"`, `"critical"`
**profile**: One of `"ml_compute"`, `"database"`, `"web_api"`, `"conductor_mgmt"`, `"data_ingest"`, `"risk_analytics"`, `"generic"`

### Data Validation Rules

```python
# Validation rules applied by inference daemon:
VALIDATION_RULES = {
    # Percentages: 0-100
    'cpu_pct': (0, 100),
    'memory_pct': (0, 100),
    'disk_pct': (0, 100),

    # Throughput: 0-10000 (Mbps or MB/s)
    'network_in_mbps': (0, 10000),
    'network_out_mbps': (0, 10000),
    'disk_read_mbps': (0, 10000),
    'disk_write_mbps': (0, 10000),

    # String validations
    'server_name': 'non-empty string',
    'timestamp': 'ISO 8601 format'
}
```

**The inference daemon will:**
- ✅ Fill missing `status` with `"healthy"`
- ✅ Auto-detect `profile` if missing
- ❌ Reject records with missing mandatory fields
- ❌ Reject records with invalid value ranges

---

## Complete Template Code

### Minimal Adapter (Polling Model)

```python
#!/usr/bin/env python3
"""
Generic Data Adapter for NordIQ Inference Engine
Polls data source and feeds to inference daemon.

Customize the poll_data_source() function for your data source.
"""

import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# ============================================================================
# CONFIGURATION
# ============================================================================

# NordIQ Inference Daemon
INFERENCE_DAEMON_URL = "http://localhost:8000"
API_KEY_FILE = ".nordiq_key"  # Relative to NordIQ root

# Polling interval
POLL_INTERVAL_SECONDS = 5

# Data source (customize these)
DATA_SOURCE_URL = "http://your-monitoring-system/api/metrics"
DATA_SOURCE_AUTH = None  # Or {"username": "...", "password": "..."}

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
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

def match_profile(server_name: str) -> str:
    """
    Match server name to profile.

    CUSTOMIZE THIS for your environment!

    Options:
    1. Return 'generic' and let inference daemon handle it
    2. Implement custom logic based on CMDB, tags, etc.
    """
    # Option 1: Let inference daemon handle it
    # return 'generic'  # Inference daemon will auto-detect

    # Option 2: Custom matching (example)
    name_lower = server_name.lower()

    if 'db' in name_lower or 'sql' in name_lower:
        return 'database'
    if 'web' in name_lower or 'api' in name_lower:
        return 'web_api'
    if 'ml' in name_lower or 'gpu' in name_lower:
        return 'ml_compute'
    if 'etl' in name_lower or 'kafka' in name_lower:
        return 'data_ingest'
    if 'risk' in name_lower:
        return 'risk_analytics'
    if 'conductor' in name_lower or 'mgmt' in name_lower:
        return 'conductor_mgmt'

    return 'generic'

def transform_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform raw data source record to NordIQ format.

    CUSTOMIZE THIS for your data source format!

    Args:
        raw_record: Record from your data source (unknown format)

    Returns:
        Record in NordIQ format (see API contract above)
    """
    # EXAMPLE: Adjust field mappings to match your data source

    # Your data source might have different field names:
    # - timestamp: "collected_at", "event_time", "@timestamp"
    # - server_name: "hostname", "host", "server_id"
    # - cpu_pct: "cpu_usage", "cpu", "processor_percent"
    # - etc.

    # Example transformation (CUSTOMIZE THIS):
    return {
        # Required fields
        'timestamp': raw_record.get('collected_at', datetime.now().isoformat()),
        'server_name': raw_record.get('hostname', 'unknown'),
        'cpu_pct': raw_record.get('cpu_usage', 0.0),
        'memory_pct': raw_record.get('mem_usage', 0.0),
        'disk_pct': raw_record.get('disk_usage', 0.0),
        'network_in_mbps': raw_record.get('net_in', 0.0),
        'network_out_mbps': raw_record.get('net_out', 0.0),
        'disk_read_mbps': raw_record.get('disk_read', 0.0),
        'disk_write_mbps': raw_record.get('disk_write', 0.0),

        # Optional fields
        'status': raw_record.get('health_status', 'healthy'),
        'profile': match_profile(raw_record.get('hostname', 'unknown'))
    }

# ============================================================================
# DATA SOURCE INTEGRATION (CUSTOMIZE THIS)
# ============================================================================

def poll_data_source() -> List[Dict[str, Any]]:
    """
    Poll your data source and return raw records.

    CUSTOMIZE THIS ENTIRE FUNCTION for your data source!

    Examples:
    - Query Linborg API
    - Query Elasticsearch
    - Query MongoDB
    - Query InfluxDB
    - Read from Kafka
    - Query Prometheus

    Returns:
        List of raw records (any format)
    """
    # EXAMPLE 1: REST API query
    try:
        response = requests.get(
            DATA_SOURCE_URL,
            auth=DATA_SOURCE_AUTH,
            timeout=10
        )
        response.raise_for_status()

        # Adjust based on your API response format
        data = response.json()

        # Your API might return:
        # - {"servers": [...]}
        # - {"metrics": [...]}
        # - {"data": [...]}
        # - [...]  (direct array)

        return data.get('servers', [])  # CUSTOMIZE THIS

    except Exception as e:
        logger.error(f"Failed to poll data source: {e}")
        return []

    # EXAMPLE 2: Elasticsearch query (see ELASTICSEARCH_INTEGRATION.md)
    # from elasticsearch import Elasticsearch
    # es = Elasticsearch(['http://localhost:9200'])
    # result = es.search(index="metrics-*", body={...})
    # return [hit['_source'] for hit in result['hits']['hits']]

    # EXAMPLE 3: MongoDB query (see MONGODB_INTEGRATION.md)
    # from pymongo import MongoClient
    # client = MongoClient('mongodb://localhost:27017')
    # db = client.monitoring
    # return list(db.metrics.find({'timestamp': {'$gt': last_poll_time}}))

    # EXAMPLE 4: InfluxDB query
    # from influxdb_client import InfluxDBClient
    # client = InfluxDBClient(url="http://localhost:8086", token="...")
    # query_api = client.query_api()
    # result = query_api.query('from(bucket:"metrics") |> range(start: -5s)')
    # return [record.values for table in result for record in table.records]

# ============================================================================
# NORDIQ INTEGRATION
# ============================================================================

def send_to_inference_daemon(records: List[Dict[str, Any]], api_key: str) -> bool:
    """
    Send batch of records to NordIQ inference daemon.

    Args:
        records: List of records in NordIQ format
        api_key: NordIQ API key

    Returns:
        True if successful, False otherwise
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
            f"Sent {result.get('records_received', 0)} records, "
            f"updated {len(result.get('servers_updated', []))} servers"
        )

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to send data to inference daemon: {e}")
        return False

# ============================================================================
# MAIN LOOP
# ============================================================================

def run_adapter():
    """Main adapter loop."""
    logger.info("=== NordIQ Data Adapter Starting ===")

    # Load API key
    try:
        api_key = load_api_key()
        logger.info("API key loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load API key: {e}")
        return

    # Verify connection to inference daemon
    try:
        response = requests.get(f"{INFERENCE_DAEMON_URL}/health", timeout=5)
        response.raise_for_status()
        logger.info("Connected to inference daemon")
    except Exception as e:
        logger.error(f"Cannot reach inference daemon: {e}")
        logger.error(f"URL: {INFERENCE_DAEMON_URL}")
        return

    logger.info(f"Polling every {POLL_INTERVAL_SECONDS} seconds")
    logger.info("Press Ctrl+C to stop")

    # Main polling loop
    while True:
        try:
            # Step 1: Poll data source
            raw_records = poll_data_source()

            if not raw_records:
                logger.debug("No records from data source")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            logger.info(f"Polled {len(raw_records)} raw records")

            # Step 2: Transform to NordIQ format
            nordiq_records = []
            for raw in raw_records:
                try:
                    transformed = transform_record(raw)
                    nordiq_records.append(transformed)
                except Exception as e:
                    logger.warning(f"Failed to transform record: {e}")
                    continue

            if not nordiq_records:
                logger.warning("No valid records after transformation")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Step 3: Send to inference daemon
            send_to_inference_daemon(nordiq_records, api_key)

            # Step 4: Wait for next poll
            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("\nShutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL_SECONDS)

    logger.info("=== NordIQ Data Adapter Stopped ===")

if __name__ == "__main__":
    run_adapter()
```

### Advanced Adapter (Streaming/Subscription Model)

```python
#!/usr/bin/env python3
"""
Advanced Data Adapter for NordIQ - Streaming Model
Subscribe to event stream and feed to inference daemon in real-time.

Use this for: Kafka, MQTT, WebSocket, or other streaming sources.
"""

import asyncio
import requests
from datetime import datetime
from typing import List, Dict, Any
import logging
from collections import defaultdict
import threading

# Configuration
INFERENCE_DAEMON_URL = "http://localhost:8000"
API_KEY_FILE = ".nordiq_key"
BATCH_SIZE = 50  # Send to inference daemon every N records
BATCH_TIMEOUT_SECONDS = 5  # Or every N seconds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# BATCHING LOGIC
# ============================================================================

class RecordBatcher:
    """Batches records for efficient sending."""

    def __init__(self, batch_size: int, timeout: float, callback):
        self.batch_size = batch_size
        self.timeout = timeout
        self.callback = callback
        self.buffer = []
        self.lock = threading.Lock()
        self.last_flush = datetime.now()

    def add(self, record: Dict[str, Any]):
        """Add record to batch."""
        with self.lock:
            self.buffer.append(record)

            # Flush if batch size reached
            if len(self.buffer) >= self.batch_size:
                self._flush()

            # Flush if timeout reached
            elif (datetime.now() - self.last_flush).total_seconds() >= self.timeout:
                self._flush()

    def _flush(self):
        """Send batch to callback."""
        if self.buffer:
            try:
                self.callback(self.buffer.copy())
            except Exception as e:
                logger.error(f"Batch callback failed: {e}")

            self.buffer.clear()
            self.last_flush = datetime.now()

    def flush(self):
        """Manually flush remaining records."""
        with self.lock:
            self._flush()

# ============================================================================
# STREAMING DATA SOURCE (CUSTOMIZE THIS)
# ============================================================================

async def subscribe_to_stream(batcher: RecordBatcher):
    """
    Subscribe to streaming data source.

    CUSTOMIZE THIS for your streaming source!

    Examples:
    - Kafka consumer
    - MQTT subscriber
    - WebSocket connection
    - Redis pub/sub
    - RabbitMQ consumer
    """
    # EXAMPLE 1: Kafka consumer
    # from confluent_kafka import Consumer
    #
    # consumer = Consumer({
    #     'bootstrap.servers': 'localhost:9092',
    #     'group.id': 'nordiq-adapter',
    #     'auto.offset.reset': 'latest'
    # })
    # consumer.subscribe(['metrics-topic'])
    #
    # while True:
    #     msg = consumer.poll(timeout=1.0)
    #     if msg is None:
    #         continue
    #     if msg.error():
    #         logger.error(f"Kafka error: {msg.error()}")
    #         continue
    #
    #     raw_record = json.loads(msg.value().decode('utf-8'))
    #     nordiq_record = transform_record(raw_record)
    #     batcher.add(nordiq_record)

    # EXAMPLE 2: WebSocket stream
    # import websocket
    #
    # def on_message(ws, message):
    #     raw_record = json.loads(message)
    #     nordiq_record = transform_record(raw_record)
    #     batcher.add(nordiq_record)
    #
    # ws = websocket.WebSocketApp(
    #     "ws://your-monitoring-system/stream",
    #     on_message=on_message
    # )
    # ws.run_forever()

    # EXAMPLE 3: MQTT subscriber
    # import paho.mqtt.client as mqtt
    #
    # def on_message(client, userdata, msg):
    #     raw_record = json.loads(msg.payload.decode())
    #     nordiq_record = transform_record(raw_record)
    #     batcher.add(nordiq_record)
    #
    # client = mqtt.Client()
    # client.on_message = on_message
    # client.connect("mqtt.example.com", 1883, 60)
    # client.subscribe("metrics/#")
    # client.loop_forever()

    logger.error("subscribe_to_stream() not implemented - customize for your source!")

# ============================================================================
# MAIN
# ============================================================================

def send_batch(records: List[Dict[str, Any]]):
    """Callback to send batch to inference daemon."""
    try:
        api_key = load_api_key()
        response = requests.post(
            f"{INFERENCE_DAEMON_URL}/feed/data",
            json={"records": records},
            headers={"X-API-Key": api_key},
            timeout=30
        )
        response.raise_for_status()
        logger.info(f"Sent batch of {len(records)} records")
    except Exception as e:
        logger.error(f"Failed to send batch: {e}")

async def main():
    """Main streaming adapter loop."""
    logger.info("=== NordIQ Streaming Adapter Starting ===")

    # Create batcher
    batcher = RecordBatcher(
        batch_size=BATCH_SIZE,
        timeout=BATCH_TIMEOUT_SECONDS,
        callback=send_batch
    )

    # Subscribe to stream
    try:
        await subscribe_to_stream(batcher)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        batcher.flush()  # Send remaining records

    logger.info("=== NordIQ Streaming Adapter Stopped ===")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Implementation Checklist

Use this checklist when building your adapter:

### Phase 1: Setup (15 minutes)

- [ ] Copy template code to `src/daemons/your_adapter_daemon.py`
- [ ] Update `DATA_SOURCE_URL` configuration
- [ ] Test connection to data source manually
- [ ] Test connection to inference daemon: `curl http://localhost:8000/health`
- [ ] Verify API key exists: `cat .nordiq_key`

### Phase 2: Data Source Integration (1-4 hours)

- [ ] Implement `poll_data_source()` for your source
- [ ] Test: Print raw records to verify format
- [ ] Document your data source's field names
- [ ] Handle authentication (if needed)
- [ ] Handle pagination (if needed)
- [ ] Handle rate limiting (if needed)

### Phase 3: Transformation (1-2 hours)

- [ ] Implement `transform_record()` for your format
- [ ] Map all 9 required fields (see Required Metrics Fields)
- [ ] Handle missing/null values
- [ ] Validate data ranges (0-100 for percentages, etc.)
- [ ] Test: Print transformed records
- [ ] Verify timestamp format (ISO 8601)

### Phase 4: Profile Matching (30 minutes - 2 hours)

Choose one approach:

**Option A: Let inference daemon handle it**
- [ ] Don't include `profile` field
- [ ] Verify server names follow patterns (ppml*, ppdb*, etc.)
- [ ] Test with sample data

**Option B: Custom profile matching**
- [ ] Implement `match_profile()` function
- [ ] Test with all server types
- [ ] Verify all 7 profiles covered
- [ ] Document your matching logic

### Phase 5: Integration Testing (1 hour)

- [ ] Start inference daemon: `./start_all.sh`
- [ ] Run adapter: `python src/daemons/your_adapter_daemon.py`
- [ ] Verify logs show successful sends
- [ ] Check inference daemon logs: `tail -f logs/inference_daemon.log`
- [ ] Open dashboard: http://localhost:8050
- [ ] Verify servers appear on dashboard
- [ ] Wait 20-30 minutes for warmup
- [ ] Verify predictions appear

### Phase 6: Error Handling (1 hour)

- [ ] Test: Inference daemon down (should log error, retry)
- [ ] Test: Invalid data (should skip bad records)
- [ ] Test: Network timeout (should retry)
- [ ] Test: API key invalid (should fail with clear message)
- [ ] Add retry logic with exponential backoff
- [ ] Add circuit breaker pattern (optional)

### Phase 7: Production Readiness (2 hours)

- [ ] Convert to daemon mode (nohup, PID tracking)
- [ ] Add to `start_all.sh` script
- [ ] Add to `stop_all.sh` script
- [ ] Add to `status.sh` script
- [ ] Configure log rotation
- [ ] Document configuration options
- [ ] Write deployment guide

---

## Testing & Validation

### Manual Testing

**Step 1: Test Data Source Connection**
```python
# Add to your adapter for testing
if __name__ == "__main__":
    # Test mode
    raw = poll_data_source()
    print(f"Got {len(raw)} records")
    print(f"Sample record: {raw[0] if raw else 'None'}")
```

**Step 2: Test Transformation**
```python
# Test transformation
raw = poll_data_source()
if raw:
    transformed = transform_record(raw[0])
    print(f"Transformed: {transformed}")

    # Validate required fields
    required = ['timestamp', 'server_name', 'cpu_pct', 'memory_pct',
                'disk_pct', 'network_in_mbps', 'network_out_mbps',
                'disk_read_mbps', 'disk_write_mbps']

    missing = [f for f in required if f not in transformed]
    if missing:
        print(f"ERROR: Missing fields: {missing}")
    else:
        print("✓ All required fields present")
```

**Step 3: Test Inference Daemon Integration**
```bash
# Send test data via curl
API_KEY=$(cat .nordiq_key)

curl -X POST http://localhost:8000/feed/data \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "timestamp": "2025-10-30T14:35:00",
        "server_name": "test001",
        "cpu_pct": 45.2,
        "memory_pct": 78.5,
        "disk_pct": 62.3,
        "network_in_mbps": 125.4,
        "network_out_mbps": 89.2,
        "disk_read_mbps": 42.1,
        "disk_write_mbps": 18.3
      }
    ]
  }'
```

**Step 4: Verify Dashboard**
```bash
# Open dashboard
open http://localhost:8050

# Or check predictions via API
curl -H "X-API-Key: $API_KEY" http://localhost:8000/predictions/current
```

### Automated Testing

```python
# test_adapter.py
import pytest
from your_adapter_daemon import transform_record, match_profile

def test_transform_basic():
    """Test basic transformation."""
    raw = {
        'hostname': 'ppdb001',
        'cpu_usage': 45.2,
        'mem_usage': 78.5,
        # ... other fields
    }

    result = transform_record(raw)

    assert 'server_name' in result
    assert result['server_name'] == 'ppdb001'
    assert result['cpu_pct'] == 45.2

def test_profile_matching():
    """Test profile matching logic."""
    assert match_profile('ppdb001') == 'database'
    assert match_profile('ppweb003') == 'web_api'
    assert match_profile('unknown') == 'generic'

def test_handles_missing_fields():
    """Test handling of missing data."""
    raw = {'hostname': 'test001'}  # Missing most fields

    result = transform_record(raw)

    # Should fill with defaults
    assert result['cpu_pct'] == 0.0
    assert result['memory_pct'] == 0.0
```

---

## Production Deployment

### Add to Startup Scripts

**1. Update `start_all.sh`**:
```bash
# Add after metrics daemon section
echo "Starting Your Adapter Daemon..."
nohup python src/daemons/your_adapter_daemon.py \
    > "$LOG_DIR/adapter_daemon.log" 2>&1 &
ADAPTER_PID=$!
echo $ADAPTER_PID > "$PID_DIR/adapter.pid"

wait_for_service() {
    # Add health check if your adapter has one
    sleep 2
}

wait_for_service
echo "  ✓ Adapter Daemon started (PID: $ADAPTER_PID)"
```

**2. Update `stop_all.sh`**:
```bash
# Add after metrics daemon section
stop_service "adapter"
```

**3. Update `status.sh`**:
```bash
# Add after metrics daemon section
check_service "adapter" "adapter.pid" "8002"  # Use your port
```

### Configuration Management

Create a config file:

```python
# config/adapter_config.yaml
data_source:
  type: "linborg"  # or "elasticsearch", "mongodb", etc.
  url: "http://linborg.wellsfargo.com/api"
  auth:
    method: "bearer"
    token_file: "/path/to/token"

nordiq:
  inference_url: "http://localhost:8000"
  api_key_file: ".nordiq_key"

adapter:
  poll_interval: 5  # seconds
  batch_size: 100   # records per batch
  retry_attempts: 3
  retry_backoff: 2  # seconds

profiles:
  enable_custom_matching: true
  cmdb_lookup: true
  cache_ttl: 3600  # seconds

logging:
  level: "INFO"
  file: "logs/adapter_daemon.log"
  rotation: "daily"
  retention: 7  # days
```

Load config:
```python
import yaml

with open('config/adapter_config.yaml') as f:
    config = yaml.safe_load(f)

DATA_SOURCE_URL = config['data_source']['url']
POLL_INTERVAL = config['adapter']['poll_interval']
# etc.
```

---

## Troubleshooting

### Issue: "Cannot reach inference daemon"

**Symptoms**: Adapter logs show connection errors.

**Solutions**:
```bash
# Check if inference daemon is running
./status.sh

# Check if port 8000 is open
netstat -tuln | grep 8000

# Check inference daemon logs
tail -f logs/inference_daemon.log

# Test directly
curl http://localhost:8000/health
```

### Issue: "API key invalid"

**Symptoms**: 401 Unauthorized errors.

**Solutions**:
```bash
# Verify key exists
cat .nordiq_key

# Regenerate key
python bin/generate_api_key.py

# Update your adapter code to use new key
```

### Issue: "Records not appearing on dashboard"

**Symptoms**: Adapter sends successfully, but dashboard empty.

**Solutions**:
1. Wait 20-30 minutes for warmup
2. Check server names match expectations
3. Verify data format:
```bash
# Check what inference daemon sees
tail -f logs/inference_daemon.log | grep "records_received"
```

### Issue: "Profile matching not working"

**Symptoms**: All servers show as "Generic" profile.

**Solutions**:
1. Check server name patterns:
```python
# Add debug logging
logger.info(f"Server: {server_name} -> Profile: {match_profile(server_name)}")
```

2. Verify profile values are valid:
```python
VALID_PROFILES = [
    'ml_compute', 'database', 'web_api', 'conductor_mgmt',
    'data_ingest', 'risk_analytics', 'generic'
]
```

### Issue: "Data source returns unexpected format"

**Symptoms**: Transformation errors, missing fields.

**Solutions**:
```python
# Add extensive logging
def transform_record(raw):
    logger.debug(f"Raw record: {raw}")

    try:
        transformed = {
            'timestamp': raw.get('collected_at'),
            # ...
        }
        logger.debug(f"Transformed: {transformed}")
        return transformed
    except Exception as e:
        logger.error(f"Transform failed: {e}", exc_info=True)
        raise
```

### Issue: "Too many records, rate limiting"

**Symptoms**: HTTP 429 errors from inference daemon.

**Solutions**:
```python
# Batch records to stay under 60/minute limit
def send_in_batches(records, batch_size=50):
    """Send records in batches with rate limiting."""
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        send_to_inference_daemon(batch, api_key)

        # Rate limit: max 1 per second
        time.sleep(1.1)
```

### Issue: "Memory leak in adapter"

**Symptoms**: Adapter memory grows over time.

**Solutions**:
```python
# Ensure batches are cleared
batcher.buffer.clear()

# Limit buffer sizes
from collections import deque
buffer = deque(maxlen=1000)  # Auto-prune old records

# Monitor memory
import psutil
import os

process = psutil.Process(os.getpid())
logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

---

## Summary

### What You Need to Build

1. **Data source connection** - Connect to Linborg/monitoring system
2. **Polling/streaming logic** - Get metrics continuously
3. **Transformation layer** - Convert to NordIQ format (9 required fields)
4. **Profile matching** - Map servers to profiles (or skip, let inference do it)
5. **HTTP POST** - Send batches to `/feed/data` endpoint
6. **Error handling** - Retry logic, logging

### What's Already Built (No Work Needed)

1. ✅ Inference daemon API (`/feed/data` endpoint)
2. ✅ Auto profile detection (if you skip matching)
3. ✅ Rolling window management
4. ✅ TFT predictions
5. ✅ Risk scoring
6. ✅ Alert generation
7. ✅ Dashboard serving
8. ✅ Data buffer for retraining

### Key Decision: Profile Matching

**Easiest**: Don't include `profile` field, let inference daemon auto-detect.
**More Control**: Implement custom matching based on CMDB/tags/etc.

### Estimated Development Time

- **Minimal adapter** (polling, no custom profiles): 4-8 hours
- **Production adapter** (streaming, custom profiles, error handling): 2-3 days
- **Enterprise adapter** (CMDB integration, monitoring, HA): 1 week

---

## Next Steps for Claude 3.7

When you (Claude 3.7) are asked to build a data adapter:

1. **Ask the user these questions**:
   - What is your data source? (Linborg, Elasticsearch, etc.)
   - How does data flow? (REST API, streaming, database?)
   - What are the field names in your source data?
   - Do you have CMDB or service tags for profile matching?
   - How many servers? (affects batching strategy)

2. **Use the template**:
   - Copy minimal or streaming template
   - Customize `poll_data_source()` for their source
   - Customize `transform_record()` for their fields
   - Optionally customize `match_profile()`

3. **Test incrementally**:
   - Test data source connection first
   - Test transformation with sample data
   - Test inference daemon integration
   - Then run full loop

4. **Document everything**:
   - Field mappings
   - Profile matching logic
   - Configuration options
   - Deployment steps

---

© 2025 NordIQ AI, LLC. All rights reserved.
