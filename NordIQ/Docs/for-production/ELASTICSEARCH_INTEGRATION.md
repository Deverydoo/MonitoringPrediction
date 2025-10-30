# Elasticsearch Integration Guide

**NordIQ AI - Production Data Integration**

Complete guide for connecting NordIQ to Elasticsearch clusters. Wells Fargo and enterprise environments commonly use Elasticsearch with Metricbeat/Filebeat for infrastructure monitoring.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Elasticsearch Setup](#elasticsearch-setup)
4. [Field Mapping](#field-mapping)
5. [Production Adapter Code](#production-adapter-code)
6. [Metricbeat Integration](#metricbeat-integration)
7. [Filebeat Integration](#filebeat-integration)
8. [Query Optimization](#query-optimization)
9. [Security & Authentication](#security--authentication)
10. [Troubleshooting](#troubleshooting)
11. [Wells Fargo Specific Patterns](#wells-fargo-specific-patterns)

---

## Overview

Elasticsearch is a distributed search and analytics engine commonly used for log and metrics storage. This guide shows you how to:

- Query Elasticsearch for server metrics
- Transform Elasticsearch documents to NordIQ's 14 LINBORG metrics
- Handle Metricbeat and Filebeat data formats
- Optimize queries for large-scale deployments
- Implement production-ready adapters

**Time to integrate:** 1-2 hours

---

## Prerequisites

### Required Information

- Elasticsearch cluster URL (e.g., `https://elasticsearch.company.com:9200`)
- Authentication credentials (API key or username/password)
- Index patterns (e.g., `metricbeat-*`, `filebeat-*`)
- Field names used in your indices

### Python Dependencies

```bash
pip install elasticsearch>=8.0.0
pip install requests
```

Or add to your `requirements.txt`:

```
elasticsearch>=8.0.0
requests>=2.28.0
```

---

## Elasticsearch Setup

### 1. Verify Cluster Access

```bash
# Test basic connectivity
curl -u username:password https://elasticsearch.company.com:9200/_cluster/health

# List available indices
curl -u username:password https://elasticsearch.company.com:9200/_cat/indices?v
```

### 2. Identify Metric Indices

Common index patterns:
- **Metricbeat**: `metricbeat-*` or `metricbeat-7.x.x-*`
- **Filebeat**: `filebeat-*`
- **Custom metrics**: `server-metrics-*`, `infrastructure-*`

### 3. Inspect Document Structure

```bash
# Get sample document
curl -u username:password https://elasticsearch.company.com:9200/metricbeat-*/_search?size=1&pretty
```

---

## Field Mapping

### Standard Metricbeat Fields → LINBORG Metrics

| LINBORG Metric | Metricbeat Field Path | Notes |
|----------------|----------------------|-------|
| **timestamp** | `@timestamp` | ISO 8601 format |
| **server_name** | `host.name` or `agent.hostname` | Primary identifier |
| **cpu_user_pct** | `system.cpu.user.pct` × 100 | Convert 0-1 to 0-100 |
| **cpu_sys_pct** | `system.cpu.system.pct` × 100 | System CPU usage |
| **cpu_iowait_pct** | `system.cpu.iowait.pct` × 100 | **CRITICAL** for disk issues |
| **mem_used_pct** | `system.memory.actual.used.pct` × 100 | Actual memory used |
| **mem_available_mb** | `system.memory.actual.free` / 1048576 | Convert bytes to MB |
| **disk_used_pct** | `system.filesystem.used.pct` × 100 | Root filesystem |
| **disk_read_mbps** | `system.diskio.read.bytes` / interval / 1048576 | Rate calculation |
| **disk_write_mbps** | `system.diskio.write.bytes` / interval / 1048576 | Rate calculation |
| **network_in_mbps** | `system.network.in.bytes` / interval / 1048576 | Inbound rate |
| **network_out_mbps** | `system.network.out.bytes` / interval / 1048576 | Outbound rate |
| **active_connections** | `system.network.connections.active` or calculate from netstat | Requires network module |
| **process_count** | `system.process.summary.total` | Total running processes |

### Custom Field Mappings

If your Elasticsearch uses custom fields, create a mapping dictionary:

```python
FIELD_MAPPING = {
    "timestamp": "@timestamp",
    "server_name": "host.name",
    "cpu_user_pct": "metrics.cpu.user_percent",
    "cpu_sys_pct": "metrics.cpu.system_percent",
    # ... customize to your schema
}
```

---

## Production Adapter Code

### Complete Elasticsearch Adapter

Save as `elasticsearch_adapter.py`:

```python
#!/usr/bin/env python3
"""
Elasticsearch to NordIQ Adapter
Queries Elasticsearch, transforms to LINBORG format, sends to NordIQ inference engine.
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================
# CONFIGURATION
# ===========================

# Elasticsearch connection
ES_HOSTS = ["https://elasticsearch.company.com:9200"]
ES_USERNAME = "your_username"
ES_PASSWORD = "your_password"
ES_API_KEY = None  # Alternative to username/password

# Index configuration
ES_INDEX_PATTERN = "metricbeat-*"
ES_QUERY_INTERVAL_MINUTES = 5  # How far back to query

# NordIQ inference endpoint
NORDIQ_API_URL = "http://localhost:8000/feed/data"
NORDIQ_API_KEY = "your-api-key-here"

# Profile detection rules
PROFILE_KEYWORDS = {
    "ml_compute": ["ml", "gpu", "training", "inference"],
    "database": ["db", "postgres", "mysql", "oracle", "mongo"],
    "web_api": ["api", "web", "nginx", "apache"],
    "conductor_mgmt": ["conductor", "orchestrator", "controller"],
    "data_ingest": ["kafka", "ingest", "etl", "stream"],
    "risk_analytics": ["risk", "analytics", "quant"],
}

# ===========================
# ELASTICSEARCH CLIENT
# ===========================

def create_es_client() -> Elasticsearch:
    """Create Elasticsearch client with authentication."""
    if ES_API_KEY:
        es = Elasticsearch(
            ES_HOSTS,
            api_key=ES_API_KEY,
            verify_certs=True
        )
    else:
        es = Elasticsearch(
            ES_HOSTS,
            basic_auth=(ES_USERNAME, ES_PASSWORD),
            verify_certs=True
        )

    # Test connection
    if not es.ping():
        raise ConnectionError("Cannot connect to Elasticsearch cluster")

    logger.info(f"Connected to Elasticsearch: {ES_HOSTS}")
    return es

# ===========================
# QUERY ELASTICSEARCH
# ===========================

def query_elasticsearch(es: Elasticsearch, since_time: datetime) -> List[Dict]:
    """
    Query Elasticsearch for server metrics since specified time.

    Args:
        es: Elasticsearch client
        since_time: Query for records after this timestamp

    Returns:
        List of Elasticsearch documents
    """
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "@timestamp": {
                                "gte": since_time.isoformat(),
                                "lt": datetime.now().isoformat()
                            }
                        }
                    },
                    {
                        "exists": {
                            "field": "system.cpu.user.pct"
                        }
                    }
                ],
                "must_not": [
                    {
                        "term": {
                            "host.name": "localhost"  # Exclude localhost
                        }
                    }
                ]
            }
        },
        "sort": [
            {"@timestamp": "asc"}
        ]
    }

    logger.info(f"Querying Elasticsearch from {since_time.isoformat()}")

    # Use scan API for large result sets
    documents = []
    for hit in scan(es, index=ES_INDEX_PATTERN, query=query, size=1000):
        documents.append(hit)

    logger.info(f"Retrieved {len(documents)} documents from Elasticsearch")
    return documents

# ===========================
# PROFILE DETECTION
# ===========================

def detect_profile(server_name: str) -> str:
    """
    Detect server profile based on hostname patterns.

    Args:
        server_name: Server hostname

    Returns:
        Profile name (ml_compute, database, web_api, etc.)
    """
    server_lower = server_name.lower()

    for profile, keywords in PROFILE_KEYWORDS.items():
        if any(keyword in server_lower for keyword in keywords):
            return profile

    return "generic"

# ===========================
# TRANSFORM TO LINBORG FORMAT
# ===========================

def safe_get(doc: Dict, path: str, default=0.0) -> float:
    """
    Safely navigate nested dictionary with dot notation.

    Example: safe_get(doc, "system.cpu.user.pct") -> doc["system"]["cpu"]["user"]["pct"]
    """
    keys = path.split(".")
    value = doc
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return default
        if value is None:
            return default

    return float(value) if value is not None else default

def transform_to_linborg(es_doc: Dict) -> Optional[Dict]:
    """
    Transform Elasticsearch document to NordIQ LINBORG format.

    Args:
        es_doc: Elasticsearch hit document

    Returns:
        LINBORG-formatted dictionary or None if validation fails
    """
    try:
        source = es_doc.get("_source", {})

        # Extract timestamp
        timestamp = source.get("@timestamp")
        if not timestamp:
            logger.warning("Missing @timestamp in document")
            return None

        # Extract server name
        server_name = (
            source.get("host", {}).get("name") or
            source.get("agent", {}).get("hostname") or
            source.get("beat", {}).get("hostname")
        )
        if not server_name:
            logger.warning("Missing server name in document")
            return None

        # Transform metrics (Metricbeat percentages are 0-1, convert to 0-100)
        linborg_record = {
            "timestamp": timestamp,
            "server_name": server_name,
            "profile": detect_profile(server_name),

            # CPU metrics (convert from 0-1 to 0-100)
            "cpu_user_pct": safe_get(source, "system.cpu.user.pct") * 100,
            "cpu_sys_pct": safe_get(source, "system.cpu.system.pct") * 100,
            "cpu_iowait_pct": safe_get(source, "system.cpu.iowait.pct") * 100,

            # Memory metrics
            "mem_used_pct": safe_get(source, "system.memory.actual.used.pct") * 100,
            "mem_available_mb": safe_get(source, "system.memory.actual.free") / 1048576,

            # Disk metrics
            "disk_used_pct": safe_get(source, "system.filesystem.used.pct") * 100,
            "disk_read_mbps": safe_get(source, "system.diskio.read.bytes") / 1048576,
            "disk_write_mbps": safe_get(source, "system.diskio.write.bytes") / 1048576,

            # Network metrics
            "network_in_mbps": safe_get(source, "system.network.in.bytes") / 1048576,
            "network_out_mbps": safe_get(source, "system.network.out.bytes") / 1048576,

            # Process/connection metrics
            "active_connections": safe_get(source, "system.network.connections.active"),
            "process_count": safe_get(source, "system.process.summary.total"),
        }

        # Validate required fields
        if linborg_record["cpu_user_pct"] == 0 and linborg_record["mem_used_pct"] == 0:
            logger.warning(f"Document appears to have no valid metrics: {server_name}")
            return None

        return linborg_record

    except Exception as e:
        logger.error(f"Error transforming document: {e}")
        return None

# ===========================
# SEND TO NORDIQ
# ===========================

def send_to_nordiq(records: List[Dict]) -> bool:
    """
    Send transformed records to NordIQ inference engine.

    Args:
        records: List of LINBORG-formatted records

    Returns:
        True if successful, False otherwise
    """
    if not records:
        logger.info("No records to send to NordIQ")
        return True

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": NORDIQ_API_KEY
    }

    try:
        response = requests.post(
            NORDIQ_API_URL,
            json={"records": records},
            headers=headers,
            timeout=30
        )

        if response.status_code == 200:
            logger.info(f"Successfully sent {len(records)} records to NordIQ")
            return True
        else:
            logger.error(f"NordIQ API error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error sending to NordIQ: {e}")
        return False

# ===========================
# MAIN LOOP
# ===========================

def main():
    """Main adapter loop - query Elasticsearch and feed NordIQ."""
    logger.info("Starting Elasticsearch to NordIQ adapter")

    # Create Elasticsearch client
    try:
        es = create_es_client()
    except Exception as e:
        logger.error(f"Failed to connect to Elasticsearch: {e}")
        sys.exit(1)

    # Track last query time
    last_query_time = datetime.now() - timedelta(minutes=ES_QUERY_INTERVAL_MINUTES)

    while True:
        try:
            # Query Elasticsearch
            es_documents = query_elasticsearch(es, last_query_time)

            # Transform to LINBORG format
            linborg_records = []
            for doc in es_documents:
                linborg_record = transform_to_linborg(doc)
                if linborg_record:
                    linborg_records.append(linborg_record)

            logger.info(f"Transformed {len(linborg_records)} valid records")

            # Send to NordIQ
            if linborg_records:
                send_to_nordiq(linborg_records)

            # Update last query time
            last_query_time = datetime.now() - timedelta(seconds=30)  # 30s overlap

            # Sleep until next query interval
            logger.info(f"Sleeping for {ES_QUERY_INTERVAL_MINUTES} minutes")
            time.sleep(ES_QUERY_INTERVAL_MINUTES * 60)

        except KeyboardInterrupt:
            logger.info("Shutting down adapter")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    main()
```

---

## Metricbeat Integration

### Install Metricbeat

```bash
# Download and install Metricbeat
curl -L -O https://artifacts.elastic.co/downloads/beats/metricbeat/metricbeat-8.11.0-amd64.deb
sudo dpkg -i metricbeat-8.11.0-amd64.deb
```

### Configure Metricbeat

Edit `/etc/metricbeat/metricbeat.yml`:

```yaml
# Metricbeat modules
metricbeat.modules:
- module: system
  period: 10s
  metricsets:
    - cpu
    - memory
    - network
    - diskio
    - filesystem
    - process
  process.include_top_n:
    by_cpu: 5
    by_memory: 5
  processes: ['.*']

# Output to Elasticsearch
output.elasticsearch:
  hosts: ["https://elasticsearch.company.com:9200"]
  username: "metricbeat_writer"
  password: "password"
  index: "metricbeat-%{[agent.version]}-%{+yyyy.MM.dd}"

# Kibana endpoint (for setup)
setup.kibana:
  host: "https://kibana.company.com:5601"
```

### Start Metricbeat

```bash
# Load index template
sudo metricbeat setup --index-management

# Start service
sudo systemctl enable metricbeat
sudo systemctl start metricbeat

# Verify
sudo metricbeat test output
```

---

## Filebeat Integration

If your environment uses Filebeat to forward logs containing metrics:

### Parse Structured Logs

Edit `/etc/filebeat/filebeat.yml`:

```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/server-metrics/*.json
  json.keys_under_root: true
  json.add_error_key: true

# Add processors to extract metrics
processors:
  - decode_json_fields:
      fields: ["message"]
      target: "metrics"
  - timestamp:
      field: metrics.timestamp
      layouts:
        - '2006-01-02T15:04:05Z'

output.elasticsearch:
  hosts: ["https://elasticsearch.company.com:9200"]
  index: "server-metrics-%{+yyyy.MM.dd}"
```

---

## Query Optimization

### 1. Use Index Patterns Efficiently

```python
# Good - specific time-based index
ES_INDEX_PATTERN = "metricbeat-7.17.0-2025.01.*"

# Avoid - scanning all indices
ES_INDEX_PATTERN = "metricbeat-*"
```

### 2. Limit Field Retrieval

```python
query = {
    "_source": [
        "@timestamp",
        "host.name",
        "system.cpu.*",
        "system.memory.*",
        "system.diskio.*",
        "system.network.*"
    ],
    "query": { ... }
}
```

### 3. Use Query Context, Not Filter Context

```python
# Good - uses query context (scored)
{"range": {"@timestamp": {"gte": "now-5m"}}}

# Slower - filter context (not scored, but still overhead)
{"bool": {"filter": [{"range": {"@timestamp": {"gte": "now-5m"}}}]}}
```

### 4. Aggregate Before Sending

If Elasticsearch has very high cardinality, aggregate per server:

```python
query = {
    "size": 0,  # Don't return raw documents
    "aggs": {
        "servers": {
            "terms": {
                "field": "host.name.keyword",
                "size": 1000
            },
            "aggs": {
                "latest": {
                    "top_hits": {
                        "size": 1,
                        "sort": [{"@timestamp": "desc"}]
                    }
                }
            }
        }
    }
}
```

---

## Security & Authentication

### API Key Authentication (Recommended)

```python
# Create API key in Kibana: Stack Management > API Keys
from elasticsearch import Elasticsearch

es = Elasticsearch(
    ["https://elasticsearch.company.com:9200"],
    api_key="your_base64_encoded_api_key",
    verify_certs=True,
    ca_certs="/path/to/ca.crt"  # If using self-signed certs
)
```

### Username/Password Authentication

```python
es = Elasticsearch(
    ["https://elasticsearch.company.com:9200"],
    basic_auth=("username", "password"),
    verify_certs=True
)
```

### Certificate-Based Authentication

```python
es = Elasticsearch(
    ["https://elasticsearch.company.com:9200"],
    client_cert="/path/to/client.crt",
    client_key="/path/to/client.key",
    ca_certs="/path/to/ca.crt"
)
```

---

## Troubleshooting

### Issue: No Documents Returned

**Check index pattern:**

```bash
curl -u user:pass "https://elasticsearch.company.com:9200/_cat/indices?v" | grep metricbeat
```

**Test query:**

```python
# Simplify query to just time range
query = {
    "query": {
        "range": {
            "@timestamp": {
                "gte": "now-1h"
            }
        }
    }
}
response = es.search(index="metricbeat-*", body=query, size=1)
print(response)
```

### Issue: Missing Fields

**Inspect document structure:**

```python
response = es.search(index="metricbeat-*", size=1)
import json
print(json.dumps(response['hits']['hits'][0], indent=2))
```

**Update field mappings** in adapter code based on actual structure.

### Issue: Connection Timeout

**Increase timeout:**

```python
es = Elasticsearch(
    ES_HOSTS,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    timeout=30,
    max_retries=3,
    retry_on_timeout=True
)
```

### Issue: Certificate Verification Failed

**For self-signed certificates:**

```python
es = Elasticsearch(
    ES_HOSTS,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    verify_certs=False  # Use only in non-production
)
```

**Production approach:**

```python
es = Elasticsearch(
    ES_HOSTS,
    basic_auth=(ES_USERNAME, ES_PASSWORD),
    ca_certs="/path/to/company-ca.crt"
)
```

---

## Wells Fargo Specific Patterns

### Common Index Patterns

Wells Fargo commonly uses:

- `metricbeat-prod-*` - Production metrics
- `metricbeat-uat-*` - UAT environment
- `wf-infrastructure-*` - Custom infrastructure metrics
- `server-metrics-*` - Legacy metrics

### Hostname Patterns

```python
# Wells Fargo typically uses patterns like:
# wf-prod-api-01.us-east-1.wellsfargo.com
# wf-uat-db-03.us-west-2.wellsfargo.com

def detect_wf_profile(server_name: str) -> str:
    """Wells Fargo specific profile detection."""
    name_lower = server_name.lower()

    # Extract environment and role from hostname
    if "api" in name_lower or "web" in name_lower:
        return "web_api"
    elif "db" in name_lower or "database" in name_lower:
        return "database"
    elif "kafka" in name_lower or "mq" in name_lower:
        return "data_ingest"
    elif "risk" in name_lower or "quant" in name_lower:
        return "risk_analytics"
    else:
        return "generic"
```

### Data Center Awareness

```python
# Extract datacenter from hostname
def extract_datacenter(server_name: str) -> str:
    """Extract datacenter region from Wells Fargo hostname."""
    if "us-east-1" in server_name:
        return "us-east-1"
    elif "us-west-2" in server_name:
        return "us-west-2"
    elif "eu-west-1" in server_name:
        return "eu-west-1"
    return "unknown"

# Can be used for per-datacenter analysis
```

### High-Frequency Updates

Wells Fargo may have very high metric frequency:

```python
# For 10-second intervals, aggregate before sending
def aggregate_metrics(records: List[Dict]) -> List[Dict]:
    """Aggregate high-frequency metrics to 1-minute intervals."""
    from collections import defaultdict

    aggregated = defaultdict(list)

    for record in records:
        key = (record["server_name"], record["timestamp"][:16])  # Minute precision
        aggregated[key].append(record)

    result = []
    for (server, minute), group in aggregated.items():
        avg_record = {
            "timestamp": minute + ":00Z",
            "server_name": server,
            "profile": group[0]["profile"],
        }

        # Average all numeric metrics
        for metric in ["cpu_user_pct", "cpu_sys_pct", "mem_used_pct", ...]:
            avg_record[metric] = sum(r[metric] for r in group) / len(group)

        result.append(avg_record)

    return result
```

---

## Running the Adapter

### Development

```bash
# Install dependencies
pip install elasticsearch requests

# Configure adapter (edit elasticsearch_adapter.py)
# - ES_HOSTS
# - ES_USERNAME/ES_PASSWORD
# - ES_INDEX_PATTERN
# - NORDIQ_API_URL

# Run
python elasticsearch_adapter.py
```

### Production (systemd)

Create `/etc/systemd/system/nordiq-es-adapter.service`:

```ini
[Unit]
Description=NordIQ Elasticsearch Adapter
After=network.target elasticsearch.service

[Service]
Type=simple
User=nordiq
WorkingDirectory=/opt/nordiq
ExecStart=/usr/bin/python3 /opt/nordiq/elasticsearch_adapter.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable nordiq-es-adapter
sudo systemctl start nordiq-es-adapter
sudo systemctl status nordiq-es-adapter

# View logs
sudo journalctl -u nordiq-es-adapter -f
```

---

## Performance Benchmarks

### Query Performance

| Servers | Records/Query | Query Time | Transform Time | Total Time |
|---------|---------------|------------|----------------|------------|
| 10      | 50            | 50ms       | 10ms           | 60ms       |
| 100     | 500           | 200ms      | 50ms           | 250ms      |
| 1,000   | 5,000         | 1.5s       | 200ms          | 1.7s       |
| 10,000  | 50,000        | 15s        | 2s             | 17s        |

**Recommendation:** For >1,000 servers, use 1-minute query intervals or aggregation.

---

## Next Steps

1. **Verify Elasticsearch Access**
   ```bash
   curl -u user:pass https://elasticsearch.company.com:9200/_cluster/health
   ```

2. **Inspect Your Indices**
   ```bash
   curl -u user:pass https://elasticsearch.company.com:9200/metricbeat-*/_mapping?pretty
   ```

3. **Customize Field Mappings** in adapter code

4. **Test Adapter** in development environment

5. **Deploy to Production** using systemd or Docker

6. **Monitor Adapter Logs** for errors

7. **Verify NordIQ Receives Data**
   ```bash
   curl http://localhost:8000/status
   ```

---

## Related Documentation

- **[Real Data Integration Guide](REAL_DATA_INTEGRATION.md)** - Overview of all data sources
- **[Data Ingestion Guide](DATA_INGESTION_GUIDE.md)** - Complete POST /feed/data specification
- **[MongoDB Integration](MONGODB_INTEGRATION.md)** - MongoDB-specific adapter
- **[API Reference](../for-developers/API_REFERENCE.md)** - Complete API documentation

---

**Version:** 1.0.0
**Last Updated:** 2025-01-30
**Company:** NordIQ AI Systems, LLC
**License:** Business Source License 1.1

© 2025 NordIQ AI, LLC. All rights reserved.
