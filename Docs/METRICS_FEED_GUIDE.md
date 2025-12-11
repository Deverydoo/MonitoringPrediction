# Tachyon Argus Inference Engine - Metrics Feed Guide

## Overview

The TFT Inference Daemon accepts server metrics via a REST API endpoint. You can feed data from any monitoring system (Prometheus, Zabbix, Datadog, custom collectors, etc.) by formatting it to match the expected schema.

**Endpoint:** `POST /feed/data`
**Base URL:** `http://localhost:8000` (configurable via `--port`)
**Rate Limit:** 60 requests/minute

---

## Authentication

Include your API key in the header:
```
X-API-Key: your-api-key
```

See [DASHBOARD_INTEGRATION_GUIDE.md](DASHBOARD_INTEGRATION_GUIDE.md) for API key setup.

---

## Data Schema (v2.0.0)

### Required Fields

Each metric record must contain these 16 fields:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `timestamp` | ISO 8601 string | - | e.g., "2025-01-15T10:00:00" |
| `server_name` | string | - | Unique server identifier |
| `status` | string | see below | Current server state |
| **CPU Metrics** |
| `cpu_user_pct` | float | 0-100 | User CPU percentage |
| `cpu_sys_pct` | float | 0-100 | System CPU percentage |
| `cpu_iowait_pct` | float | 0-100 | I/O wait percentage |
| `cpu_idle_pct` | float | 0-100 | Idle CPU percentage |
| `java_cpu_pct` | float | 0-100 | Java process CPU (0 if N/A) |
| **Memory Metrics** |
| `mem_used_pct` | float | 0-100 | Memory usage percentage |
| `swap_used_pct` | float | 0-100 | Swap usage percentage |
| **Disk Metrics** |
| `disk_usage_pct` | float | 0-100 | Disk usage percentage |
| **Network Metrics** |
| `net_in_mb_s` | float | 0+ | Inbound traffic (MB/s) |
| `net_out_mb_s` | float | 0+ | Outbound traffic (MB/s) |
| **Connection Metrics** |
| `back_close_wait` | int | 0+ | Backend CLOSE_WAIT connections |
| `front_close_wait` | int | 0+ | Frontend CLOSE_WAIT connections |
| **System Metrics** |
| `load_average` | float | 0+ | System load average |
| `uptime_days` | int | 0-365 | Days since last reboot |

### Valid Status Values

```
critical_issue   - Server experiencing critical problems
healthy          - Normal operation
heavy_load       - High resource utilization
idle             - Low activity
maintenance      - Scheduled maintenance
morning_spike    - Expected morning load increase
offline          - Server unreachable
recovery         - Recovering from issue
```

---

## Request Format

```json
POST /feed/data
Content-Type: application/json
X-API-Key: your-api-key

{
  "records": [
    {
      "timestamp": "2025-01-15T10:00:00",
      "server_name": "ppdb001",
      "status": "healthy",
      "cpu_user_pct": 25.3,
      "cpu_sys_pct": 8.2,
      "cpu_iowait_pct": 2.1,
      "cpu_idle_pct": 64.4,
      "java_cpu_pct": 15.5,
      "mem_used_pct": 72.1,
      "swap_used_pct": 5.2,
      "disk_usage_pct": 45.8,
      "net_in_mb_s": 12.5,
      "net_out_mb_s": 8.3,
      "back_close_wait": 5,
      "front_close_wait": 3,
      "load_average": 2.4,
      "uptime_days": 45
    },
    {
      "timestamp": "2025-01-15T10:00:00",
      "server_name": "ppdb002",
      "status": "healthy",
      ...
    }
  ]
}
```

**Key Points:**
- Send all servers in a single batch (same timestamp)
- Recommended interval: every 5 minutes (matches model's training resolution)
- The daemon maintains a rolling window of the last 2880 records (~24 hours for 45 servers)

---

## Response Format

**Success:**
```json
{
  "status": "accepted",
  "records_received": 45,
  "tick": 1234,
  "rolling_window_size": 2880,
  "warmup_complete": true
}
```

**Warmup Period:**
```json
{
  "status": "warming_up",
  "records_received": 45,
  "tick": 50,
  "rolling_window_size": 2250,
  "warmup_complete": false,
  "progress": "45% (need 100 records minimum)"
}
```

**Empty Batch:**
```json
{
  "status": "ignored",
  "reason": "empty batch"
}
```

---

## Implementation Examples

### Python with Requests

```python
import requests
from datetime import datetime

API_URL = "http://localhost:8000/feed/data"
API_KEY = "your-api-key"

def collect_server_metrics(server_name):
    """Collect metrics from a server (implement your logic)."""
    # Example: query Prometheus, read from /proc, call monitoring API
    return {
        "timestamp": datetime.now().isoformat(),
        "server_name": server_name,
        "status": "healthy",
        "cpu_user_pct": 25.3,
        "cpu_sys_pct": 8.2,
        "cpu_iowait_pct": 2.1,
        "cpu_idle_pct": 64.4,
        "java_cpu_pct": 15.5,
        "mem_used_pct": 72.1,
        "swap_used_pct": 5.2,
        "disk_usage_pct": 45.8,
        "net_in_mb_s": 12.5,
        "net_out_mb_s": 8.3,
        "back_close_wait": 5,
        "front_close_wait": 3,
        "load_average": 2.4,
        "uptime_days": 45
    }

def send_metrics(servers):
    """Collect and send metrics for all servers."""
    records = [collect_server_metrics(s) for s in servers]

    response = requests.post(
        API_URL,
        json={"records": records},
        headers={"X-API-Key": API_KEY}
    )

    return response.json()

# Run every 5 minutes
if __name__ == "__main__":
    servers = ["ppdb001", "ppdb002", "ppml001", "ppapi001"]
    result = send_metrics(servers)
    print(f"Fed {result.get('records_received', 0)} records")
```

### Bash/cURL

```bash
#!/bin/bash
API_URL="http://localhost:8000/feed/data"
API_KEY="your-api-key"
TIMESTAMP=$(date -Iseconds)

# Build JSON payload
read -r -d '' PAYLOAD << EOF
{
  "records": [
    {
      "timestamp": "$TIMESTAMP",
      "server_name": "ppdb001",
      "status": "healthy",
      "cpu_user_pct": 25.3,
      "cpu_sys_pct": 8.2,
      "cpu_iowait_pct": 2.1,
      "cpu_idle_pct": 64.4,
      "java_cpu_pct": 15.5,
      "mem_used_pct": 72.1,
      "swap_used_pct": 5.2,
      "disk_usage_pct": 45.8,
      "net_in_mb_s": 12.5,
      "net_out_mb_s": 8.3,
      "back_close_wait": 5,
      "front_close_wait": 3,
      "load_average": 2.4,
      "uptime_days": 45
    }
  ]
}
EOF

curl -X POST "$API_URL" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "$PAYLOAD"
```

### From Prometheus

```python
import requests
from datetime import datetime
from prometheus_api_client import PrometheusConnect

PROM_URL = "http://prometheus:9090"
TACHYON_URL = "http://localhost:8000/feed/data"
API_KEY = "your-api-key"

prom = PrometheusConnect(url=PROM_URL)

def query_metric(query):
    """Query Prometheus and return value."""
    result = prom.custom_query(query)
    if result:
        return float(result[0]['value'][1])
    return 0.0

def collect_from_prometheus(server):
    """Collect metrics from Prometheus for a server."""
    return {
        "timestamp": datetime.now().isoformat(),
        "server_name": server,
        "status": "healthy",
        "cpu_user_pct": query_metric(f'cpu_user_percent{{instance="{server}"}}'),
        "cpu_sys_pct": query_metric(f'cpu_system_percent{{instance="{server}"}}'),
        "cpu_iowait_pct": query_metric(f'cpu_iowait_percent{{instance="{server}"}}'),
        "cpu_idle_pct": query_metric(f'cpu_idle_percent{{instance="{server}"}}'),
        "java_cpu_pct": query_metric(f'process_cpu_percent{{instance="{server}",job="java"}}'),
        "mem_used_pct": query_metric(f'memory_used_percent{{instance="{server}"}}'),
        "swap_used_pct": query_metric(f'swap_used_percent{{instance="{server}"}}'),
        "disk_usage_pct": query_metric(f'disk_used_percent{{instance="{server}",mountpoint="/"}}'),
        "net_in_mb_s": query_metric(f'rate(network_receive_bytes{{instance="{server}"}}[1m])/1048576'),
        "net_out_mb_s": query_metric(f'rate(network_transmit_bytes{{instance="{server}"}}[1m])/1048576'),
        "back_close_wait": int(query_metric(f'tcp_close_wait{{instance="{server}",direction="backend"}}')),
        "front_close_wait": int(query_metric(f'tcp_close_wait{{instance="{server}",direction="frontend"}}')),
        "load_average": query_metric(f'node_load1{{instance="{server}"}}'),
        "uptime_days": int(query_metric(f'node_time_seconds{{instance="{server}"}}-node_boot_time_seconds{{instance="{server}"}}')/86400)
    }

def feed_tachyon(servers):
    records = [collect_from_prometheus(s) for s in servers]
    response = requests.post(
        TACHYON_URL,
        json={"records": records},
        headers={"X-API-Key": API_KEY}
    )
    return response.json()
```

### From Zabbix API

```python
import requests
from datetime import datetime
from pyzabbix import ZabbixAPI

ZABBIX_URL = "http://zabbix/api_jsonrpc.php"
ZABBIX_USER = "api_user"
ZABBIX_PASS = "api_pass"
TACHYON_URL = "http://localhost:8000/feed/data"
API_KEY = "your-api-key"

zapi = ZabbixAPI(ZABBIX_URL)
zapi.login(ZABBIX_USER, ZABBIX_PASS)

def get_item_value(host_id, item_key):
    """Get latest value for a Zabbix item."""
    items = zapi.item.get(hostids=host_id, search={"key_": item_key}, output=["lastvalue"])
    if items:
        return float(items[0]['lastvalue'])
    return 0.0

def collect_from_zabbix(host_name, host_id):
    """Collect metrics from Zabbix for a host."""
    return {
        "timestamp": datetime.now().isoformat(),
        "server_name": host_name,
        "status": "healthy",
        "cpu_user_pct": get_item_value(host_id, "system.cpu.util[,user]"),
        "cpu_sys_pct": get_item_value(host_id, "system.cpu.util[,system]"),
        "cpu_iowait_pct": get_item_value(host_id, "system.cpu.util[,iowait]"),
        "cpu_idle_pct": get_item_value(host_id, "system.cpu.util[,idle]"),
        "java_cpu_pct": get_item_value(host_id, "proc.cpu.util[java]"),
        "mem_used_pct": get_item_value(host_id, "vm.memory.utilization"),
        "swap_used_pct": get_item_value(host_id, "system.swap.pused"),
        "disk_usage_pct": get_item_value(host_id, "vfs.fs.pused[/]"),
        "net_in_mb_s": get_item_value(host_id, "net.if.in[eth0]") / 1048576,
        "net_out_mb_s": get_item_value(host_id, "net.if.out[eth0]") / 1048576,
        "back_close_wait": int(get_item_value(host_id, "net.tcp.count[CLOSE_WAIT]")),
        "front_close_wait": 0,  # Combine or split as needed
        "load_average": get_item_value(host_id, "system.cpu.load[all,avg1]"),
        "uptime_days": int(get_item_value(host_id, "system.uptime") / 86400)
    }
```

---

## Feeding Strategy

### Recommended Interval

**5 minutes** - Matches the model's training resolution. The TFT model was trained on 5-minute intervals, so feeding at the same rate gives best prediction accuracy.

### Batch Size

Send all servers in a single request per tick. For example, if you have 45 servers:
- 1 request every 5 minutes
- 45 records per request
- Rolling window holds ~64 ticks of data (5+ hours)

### Warmup Period

The daemon needs at least **100 records** before it can make predictions. For 45 servers at 5-minute intervals:
- Minimum warmup: ~3 ticks (15 minutes)
- Recommended warmup: 12 ticks (1 hour) for stable predictions

Check warmup status via `/status` endpoint:
```json
{
  "warmup_complete": true,
  "rolling_window_size": 2880,
  "servers_tracked": 45
}
```

---

## Server Naming Conventions

The model recognizes server profile prefixes for display purposes:

| Prefix | Profile | Description |
|--------|---------|-------------|
| `ppdb` | Database | PostgreSQL, MySQL, etc. |
| `ppml` | ML Compute | Machine learning workloads |
| `ppapi` | Web API | REST/GraphQL servers |
| `ppcond` | Conductor Mgmt | Orchestration services |
| `ppetl` | ETL/Ingest | Data pipeline workers |
| `pprisk` | Risk Analytics | Financial/risk compute |
| Other | Generic | Unspecified workload |

You can use any naming convention - these prefixes just affect the UI labels.

---

## Handling Missing Metrics

If a metric isn't available from your monitoring system:

1. **Use sensible defaults:**
   - `java_cpu_pct`: 0.0 if no Java process
   - `swap_used_pct`: 0.0 if no swap configured
   - `back_close_wait` / `front_close_wait`: 0 if not tracked

2. **Don't omit required fields** - the validator will reject incomplete records

3. **Estimate from related metrics:**
   - `cpu_idle_pct` = 100 - user - sys - iowait
   - `load_average`: derive from CPU metrics if not available

---

## Error Handling

### Validation Errors

If data doesn't match the schema:
```json
{
  "status": "error",
  "message": "Validation failed",
  "errors": [
    "Missing required columns: ['cpu_iowait_pct']",
    "Invalid status value 'unknown' - must be one of: healthy, critical_issue, ..."
  ]
}
```

### Connection Errors

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount('http://', HTTPAdapter(max_retries=retries))

try:
    response = session.post(API_URL, json={"records": records}, headers=headers, timeout=10)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"Failed to feed metrics: {e}")
    # Buffer locally and retry later
```

---

## Monitoring the Feed

Check that data is flowing:

```bash
# Check daemon status
curl http://localhost:8000/status

# Response shows:
# - rolling_window_size: how many records buffered
# - tick_count: how many batches received
# - last_prediction: when predictions last ran
# - servers_tracked: unique servers seen
```

---

## Testing Your Integration

1. **Start the daemon:**
   ```bash
   cd Argus
   python src/daemons/tft_inference_daemon.py
   ```

2. **Send test data:**
   ```bash
   curl -X POST http://localhost:8000/feed/data \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-key" \
     -d '{"records": [{"timestamp": "2025-01-15T10:00:00", "server_name": "test001", "status": "healthy", "cpu_user_pct": 25, "cpu_sys_pct": 5, "cpu_iowait_pct": 2, "cpu_idle_pct": 68, "java_cpu_pct": 10, "mem_used_pct": 60, "swap_used_pct": 0, "disk_usage_pct": 40, "net_in_mb_s": 5, "net_out_mb_s": 3, "back_close_wait": 0, "front_close_wait": 0, "load_average": 1.5, "uptime_days": 30}]}'
   ```

3. **Verify status:**
   ```bash
   curl http://localhost:8000/status
   ```

4. **After warmup, check predictions:**
   ```bash
   curl -H "X-API-Key: your-key" http://localhost:8000/predictions/current
   ```
