# Data Format Specification
**Complete Schema Reference for Input and Output Data**

**Version:** 3.0
**Audience:** Software Developers, Data Engineers
**Purpose:** Technical reference for all data structures, schemas, and formats

---

## Table of Contents

1. [Overview](#overview)
2. [Input Format (Metrics)](#input-format-metrics)
3. [Output Format (Predictions)](#output-format-predictions)
4. [Enumerations](#enumerations)
5. [Data Types](#data-types)
6. [JSON Schemas](#json-schemas)
7. [Examples](#examples)

---

## Overview

This document provides the complete technical specification for all data formats used by the NordIQ system.

**Key Specifications:**
- Input: Server metrics (POST `/feed/data`)
- Output: Predictions and risk scores (GET `/predictions/*`)
- Format: JSON
- Encoding: UTF-8
- Timestamps: ISO 8601 with timezone

---

## Input Format (Metrics)

### Metrics Record Schema

**Complete specification for a single metrics record**

```typescript
interface MetricsRecord {
  // Required: Core Identification
  timestamp: string;        // ISO 8601 format with timezone
  server_name: string;      // 3-64 chars, alphanumeric + [-_.]
  profile: ServerProfile;   // Enum: see ServerProfile below

  // Required: CPU Metrics (percentages 0.0-100.0)
  cpu_user_pct: number;     // User space CPU %
  cpu_sys_pct: number;      // Kernel space CPU %
  cpu_iowait_pct: number;   // I/O wait % (CRITICAL metric)
  cpu_idle_pct: number;     // Idle CPU %
  java_cpu_pct: number;     // Java process CPU % (0 if no Java)

  // Required: Memory Metrics (percentages 0.0-100.0)
  mem_used_pct: number;     // RAM usage %
  swap_used_pct: number;    // Swap usage %

  // Required: Disk Metrics
  disk_usage_pct: number;   // Filesystem usage % (0.0-100.0)

  // Required: Network Metrics (MB/s)
  net_in_mb_s: number;      // Inbound throughput (0.0-10000.0)
  net_out_mb_s: number;     // Outbound throughput (0.0-10000.0)

  // Required: Connection Metrics (counts)
  back_close_wait: number;  // Backend TCP CLOSE_WAIT (0-100000)
  front_close_wait: number; // Frontend TCP CLOSE_WAIT (0-100000)

  // Required: System Metrics
  load_average: number;     // 1-minute load average (0.0-1000.0)
  uptime_days: number;      // Days since boot (0-36500)

  // Optional: Operational State
  state?: ServerState;      // Enum: see ServerState below
  problem_child?: boolean;  // Flag for problematic servers
  notes?: string;           // Freeform notes (max 500 chars)
}
```

### Batch Request Schema

**Wrapper for sending multiple records**

```typescript
interface FeedDataRequest {
  records: MetricsRecord[];  // 1-1000 records per request
}
```

### Validation Rules

```typescript
interface ValidationRules {
  timestamp: {
    format: "ISO 8601",
    timezone_required: true,
    examples: [
      "2025-10-30T15:30:00Z",
      "2025-10-30T15:30:00+00:00",
      "2025-10-30T10:30:00-05:00"
    ]
  };

  server_name: {
    min_length: 3,
    max_length: 64,
    allowed_chars: "a-z A-Z 0-9 - _ .",
    case_sensitive: true
  };

  cpu_metrics: {
    range: [0.0, 100.0],
    sum_constraint: "user + sys + iowait + idle ≈ 100 (±5% tolerance)"
  };

  memory_metrics: {
    range: [0.0, 100.0],
    warnings: {
      mem_used_pct: "> 95",
      swap_used_pct: "> 10"
    }
  };

  network_metrics: {
    range: [0.0, 10000.0],
    unit: "MB/s",
    warnings: {
      net_in_mb_s: "> 1000 (check units)",
      net_out_mb_s: "> 1000 (check units)"
    }
  };

  connection_metrics: {
    range: [0, 100000],
    warnings: {
      back_close_wait: "> 1000 (connection leak)",
      front_close_wait: "> 1000 (connection leak)"
    }
  };

  system_metrics: {
    load_average: {
      range: [0.0, 1000.0],
      interpretation: "Higher = more CPU queuing"
    },
    uptime_days: {
      range: [0, 36500],
      interpretation: "Time since last reboot"
    }
  };
}
```

---

## Output Format (Predictions)

### Prediction Response Schema

**Complete specification for prediction output**

```typescript
interface PredictionResponse {
  timestamp: string;                    // When predictions were generated
  summary: PredictionSummary;           // Fleet-wide statistics
  predictions: Record<string, ServerPrediction>;  // Per-server predictions
}

interface PredictionSummary {
  total_servers: number;                // Total servers tracked
  critical: number;                     // Servers in critical state
  warning: number;                      // Servers in warning state
  degrading: number;                    // Servers degrading
  healthy: number;                      // Servers healthy
  avg_risk_score: number;               // Average risk across fleet (0-100)
}

interface ServerPrediction {
  server_name: string;                  // Server identifier
  profile: ServerProfile;               // Server profile
  risk_score: number;                   // Current risk score (0-100)
  risk_level: RiskLevel;                // Risk category
  current_metrics: CurrentMetrics;      // Latest observed metrics
  predicted_failures: PredictedFailure[]; // Predicted issues
  predictions: FuturePrediction[];      // Time series predictions (96 steps)
}

interface CurrentMetrics {
  cpu_user_pct: number;
  cpu_sys_pct: number;
  cpu_iowait_pct: number;
  mem_used_pct: number;
  disk_usage_pct: number;
  load_average: number;
  // ... all 14 LINBORG metrics
}

interface PredictedFailure {
  metric: string;                       // Which metric will fail
  current_value: number;                // Current observed value
  predicted_value: number;              // Predicted peak value
  threshold: number;                    // Failure threshold
  time_to_failure_minutes: number;      // Minutes until failure
  confidence: number;                   // Confidence (0.0-1.0)
  severity: "critical" | "warning" | "info";
  recommendation?: string;              // Suggested action
}

interface FuturePrediction {
  timestamp: string;                    // Future timestamp (ISO 8601)
  step: number;                         // Prediction step (1-96)
  cpu_user_pct: number;                 // Predicted CPU user %
  cpu_sys_pct: number;                  // Predicted CPU sys %
  mem_used_pct: number;                 // Predicted memory %
  disk_usage_pct: number;               // Predicted disk %
  load_average: number;                 // Predicted load
  risk_score: number;                   // Risk score at this step
  confidence_interval?: {
    lower: number;                      // p10 quantile
    upper: number;                      // p90 quantile
  };
}
```

### Alert Response Schema

**Schema for active alerts endpoint**

```typescript
interface AlertResponse {
  timestamp: string;                    // When alerts were generated
  total_alerts: number;                 // Total active alerts
  alerts: Alert[];                      // List of alerts
}

interface Alert {
  server_name: string;                  // Server identifier
  profile: ServerProfile;               // Server profile
  risk_score: number;                   // Current risk (0-100)
  level: RiskLevel;                     // Alert severity
  message: string;                      // Human-readable alert message
  predicted_failure: PredictedFailure;  // Details of predicted issue
  timestamp: string;                    // When alert was triggered
  alert_id?: string;                    // Unique alert identifier
}
```

### XAI Response Schema

**Schema for explainable AI endpoint**

```typescript
interface XAIResponse {
  server_name: string;                  // Server identifier
  timestamp: string;                    // Analysis timestamp
  prediction_step: number;              // Which step explained (0-95)
  predicted_risk_score: number;         // Risk at that step
  explanation: XAIExplanation;          // Detailed explanation
  recommendation: string;               // Suggested action
}

interface XAIExplanation {
  top_contributing_factors: ContributingFactor[];
  profile_baseline: ProfileBaseline;
  attention_weights: AttentionWeights;
}

interface ContributingFactor {
  feature: string;                      // Metric name
  importance: number;                   // Feature importance (0.0-1.0)
  current_value: number;                // Current metric value
  direction: "increasing" | "decreasing" | "stable";
  contribution_to_risk: string;         // Human explanation
}

interface ProfileBaseline {
  profile: ServerProfile;               // Server profile
  normal_value: Record<string, number>; // Typical values for profile
  current_deviation: string;            // How far from normal
}

interface AttentionWeights {
  recent_history: number;               // Weight on recent data (0.0-1.0)
  temporal_patterns: number;            // Weight on time patterns
  profile_knowledge: number;            // Weight on profile knowledge
}
```

---

## Enumerations

### ServerProfile

**Valid server profile values**

```typescript
enum ServerProfile {
  ML_COMPUTE = "ml_compute",           // ML training nodes
  DATABASE = "database",               // Database servers
  WEB_API = "web_api",                 // Web/API servers
  CONDUCTOR_MGMT = "conductor_mgmt",   // Orchestration servers
  DATA_INGEST = "data_ingest",         // ETL/streaming servers
  RISK_ANALYTICS = "risk_analytics",   // Risk calculation servers
  GENERIC = "generic"                  // Generic/utility servers
}
```

**Profile Characteristics:**

| Profile | Typical CPU | Typical Memory | Typical I/O | Use Case |
|---------|-------------|----------------|-------------|----------|
| ml_compute | 60-90% | 70-95% | Moderate | Training, inference |
| database | 30-70% | 60-85% | Very High | CRUD operations |
| web_api | 20-50% | 40-60% | Low | Request handling |
| conductor_mgmt | 15-40% | 30-50% | Low-Moderate | Job scheduling |
| data_ingest | 40-80% | 50-75% | High | Stream processing |
| risk_analytics | 70-95% | 60-85% | Moderate | Calculations |
| generic | 10-30% | 20-40% | Low | Utility services |

### ServerState

**Valid operational state values (optional field)**

```typescript
enum ServerState {
  HEALTHY = "healthy",             // Normal operation
  DEGRADING = "degrading",         // Performance declining
  STRESSED = "stressed",           // High load, near limits
  CRITICAL = "critical",           // Severe issues
  RECOVERING = "recovering",       // Coming back online
  MAINTENANCE = "maintenance",     // Planned downtime
  OFFLINE = "offline",             // Server unavailable
  UNKNOWN = "unknown"              // State unclear
}
```

### RiskLevel

**Risk categorization in predictions**

```typescript
enum RiskLevel {
  HEALTHY = "Healthy",             // Risk score 0-30
  DEGRADING = "Degrading",         // Risk score 30-60
  WARNING = "Warning",             // Risk score 60-80
  CRITICAL = "Critical"            // Risk score 80-100
}
```

**Risk Level Mapping:**

```typescript
function getRiskLevel(riskScore: number): RiskLevel {
  if (riskScore < 30) return RiskLevel.HEALTHY;
  if (riskScore < 60) return RiskLevel.DEGRADING;
  if (riskScore < 80) return RiskLevel.WARNING;
  return RiskLevel.CRITICAL;
}
```

---

## Data Types

### Timestamp Format

**ISO 8601 with timezone**

```typescript
type Timestamp = string;  // ISO 8601 format

// Valid examples:
const examples: Timestamp[] = [
  "2025-10-30T15:30:00Z",                 // UTC (Z suffix)
  "2025-10-30T15:30:00+00:00",            // UTC (offset notation)
  "2025-10-30T10:30:00-05:00",            // EST
  "2025-10-30T15:30:00.123Z",             // With milliseconds
  "2025-10-30T15:30:00.123456+00:00"      // With microseconds
];

// Invalid examples (will be rejected):
const invalid: string[] = [
  "2025-10-30 15:30:00",                  // Missing 'T'
  "10/30/2025 3:30 PM",                   // US date format
  "1730300400",                           // Unix timestamp
  "2025-10-30T15:30:00"                   // Missing timezone
];
```

### Numeric Ranges

**Type definitions with validation ranges**

```typescript
type Percentage = number;      // 0.0 - 100.0
type MegabytesPerSecond = number;  // 0.0 - 10000.0
type ConnectionCount = number; // 0 - 100000 (integer)
type LoadAverage = number;     // 0.0 - 1000.0
type Days = number;            // 0 - 36500 (integer)
type RiskScore = number;       // 0 - 100
type Confidence = number;      // 0.0 - 1.0
type Importance = number;      // 0.0 - 1.0
```

---

## JSON Schemas

### Input Schema (JSON Schema v7)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "MetricsRecord",
  "type": "object",
  "required": [
    "timestamp", "server_name", "profile",
    "cpu_user_pct", "cpu_sys_pct", "cpu_iowait_pct", "cpu_idle_pct", "java_cpu_pct",
    "mem_used_pct", "swap_used_pct", "disk_usage_pct",
    "net_in_mb_s", "net_out_mb_s",
    "back_close_wait", "front_close_wait",
    "load_average", "uptime_days"
  ],
  "properties": {
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp with timezone"
    },
    "server_name": {
      "type": "string",
      "minLength": 3,
      "maxLength": 64,
      "pattern": "^[a-zA-Z0-9._-]+$"
    },
    "profile": {
      "type": "string",
      "enum": ["ml_compute", "database", "web_api", "conductor_mgmt", "data_ingest", "risk_analytics", "generic"]
    },
    "cpu_user_pct": {"type": "number", "minimum": 0, "maximum": 100},
    "cpu_sys_pct": {"type": "number", "minimum": 0, "maximum": 100},
    "cpu_iowait_pct": {"type": "number", "minimum": 0, "maximum": 100},
    "cpu_idle_pct": {"type": "number", "minimum": 0, "maximum": 100},
    "java_cpu_pct": {"type": "number", "minimum": 0, "maximum": 100},
    "mem_used_pct": {"type": "number", "minimum": 0, "maximum": 100},
    "swap_used_pct": {"type": "number", "minimum": 0, "maximum": 100},
    "disk_usage_pct": {"type": "number", "minimum": 0, "maximum": 100},
    "net_in_mb_s": {"type": "number", "minimum": 0, "maximum": 10000},
    "net_out_mb_s": {"type": "number", "minimum": 0, "maximum": 10000},
    "back_close_wait": {"type": "integer", "minimum": 0, "maximum": 100000},
    "front_close_wait": {"type": "integer", "minimum": 0, "maximum": 100000},
    "load_average": {"type": "number", "minimum": 0, "maximum": 1000},
    "uptime_days": {"type": "integer", "minimum": 0, "maximum": 36500},
    "state": {
      "type": "string",
      "enum": ["healthy", "degrading", "stressed", "critical", "recovering", "maintenance", "offline", "unknown"]
    },
    "problem_child": {"type": "boolean"},
    "notes": {"type": "string", "maxLength": 500}
  }
}
```

### Output Schema (JSON Schema v7)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "PredictionResponse",
  "type": "object",
  "required": ["timestamp", "summary", "predictions"],
  "properties": {
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "summary": {
      "type": "object",
      "properties": {
        "total_servers": {"type": "integer"},
        "critical": {"type": "integer"},
        "warning": {"type": "integer"},
        "degrading": {"type": "integer"},
        "healthy": {"type": "integer"},
        "avg_risk_score": {"type": "number", "minimum": 0, "maximum": 100}
      }
    },
    "predictions": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "server_name": {"type": "string"},
          "profile": {"type": "string"},
          "risk_score": {"type": "number", "minimum": 0, "maximum": 100},
          "risk_level": {"type": "string", "enum": ["Healthy", "Degrading", "Warning", "Critical"]},
          "predicted_failures": {"type": "array"},
          "predictions": {"type": "array"}
        }
      }
    }
  }
}
```

---

## Examples

### Complete Input Example

```json
{
  "records": [
    {
      "timestamp": "2025-10-30T15:30:00Z",
      "server_name": "ppdb001",
      "profile": "database",
      "cpu_user_pct": 45.2,
      "cpu_sys_pct": 15.3,
      "cpu_iowait_pct": 18.5,
      "cpu_idle_pct": 21.0,
      "java_cpu_pct": 0.0,
      "mem_used_pct": 78.4,
      "swap_used_pct": 3.2,
      "disk_usage_pct": 82.1,
      "net_in_mb_s": 45.3,
      "net_out_mb_s": 38.7,
      "back_close_wait": 25,
      "front_close_wait": 12,
      "load_average": 8.5,
      "uptime_days": 45,
      "state": "stressed",
      "problem_child": false,
      "notes": "High I/O wait detected during backup window"
    },
    {
      "timestamp": "2025-10-30T15:30:00Z",
      "server_name": "ppweb002",
      "profile": "web_api",
      "cpu_user_pct": 32.1,
      "cpu_sys_pct": 8.4,
      "cpu_iowait_pct": 1.2,
      "cpu_idle_pct": 58.3,
      "java_cpu_pct": 28.5,
      "mem_used_pct": 55.2,
      "swap_used_pct": 0.1,
      "disk_usage_pct": 45.8,
      "net_in_mb_s": 125.4,
      "net_out_mb_s": 98.3,
      "back_close_wait": 8,
      "front_close_wait": 15,
      "load_average": 3.2,
      "uptime_days": 28,
      "state": "healthy"
    }
  ]
}
```

### Complete Output Example

```json
{
  "timestamp": "2025-10-30T15:30:00Z",
  "summary": {
    "total_servers": 25,
    "critical": 2,
    "warning": 5,
    "degrading": 8,
    "healthy": 10,
    "avg_risk_score": 45.3
  },
  "predictions": {
    "ppdb001": {
      "server_name": "ppdb001",
      "profile": "database",
      "risk_score": 85.2,
      "risk_level": "Critical",
      "current_metrics": {
        "cpu_user_pct": 45.2,
        "cpu_sys_pct": 15.3,
        "cpu_iowait_pct": 18.5,
        "mem_used_pct": 78.4,
        "disk_usage_pct": 82.1,
        "load_average": 8.5
      },
      "predicted_failures": [
        {
          "metric": "mem_used_pct",
          "current_value": 78.4,
          "predicted_value": 95.2,
          "threshold": 90.0,
          "time_to_failure_minutes": 45,
          "confidence": 0.89,
          "severity": "critical",
          "recommendation": "Memory exhaustion predicted. Consider: 1) Restart memory-heavy processes, 2) Add RAM, 3) Review query performance"
        }
      ],
      "predictions": [
        {
          "timestamp": "2025-10-30T15:35:00Z",
          "step": 1,
          "cpu_user_pct": 46.1,
          "cpu_sys_pct": 15.5,
          "mem_used_pct": 79.2,
          "disk_usage_pct": 82.3,
          "load_average": 8.7,
          "risk_score": 86.0,
          "confidence_interval": {
            "lower": 75.3,
            "upper": 92.1
          }
        },
        {
          "timestamp": "2025-10-30T15:40:00Z",
          "step": 2,
          "cpu_user_pct": 47.3,
          "cpu_sys_pct": 15.8,
          "mem_used_pct": 82.5,
          "disk_usage_pct": 82.5,
          "load_average": 9.1,
          "risk_score": 88.5,
          "confidence_interval": {
            "lower": 78.2,
            "upper": 94.3
          }
        }
      ]
    },
    "ppweb002": {
      "server_name": "ppweb002",
      "profile": "web_api",
      "risk_score": 25.1,
      "risk_level": "Healthy",
      "current_metrics": {
        "cpu_user_pct": 32.1,
        "mem_used_pct": 55.2,
        "disk_usage_pct": 45.8,
        "load_average": 3.2
      },
      "predicted_failures": [],
      "predictions": [
        {
          "timestamp": "2025-10-30T15:35:00Z",
          "step": 1,
          "cpu_user_pct": 32.5,
          "mem_used_pct": 55.5,
          "disk_usage_pct": 45.8,
          "load_average": 3.3,
          "risk_score": 25.3
        }
      ]
    }
  }
}
```

---

## See Also

- [API Reference](API_REFERENCE.md) - All API endpoints
- [Data Ingestion Guide](../for-production/DATA_INGESTION_GUIDE.md) - How to send data
- [Real Data Integration](../for-production/REAL_DATA_INTEGRATION.md) - Production setup

---

**Questions?** See [Troubleshooting Guide](../operations/TROUBLESHOOTING.md) or contact support.
