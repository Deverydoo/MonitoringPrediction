# Tachyon Argus Documentation

**Predictive Infrastructure Monitoring**

---

## Quick Navigation

### Getting Started
- **[QUICKSTART.md](QUICKSTART.md)** - Get running in 30 seconds

### Core Guides (Consolidated)
| Document | Purpose | When to Read |
|----------|---------|--------------|
| **[ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)** | System design, data flow, deployment | Understanding the system architecture |
| **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** | Model training, retraining, drift detection | Training or retraining models |
| **[PERFORMANCE_COMPLETE.md](PERFORMANCE_COMPLETE.md)** | Optimization guide, caching, scalability | Performance tuning |

### Technical References
| Document | Purpose |
|----------|---------|
| **[UNKNOWN_SERVER_HANDLING.md](UNKNOWN_SERVER_HANDLING.md)** | How inference handles new/unknown servers via hash encoding |
| **[SPARSE_DATA_HANDLING.md](SPARSE_DATA_HANDLING.md)** | How the system handles offline servers and data gaps |

### For Teams
| Document | Purpose |
|----------|---------|
| **[HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md)** | Complete team handoff document with integration guides |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | Contribution guidelines and code standards |

### For AI Assistants
| Document | Purpose |
|----------|---------|
| **[RAG/CURRENT_STATE.md](RAG/CURRENT_STATE.md)** | Current system state and context |
| **[RAG/PROJECT_CODEX.md](RAG/PROJECT_CODEX.md)** | Development rules and conventions |

---

## Integration Quick Reference

### Sending Data to Inference Engine

**POST to `/feed` endpoint:**
```bash
curl -X POST http://localhost:8000/feed \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '[{
    "timestamp": "2025-12-02T16:00:00Z",
    "server_name": "ppdb001",
    "cpu_user_pct": 45.2,
    "mem_used_pct": 67.8,
    "state": "healthy"
  }]'
```

See [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) Section 2-3 for complete API reference.

### Connecting to Grafana

The inference daemon exposes a REST API that Grafana can query:

1. Install **Grafana JSON API** plugin
2. Add data source pointing to `http://localhost:8000`
3. Query `/predictions/current` for real-time predictions

See [HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md) for detailed integration patterns.

---

## Document Structure

```
Docs/
├── README.md                    # This file - main navigation
├── QUICKSTART.md               # Fast setup guide
│
├── ARCHITECTURE_GUIDE.md       # System architecture (consolidated)
├── TRAINING_GUIDE.md           # Model training (consolidated)
├── PERFORMANCE_COMPLETE.md     # Performance optimization (consolidated)
│
├── UNKNOWN_SERVER_HANDLING.md  # Hash-based server encoding
├── SPARSE_DATA_HANDLING.md     # Offline server handling
│
├── HANDOFF_SUMMARY.md          # Team handoff document
├── CONTRIBUTING.md             # Contribution guidelines
│
├── RAG/                        # AI assistant context
│   ├── CURRENT_STATE.md
│   ├── PROJECT_CODEX.md
│   └── ...
│
└── archive/                    # Historical documents
    └── ...
```

---

## Common Tasks

### I want to...

**...get the system running quickly**
> [QUICKSTART.md](QUICKSTART.md)

**...understand the architecture**
> [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)

**...train or retrain a model**
> [TRAINING_GUIDE.md](TRAINING_GUIDE.md)

**...integrate with production data sources**
> [HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md) - See "Production Integration" section

**...connect to Grafana or external dashboards**
> [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) - See API endpoints section

**...understand how new servers are handled**
> [UNKNOWN_SERVER_HANDLING.md](UNKNOWN_SERVER_HANDLING.md)

---

## Key Concepts

### NordIQ Metrics Framework
The system uses 14 production metrics:
- CPU: `cpu_user_pct`, `cpu_sys_pct`, `cpu_iowait_pct`, `cpu_idle_pct`, `java_cpu_pct`
- Memory: `mem_used_pct`, `swap_used_pct`
- Disk: `disk_usage_pct`
- Network: `net_in_mb_s`, `net_out_mb_s`
- Connections: `back_close_wait`, `front_close_wait`
- System: `load_average`, `uptime_days`

### Server Profiles
7 profiles for transfer learning:
- ML_COMPUTE, DATABASE, WEB_API, CONDUCTOR_MGMT
- DATA_INGEST, RISK_ANALYTICS, GENERIC

### Architecture Pattern
```
Data Source → Adapter (push) → Inference Daemon → Dashboard
                                    ↓
                              REST API (8000)
                                    ↓
                              Grafana / Custom
```

---

**Last Updated:** December 2025
**Maintained By:** Project Team
