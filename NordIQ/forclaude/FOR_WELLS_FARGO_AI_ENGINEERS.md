# For Wells Fargo AI Engineers

**Purpose**: This package enables your Claude AI assistant (via Wells Fargo's corporate interface) to build data adapters for the NordIQ inference engine.

**Use Case**: Connect Wells Fargo monitoring systems (Linborg, Elasticsearch, etc.) to NordIQ for predictive infrastructure monitoring.

---

## What This Package Does

Provides complete specifications and templates for Claude to autonomously build data adapter daemons that:
1. Connect to Wells Fargo monitoring infrastructure
2. Transform metrics to NordIQ format
3. Feed real-time data to NordIQ inference engine
4. Enable predictive alerts and risk scoring

**No manual coding required** - Claude does it all from these specifications.

---

## Package Contents (8 Files, 92 KB)

```
forclaude/
├── UPLOAD_THESE_FILES.txt            # Upload instructions
├── FOR_WELLS_FARGO_AI_ENGINEERS.md   # This file (deployment guide)
├── README.md                         # Package overview
├── 00_READ_ME_FIRST.md               # Navigation guide for Claude
├── 01_QUICK_START.md                 # Quick overview and examples
├── 02_API_CONTRACT.md                # Complete API specification
├── 03_MINIMAL_TEMPLATE.py            # Ready-to-use Python template
├── 04_TESTING_GUIDE.md               # Testing procedures
└── 05_SUMMARY_FOR_CLAUDE.md          # Complete summary for Claude
```

---

## How Wells Engineers Should Use This

### Step 1: Upload Files to Claude Interface

Upload all 8 files to your Wells Fargo Claude conversation interface.

**Note**: The Claude interface at Wells may handle file uploads differently than Anthropic's. Confirm all files are successfully uploaded before proceeding.

### Step 2: Initialize Claude

Provide this prompt to Claude:

```
"I am a Wells Fargo AI engineer. I need you to build a data adapter daemon
that connects our monitoring infrastructure to the NordIQ inference engine.

I've uploaded 8 documentation files that contain:
- Complete API specifications
- Ready-to-use Python template
- Testing procedures
- Examples for common data sources

Start by reading 05_SUMMARY_FOR_CLAUDE.md for a complete overview.

Our data source is: [SPECIFY: Linborg / Elasticsearch / Custom / etc.]

The adapter must:
1. Connect to our monitoring system
2. Transform 9 required metric fields
3. POST to NordIQ inference daemon REST API
4. Run as a daemon process

Can you help me build this adapter?"
```

### Step 3: Provide Wells-Specific Context

Claude will ask questions. Provide:

**Data Source Details**:
- System name (Linborg, Elasticsearch, Splunk, etc.)
- API endpoint or connection string
- Authentication method (tokens, certs, Kerberos, etc.)

**Sample Data**:
- Provide actual JSON/data structure from your monitoring system
- Claude needs to see field names to map them correctly

**Environment Details**:
- Server count (affects batching strategy)
- Network constraints (firewalls, proxies)
- Deployment environment (Linux version, Python version)

**Security Requirements**:
- Credential storage approach
- Network segmentation
- Compliance requirements (SOX, PCI, etc.)

### Step 4: Review Claude's Implementation

Claude will:
1. Analyze your data structure
2. Create field mapping table
3. Customize the Python template
4. Add Wells-specific authentication
5. Implement error handling
6. Generate deployment scripts

**Review for**:
- Correct field mappings
- Proper authentication handling
- Secure credential storage
- Error handling
- Logging compliance
- Network configuration

### Step 5: Test in Non-Production

Before production deployment:

```bash
# Test data source connection
python adapter.py --test-connection

# Test transformation
python adapter.py --test-transform

# Test inference daemon integration
python adapter.py --dry-run

# Test full flow (30 min for warmup)
python adapter.py
```

### Step 6: Production Deployment

Deploy using Wells standard procedures:
- Add to systemd/init.d
- Configure log rotation
- Set up monitoring/alerts
- Document in CMDB

---

## Technical Architecture

### Data Flow

```
Wells Monitoring (Linborg)
         ↓
    Adapter Daemon (Claude builds this)
         ↓ HTTP POST /feed/data
    NordIQ Inference Daemon
         ↓
    Predictions + Risk Scores
         ↓
    NordIQ Dashboard
```

### API Contract

The adapter sends this to NordIQ:

```json
POST http://nordiq-host:8000/feed/data
Headers: X-API-Key: {key}
Body: {
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
      "disk_write_mbps": 18.3
    }
  ]
}
```

**Just 9 fields required!** Everything else is already built.

---

## What Claude Will Generate

Claude will produce:

### 1. Adapter Daemon (`adapter_daemon.py`)
- ~150-250 lines of Python
- Polling loop (default 5 seconds)
- Field transformation logic
- Error handling and retries
- Logging
- Graceful shutdown

### 2. Configuration (`adapter_config.yaml`)
- Data source connection
- NordIQ endpoint
- Polling interval
- Batch size
- Authentication

### 3. Deployment Scripts
- `start_adapter.sh` - Start daemon
- `stop_adapter.sh` - Stop daemon
- `status_adapter.sh` - Check status

### 4. Documentation
- Field mapping table
- Deployment instructions
- Testing procedures
- Troubleshooting guide

---

## Security Considerations for Wells

### Credential Management

Claude will need guidance on Wells standards:

**Option 1: External secrets management** (Recommended)
```python
# Vault, CyberArk, etc.
from wells_secrets import get_secret
api_key = get_secret('nordiq-api-key')
```

**Option 2: Environment variables**
```python
import os
api_key = os.environ['NORDIQ_API_KEY']
```

**Option 3: Encrypted config file**
```python
from wells_crypto import decrypt_config
config = decrypt_config('/etc/nordiq/adapter.conf.enc')
```

### Network Security

Considerations:
- Adapter → Monitoring system (internal network)
- Adapter → NordIQ (may cross network zones)
- Firewall rules needed?
- Proxy configuration?
- mTLS requirements?

### Compliance

Ensure adapter meets:
- SOX compliance (if monitoring financial systems)
- PCI compliance (if monitoring payment systems)
- Wells logging standards
- Wells audit requirements
- Data retention policies

---

## Performance Specifications

### Scalability

| Servers | Batch Strategy | Rate |
|---------|----------------|------|
| 1-50 | Single batch | 1 req/min |
| 50-200 | Single batch | 1 req/min |
| 200-1000 | 4 batches | 4 req/min |
| 1000+ | 10 batches | 10 req/min |

**Rate limit**: 60 requests/minute max

### Resource Usage

**Expected**:
- CPU: <5% (polling mode)
- Memory: <100 MB
- Network: <1 Mbps (depends on server count)
- Disk: <10 MB (logs only)

### Latency

- Poll → Transform → POST: <500ms
- End-to-end (monitoring → dashboard): 5-10 seconds

---

## Common Wells-Specific Scenarios

### Scenario 1: Linborg Integration

**Data Source**: Wells Fargo internal Linborg system

**Challenge**: Linborg may have custom API format

**Solution**: Provide Claude with:
- Linborg API documentation
- Sample JSON response
- Authentication method

Claude will generate Linborg-specific adapter.

### Scenario 2: Elasticsearch (Metricbeat)

**Data Source**: Elasticsearch with Metricbeat indices

**Challenge**: Nested JSON structure

**Solution**: Use `02_API_CONTRACT.md` Elasticsearch examples

Claude has pre-built transformations for Metricbeat format.

### Scenario 3: Multiple Data Sources

**Challenge**: Need adapters for Linborg + Elasticsearch + Splunk

**Solution**: Run Claude workflow 3 times, once per source

Each adapter runs independently, all feed to same NordIQ instance.

### Scenario 4: Proxy Requirements

**Challenge**: Wells requires HTTP proxy for outbound connections

**Solution**: Claude will add proxy configuration:
```python
proxies = {
    'http': 'http://proxy.wellsfargo.com:8080',
    'https': 'http://proxy.wellsfargo.com:8080'
}
requests.post(url, proxies=proxies, ...)
```

---

## Testing in Wells Environment

### Unit Testing

```python
# Test data source connection
def test_connection():
    records = poll_data_source()
    assert len(records) > 0
    assert 'hostname' in records[0]

# Test transformation
def test_transform():
    raw = {...}  # Sample from Wells
    nordiq = transform_record(raw)
    assert len(nordiq) == 9  # All required fields
    assert 0 <= nordiq['cpu_pct'] <= 100
```

### Integration Testing

```bash
# Non-prod environment
export NORDIQ_HOST=nordiq-dev.wellsfargo.com
export MONITORING_HOST=linborg-dev.wellsfargo.com

python adapter_daemon.py
```

### Load Testing

```bash
# Verify performance with production-scale data
python adapter_daemon.py --test-load --servers=1000
```

---

## Deployment Checklist for Wells

Before production:

**Pre-Deployment**:
- [ ] Code review completed
- [ ] Security review completed
- [ ] Credentials configured in approved secrets manager
- [ ] Network firewall rules approved and configured
- [ ] CMDB entry created
- [ ] Monitoring/alerting configured
- [ ] Runbook documented

**Deployment**:
- [ ] Deploy to non-prod first (1 week minimum)
- [ ] Verify data flowing correctly
- [ ] Verify NordIQ predictions accurate
- [ ] Load test with production-scale data
- [ ] Document any tuning performed

**Post-Deployment**:
- [ ] Monitor for 24 hours
- [ ] Verify no errors in logs
- [ ] Verify dashboard shows correct data
- [ ] Verify predictions appearing
- [ ] Verify alerts functioning
- [ ] Update documentation

---

## Troubleshooting Guide for Wells Engineers

### Issue: Adapter can't connect to Linborg

**Check**:
```bash
# Test from adapter host
curl http://linborg.wellsfargo.com/api/metrics

# Check network
ping linborg.wellsfargo.com
traceroute linborg.wellsfargo.com

# Check firewall
telnet linborg.wellsfargo.com 443
```

### Issue: Authentication failures

**Check**:
```bash
# Verify credentials
echo $API_TOKEN | wc -c  # Should be non-zero

# Test auth manually
curl -H "Authorization: Bearer $API_TOKEN" http://linborg.../api/metrics

# Check token expiration
```

### Issue: NordIQ not receiving data

**Check**:
```bash
# Test NordIQ connectivity
curl http://nordiq-host:8000/health

# Check adapter logs
tail -f /var/log/nordiq/adapter.log

# Check NordIQ daemon logs
ssh nordiq-host "tail -f /opt/nordiq/logs/inference_daemon.log"
```

### Issue: Dashboard shows no data

**Wait**: 30 minutes minimum for warmup

**Check**:
```bash
# Verify data reaching inference daemon
curl -H "X-API-Key: $KEY" http://nordiq-host:8000/status

# Check warmup status
curl -H "X-API-Key: $KEY" http://nordiq-host:8000/status | jq '.warmup_status'
```

---

## Support and Escalation

### Claude Issues

If Claude generates incorrect code:
1. Provide more specific data samples
2. Clarify field names and formats
3. Show error messages
4. Ask Claude to fix specific issues

### NordIQ Issues

If NordIQ inference daemon issues:
1. Check logs: `/opt/nordiq/logs/inference_daemon.log`
2. Verify API key: `cat /opt/nordiq/.nordiq_key`
3. Restart if needed: `cd /opt/nordiq && ./stop_all.sh && ./start_all.sh`
4. Contact NordIQ support: [support info]

---

## Wells-Specific Best Practices

### Logging

Follow Wells logging standards:
```python
import logging
logger = logging.getLogger('wells.nordiq.adapter')
logger.setLevel(logging.INFO)

# Wells standard log format
formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
```

### Error Handling

Wells-compliant error handling:
```python
try:
    send_data(records)
except Exception as e:
    logger.error(f"Failed to send data: {e}", exc_info=True)
    # Don't expose sensitive data in logs
    # Don't crash - retry on next iteration
```

### Monitoring

Integrate with Wells monitoring:
```python
# Emit metrics to Wells monitoring
from wells_metrics import emit_metric

emit_metric('nordiq.adapter.records_sent', len(records))
emit_metric('nordiq.adapter.errors', error_count)
emit_metric('nordiq.adapter.latency_ms', latency)
```

---

## Expected Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Claude generates adapter | 4-6 hours | Working Python code |
| Code review | 1-2 days | Approved code |
| Security review | 3-5 days | Security approval |
| Non-prod deployment | 1 week | Validated in test env |
| Production deployment | 1 day | Running in production |
| Monitoring period | 1 week | Stable operation |
| **TOTAL** | **2-3 weeks** | Production adapter |

*Most time is Wells processes, not development!*

---

## Success Metrics

Adapter is production-ready when:

1. ✅ **Connectivity**: Stable connection to data source (99.9% uptime)
2. ✅ **Data Quality**: All 9 fields correctly mapped
3. ✅ **Performance**: <500ms per poll cycle
4. ✅ **Reliability**: No crashes for 7+ days
5. ✅ **Security**: Passes Wells security review
6. ✅ **Compliance**: Meets logging and audit requirements
7. ✅ **Predictions**: NordIQ dashboard shows accurate predictions
8. ✅ **Alerts**: Alerts trigger appropriately

---

## ROI for Wells

### Before NordIQ
- Reactive monitoring (respond after failures)
- Manual log analysis
- Alert fatigue (false positives)
- Average 30-60 min MTTR

### After NordIQ
- Predictive monitoring (prevent failures)
- AI-driven analysis
- Contextual risk-based alerts
- Target 5-10 min MTTR

### Value Proposition
- Reduce incidents by 50-70%
- Reduce MTTR by 80%
- Reduce false positives by 90%
- Increase team productivity by 40%

---

## Questions?

This package is designed to be self-service for Wells AI engineers. Claude should be able to build adapters with minimal human intervention.

**If you get stuck**:
1. Re-upload files to Claude
2. Provide more specific context
3. Share error messages with Claude
4. Ask Claude to read specific documentation files

**The package contains everything Claude needs to succeed.**

---

## License and Usage

© 2025 NordIQ AI, LLC. All rights reserved.

This package is provided for Wells Fargo to evaluate NordIQ integration.

For production deployment, contact NordIQ for:
- Commercial licensing
- Enterprise support
- SLA agreements
- Professional services

---

## Version

**Package Version**: 1.0
**Last Updated**: October 30, 2025
**Compatible with**: NordIQ v2.3+
**Claude Interface**: Wells Fargo Corporate API

---

**Ready to deploy!** Upload these 8 files to your Wells Claude interface and start building adapters.

Good luck! 🚀
