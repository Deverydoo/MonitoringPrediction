# Instructions for Claude 3.7 (or any AI assistant)

**Purpose**: You are building a data adapter daemon that feeds production monitoring data into the NordIQ inference engine.

**Context**: This is for Wells Fargo. The data source is either:
- Linborg (Wells internal monitoring system)
- Direct service feeds that also feed Linborg
- Or any other monitoring infrastructure they specify

**Your Job**: Build a Python daemon that:
1. Connects to their data source
2. Transforms data to NordIQ format (9 required fields)
3. POSTs to inference daemon REST API
4. Runs continuously (daemon mode)

---

## Files in This Folder

Read them in this order:

### 1. **START HERE** - Read These First

**01_QUICK_START.md** (5 min read)
- What you're building
- Why it's easy
- 5-minute overview

**02_API_CONTRACT.md** (10 min read)
- Exact API endpoint specification
- Required fields (just 9)
- Request/response formats
- Profile matching (optional - inference daemon can do it)

### 2. **IMPLEMENTATION** - Build Your Adapter

**03_MINIMAL_TEMPLATE.py** (Ready to use)
- Copy this file
- Customize 3 functions (marked with CUSTOMIZE)
- Run it

**04_STREAMING_TEMPLATE.py** (If they use Kafka/WebSocket)
- For streaming data sources
- Includes batching logic

**05_COMMON_TRANSFORMATIONS.md** (Examples)
- Elasticsearch → NordIQ
- MongoDB → NordIQ
- Generic REST API → NordIQ
- Prometheus → NordIQ

### 3. **REFERENCE** - When You Need Details

**06_FIELD_REFERENCE.md**
- All 9 required fields
- Data types, ranges, validation
- Optional fields

**07_PROFILE_MATCHING.md**
- What profiles are
- Auto-detection (built-in)
- Custom matching (if needed)

**08_ERROR_HANDLING.md**
- Common errors
- Retry logic
- Debugging tips

### 4. **TESTING** - Verify It Works

**09_TESTING_GUIDE.md**
- How to test your adapter
- Verification commands
- Dashboard validation

### 5. **DEPLOYMENT** - Production Ready

**10_PRODUCTION_DEPLOYMENT.md**
- Add to startup scripts
- Daemon mode
- Log management

---

## Quick Decision Tree

**Q: Do they have a REST API for metrics?**
- YES → Use `03_MINIMAL_TEMPLATE.py` (polling model)
- NO → Continue

**Q: Do they use Kafka/WebSocket/streaming?**
- YES → Use `04_STREAMING_TEMPLATE.py`
- NO → Continue

**Q: Is it Elasticsearch?**
- YES → See `05_COMMON_TRANSFORMATIONS.md` → Elasticsearch section
- NO → Continue

**Q: Is it MongoDB?**
- YES → See `05_COMMON_TRANSFORMATIONS.md` → MongoDB section
- NO → Continue

**Q: Is it InfluxDB/Prometheus?**
- YES → See `05_COMMON_TRANSFORMATIONS.md` → InfluxDB/Prometheus section
- NO → Ask user for details, adapt `03_MINIMAL_TEMPLATE.py`

---

## Questions to Ask the User

Before you start coding, ask:

1. **"What is your data source?"**
   - Linborg API, Elasticsearch, MongoDB, custom API, etc.?

2. **"How do I access it?"**
   - REST API endpoint? Database connection? Kafka topic?

3. **"What does your data look like?"**
   - Ask for sample JSON/data structure
   - Need to know their field names

4. **"How do I authenticate?"**
   - API token? Username/password? Certificate?

5. **"How many servers are you monitoring?"**
   - Affects batching strategy (10 vs 1000 servers)

6. **"Do you have a CMDB or service tags?"**
   - For profile matching, or we can use auto-detection

---

## What's Already Built (No Work Needed)

The NordIQ inference daemon **already handles**:
- ✅ Profile matching (auto-detection from server names)
- ✅ Rolling window management
- ✅ TFT predictions
- ✅ Risk scoring
- ✅ Alert generation
- ✅ Dashboard serving
- ✅ Data validation

**You only need to build the data adapter!**

---

## Estimated Time

- **Minimal adapter**: 4-6 hours
- **Production adapter**: 1-2 days
- **Enterprise adapter**: 3-5 days

Most use cases: **Minimal adapter is sufficient!**

---

## Architecture Diagram

```
[Your Data Source]
       ↓
  Your Adapter (you build this)
       ↓
  POST /feed/data
       ↓
  NordIQ Inference Daemon (already exists)
       ↓
  Dashboard (already exists)
```

---

## Success Criteria

Your adapter is complete when:
1. ✅ Connects to data source successfully
2. ✅ Transforms data to 9 required fields
3. ✅ POSTs to inference daemon without errors
4. ✅ Dashboard shows servers (wait 20-30 min for warmup)
5. ✅ Predictions appear on dashboard
6. ✅ Runs continuously without crashes

---

## Getting Help

If you get stuck:
1. Check `08_ERROR_HANDLING.md` for common issues
2. Check inference daemon logs: `logs/inference_daemon.log`
3. Test inference daemon directly: `curl http://localhost:8000/health`
4. Verify API key: `cat .nordiq_key`

---

## Start Here

1. Read `01_QUICK_START.md` (5 min)
2. Read `02_API_CONTRACT.md` (10 min)
3. Copy `03_MINIMAL_TEMPLATE.py`
4. Customize the 3 functions marked `# CUSTOMIZE THIS`
5. Test it
6. Deploy it

**You'll be done in 4-6 hours!**

---

© 2025 NordIQ AI, LLC. All rights reserved.
