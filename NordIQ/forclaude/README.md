# For Claude 3.7 - Data Adapter Development Kit

This folder contains everything Claude 3.7 (or any AI assistant) needs to build a data adapter for the NordIQ inference engine.

---

## What's in This Folder

| File | Size | Purpose |
|------|------|---------|
| **README.md** | - | This file (index) |
| **00_READ_ME_FIRST.md** | 5 KB | Navigation and decision tree |
| **01_QUICK_START.md** | 8 KB | Overview and examples (read first!) |
| **02_API_CONTRACT.md** | 14 KB | Exact API specification |
| **03_MINIMAL_TEMPLATE.py** | 12 KB | Ready-to-use Python template |
| **04_TESTING_GUIDE.md** | 13 KB | Testing and verification |
| **05_SUMMARY_FOR_CLAUDE.md** | 12 KB | Complete summary for AI |

**Total**: 6 files, ~76 KB, everything you need!

---

## Quick Start

### For Claude 3.7 (or any AI assistant):

1. Read **`05_SUMMARY_FOR_CLAUDE.md`** (3 min) - Complete overview
2. Read **`01_QUICK_START.md`** (5 min) - Understand the task
3. Read **`02_API_CONTRACT.md`** (10 min) - Learn the specs
4. Copy **`03_MINIMAL_TEMPLATE.py`** - Customize 3 functions
5. Follow **`04_TESTING_GUIDE.md`** - Verify it works

**Total time**: 4-6 hours

### For humans:

Same process! These docs are for both AI and humans.

---

## What You're Building

A **data adapter daemon** that:
1. Connects to Wells Fargo's data source (Linborg, Elasticsearch, etc.)
2. Transforms 9 metric fields to NordIQ format
3. POSTs to NordIQ inference daemon REST API
4. Runs continuously

**That's it!** Everything else is already built.

---

## File Descriptions

### 00_READ_ME_FIRST.md
- Navigation guide
- Decision tree (REST API? Streaming? Elasticsearch?)
- Questions to ask user
- File reading order

### 01_QUICK_START.md
- What you're building (overview)
- Why it's easy (3 functions to customize)
- Complete examples
- Common data sources
- Timeline (4-6 hours)

### 02_API_CONTRACT.md
- Exact API endpoint specification
- 9 required fields (types, ranges, units)
- Request/response formats
- Profile matching (auto-detection built-in!)
- Unit conversions
- Testing commands

### 03_MINIMAL_TEMPLATE.py
- Complete working Python script
- 3 functions marked "CUSTOMIZE THIS"
- Everything else ready to use
- Includes error handling, logging, batching
- Just copy and customize!

### 04_TESTING_GUIDE.md
- 5-step testing process
- Verification commands
- Common errors and solutions
- Dashboard validation
- Performance testing
- Final checklist

### 05_SUMMARY_FOR_CLAUDE.md
- Complete summary for AI assistants
- All key information in one place
- Common mistakes to avoid
- Debugging tips
- Success criteria

---

## The 9 Required Fields

Your only job is to map these from the user's data source:

```python
{
    "timestamp": "2025-10-30T14:35:00",  # ISO 8601
    "server_name": "ppdb001",            # Unique ID
    "cpu_pct": 45.2,                     # 0-100 %
    "memory_pct": 78.5,                  # 0-100 %
    "disk_pct": 62.3,                    # 0-100 %
    "network_in_mbps": 125.4,            # Mbps
    "network_out_mbps": 89.2,            # Mbps
    "disk_read_mbps": 42.1,              # MB/s
    "disk_write_mbps": 18.3              # MB/s
}
```

---

## Profile Matching - YOU DON'T NEED TO DO IT!

The inference daemon already auto-detects profiles from server names:
- `ppdb*` → database
- `ppweb*` → web_api
- `ppml*` → ml_compute

**Just omit the `profile` field unless user has special requirements!**

---

## How to Use This Folder

### Step 1: Upload to Claude 3.7

Upload all 6 files to Claude 3.7 conversation:
- 00_READ_ME_FIRST.md
- 01_QUICK_START.md
- 02_API_CONTRACT.md
- 03_MINIMAL_TEMPLATE.py
- 04_TESTING_GUIDE.md
- 05_SUMMARY_FOR_CLAUDE.md

### Step 2: Give Claude Context

Tell Claude:
```
"I need you to build a data adapter daemon that feeds data from our
monitoring system to the NordIQ inference engine. I've uploaded 6 files
that contain everything you need. Start by reading 05_SUMMARY_FOR_CLAUDE.md."
```

### Step 3: Answer Claude's Questions

Claude will ask:
- What is your data source? (Linborg, Elasticsearch, etc.)
- What does your data look like? (show sample JSON)
- How to authenticate?
- How many servers?

### Step 4: Review Claude's Code

Claude will customize the template. Review and test it.

### Step 5: Deploy

Add to startup scripts, done!

---

## What's Already Built (No Work Needed)

The NordIQ inference daemon already has:
- ✅ Profile auto-detection
- ✅ Data validation
- ✅ Rolling window management
- ✅ TFT predictions
- ✅ Risk scoring
- ✅ Alert generation
- ✅ Dashboard
- ✅ Hot model reload
- ✅ Automated retraining

**You only build the data adapter (100-200 lines of code)!**

---

## Estimated Time

| Task | Time |
|------|------|
| Reading docs | 30 min |
| Getting sample data from user | 15 min |
| Mapping fields | 15 min |
| Customizing template | 2 hours |
| Testing | 1 hour |
| Deployment | 30 min |
| **TOTAL** | **4-5 hours** |

---

## Success Criteria

Adapter is complete when:
1. ✅ Connects to data source
2. ✅ Transforms 9 fields correctly
3. ✅ POSTs to inference daemon
4. ✅ Dashboard shows servers (after 30 min warmup)
5. ✅ Predictions appear
6. ✅ Runs continuously without errors

---

## Support

If stuck:
1. Check API contract: `02_API_CONTRACT.md`
2. Check testing guide: `04_TESTING_GUIDE.md`
3. Check inference daemon logs: `logs/inference_daemon.log`
4. Verify daemon running: `./status.sh`

---

## Key Insight

**This is a simple ETL pipeline:**
- **Extract**: Query their data source
- **Transform**: Map 9 field names
- **Load**: POST to REST API

**Don't over-engineer it!**

The template does 90% of the work. Just customize 3 functions:
1. `poll_data_source()` - Connect to their source
2. `transform_record()` - Map field names
3. `match_profile()` - (optional, can skip)

**That's it!**

---

## Wells Fargo Corporate API Note

This package is designed to work with Wells Fargo's corporate Claude API, which may have different constraints than standard Claude. All instructions assume you can upload these files to your Claude conversation.

---

## License

© 2025 NordIQ AI, LLC. All rights reserved.

These files are provided for the specific purpose of building data adapters for the NordIQ inference engine.

---

**Ready to build? Start with `05_SUMMARY_FOR_CLAUDE.md`!**
