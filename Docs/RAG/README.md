# RAG (Retrieval-Augmented Generation) Folder

**Purpose**: Context documents for AI assistants working on the TFT Monitoring Prediction System

**Last Cleanup**: October 17, 2025

---

## 📚 Document Overview

### Primary Documents (Read First)

**[CURRENT_STATE.md](CURRENT_STATE.md)** (484 lines)
- ⭐ **Single source of truth** for current system state
- Combined and consolidated from ESSENTIAL_RAG + CURRENT_STATE_RAG
- Contains: Architecture, NordIQ Metrics Framework metrics, quick start, recent changes
- **Start here** for understanding the current system

**[PROJECT_CODEX.md](PROJECT_CODEX.md)** (844 lines)
- Development rules and conventions
- Schema requirements and validation rules
- Naming conventions and best practices
- **Read this** for understanding development standards

### Supporting Documents

**[CLAUDE_SESSION_GUIDELINES.md](CLAUDE_SESSION_GUIDELINES.md)** (430 lines)
- Session management protocol
- How to start/end sessions properly
- Documentation standards
- **For AI assistants** managing work sessions

**[MODULAR_REFACTOR_COMPLETE.md](MODULAR_REFACTOR_COMPLETE.md)** (262 lines)
- Details of the modular architecture refactor
- 84.8% code reduction achievement
- Modular structure explanation
- **For understanding** the dashboard architecture

**[TIME_TRACKING.md](TIME_TRACKING.md)** (201 lines)
- High-level development time summary
- Key achievements and ROI metrics
- Performance benchmarks
- **For reference** on project timeline

---

## 📊 Cleanup Results

**Before**: 3,971 lines across 7 files
**After**: 2,221 lines across 5 files
**Reduction**: 44% (1,750 lines removed)

### Changes Made:
- ✅ Merged ESSENTIAL_RAG.md + CURRENT_STATE_RAG.md → **CURRENT_STATE.md**
- ✅ Simplified TIME_TRACKING.md (634 → 201 lines, 68% reduction)
- ✅ Moved SESSION_2025-10-13_NordIQ Metrics Framework_METRICS_REFACTOR.md to Archive
- ✅ Updated all cross-references to new structure
- ✅ Eliminated redundancy and outdated information

---

## 🎯 How to Use These Documents

### For AI Assistants (New Session)
1. Read **CURRENT_STATE.md** - Get up to speed on latest status
2. Review **PROJECT_CODEX.md** - Understand development rules
3. Check **CLAUDE_SESSION_GUIDELINES.md** - Follow session protocol
4. Reference **MODULAR_REFACTOR_COMPLETE.md** - If working on dashboard code

### For Human Developers
- **CURRENT_STATE.md** provides the big picture
- **PROJECT_CODEX.md** defines the rules to follow
- Other docs provide historical context and architecture details

### What NOT to Read
- ⚠️ **DO NOT** read `Docs/Archive/` - Historical documents only
- ⚠️ **DO NOT** search for deprecated docs (ESSENTIAL_RAG.md, old CURRENT_STATE_RAG.md)
- ⚠️ **DO NOT** read old session notes unless specifically requested

---

## 🔄 Maintenance

**When to Update**:
- After major features or refactors → Update CURRENT_STATE.md
- When changing development rules → Update PROJECT_CODEX.md
- At end of significant sessions → May create session summary in Docs/

**Cleanup Schedule**:
- Review quarterly for outdated information
- Consolidate when multiple docs cover same topics
- Archive historical session notes

---

## 📁 Full Documentation Structure

```
Docs/
├── RAG/                           # You are here
│   ├── CURRENT_STATE.md          # ⭐ Single source of truth
│   ├── PROJECT_CODEX.md          # Development rules
│   ├── CLAUDE_SESSION_GUIDELINES.md  # Session management
│   ├── MODULAR_REFACTOR_COMPLETE.md  # Architecture details
│   └── TIME_TRACKING.md          # Development hours
├── Archive/                       # Historical documents
│   └── SESSION_*.md              # Old session notes
├── DATA_CONTRACT.md              # Schema specification
├── SERVER_PROFILES.md            # Profile system
├── INDEX.md                      # Navigation
└── [Other technical docs]
```

---

**Maintained By**: Project Team
**Last Updated**: October 17, 2025
**Next Review**: After Phase 2 completion
