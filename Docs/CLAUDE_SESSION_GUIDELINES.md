# Claude Assistant Session Guidelines

**For Claude Code Assistant working on the TFT Monitoring Prediction System**

---

## 📋 Session Management Protocol

### **Session Start Checklist**

When beginning a new session:

1. **Record Session Start Time**
   ```markdown
   **Session Start:** [TIME] (e.g., 7:30 AM, 2:15 PM)
   **Date:** YYYY-MM-DD
   ```

2. **Quick Status Check**
   - Review last session summary (if exists)
   - Check git status
   - Review recent commits
   - Identify any incomplete tasks

3. **Greet and Orient**
   - Acknowledge the time of day
   - Reference what was accomplished in previous session
   - Ask about priorities for this session

4. **Read Core Documentation (ESSENTIAL ONLY)**
   - [ESSENTIAL_RAG.md](ESSENTIAL_RAG.md) - ⭐ Complete system reference
   - [SESSION_2025-10-11_SUMMARY.md](SESSION_2025-10-11_SUMMARY.md) - Latest session
   - [PROJECT_CODEX.md](PROJECT_CODEX.md) - Development rules
   - ⚠️ **DO NOT** read `archive/` directory - Historical docs only

---

### **During Session**

1. **Track Progress**
   - Use TodoWrite for multi-step tasks
   - Mark tasks complete as soon as finished
   - Update session notes as you work

2. **Document Decisions**
   - Record why certain approaches were chosen
   - Note any technical debt or "TODO" items
   - Save important findings immediately

3. **Create Clear Artifacts**
   - Code changes with clear comments
   - Documentation updates
   - Validation reports for major changes

---

### **Session End Protocol**

When wrapping up (user says "call it a night", "done for today", etc.):

1. **Record Session End Time**
   ```markdown
   **Session End:** [TIME] (e.g., 5:45 PM, 11:30 PM)
   **Duration:** [HOURS] hours (e.g., 4.5 hours)
   ```

2. **Create Session Summary Document**
   - Filename: `SESSION_YYYY-MM-DD_SUMMARY.md`
   - Location: `Docs/`
   - Template: See [Session Summary Template](#session-summary-template) below

3. **Update PROJECT_SUMMARY.md**
   - Add session changes to "Change Notes" section
   - Update "Current System State" if applicable
   - Update "Last Updated" timestamp

4. **Git Status Check**
   - List modified files
   - Suggest commit message if changes warrant it
   - Note any uncommitted work

5. **Set Clear Next Steps**
   - List 3-5 actionable next steps
   - Mark priority/blocking tasks
   - Note what's ready vs. what needs work

---

## 📝 Session Summary Template

```markdown
# Session Summary - [DATE]

**Session Start:** [TIME] (e.g., 7:30 AM)
**Session End:** [TIME] (e.g., 5:45 PM)
**Duration:** [HOURS] hours
**Status:** ✅ COMPLETE | 🔄 IN PROGRESS | ⚠️ BLOCKED

---

## 🎯 What Was Accomplished

### 1. [Major Task 1] ✅
- Detail what was done
- Files modified
- Results/outcomes

### 2. [Major Task 2] ✅
- Detail what was done
- Files modified
- Results/outcomes

### 3. [Major Task 3] 🔄
- What was started
- Current status
- What remains

---

## 📊 Current System State

### What Works ✅
- Feature/component 1
- Feature/component 2
- Feature/component 3

### What's Ready ⏭️
- Next task 1 - description
- Next task 2 - description

### What's Blocked ⚠️
- Issue 1 - why blocked
- Issue 2 - why blocked

---

## 🚀 Next Steps

### Immediate (Next Session)
1. **[Task 1]** - Description and why it's next
2. **[Task 2]** - Description
3. **[Task 3]** - Description

### Soon (This Week)
- Task A
- Task B

### Future (Nice to Have)
- Enhancement 1
- Enhancement 2

---

## 📁 Files Modified

### Code Updates
- `file1.py` - What changed
- `file2.py` - What changed

### Documentation Created/Updated
- `DOC1.md` - Purpose
- `DOC2.md` - Purpose

---

## 🔑 Key Technical Details

### Important Decisions Made
1. **Decision 1** - Why and what
2. **Decision 2** - Why and what

### Issues Resolved
1. **Issue 1** - How it was fixed
2. **Issue 2** - How it was fixed

### Known Issues/Technical Debt
1. **TODO 1** - What needs attention
2. **TODO 2** - What needs attention

---

## ✅ Validation Checklist

- [ ] All code tested
- [ ] Documentation updated
- [ ] Git status clean (or intentional uncommitted work noted)
- [ ] Next steps clearly defined
- [ ] No blockers for next session

---

**Session End:** [TIME]
**Next Phase:** [Brief description of what's next]
**Status:** [GREEN/YELLOW/RED] - [Brief status]
```

---

## 🎯 Best Practices

### **Time Tracking**
- Always ask for start time if not provided
- Record when user indicates end of session
- Calculate and note session duration
- Track cumulative time on major features

### **Session Types**

**Morning Sessions (AM)**
- Focus on planning and architecture
- Review overnight thoughts/ideas
- Set clear goals for the day
- Fresh perspective on hard problems

**Afternoon Sessions (PM)**
- Implementation and testing
- Building on morning plans
- Incremental progress
- Mid-session check-ins

**Evening Sessions (PM/Night)**
- Wrap up in-progress work
- Document decisions made
- Prepare for next session
- Avoid starting major new features

### **Communication Style by Time**

**Morning (Before 12 PM)**
```
"Good morning! I see we left off with [X] yesterday around [TIME].
Ready to tackle [Y] today?"
```

**Afternoon (12 PM - 5 PM)**
```
"We've been working for [X] hours. Made good progress on [Y].
Should we continue or switch to [Z]?"
```

**Evening (After 5 PM)**
```
"We've been at this for [X] hours since [START_TIME].
Getting close to wrapping up? Let me summarize what we've done..."
```

---

## 📊 Session Metrics to Track

In session summaries, include:

1. **Time Metrics**
   - Session duration
   - Time on each major task
   - Cumulative project time

2. **Productivity Metrics**
   - Tasks completed vs. planned
   - Files modified
   - Lines of code changed
   - Tests written/fixed

3. **Quality Metrics**
   - Issues resolved
   - Bugs fixed
   - Documentation pages created

4. **Progress Metrics**
   - % of feature complete
   - Blockers removed
   - Technical debt addressed

---

## 🔄 Cross-Session Continuity

### **Handoff Between Sessions**

**End of Session N:**
```markdown
## 🎯 Ready for Next Session

**Quick Start Commands:**
```bash
conda activate py310
python main.py status
git status
```

**Where We Left Off:**
- [Specific file/function] - [Exact state]
- [Next logical step] - [Why it's next]

**Context for Next Time:**
- Decision made: [X] because [Y]
- Tried [A] but [B] happened
- Need to remember: [Important detail]
```

**Start of Session N+1:**
```markdown
## 📍 Picking Up From Last Session

**Last Session:** [DATE] [START]-[END] ([DURATION])
**Left Off:** [Brief description]
**Today's Goal:** [What we're tackling]
```

---

## ⚠️ Important Reminders

### **Always Include**
- Session start and end times
- Duration calculation
- Clear next steps
- File modification list
- Status indicators (✅ 🔄 ⚠️ ❌)

### **Never Forget**
- Update PROJECT_SUMMARY.md
- Create session summary document
- Note any blocking issues
- Record important decisions
- Set clear next steps

### **User Signals for End of Session**
- "Call it a night"
- "Let's wrap up"
- "Done for today"
- "That's enough for now"
- "I'm heading out"
- Asking for summary
- Time references like "it's getting late"

---

## 📚 Related Documentation

### Active Documentation (Read These)
- [ESSENTIAL_RAG.md](ESSENTIAL_RAG.md) - ⭐ Complete reference
- [PROJECT_CODEX.md](PROJECT_CODEX.md) - Development rules
- [SESSION_2025-10-11_SUMMARY.md](SESSION_2025-10-11_SUMMARY.md) - Latest work
- [INDEX.md](INDEX.md) - Doc navigation
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [TIME_TRACKING.md](TIME_TRACKING.md) - Hours tracking

### Archived Documentation (DO NOT READ)
- `archive/` - Historical docs, superseded information
- Only access when user explicitly asks about past decisions
- Never read during session start/refresh

---

## 🎓 Example Session Markers

### **Good Session Start**
```
👋 Good morning!

I see we started yesterday around 7:30 AM and wrapped up at 5:45 PM
(~10 hours). We successfully fixed the schema mapping and validated
training works.

Today's session start: 8:15 AM

What would you like to tackle first today? I recommend:
1. Training the new model (30-40 min)
2. Testing end-to-end workflow
3. Exploring online learning implementation

What's your priority?
```

### **Good Session End**
```
📝 Session Summary

**Today:** October 10, 2025
**Session:** 7:30 AM - 5:45 PM (10.25 hours)

✅ Accomplished:
- Fixed schema mapping in tft_trainer.py
- Validated training pipeline works
- Created comprehensive documentation

⏭️ Ready for Next Session:
- Train new model (command ready)
- Test end-to-end workflow
- All blockers removed

Creating session summary document: SESSION_2025-10-10_SUMMARY.md
Updating PROJECT_SUMMARY.md with today's changes...

🎯 Status: GREEN - System ready for model training
```

---

**Document Version:** 1.0
**Created:** 2025-10-10
**Purpose:** Ensure consistent, high-quality session management and documentation
**Audience:** Claude Code Assistant (and human reference)

---

## 📌 Quick Checklist for Claude

**Every Session Start:**
- [ ] Record start time
- [ ] Read ESSENTIAL_RAG.md
- [ ] Read latest SESSION summary (not archived ones)
- [ ] Review PROJECT_CODEX.md
- [ ] Check git status
- [ ] ⚠️ SKIP archive/ directory entirely
- [ ] Greet with context

**Every Session End:**
- [ ] Record end time
- [ ] Calculate duration
- [ ] Create SESSION_YYYY-MM-DD_SUMMARY.md
- [ ] Update PROJECT_SUMMARY.md
- [ ] List clear next steps
- [ ] Note git status

**Quality Markers:**
- ✅ Done
- 🔄 In Progress
- ⏭️ Ready
- ⚠️ Blocked
- ❌ Failed/Rejected
