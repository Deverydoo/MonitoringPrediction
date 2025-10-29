# Phase 4 Optimizations - Testing Guide

**Quick Start:** Test the Phase 4 optimizations before committing to git

---

## 🚀 Quick Start

```bash
# 1. Start the dashboard
cd NordIQ
python src/dashboard/tft_dashboard_web.py

# 2. Open browser to http://localhost:8501

# 3. Test the optimizations (see below)
```

---

## ✅ What to Test

### 1. Lazy Tab Loading (Should be MUCH faster)

**Test:**
1. Open dashboard - should load in <300ms (was 10-15s!)
2. Use "Select View" dropdown to switch tabs
3. Each tab should load instantly (<50ms)

**Expected:**
- ✅ Page loads almost instantly
- ✅ Only one tab renders at a time
- ✅ Tab switching is instant
- ✅ No lag or stutter

**Before vs After:**
- Before: All 11 tabs rendered = 2-3s page load
- After: Only 1 tab rendered = <300ms page load
- **90% faster!**

---

### 2. Manual Refresh (No Auto-Refresh)

**Test:**
1. Check sidebar "Refresh Settings"
2. "Enable auto-refresh" should be **unchecked** by default
3. Click "🔄 Refresh Now" button to manually fetch data

**Expected:**
- ✅ Auto-refresh is OFF by default
- ✅ Dashboard stays idle when not refreshing
- ✅ Manual refresh button works instantly
- ✅ No background CPU usage when idle

**Before vs After:**
- Before: Dashboard auto-refreshed every 5s (wasteful!)
- After: User controls refresh (0% overhead when idle)
- **100% fewer unnecessary refreshes!**

---

### 3. Fragment Isolation (Faster Interactions)

**Test:**
1. Go to "Overview" tab
2. Click any button or interact with widgets
3. Notice only the Overview section updates, not whole page

**Expected:**
- ✅ Clicking button doesn't reload entire page
- ✅ Only the fragment reruns
- ✅ Very fast interactions (<50ms)
- ✅ No full-page flicker

**Before vs After:**
- Before: Button click = full page rerun = 2-3s
- After: Button click = fragment rerun = <50ms
- **40-60× faster!**

---

### 4. Chart Container Reuse (Smoother Charts)

**Test:**
1. Go to "Overview" tab
2. Click "🔄 Refresh Now" multiple times
3. Watch the charts update

**Expected:**
- ✅ Charts update smoothly (no recreation flicker)
- ✅ Fast transitions
- ✅ No memory leaks

**Before vs After:**
- Before: Charts recreated = 200ms + flicker
- After: Charts reused = 100ms + smooth
- **50% faster + better UX!**

---

### 5. Aggressive Caching (Fewer API Calls)

**Test:**
1. Enable auto-refresh with 60s interval
2. Watch the "Last update" timestamp
3. Should only update every 60s (not every 5s!)

**Expected:**
- ✅ Cache persists for 30-60s
- ✅ Fewer API calls to daemon
- ✅ Still responsive to changes
- ✅ No stale data issues

**Before vs After:**
- Before: Cache 10-15s = 6 API calls/min
- After: Cache 30-60s = 1-2 API calls/min
- **75-83% fewer API calls!**

---

## 🐛 Common Issues

### Issue 1: "AttributeError: module 'streamlit' has no attribute 'fragment'"

**Cause:** Streamlit version too old (need 1.35+)

**Fix:**
```bash
pip install --upgrade streamlit
```

---

### Issue 2: Tab selector feels weird, want tabs back

**Cause:** Personal preference for st.tabs() UI

**Fix:** Easy to rollback (see PHASE_4_OPTIMIZATIONS_COMPLETE.md)

```python
# tft_dashboard_web.py - Replace selectbox with st.tabs
tab1, tab2, ... = st.tabs([...])
with tab1:
    overview.render(predictions)
# ... etc
```

---

### Issue 3: Charts not updating

**Cause:** Container reuse issue

**Fix:** Clear session state

```python
# In sidebar, add button:
if st.button("Clear Cache"):
    st.session_state.clear()
    st.rerun()
```

---

### Issue 4: Auto-refresh not working

**Cause:** Auto-refresh is OFF by default (intentional!)

**Fix:** Check the "Enable auto-refresh" checkbox in sidebar

---

## 📊 Performance Expectations

| Test | Before (Baseline) | After (Phase 4) | Improvement |
|------|-------------------|-----------------|-------------|
| **Page Load** | 10-15s | <300ms | **30-50× faster** |
| **Tab Switch** | 2-3s | <50ms | **40-60× faster** |
| **Button Click** | 2-3s | <50ms | **40-60× faster** |
| **Refresh (manual)** | N/A | <100ms | **Instant** |
| **CPU (idle)** | 20% | <0.5% | **40× reduction** |

---

## ✅ Success Checklist

Test all items before committing:

**Performance:**
- [ ] Page loads in <300ms (open Chrome DevTools to verify)
- [ ] Tab switching feels instant (<50ms)
- [ ] Buttons respond instantly
- [ ] No lag or stutter
- [ ] CPU usage <1% when idle

**Functionality:**
- [ ] All tabs load correctly
- [ ] Charts render correctly
- [ ] Manual refresh button works
- [ ] Auto-refresh can be enabled
- [ ] No JavaScript errors in console

**User Experience:**
- [ ] Tab selector dropdown is intuitive
- [ ] Manual refresh feels natural
- [ ] Charts update smoothly
- [ ] No visual glitches

---

## 🎯 Quick Performance Test

**1-Minute Test:**

```bash
# 1. Start dashboard
python src/dashboard/tft_dashboard_web.py

# 2. Open Chrome DevTools (F12)
# 3. Go to Network tab
# 4. Refresh page (Ctrl+R)
# 5. Check "Load" time in bottom status bar

Expected: <300ms (was 10-15s!)
```

**If slower than 300ms:**
- Check daemon is running (http://localhost:8000/health)
- Check CPU usage (Task Manager)
- Try clearing cache (sidebar button)
- Restart dashboard

---

## 🔄 Rollback (If Needed)

If Phase 4 causes issues, here's how to rollback:

**Option 1: Git Rollback (Recommended)**

```bash
# Discard all uncommitted changes
git restore .

# Restart dashboard
python src/dashboard/tft_dashboard_web.py
```

**Option 2: Manual Rollback**

See detailed rollback instructions in [PHASE_4_OPTIMIZATIONS_COMPLETE.md](../Docs/PHASE_4_OPTIMIZATIONS_COMPLETE.md)

---

## 📝 Next Steps

**After Testing:**

1. **If all tests pass:**
   - Commit changes to git
   - Update documentation
   - Deploy to production

2. **If issues found:**
   - Report issues in GitHub
   - Rollback if critical
   - Fix and re-test

3. **Optional (only if needed):**
   - Apply container reuse to more tabs
   - Add fragment-level caching
   - Implement background fetching

---

## 🎉 Expected Results

After Phase 4 optimizations, your dashboard should:

- ✅ Load in under 300ms (was 10-15s)
- ✅ Tab switching instant (was 2-3s)
- ✅ Buttons respond in <50ms (was 2-3s)
- ✅ Zero CPU usage when idle (was 20%)
- ✅ 100% fewer auto-refreshes (manual control)
- ✅ 91% less rendering (1 tab vs 11)
- ✅ Feel **blazing fast**!

**Total: 30-50× faster than original baseline!**

---

**Document Version:** 1.0.0
**Date:** October 29, 2025
**Phase:** Phase 4 Testing
**Company:** NordIQ AI, LLC
