# Dashboard Security Audit Report

**Date**: 2025-10-16
**Target**: `tft_dashboard_web.py` + `Dashboard/utils/api_client.py`
**Scope**: Security vulnerabilities, compatibility with secured inference daemon
**Status**: ⚠️ **CRITICAL ISSUE FOUND** - API Key Missing

---

## Executive Summary

The dashboard has **ONE CRITICAL vulnerability**: it does not send the API key when communicating with the now-secured inference daemon. This will cause all protected endpoint requests to fail with HTTP 403 Forbidden.

**Risk Level**: 🔴 **CRITICAL** (Dashboard completely broken with secured daemon)

**Other Security Posture**: ✅ **GOOD** (No XSS, SQL injection, or code execution vulnerabilities)

---

## 🔴 Critical Findings

### 1. Missing API Key Authentication (CRITICAL)

**File**: [Dashboard/utils/api_client.py](Dashboard/utils/api_client.py)
**Lines**: 30-64 (all API methods)

**Problem**: The `DaemonClient` class makes requests to protected endpoints without sending the `X-API-Key` header.

**Impact**:
- ❌ `get_predictions()` will fail (HTTP 403)
- ❌ `get_alerts()` will fail (HTTP 403)
- ❌ `feed_data()` will fail (HTTP 403)
- ✅ `check_health()` will work (unprotected endpoint)

**Current Code**:
```python
def get_predictions(self) -> Optional[Dict]:
    """Get current predictions from daemon."""
    try:
        response = requests.get(f"{self.base_url}/predictions/current", timeout=5)
        # ❌ No X-API-Key header!
        if response.ok:
            return response.json()
        return None
```

**Required Fix**:
```python
def get_predictions(self) -> Optional[Dict]:
    """Get current predictions from daemon."""
    try:
        headers = self._get_auth_headers()  # Add API key header
        response = requests.get(
            f"{self.base_url}/predictions/current",
            headers=headers,  # ✅ Send API key
            timeout=5
        )
        if response.ok:
            return response.json()
        return None
```

**Affected Endpoints**:
1. `GET /predictions/current` - Used by all dashboard tabs
2. `GET /alerts/active` - Used by Overview and Alerting tabs
3. `POST /feed/data` - Used by Demo Mode

**Recommended Solution**:
- Add API key configuration to Streamlit secrets or environment variable
- Modify `DaemonClient.__init__()` to accept API key
- Add `_get_auth_headers()` helper method
- Update all protected endpoint calls to include headers

---

## ✅ Security Strengths

### 1. No Direct User Input to Backend
**Good**: Dashboard doesn't allow arbitrary user input that gets sent to daemon (no SQL injection, command injection risk)

### 2. No XSS Vulnerabilities
**Good**: All user-facing data is rendered through Streamlit's safe rendering (no raw HTML injection)

### 3. No Secrets Hardcoded
**Good**: No API keys, passwords, or credentials hardcoded in source

### 4. No Pickle Deserialization
**Good**: Dashboard only uses JSON (no pickle security risk)

### 5. Read-Only API Access (Mostly)
**Good**: Dashboard primarily reads data, minimal write operations
- Only write: `POST /feed/data` for demo mode (safe)
- No database writes, file system access, or shell commands

### 6. CORS-Friendly
**Good**: Dashboard makes same-origin requests (localhost:8501 → localhost:8000)

### 7. No Authentication Required
**Good**: Dashboard itself doesn't need user login (internal tool)
**Note**: This is acceptable for internal corporate networks, but consider adding authentication if exposed to wider network

---

## 🟡 Medium Priority Findings

### 1. API Key Storage Method Needed

**Issue**: No mechanism to configure API key for production

**Recommendations** (pick one):

**Option A: Streamlit Secrets** (Best for production)
```toml
# .streamlit/secrets.toml
[daemon]
api_key = "your-secret-key-here"
url = "http://localhost:8000"
```

```python
# In api_client.py
import streamlit as st
api_key = st.secrets["daemon"]["api_key"]
```

**Option B: Environment Variable** (Best for Docker/systemd)
```python
import os
api_key = os.getenv("TFT_API_KEY")
```

**Option C: Configuration File** (Best for traditional deployment)
```python
# Dashboard/config/dashboard_config.py
DAEMON_API_KEY = os.getenv("TFT_API_KEY", "")
```

**Recommendation**: Use **Option A (Streamlit Secrets)** for production, with fallback to env var:
```python
try:
    api_key = st.secrets["daemon"]["api_key"]
except:
    api_key = os.getenv("TFT_API_KEY", "")
```

---

### 2. Error Message Information Disclosure

**File**: [Dashboard/utils/api_client.py:38, 48, 60, 63](Dashboard/utils/api_client.py)

**Issue**: Error messages may expose internal details (URLs, stack traces)

**Current Code**:
```python
st.error(f"Error fetching predictions: {e}")  # May expose sensitive error details
st.error(f"Daemon returned {response.status_code}: {response.text[:200]}")  # Exposes error response
```

**Risk**: Low (internal tool), but may confuse users

**Recommendation**: Use generic error messages for production:
```python
# Development mode: detailed errors
if os.getenv("DEBUG", "false").lower() == "true":
    st.error(f"Error fetching predictions: {e}")
else:
    # Production: generic message
    st.error("Unable to fetch predictions. Check daemon connection.")
```

---

### 3. No Request Timeout Consistency

**Issue**: Different timeouts used across methods (2s, 5s)

**Recommendation**: Standardize timeouts:
```python
class DaemonClient:
    TIMEOUT_HEALTH = 2    # Fast health checks
    TIMEOUT_DATA = 10     # Slower data operations (large payloads)

    def check_health(self):
        response = requests.get(url, timeout=self.TIMEOUT_HEALTH)

    def get_predictions(self):
        response = requests.get(url, timeout=self.TIMEOUT_DATA)
```

---

### 4. No Retry Logic for Transient Failures

**Issue**: Single request failure causes immediate error (no retry)

**Recommendation**: Add simple retry for transient network errors:
```python
import time

def get_predictions(self, retries=3) -> Optional[Dict]:
    """Get current predictions with retry logic."""
    for attempt in range(retries):
        try:
            response = requests.get(...)
            if response.ok:
                return response.json()
            if response.status_code == 403:
                # Don't retry auth failures
                st.error("Authentication failed - check API key")
                return None
        except requests.exceptions.ConnectionError:
            if attempt < retries - 1:
                time.sleep(1)  # Brief wait before retry
                continue
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    return None
```

---

### 5. Metrics Generator Control - No Authentication

**File**: [tft_dashboard_web.py:206-248](tft_dashboard_web.py#L206-L248)

**Issue**: Dashboard sends scenario control commands to `http://localhost:8001` without authentication

**Current Code**:
```python
response = requests.post(
    f"{generator_url}/scenario/set",
    json={"scenario": "healthy"},
    timeout=2
)
```

**Risk**: Low (localhost only), but if metrics generator is network-accessible, anyone can change scenarios

**Recommendation**: If metrics generator adds API key auth in future, update dashboard to send key:
```python
response = requests.post(
    f"{generator_url}/scenario/set",
    json={"scenario": "healthy"},
    headers={"X-API-Key": api_key},
    timeout=2
)
```

---

## 🟢 Low Priority Findings

### 1. Demo Mode Uses Same API Key as Production

**Issue**: No separation between demo and production API keys

**Risk**: Very Low (demo mode is optional feature)

**Recommendation**: Consider separate keys if needed:
```python
demo_api_key = st.secrets.get("daemon_demo_api_key", api_key)
```

---

### 2. No Rate Limiting on Dashboard Side

**Issue**: Dashboard can spam daemon with requests (auto-refresh + manual refresh)

**Risk**: Low (inference daemon has rate limiting)

**Note**: Daemon rate limiting (30 req/min) will handle this. No dashboard-side limiting needed.

---

### 3. Session State Not Encrypted

**Issue**: Streamlit session state stores predictions in memory (not encrypted)

**Risk**: Very Low (server-side memory only, not persisted)

**Note**: This is standard Streamlit behavior and acceptable for internal tools.

---

## 📋 Compatibility Checklist

**Dashboard compatibility with secured inference daemon**:

- ❌ **API Key Header** - NOT IMPLEMENTED (critical)
- ✅ **CORS Origin** - Compatible (dashboard on :8501, daemon expects localhost:8501)
- ✅ **HTTP Methods** - Compatible (uses GET and POST only)
- ✅ **Request Format** - Compatible (JSON payloads)
- ✅ **Unprotected Endpoints** - `/health` still works (no API key needed)
- ❌ **Protected Endpoints** - Will fail without API key (`/predictions/current`, `/alerts/active`, `/feed/data`)

---

## 🔧 Required Fixes

### Priority 1: Add API Key Support (CRITICAL)

**File**: `Dashboard/utils/api_client.py`

**Changes Required**:
1. Add API key parameter to `__init__()`
2. Create `_get_auth_headers()` helper method
3. Update all protected endpoint methods to include headers
4. Add error handling for 403 Forbidden responses

**Implementation** (see next section)

---

### Priority 2: Add API Key Configuration (HIGH)

**File**: `.streamlit/secrets.toml` (create if doesn't exist)

**Add**:
```toml
[daemon]
api_key = "your-secret-key-here"
url = "http://localhost:8000"
```

**File**: `Dashboard/config/dashboard_config.py`

**Add**:
```python
import os
import streamlit as st

# API Key configuration (Streamlit secrets with env var fallback)
try:
    DAEMON_API_KEY = st.secrets["daemon"]["api_key"]
except:
    DAEMON_API_KEY = os.getenv("TFT_API_KEY", "")

# Warn if no API key in production
if not DAEMON_API_KEY:
    print("[WARNING] No API key configured - daemon may reject requests")
```

---

### Priority 3: Update Dashboard to Use API Key (HIGH)

**File**: `tft_dashboard_web.py`

**Change** (line 134):
```python
# OLD
client = DaemonClient(daemon_url)

# NEW
from Dashboard.config.dashboard_config import DAEMON_API_KEY
client = DaemonClient(daemon_url, api_key=DAEMON_API_KEY)
```

---

## 📝 Summary of Security Issues

| Issue | Severity | Impact | Status |
|-------|----------|--------|--------|
| **Missing API Key** | 🔴 CRITICAL | Dashboard broken with secured daemon | ❌ NOT FIXED |
| Error message disclosure | 🟡 MEDIUM | Information leakage | ⚠️ LOW RISK |
| No retry logic | 🟡 MEDIUM | Poor UX on network issues | ⚠️ MINOR |
| Inconsistent timeouts | 🟡 MEDIUM | Unpredictable behavior | ⚠️ MINOR |
| Generator auth missing | 🟡 MEDIUM | Scenario control unsecured | ⚠️ LOW RISK |
| No XSS vulnerabilities | ✅ GOOD | N/A | ✅ SECURE |
| No code injection | ✅ GOOD | N/A | ✅ SECURE |
| No hardcoded secrets | ✅ GOOD | N/A | ✅ SECURE |

---

## 🎯 Recommendation

**IMMEDIATE ACTION REQUIRED**: Update `DaemonClient` to support API key authentication before deploying secured inference daemon.

**Timeline**:
1. **Now**: Implement API key support in `DaemonClient` (15 minutes)
2. **Before Production**: Configure API key in `.streamlit/secrets.toml` (5 minutes)
3. **Optional**: Add retry logic, error message improvements (30 minutes)

**Without Fix**: Dashboard will be completely non-functional when connecting to secured daemon (all requests return HTTP 403).

---

## 📌 Next Steps

1. Implement API key support in `DaemonClient` class
2. Add API key configuration to dashboard config
3. Update dashboard instantiation to pass API key
4. Test dashboard with secured daemon
5. Verify all tabs work correctly
6. Document API key setup in deployment guide

---

**Last Updated**: 2025-10-16
**Audited By**: Claude (Security Analysis Agent)
**Next Review**: After API key implementation
