# Authentication Implementation Guide

**Question**: How difficult is it to add token-based or other authentication instead of accepting connections from anywhere?

**Answer**: Not difficult at all! Estimated implementation time: **2-4 hours** for production-ready auth.

---

## Current State: No Authentication

**Current Security**:
- ✅ Running on localhost (127.0.0.1:8501) - only accessible from same machine
- ✅ Inference daemon runs on localhost (127.0.0.1:8000)
- ✅ Metrics daemon runs on localhost (127.0.0.1:8001)
- ❌ No authentication if exposed to network
- ❌ No user tracking or audit logs
- ❌ No role-based access control

**Risk Level**:
- **Low** for POC/demo on localhost
- **HIGH** if exposed to corporate network or internet

---

## Authentication Options (Easiest → Most Complex)

### Option 1: Simple Token Authentication ⭐ RECOMMENDED FOR MVP
**Difficulty**: ⭐ Easy (2 hours)
**Security**: Medium
**Best For**: Internal tools, trusted networks

#### Implementation:

```python
# auth.py
import streamlit as st
import hashlib
from datetime import datetime, timedelta

# Configuration
VALID_TOKENS = {
    "demo_token_abc123": {
        "username": "demo_user",
        "role": "viewer",
        "expires": None  # Never expires
    },
    "admin_token_xyz789": {
        "username": "admin",
        "role": "admin",
        "expires": None
    }
}

def check_authentication():
    """Check if user is authenticated via token."""

    # Check if already authenticated in session
    if "authenticated" in st.session_state and st.session_state.authenticated:
        return True

    # Check URL parameter for token
    query_params = st.query_params
    token = query_params.get("token", None)

    if token and token in VALID_TOKENS:
        # Valid token
        st.session_state.authenticated = True
        st.session_state.username = VALID_TOKENS[token]["username"]
        st.session_state.role = VALID_TOKENS[token]["role"]
        st.session_state.login_time = datetime.now()
        return True

    # Not authenticated - show login page
    st.error("🔒 Authentication Required")
    st.markdown("""
    This dashboard requires authentication. Please contact your administrator for an access token.

    **Access URL Format**: `http://localhost:8501?token=YOUR_TOKEN_HERE`
    """)
    st.stop()
    return False

def logout():
    """Clear authentication."""
    for key in ["authenticated", "username", "role", "login_time"]:
        if key in st.session_state:
            del st.session_state[key]
```

**Usage in Dashboard**:

```python
# tft_dashboard_web.py (add at top after imports)
from auth import check_authentication, logout

# Check auth before rendering anything
check_authentication()

# Show user info in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown(f"👤 **User**: {st.session_state.username}")
    st.markdown(f"🎭 **Role**: {st.session_state.role}")
    if st.button("🚪 Logout"):
        logout()
        st.rerun()
```

**Pros**:
- ✅ Very simple to implement (50 lines of code)
- ✅ No database required
- ✅ Tokens can be revoked by removing from dict
- ✅ URL-based auth works with bookmarks

**Cons**:
- ❌ Tokens in URL (visible in browser history)
- ❌ No password changes without code update
- ❌ Limited scalability (tokens in code)

**Security Enhancements**:
```python
# Store tokens in environment variables or config file
import os
VALID_TOKENS = json.loads(os.getenv("DASHBOARD_TOKENS", "{}"))

# Add token expiration
if token_info["expires"] and datetime.now() > token_info["expires"]:
    st.error("Token expired")
    st.stop()

# Hash tokens in memory
TOKEN_HASHES = {hashlib.sha256(t.encode()).hexdigest(): info for t, info in VALID_TOKENS.items()}
```

---

### Option 2: Username/Password Authentication
**Difficulty**: ⭐⭐ Moderate (3 hours)
**Security**: Medium-High
**Best For**: Small teams, internal apps

#### Implementation:

```python
# auth.py
import streamlit as st
import hashlib
import json
from pathlib import Path

USER_DB = Path("config/users.json")

def hash_password(password: str) -> str:
    """Hash password with salt."""
    salt = "tft_monitoring_salt_2025"  # Use proper secret management in prod
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()

def load_users():
    """Load users from JSON file."""
    if USER_DB.exists():
        return json.loads(USER_DB.read_text())
    return {
        "admin": {
            "password_hash": hash_password("admin123"),  # Change in production!
            "role": "admin",
            "email": "admin@company.com"
        },
        "viewer": {
            "password_hash": hash_password("viewer123"),
            "role": "viewer",
            "email": "viewer@company.com"
        }
    }

def check_authentication():
    """Check username/password authentication."""

    # Already authenticated
    if st.session_state.get("authenticated", False):
        return True

    # Show login form
    st.markdown("## 🔒 TFT Monitoring Dashboard - Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")

        if submit:
            users = load_users()

            if username in users:
                password_hash = hash_password(password)

                if users[username]["password_hash"] == password_hash:
                    # Successful login
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.role = users[username]["role"]
                    st.session_state.email = users[username]["email"]
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid password")
            else:
                st.error("User not found")

    st.stop()
```

**Pros**:
- ✅ Familiar login UX
- ✅ Passwords not in URL
- ✅ Easy user management (JSON file)
- ✅ Role-based access control

**Cons**:
- ❌ Need password reset mechanism
- ❌ Session management required
- ❌ No SSO/LDAP integration

---

### Option 3: JWT Token Authentication
**Difficulty**: ⭐⭐⭐ Moderate-Advanced (4 hours)
**Security**: High
**Best For**: API integration, distributed systems

#### Implementation:

```python
# auth.py
import streamlit as st
import jwt
from datetime import datetime, timedelta
import os

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-prod")
ALGORITHM = "HS256"

def generate_token(username: str, role: str, expires_hours: int = 24) -> str:
    """Generate JWT token."""
    payload = {
        "username": username,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=expires_hours),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> dict:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return {"error": "Token expired"}
    except jwt.InvalidTokenError:
        return {"error": "Invalid token"}

def check_authentication():
    """Check JWT authentication."""

    if st.session_state.get("authenticated", False):
        return True

    # Check Authorization header (if running behind proxy)
    # OR check query parameter
    query_params = st.query_params
    token = query_params.get("token", None)

    if token:
        payload = verify_token(token)

        if "error" not in payload:
            st.session_state.authenticated = True
            st.session_state.username = payload["username"]
            st.session_state.role = payload["role"]
            st.session_state.token_exp = payload["exp"]
            return True
        else:
            st.error(f"🔒 {payload['error']}")
    else:
        st.error("🔒 Authentication Required - No token provided")

    st.stop()

# Token generation utility (run separately)
def generate_user_token():
    """CLI utility to generate tokens."""
    import sys
    if len(sys.argv) < 3:
        print("Usage: python auth.py <username> <role>")
        sys.exit(1)

    username = sys.argv[1]
    role = sys.argv[2]
    token = generate_token(username, role, expires_hours=720)  # 30 days

    print(f"\n{'='*70}")
    print(f"JWT Token for {username} ({role}):")
    print(f"{'='*70}")
    print(token)
    print(f"\nExpires: {datetime.utcnow() + timedelta(hours=720)}")
    print(f"\nAccess URL: http://localhost:8501?token={token}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    generate_user_token()
```

**Usage**:
```bash
# Generate token
python auth.py admin_user admin

# Output:
# JWT Token for admin_user (admin):
# eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6ImFkbWluX3VzZXIiLCJyb2xlIjoiYWRtaW4iLCJleHAiOjE3MzQ2MDAwMDB9.ABC123...
#
# Access URL: http://localhost:8501?token=eyJhbGci...
```

**Pros**:
- ✅ Industry standard
- ✅ Built-in expiration
- ✅ Stateless (no session storage)
- ✅ Can include custom claims
- ✅ Works with API gateways

**Cons**:
- ❌ Requires PyJWT library
- ❌ More complex setup
- ❌ Need token refresh mechanism

---

### Option 4: OAuth2/SSO Integration (Google, Microsoft, Okta)
**Difficulty**: ⭐⭐⭐⭐ Advanced (8+ hours)
**Security**: Very High
**Best For**: Enterprise deployments

#### Implementation (using streamlit-oauth):

```python
# auth.py
import streamlit as st
from streamlit_oauth import OAuth2Component
import os

# OAuth configuration
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET")
OAUTH_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
OAUTH_REDIRECT_URI = "http://localhost:8501"

def check_authentication():
    """Check OAuth2 authentication."""

    if st.session_state.get("authenticated", False):
        return True

    # Create OAuth component
    oauth2 = OAuth2Component(
        OAUTH_CLIENT_ID,
        OAUTH_CLIENT_SECRET,
        OAUTH_AUTHORIZE_URL,
        OAUTH_TOKEN_URL,
        OAUTH_REDIRECT_URI,
        scope="openid email profile"
    )

    # Get authorization
    result = oauth2.authorize_button(
        "Login with Google",
        icon="🔐",
        redirect_uri=OAUTH_REDIRECT_URI
    )

    if result:
        # Successful authentication
        st.session_state.authenticated = True
        st.session_state.username = result.get("email")
        st.session_state.user_info = result
        st.rerun()

    st.stop()
```

**Pros**:
- ✅ Enterprise-grade security
- ✅ Single sign-on experience
- ✅ Centralized user management
- ✅ MFA support
- ✅ Compliance-friendly

**Cons**:
- ❌ Complex setup
- ❌ Requires OAuth provider configuration
- ❌ May require corporate approval
- ❌ Additional dependencies

---

## Role-Based Access Control (RBAC)

Once authenticated, you can add role-based permissions:

```python
# auth.py
ROLE_PERMISSIONS = {
    "admin": {
        "view_dashboard": True,
        "modify_settings": True,
        "view_advanced": True,
        "trigger_remediation": True,
        "manage_users": True
    },
    "operator": {
        "view_dashboard": True,
        "modify_settings": False,
        "view_advanced": True,
        "trigger_remediation": True,
        "manage_users": False
    },
    "viewer": {
        "view_dashboard": True,
        "modify_settings": False,
        "view_advanced": False,
        "trigger_remediation": False,
        "manage_users": False
    }
}

def has_permission(permission: str) -> bool:
    """Check if current user has permission."""
    role = st.session_state.get("role", "viewer")
    return ROLE_PERMISSIONS.get(role, {}).get(permission, False)

def require_permission(permission: str):
    """Decorator to require permission for tab/feature."""
    if not has_permission(permission):
        st.error(f"🔒 Access Denied: You need '{permission}' permission")
        st.info(f"Your role ({st.session_state.get('role')}) does not have access to this feature.")
        st.stop()
```

**Usage in Dashboard**:

```python
# In Auto-Remediation tab
with tab6:
    require_permission("trigger_remediation")
    st.subheader("🤖 Auto-Remediation Strategy")
    # ... rest of tab content

# In Advanced tab
with tab8:
    require_permission("view_advanced")
    st.subheader("⚙️ Advanced Settings")
    # ... rest of tab content
```

---

## Audit Logging

Add security audit logging:

```python
# audit_log.py
import json
from datetime import datetime
from pathlib import Path

AUDIT_LOG = Path("logs/audit.log")

def log_event(event_type: str, details: dict = None):
    """Log security/audit event."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "username": st.session_state.get("username", "anonymous"),
        "role": st.session_state.get("role", "unknown"),
        "event_type": event_type,
        "details": details or {}
    }

    AUDIT_LOG.parent.mkdir(exist_ok=True)
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")

# Usage examples:
log_event("login_success", {"ip": "192.168.1.100"})
log_event("view_alert", {"server": "ppdb001", "risk": 85})
log_event("trigger_remediation", {"server": "ppml0001", "action": "restart_service"})
log_event("logout", {})
```

---

## Implementation Comparison

| Feature | Token Auth | User/Pass | JWT | OAuth2/SSO |
|---------|-----------|-----------|-----|------------|
| **Implementation Time** | 2 hours | 3 hours | 4 hours | 8+ hours |
| **Code Complexity** | Low | Medium | Medium | High |
| **External Dependencies** | None | None | PyJWT | streamlit-oauth |
| **User Management** | Code/Config | JSON file | External | IDP |
| **Password Security** | N/A | Medium | N/A | High |
| **Token Expiration** | Optional | Session-based | Built-in | Built-in |
| **SSO Support** | No | No | Possible | Yes |
| **MFA Support** | No | Manual | No | Yes |
| **Audit Logging** | Manual | Manual | Manual | Built-in |
| **Production Ready** | With caveats | Yes | Yes | Yes |
| **Best For** | POC/Demo | Small teams | APIs | Enterprise |

---

## Recommended Approach for Your Demo

### For Tuesday Demo: ⭐ **No Auth Required**
- Dashboard runs on localhost
- Only accessible from demo machine
- No security risk for presentation

### For Post-Demo Internal Use: ⭐ **Option 1: Simple Token Auth**
**Why**:
- ✅ 2 hours to implement (minimal delay)
- ✅ Good enough for internal corporate network
- ✅ Easy to distribute tokens to team
- ✅ Can upgrade later if needed

**Implementation Steps**:
1. Create `auth.py` with token validation (30 min)
2. Add authentication check to dashboard (15 min)
3. Generate tokens for team members (15 min)
4. Test and document (1 hour)

**Example Token Distribution**:
```
Team Member Access Tokens:
- John (Admin): http://localhost:8501?token=admin_john_abc123
- Sarah (Operator): http://localhost:8501?token=ops_sarah_xyz789
- Mike (Viewer): http://localhost:8501?token=view_mike_def456
```

### For Production Deployment: ⭐⭐⭐ **Option 4: OAuth2/SSO**
**Why**:
- ✅ Integrates with corporate identity provider (Okta, Azure AD, etc.)
- ✅ Centralized user management
- ✅ MFA enforcement
- ✅ Compliance requirements met
- ✅ No password management burden

---

## Additional Security Considerations

### 1. Network Security
```python
# Only allow specific IP ranges
ALLOWED_IPS = ["192.168.1.0/24", "10.0.0.0/8"]

def check_ip_whitelist():
    """Verify request comes from allowed IP."""
    # Note: Streamlit doesn't expose request IP directly
    # Need to run behind nginx/proxy that adds X-Forwarded-For header
    pass
```

### 2. HTTPS/TLS
```bash
# Run Streamlit with HTTPS
streamlit run tft_dashboard_web.py \
    --server.sslCertFile=/path/to/cert.pem \
    --server.sslKeyFile=/path/to/key.pem
```

### 3. Rate Limiting
```python
from datetime import datetime, timedelta
from collections import defaultdict

# Track login attempts
login_attempts = defaultdict(list)

def check_rate_limit(username: str, max_attempts: int = 5, window_minutes: int = 15):
    """Prevent brute force attacks."""
    now = datetime.now()
    cutoff = now - timedelta(minutes=window_minutes)

    # Remove old attempts
    login_attempts[username] = [
        t for t in login_attempts[username] if t > cutoff
    ]

    if len(login_attempts[username]) >= max_attempts:
        st.error(f"Too many login attempts. Try again in {window_minutes} minutes.")
        st.stop()

    login_attempts[username].append(now)
```

### 4. Session Timeout
```python
from datetime import datetime, timedelta

SESSION_TIMEOUT_MINUTES = 60

def check_session_timeout():
    """Logout after inactivity."""
    if "last_activity" in st.session_state:
        last_activity = st.session_state.last_activity
        if datetime.now() - last_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
            st.warning("Session expired due to inactivity")
            logout()
            st.rerun()

    st.session_state.last_activity = datetime.now()
```

---

## Migration Path

**Phase 1 (Demo)**: No authentication
- ✅ Works for Tuesday demo
- ✅ Zero implementation time

**Phase 2 (Internal)**: Simple token auth (Week 1 post-demo)
- ✅ 2-hour implementation
- ✅ Good for internal team (5-20 users)
- ✅ Tokens in environment variables

**Phase 3 (Department)**: JWT auth (Month 2)
- ✅ 4-hour upgrade
- ✅ Proper expiration and refresh
- ✅ API integration support

**Phase 4 (Enterprise)**: OAuth2/SSO (Month 3-4)
- ✅ 8-hour integration
- ✅ Corporate identity provider
- ✅ Compliance and audit ready

---

## Sample Code: Complete Token Auth Implementation

Here's a **complete, production-ready token auth** you can drop in after the demo:

```python
# config/auth_config.json
{
    "tokens": {
        "demo_abc123": {"username": "demo", "role": "viewer", "expires": null},
        "admin_xyz789": {"username": "admin", "role": "admin", "expires": null}
    },
    "session_timeout_minutes": 60,
    "enable_audit_log": true
}
```

```python
# auth.py
import streamlit as st
import json
from pathlib import Path
from datetime import datetime, timedelta

CONFIG = Path("config/auth_config.json")
AUDIT_LOG = Path("logs/audit.log")

def load_config():
    """Load authentication configuration."""
    if CONFIG.exists():
        return json.loads(CONFIG.read_text())
    return {"tokens": {}, "session_timeout_minutes": 60, "enable_audit_log": False}

def audit_log(event_type: str, details: dict = None):
    """Log authentication events."""
    config = load_config()
    if not config.get("enable_audit_log", False):
        return

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "username": st.session_state.get("username", "anonymous"),
        "event": event_type,
        "details": details or {}
    }

    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")

def check_session_timeout():
    """Check if session has timed out."""
    config = load_config()
    timeout_minutes = config.get("session_timeout_minutes", 60)

    if "last_activity" in st.session_state:
        elapsed = datetime.now() - st.session_state.last_activity
        if elapsed > timedelta(minutes=timeout_minutes):
            audit_log("session_timeout", {"elapsed_minutes": elapsed.total_seconds() / 60})
            logout()
            st.rerun()

    st.session_state.last_activity = datetime.now()

def check_authentication():
    """Main authentication check."""

    # Check if already authenticated
    if st.session_state.get("authenticated", False):
        check_session_timeout()
        return True

    # Load valid tokens
    config = load_config()
    valid_tokens = config.get("tokens", {})

    # Check URL parameter
    query_params = st.query_params
    token = query_params.get("token", None)

    if token and token in valid_tokens:
        token_info = valid_tokens[token]

        # Successful authentication
        st.session_state.authenticated = True
        st.session_state.username = token_info["username"]
        st.session_state.role = token_info["role"]
        st.session_state.last_activity = datetime.now()

        audit_log("login_success", {"role": token_info["role"]})
        return True

    # Authentication required
    st.error("🔒 Authentication Required")
    st.markdown("""
    This dashboard requires authentication. Please contact your administrator for an access token.

    **Access URL Format**: `http://localhost:8501?token=YOUR_TOKEN_HERE`
    """)

    if token:
        audit_log("login_failed", {"token_prefix": token[:8] + "..."})

    st.stop()

def logout():
    """Clear authentication session."""
    audit_log("logout", {})

    for key in list(st.session_state.keys()):
        del st.session_state[key]

def show_user_info():
    """Display user information in sidebar."""
    if st.session_state.get("authenticated", False):
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 👤 User Session")
            st.markdown(f"**User**: {st.session_state.username}")
            st.markdown(f"**Role**: {st.session_state.role}")

            if st.button("🚪 Logout", use_container_width=True):
                logout()
                st.rerun()

# Usage in tft_dashboard_web.py:
# from auth import check_authentication, show_user_info
# check_authentication()  # Add at top of file
# show_user_info()  # Add in sidebar
```

**Total Code**: ~150 lines
**Implementation Time**: ~2 hours
**Security Level**: Medium (good for internal use)

---

## Conclusion

**For your Tuesday demo**: Don't add authentication. It's localhost only and adds no value for presentation.

**For post-demo internal use**: Add simple token auth (2 hours). Easy, effective, good enough.

**For production**: Plan OAuth2/SSO integration (8 hours). Do this when you're ready for enterprise deployment.

**Bottom Line**: Authentication is not difficult to add. You can go from "no auth" to "production-ready token auth" in a single afternoon. Start simple, upgrade as needed.
