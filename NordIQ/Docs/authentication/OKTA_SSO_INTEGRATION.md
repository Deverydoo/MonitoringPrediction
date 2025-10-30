# Okta SSO Integration for TFT Dashboard

**Current Environment**: Corporate Okta SSO with passthrough authentication
**Integration Difficulty**: ⭐⭐⭐ Moderate (4-6 hours)
**Benefit**: Seamless single sign-on, no separate passwords, corporate identity management

---

## How Okta SSO Works

**User Experience**:
1. User navigates to dashboard URL: `https://monitoring.company.com`
2. Not authenticated → redirects to Okta login page
3. User logs in with corporate credentials (or already authenticated via SSO)
4. Okta redirects back to dashboard with authentication token
5. Dashboard validates token and grants access
6. **"Weird passthrough"**: If user already logged into other corporate apps, they get automatic access (no login prompt!)

**Technical Flow**:
```
User → Dashboard → Okta Login → Corporate Identity Provider → Okta → Dashboard
         ↓                                                              ↓
    (needs auth)                                                  (authenticated)
```

---

## Integration Approach Options

### Option A: SAML 2.0 (Most Common for Okta)
**Difficulty**: ⭐⭐⭐ Moderate
**Best For**: Traditional enterprise SSO

### Option B: OpenID Connect (OIDC)
**Difficulty**: ⭐⭐⭐ Moderate
**Best For**: Modern applications, REST APIs

### Option C: Reverse Proxy (Nginx/Apache + mod_auth_openidc) ⭐ RECOMMENDED
**Difficulty**: ⭐⭐ Easy-Moderate
**Best For**: No code changes to dashboard, transparent authentication

---

## Recommended Approach: Reverse Proxy with Okta

This is the **easiest and most robust** approach because:
- ✅ No changes to Python dashboard code
- ✅ Proxy handles all authentication
- ✅ Dashboard just reads HTTP headers
- ✅ Can add auth to any application (inference daemon, etc.)
- ✅ Corporate IT likely already familiar with this setup

### Architecture:

```
Internet/Corporate Network
         ↓
    Nginx Reverse Proxy (with Okta auth module)
         ↓ (validates Okta token, injects headers)
    Streamlit Dashboard (reads headers)
```

### Implementation Steps:

#### Step 1: Okta Configuration (15 minutes)

Work with your IT team to create an Okta application:

**Okta Admin Console**:
1. Applications → Create App Integration
2. Sign-in method: **OIDC - OpenID Connect**
3. Application type: **Web Application**
4. App integration name: **TFT Monitoring Dashboard**
5. Sign-in redirect URIs: `https://monitoring.company.com/oauth2/callback`
6. Sign-out redirect URIs: `https://monitoring.company.com`
7. Assignments: Add appropriate groups (e.g., "Engineering", "Operations")

**You'll receive**:
- Client ID: `0oa1a2b3c4d5e6f7g8h9`
- Client Secret: `secretXYZ123...`
- Okta Domain: `company.okta.com`

#### Step 2: Install Nginx with OpenID Connect Module (30 minutes)

```bash
# Install nginx and dependencies
sudo apt-get update
sudo apt-get install nginx libnginx-mod-http-lua lua-resty-openidc

# Or on Windows (use WSL or Docker)
docker pull nginx:latest
```

#### Step 3: Configure Nginx (45 minutes)

**nginx.conf**:
```nginx
# /etc/nginx/sites-available/tft-dashboard

upstream streamlit_dashboard {
    server 127.0.0.1:8501;
}

upstream inference_daemon {
    server 127.0.0.1:8000;
}

# HTTPS server with Okta SSO
server {
    listen 443 ssl http2;
    server_name monitoring.company.com;

    # SSL certificates (from corporate IT)
    ssl_certificate /etc/ssl/certs/company.crt;
    ssl_certificate_key /etc/ssl/private/company.key;

    # Okta OpenID Connect configuration
    set $session_secret 'generate-random-secret-here';
    set $session_cookie_samesite 'Lax';

    # Okta discovery URL
    set $oidc_discovery https://company.okta.com/.well-known/openid-configuration;

    location / {
        # Require authentication via Okta
        access_by_lua_block {
            local opts = {
                discovery = "https://company.okta.com/.well-known/openid-configuration",
                client_id = "0oa1a2b3c4d5e6f7g8h9",
                client_secret = "secretXYZ123...",
                redirect_uri = "https://monitoring.company.com/oauth2/callback",
                scope = "openid email profile",
                session_contents = {id_token=true}
            }

            -- Authenticate with Okta
            local res, err = require("resty.openidc").authenticate(opts)

            if err then
                ngx.status = 403
                ngx.say("Authentication failed: ", err)
                ngx.exit(ngx.HTTP_FORBIDDEN)
            end

            -- Inject user info into headers for dashboard
            ngx.req.set_header("X-Auth-User", res.id_token.email)
            ngx.req.set_header("X-Auth-Username", res.id_token.preferred_username)
            ngx.req.set_header("X-Auth-Name", res.id_token.name)
            ngx.req.set_header("X-Auth-Groups", table.concat(res.id_token.groups or {}, ","))
        }

        # Proxy to Streamlit
        proxy_pass http://streamlit_dashboard;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Pass auth headers
        proxy_set_header X-Auth-User $http_x_auth_user;
        proxy_set_header X-Auth-Username $http_x_auth_username;
        proxy_set_header X-Auth-Name $http_x_auth_name;
        proxy_set_header X-Auth-Groups $http_x_auth_groups;

        # WebSocket support (for Streamlit)
        proxy_read_timeout 86400;
    }

    # OAuth callback endpoint
    location /oauth2/callback {
        # Handled by lua module
        access_by_lua_block {
            require("resty.openidc").authenticate({
                discovery = "https://company.okta.com/.well-known/openid-configuration",
                client_id = "0oa1a2b3c4d5e6f7g8h9",
                client_secret = "secretXYZ123...",
                redirect_uri = "https://monitoring.company.com/oauth2/callback"
            })
        }

        proxy_pass http://streamlit_dashboard;
    }

    # Logout endpoint
    location /logout {
        access_by_lua_block {
            local session = require("resty.session").open()
            session:destroy()
            ngx.redirect("https://company.okta.com/login/signout")
        }
    }
}

# HTTP redirect to HTTPS
server {
    listen 80;
    server_name monitoring.company.com;
    return 301 https://$server_name$request_uri;
}
```

#### Step 4: Update Dashboard to Read Headers (15 minutes)

The dashboard just needs to read the injected headers:

```python
# auth.py (simple version for proxy-based auth)
import streamlit as st

def get_authenticated_user():
    """
    Get authenticated user from proxy headers.
    Nginx/Apache with Okta auth injects these headers.
    """
    # Streamlit doesn't directly expose request headers
    # But we can use the headers query param approach
    # or run a simple Flask middleware

    # For now, check if running behind authenticated proxy
    # by checking for the X-Auth-User header via environment
    import os

    user_email = os.environ.get("HTTP_X_AUTH_USER")
    user_name = os.environ.get("HTTP_X_AUTH_NAME")
    user_groups = os.environ.get("HTTP_X_AUTH_GROUPS", "").split(",")

    if user_email:
        return {
            "email": user_email,
            "name": user_name,
            "groups": user_groups,
            "authenticated": True
        }

    # If no headers, we're not behind authenticated proxy
    # This could mean:
    # 1. Running locally without proxy (dev mode)
    # 2. Proxy misconfigured
    # 3. User bypassed proxy (security issue!)

    # For production, you'd fail here
    # For dev, allow local access
    if st.secrets.get("allow_local_dev", False):
        return {
            "email": "local@dev.local",
            "name": "Local Development",
            "groups": ["admin"],
            "authenticated": True
        }

    st.error("🔒 Authentication Error: Not authenticated via SSO proxy")
    st.stop()

def check_authentication():
    """Main authentication check."""
    if "user" not in st.session_state:
        user = get_authenticated_user()
        st.session_state.user = user
        st.session_state.authenticated = True

    return st.session_state.user

def show_user_info():
    """Display user information in sidebar."""
    user = st.session_state.get("user", {})

    with st.sidebar:
        st.markdown("---")
        st.markdown("### 👤 User Session")
        st.markdown(f"**Name**: {user.get('name', 'Unknown')}")
        st.markdown(f"**Email**: {user.get('email', 'Unknown')}")

        groups = user.get('groups', [])
        if groups:
            st.markdown(f"**Groups**: {', '.join(groups)}")

        if st.button("🚪 Logout", use_container_width=True):
            # Redirect to proxy logout endpoint
            st.markdown('<meta http-equiv="refresh" content="0; url=/logout">', unsafe_allow_html=True)

# Usage in tft_dashboard_web.py:
# from auth import check_authentication, show_user_info
# user = check_authentication()
# show_user_info()
```

#### Step 5: Role-Based Access Control (30 minutes)

Map Okta groups to dashboard roles:

```python
# auth.py (RBAC additions)

# Map Okta groups to dashboard roles
OKTA_GROUP_ROLES = {
    "TFT-Dashboard-Admins": "admin",
    "TFT-Dashboard-Operators": "operator",
    "Engineering-All": "viewer",
    "Operations-Team": "operator"
}

def get_user_role(user: dict) -> str:
    """Determine user role from Okta groups."""
    groups = user.get("groups", [])

    # Check in priority order
    if "TFT-Dashboard-Admins" in groups:
        return "admin"
    elif "TFT-Dashboard-Operators" in groups or "Operations-Team" in groups:
        return "operator"
    elif any(g in groups for g in ["Engineering-All", "Engineering"]):
        return "viewer"

    # Default role
    return "viewer"

def has_permission(permission: str) -> bool:
    """Check if user has permission based on role."""
    user = st.session_state.get("user", {})
    role = get_user_role(user)

    PERMISSIONS = {
        "admin": ["view", "configure", "remediate", "manage_users"],
        "operator": ["view", "configure", "remediate"],
        "viewer": ["view"]
    }

    return permission in PERMISSIONS.get(role, [])

def require_permission(permission: str):
    """Require permission for tab/feature."""
    if not has_permission(permission):
        user = st.session_state.get("user", {})
        role = get_user_role(user)

        st.error(f"🔒 Access Denied")
        st.warning(f"Your role ({role}) does not have '{permission}' permission.")
        st.info("Contact your administrator if you need access to this feature.")
        st.stop()
```

---

## Alternative: Simpler Approach Using streamlit-authenticator with Okta

If your IT team can provide Okta API access:

```python
# auth.py (Okta API approach)
import streamlit as st
import requests
from okta.client import Client as OktaClient
import asyncio

OKTA_DOMAIN = "company.okta.com"
OKTA_API_TOKEN = "YOUR_API_TOKEN"  # From IT

async def verify_okta_user(username: str, password: str) -> dict:
    """Verify user credentials against Okta."""
    config = {
        'orgUrl': f'https://{OKTA_DOMAIN}',
        'token': OKTA_API_TOKEN
    }

    client = OktaClient(config)

    # Authenticate user
    auth_response = await client.authenticate({
        'username': username,
        'password': password
    })

    if auth_response.status == 'SUCCESS':
        # Get user details
        user = await client.get_user(auth_response.embedded.user.id)

        return {
            'authenticated': True,
            'email': user.profile.email,
            'name': f"{user.profile.firstName} {user.profile.lastName}",
            'groups': [g.profile.name for g in await user.list_groups()]
        }

    return {'authenticated': False}

def check_authentication():
    """Check Okta authentication via API."""
    if st.session_state.get("authenticated", False):
        return True

    st.markdown("## 🔒 TFT Monitoring Dashboard - Okta Login")

    with st.form("okta_login"):
        username = st.text_input("Corporate Username/Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login with Okta")

        if submit:
            with st.spinner("Authenticating with Okta..."):
                result = asyncio.run(verify_okta_user(username, password))

                if result['authenticated']:
                    st.session_state.authenticated = True
                    st.session_state.user = result
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Authentication failed. Please check your credentials.")

    st.stop()
```

---

## Benefits of Okta SSO Integration

### For Users:
- ✅ **Single Sign-On**: Already logged in to corporate network = automatic access
- ✅ **No separate password**: Use same credentials as email, Slack, etc.
- ✅ **Familiar login experience**: Same Okta login page as other corporate apps
- ✅ **Automatic logout**: When Okta session ends, dashboard access ends

### For IT/Security:
- ✅ **Centralized user management**: Add/remove users in Okta, not dashboard
- ✅ **MFA enforcement**: If Okta requires MFA, dashboard inherits it
- ✅ **Audit logging**: All logins tracked in Okta audit logs
- ✅ **Password policies**: Corporate password policies automatically enforced
- ✅ **Automatic provisioning**: New hires get access via group membership
- ✅ **Instant deprovisioning**: Terminations immediately lose all access

### For You:
- ✅ **No password management**: IT handles everything
- ✅ **No user database**: Okta is source of truth
- ✅ **Compliance ready**: Inherits corporate security controls
- ✅ **Group-based RBAC**: Map Okta groups to dashboard roles

---

## Implementation Timeline

### Phase 1: IT Coordination (1 week)
- Request Okta app creation from IT
- Get client ID/secret and domain info
- Coordinate SSL certificates for HTTPS
- Test Okta app configuration

### Phase 2: Nginx Setup (1 day)
- Install nginx with OpenID Connect module
- Configure reverse proxy
- Test authentication flow
- Set up SSL/TLS

### Phase 3: Dashboard Updates (4 hours)
- Add header reading code
- Implement RBAC based on groups
- Add user info display
- Test role permissions

### Phase 4: Testing (1 day)
- Test with different user accounts
- Verify group-based permissions
- Test logout flow
- Security testing

### Phase 5: Production Deployment (1 day)
- Deploy to production server
- Update DNS/networking
- Monitor for issues
- User training

**Total Time**: ~2 weeks elapsed (mostly waiting on IT), ~2 days actual work

---

## Troubleshooting Common Issues

### Issue 1: "Weird passthrough" not working
**Problem**: User has to log in every time, not automatically authenticated

**Cause**: Session cookie not being preserved

**Fix**:
```nginx
# In nginx.conf
set $session_cookie_samesite 'Lax';  # Not 'Strict'
set $session_cookie_lifetime 28800;  # 8 hours
```

### Issue 2: Dashboard shows "not authenticated" but user logged in
**Problem**: Headers not being passed to Streamlit

**Cause**: Proxy configuration issue

**Fix**:
```nginx
# Ensure these are set
proxy_set_header X-Auth-User $http_x_auth_user;
proxy_pass_request_headers on;
```

### Issue 3: WebSocket connections failing
**Problem**: Real-time updates not working

**Cause**: WebSocket upgrade not configured

**Fix**:
```nginx
proxy_http_version 1.1;
proxy_set_header Upgrade $http_upgrade;
proxy_set_header Connection "upgrade";
proxy_read_timeout 86400;
```

---

## Development vs. Production

### Development (localhost):
```python
# .streamlit/secrets.toml
allow_local_dev = true

# Dashboard runs without proxy
# Simulates authenticated user
```

### Production (corporate network):
```python
# .streamlit/secrets.toml
allow_local_dev = false
require_proxy_auth = true

# Dashboard requires nginx proxy with Okta
# Real authentication enforced
```

---

## Cost & Maintenance

**One-Time Setup**:
- IT coordination: 4 hours (spread over 1 week)
- Nginx configuration: 4 hours
- Dashboard updates: 4 hours
- Testing: 4 hours
- **Total**: ~16 hours (~2 days of actual work)

**Ongoing Maintenance**:
- Okta config: Managed by IT (zero effort for you)
- User management: Managed by IT (zero effort for you)
- Nginx updates: ~1 hour/quarter
- Dashboard updates: No changes needed (headers remain same)

**Licensing**:
- Okta: Already paid by company ✅
- Nginx: Free (open source) ✅
- SSL certificates: Provided by IT ✅
- **Total additional cost**: $0

---

## Security Benefits

Compared to token/password auth:

| Security Feature | Token Auth | Okta SSO |
|-----------------|------------|----------|
| **MFA Support** | Manual | ✅ Automatic |
| **Password Rotation** | Manual | ✅ Automatic |
| **Instant Deprovisioning** | Manual | ✅ Automatic |
| **Audit Logging** | Manual | ✅ Built-in |
| **Compliance** | Your problem | ✅ IT's problem |
| **Single Sign-On** | ❌ No | ✅ Yes |
| **Password Reuse Prevention** | ❌ No | ✅ Yes |
| **Session Management** | Manual | ✅ Automatic |

---

## Recommendation

**For Tuesday Demo**: No auth needed (localhost only)

**Post-Demo (Next 2 Weeks)**: Coordinate with IT to set up Okta SSO
- ✅ Leverage existing corporate infrastructure
- ✅ No password management burden
- ✅ Professional, enterprise-grade security
- ✅ Makes dashboard "production ready"
- ✅ Your IT team likely already has nginx/Apache templates for this

**Pro Tip**: Frame it to IT as "we need SSO for our new monitoring dashboard, just like Jira/Confluence/Slack" - they'll know exactly what to do. This is a standard request they handle regularly.

---

## Next Steps

1. **After Tuesday demo**, email IT with:
   - Dashboard URL/hostname you want: `monitoring.company.com`
   - Application name: "TFT Monitoring Dashboard"
   - Request: "Okta SSO integration with OpenID Connect"
   - Okta groups that need access: List your team groups

2. IT will respond with:
   - Client ID
   - Client Secret
   - Okta domain
   - Any specific configuration requirements

3. You implement nginx proxy (4 hours)

4. IT tests and approves

5. Done! Everyone uses corporate credentials to access dashboard

**Bottom Line**: Since you already have Okta, this is the "proper" way to secure the dashboard for corporate use. Much better than managing separate passwords/tokens.
