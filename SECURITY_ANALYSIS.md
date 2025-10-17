# Security Analysis - TFT Inference Daemon

## Current Security Posture: ⚠️ DEVELOPMENT MODE

The inference daemon is currently configured for **local development and trusted network environments**. It lacks several production security features.

---

## Security Issues Identified

### 🔴 CRITICAL

#### 1. No Authentication
**Issue**: All endpoints are publicly accessible without authentication
```python
# tft_inference_daemon.py:1178-1249
@app.post("/feed/data")  # NO AUTH CHECK
@app.get("/predictions/current")  # NO AUTH CHECK
@app.get("/alerts/active")  # NO AUTH CHECK
```

**Risk**: Anyone with network access can:
- Feed malicious data to the model
- Extract predictions (potential IP theft)
- DoS the service with spam requests
- View sensitive server metrics

**Severity**: **CRITICAL** for production deployment

---

#### 2. CORS Wildcard (`allow_origins=["*"]`)
**Issue**: Accepts requests from ANY origin
```python
# tft_inference_daemon.py:1151-1157
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ← INSECURE!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Risk**:
- Cross-Site Request Forgery (CSRF) attacks
- Data exfiltration via malicious websites
- Unauthorized API calls from any domain

**Severity**: **CRITICAL** for internet-facing deployments

---

#### 3. No Rate Limiting
**Issue**: No throttling on endpoints

**Risk**:
- Denial of Service (DoS) attacks
- Resource exhaustion
- API abuse (unlimited predictions/data feeds)

**Severity**: **HIGH**

---

### 🟠 HIGH

#### 4. No Input Validation Beyond Schema
**Issue**: Pydantic validates structure but not content
```python
class FeedDataRequest(BaseModel):
    records: List[Dict[str, Any]]  # Any dict accepted!
```

**Risk**:
- Malicious data injection
- Type confusion attacks
- Buffer overflow via extremely large inputs

**Severity**: **HIGH**

---

#### 5. Pickle Deserialization
**Issue**: Uses pickle to load state (arbitrary code execution vector)
```python
# tft_inference_daemon.py:1057-1096
with open(self.persistence_file, 'rb') as f:
    state = pickle.load(f)  # ← DANGEROUS!
```

**Risk**:
- Arbitrary code execution if attacker replaces pickle file
- Remote Code Execution (RCE) if pickle file is network-accessible

**Severity**: **HIGH**

---

#### 6. Listens on `0.0.0.0` (All Interfaces)
**Issue**: Binds to all network interfaces
```python
# tft_inference_daemon.py:1307-1313
uvicorn.run(
    app,
    host="0.0.0.0",  # ← Exposed to network!
    port=args.port,
)
```

**Risk**:
- Accessible from any network interface
- Internet-facing if deployed on public server

**Severity**: **HIGH** (production deployments)

---

### 🟡 MEDIUM

#### 7. No TLS/HTTPS
**Issue**: Runs on plain HTTP

**Risk**:
- Man-in-the-middle (MITM) attacks
- Data interception (predictions, metrics)
- Credential sniffing (if auth added)

**Severity**: **MEDIUM** (depends on network trust)

---

#### 8. No Request Size Limits
**Issue**: No explicit limits on request body size

**Risk**:
- Memory exhaustion attacks
- Slowloris-style DoS

**Severity**: **MEDIUM**

---

#### 9. Error Messages Expose Internals
**Issue**: Stack traces and file paths in error responses
```python
# tft_inference_daemon.py:238-242
except Exception as e:
    print(f"[ERROR] Error loading model: {e}")
    import traceback
    traceback.print_exc()  # ← Exposes internals
```

**Risk**:
- Information disclosure
- Aids attackers in reconnaissance

**Severity**: **LOW-MEDIUM**

---

#### 10. No Audit Logging
**Issue**: No logging of API access or security events

**Risk**:
- Can't detect or investigate attacks
- No compliance trail

**Severity**: **MEDIUM** (compliance-critical environments)

---

## Security Recommendations

### 🔥 IMMEDIATE (Production Blockers)

#### 1. Add Authentication
**Recommended**: API Key authentication (simple) or OAuth2 (enterprise)

**Implementation** (API Key example):
```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("TFT_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key"
        )
    return api_key

@app.post("/feed/data")
async def feed_data(request: FeedDataRequest, api_key: str = Depends(verify_api_key)):
    # Protected endpoint
    ...
```

**Setup**:
```bash
# Set environment variable
export TFT_API_KEY="your-secure-random-key-here"

# Or use systemd service file
Environment="TFT_API_KEY=your-secure-random-key-here"
```

---

#### 2. Fix CORS Wildcard
**Replace**:
```python
allow_origins=["*"]
```

**With**:
```python
allow_origins=[
    "http://localhost:8501",  # Dashboard
    "https://your-domain.com"  # Production domain
]
```

**Configuration-based** (recommended):
```python
from config.api_config import API_CONFIG

allow_origins=API_CONFIG.get('cors_origins', ['http://localhost:8501'])
```

---

#### 3. Add Rate Limiting
**Install**: `slowapi`
```bash
pip install slowapi
```

**Implementation**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/feed/data")
@limiter.limit("100/minute")  # Max 100 requests per minute
async def feed_data(request: Request, data: FeedDataRequest):
    ...

@app.get("/predictions/current")
@limiter.limit("60/minute")  # Max 60 predictions per minute
async def get_predictions(request: Request):
    ...
```

---

#### 4. Replace Pickle with JSON
**Change**:
```python
# OLD (INSECURE)
with open(self.persistence_file, 'rb') as f:
    state = pickle.load(f)
```

**To**:
```python
# NEW (SECURE)
import json

def _save_state(self):
    state = {
        'rolling_window': [self._serialize_record(r) for r in self.rolling_window],
        'tick_count': self.tick_count,
        'server_timesteps': self.server_timesteps,
        'timestamp': datetime.now().isoformat()
    }

    with open(self.persistence_file, 'w') as f:
        json.dump(state, f)

def _load_state(self):
    with open(self.persistence_file, 'r') as f:
        state = json.load(f)

    self.rolling_window = deque(state['rolling_window'], maxlen=WINDOW_SIZE)
    # ...
```

**Note**: If you need binary data, use **messagepack** instead of pickle.

---

#### 5. Bind to Localhost Only (for local-only deployments)
**Change**:
```python
host="0.0.0.0"  # All interfaces
```

**To**:
```python
host="127.0.0.1"  # Localhost only
```

**Or make configurable**:
```python
host=os.getenv("TFT_BIND_HOST", "127.0.0.1")
```

---

### 🛡️ RECOMMENDED (Production Hardening)

#### 6. Add TLS/HTTPS
**Option A**: Use reverse proxy (nginx/apache) - **RECOMMENDED**
```nginx
server {
    listen 443 ssl http2;
    server_name tft-api.your-domain.com;

    ssl_certificate /etc/ssl/certs/tft-api.crt;
    ssl_certificate_key /etc/ssl/private/tft-api.key;

    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Option B**: Use uvicorn SSL (development only)
```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8443,
    ssl_keyfile="/path/to/key.pem",
    ssl_certfile="/path/to/cert.pem"
)
```

---

#### 7. Add Request Size Limits
**Uvicorn built-in**:
```python
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,
    limit_max_requests=10000,  # Max requests before restart
    limit_concurrency=100,      # Max concurrent connections
    timeout_keep_alive=30       # Keep-alive timeout
)
```

**FastAPI middleware**:
```python
from starlette.middleware.base import BaseHTTPMiddleware

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request, call_next):
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > self.max_size:
                return Response(
                    "Request body too large",
                    status_code=413
                )
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware, max_size=10 * 1024 * 1024)
```

---

#### 8. Sanitize Error Messages
**Add production error handler**:
```python
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log full error server-side
    print(f"[ERROR] {exc}")
    import traceback
    traceback.print_exc()

    # Return generic error to client (production)
    if os.getenv("ENV") == "production":
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    else:
        # Development: show details
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)}
        )
```

---

#### 9. Add Audit Logging
**Implementation**:
```python
import logging
from fastapi import Request

# Configure security logger
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/security.log")
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
security_logger.addHandler(handler)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log all API access
    security_logger.info(
        f"API Access: {request.method} {request.url.path} "
        f"from {request.client.host} "
        f"user-agent: {request.headers.get('user-agent')}"
    )

    response = await call_next(request)

    # Log failed auth attempts
    if response.status_code == 403:
        security_logger.warning(
            f"Auth failed: {request.method} {request.url.path} "
            f"from {request.client.host}"
        )

    return response
```

---

#### 10. Input Validation Hardening
**Add strict validation**:
```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict
from datetime import datetime

class MetricRecord(BaseModel):
    timestamp: datetime
    server_name: str = Field(..., regex=r'^[a-zA-Z0-9\-_]{1,50}$')
    cpu_user_pct: float = Field(..., ge=0, le=100)
    cpu_sys_pct: float = Field(..., ge=0, le=100)
    # ... all LINBORG metrics with validation

    @validator('server_name')
    def validate_server_name(cls, v):
        # Only allow known server patterns
        if not v.startswith(('ppml', 'ppdb', 'ppweb', 'ppcon', 'ppetl', 'pprisk')):
            raise ValueError(f"Invalid server name pattern: {v}")
        return v

class FeedDataRequest(BaseModel):
    records: List[MetricRecord] = Field(..., min_items=1, max_items=100)

    @validator('records')
    def validate_record_count(cls, v):
        if len(v) > 100:
            raise ValueError("Too many records (max 100 per request)")
        return v
```

---

## Security Checklist for Production

### Pre-Deployment

- [ ] **Authentication enabled** (API key minimum, OAuth2 recommended)
- [ ] **CORS restricted** to known origins only
- [ ] **Rate limiting** configured (per-IP, per-API-key)
- [ ] **Pickle replaced** with JSON/MessagePack
- [ ] **TLS/HTTPS enabled** (via reverse proxy or uvicorn)
- [ ] **Bind address** restricted (127.0.0.1 if behind proxy)
- [ ] **Request size limits** enforced
- [ ] **Error messages** sanitized for production
- [ ] **Audit logging** enabled
- [ ] **Input validation** hardened

### Network Security

- [ ] **Firewall rules** configured (allow only necessary ports)
- [ ] **Reverse proxy** deployed (nginx/apache with WAF)
- [ ] **VPN/VPC** for internal API access
- [ ] **DDoS protection** (Cloudflare, AWS Shield, etc.)
- [ ] **Network segmentation** (isolate inference daemon)

### System Security

- [ ] **Service user** (non-root, limited permissions)
- [ ] **File permissions** (640 for configs, 750 for scripts)
- [ ] **Systemd hardening** (PrivateTmp, NoNewPrivileges, ProtectSystem)
- [ ] **SELinux/AppArmor** policies (if applicable)
- [ ] **Resource limits** (memory, CPU, file descriptors)

### Operational Security

- [ ] **Secret management** (environment variables, not hardcoded)
- [ ] **Log rotation** configured
- [ ] **Monitoring** (failed auth attempts, rate limit violations)
- [ ] **Incident response** plan
- [ ] **Backup procedures** (model, config, data)
- [ ] **Patch management** (regular updates)

### Compliance

- [ ] **Data retention** policy
- [ ] **Privacy compliance** (GDPR, HIPAA, etc. if applicable)
- [ ] **Audit trail** retention
- [ ] **Encryption at rest** (if storing sensitive data)
- [ ] **Encryption in transit** (TLS 1.2+)

---

## Deployment Modes

### Mode 1: Development (Current)
**Security**: Minimal
**Use case**: Local development, trusted network
**Configuration**:
```python
host = "127.0.0.1"  # Localhost only
allow_origins = ["http://localhost:8501"]
# No auth (trusted local environment)
```

---

### Mode 2: Internal Corporate Network
**Security**: Medium
**Use case**: Behind corporate firewall, internal users
**Configuration**:
```python
host = "0.0.0.0"  # All interfaces (behind firewall)
allow_origins = ["https://dashboard.internal.corp"]
# API key authentication
# TLS via reverse proxy
# Rate limiting
```

---

### Mode 3: Internet-Facing Production
**Security**: High
**Use case**: Public API, untrusted clients
**Configuration**:
```python
host = "127.0.0.1"  # Behind reverse proxy only
allow_origins = ["https://dashboard.your-domain.com"]
# OAuth2 + API key authentication
# TLS mandatory (TLS 1.3 only)
# Strict rate limiting
# WAF (Web Application Firewall)
# DDoS protection
```

---

## Quick Security Fixes (Copy-Paste Ready)

### 1. Add API Key Auth
Create `security.py`:
```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    valid_key = os.getenv("TFT_API_KEY")
    if not valid_key:
        raise HTTPException(status_code=500, detail="API key not configured")
    if api_key != valid_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key
```

Update endpoints:
```python
from security import verify_api_key
from fastapi import Depends

@app.post("/feed/data")
async def feed_data(request: FeedDataRequest, api_key: str = Depends(verify_api_key)):
    # Protected!
    return daemon.feed_data(request.records)
```

---

### 2. Fix CORS
Update `tft_inference_daemon.py` line 1153:
```python
# OLD
allow_origins=["*"],

# NEW
allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501").split(","),
```

Set environment:
```bash
export CORS_ORIGINS="http://localhost:8501,https://dashboard.prod.com"
```

---

### 3. Add Rate Limiting
```bash
pip install slowapi
```

Add to `tft_inference_daemon.py`:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/feed/data")
@limiter.limit("100/minute")
async def feed_data(request: Request, data: FeedDataRequest):
    ...
```

---

## Summary

### Current State: ⚠️ **INSECURE FOR PRODUCTION**

| Security Feature | Status | Priority |
|------------------|--------|----------|
| Authentication | ❌ None | 🔴 CRITICAL |
| CORS | ❌ Wildcard (`*`) | 🔴 CRITICAL |
| Rate Limiting | ❌ None | 🔴 CRITICAL |
| Input Validation | ⚠️ Basic | 🟠 HIGH |
| Pickle Security | ❌ Vulnerable | 🟠 HIGH |
| Network Binding | ⚠️ All interfaces | 🟠 HIGH |
| TLS/HTTPS | ❌ Plain HTTP | 🟡 MEDIUM |
| Request Limits | ❌ None | 🟡 MEDIUM |
| Error Handling | ⚠️ Exposes internals | 🟡 MEDIUM |
| Audit Logging | ❌ None | 🟡 MEDIUM |

### Recommended Immediate Actions

1. ✅ **Add API key authentication** (30 minutes)
2. ✅ **Fix CORS wildcard** (5 minutes)
3. ✅ **Add rate limiting** (15 minutes)
4. ✅ **Replace pickle with JSON** (1 hour)
5. ✅ **Deploy behind nginx with TLS** (2 hours)

**Total effort**: ~4 hours to production-ready security

---

## Related Files

- **Daemon**: [tft_inference_daemon.py](tft_inference_daemon.py)
- **Systemd Service**: [systemd/tft-inference-daemon.service](systemd/tft-inference-daemon.service)
- **Production Guide**: [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)

---

**Need help implementing?** See the copy-paste code examples above, or ask for a complete secured version!
