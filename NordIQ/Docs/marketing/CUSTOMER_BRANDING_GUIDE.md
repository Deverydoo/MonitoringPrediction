# Customer Branding Guide

**Version:** 1.0.0
**Last Updated:** October 29, 2025
**Purpose:** Customize dashboard branding for different customers

---

## Overview

The NordIQ Dashboard supports **full customer branding** with a simple configuration file. Change colors, logos, and header text without touching any dashboard code.

### What Can Be Customized

✅ **Header bar color** - Top navigation bar background
✅ **Accent colors** - Buttons, links, metrics, borders
✅ **Customer logo/name** - Displayed in header
✅ **Theme colors** - Streamlit theme integration
✅ **Sidebar styling** - Border accents and colors

---

## Quick Start

### Change to Wells Fargo Branding (Default)

**Already done!** Wells Fargo red branding is now active.

### Switch to Different Customer

**1. Edit branding configuration:**
```bash
# Open branding config file
notepad NordIQ/src/dashboard/Dashboard/config/branding_config.py
```

**2. Change active branding:**
```python
# Line 45-46
ACTIVE_BRANDING = 'wells_fargo'  # Change this value

# Options:
# 'nordiq' - NordIQ blue branding
# 'wells_fargo' - Wells Fargo red branding
# 'generic' - Professional gray branding
```

**3. Restart dashboard:**
```bash
daemon.bat restart dashboard
# Or
./daemon.sh restart dashboard
```

**That's it!** Dashboard will display new branding.

---

## Current Branding: Wells Fargo

### Colors

| Element | Color | Hex Code |
|---------|-------|----------|
| **Primary (Header)** | Wells Fargo Red | `#D71E28` |
| **Secondary (Hover)** | Dark Red | `#B71C1C` |
| **Background** | White | `#FFFFFF` |
| **Sidebar Accent** | Wells Fargo Red | `#D71E28` |
| **Text** | Dark Gray | `#262730` |

### Visual Elements

**Header Bar:**
- 🏛️ Wells Fargo logo emoji
- Wells Fargo Red background (`#D71E28`)
- White text
- 3px darker red bottom border

**Buttons:**
- Wells Fargo Red background
- Darker red on hover
- White text

**Metrics:**
- Wells Fargo Red numbers
- Emphasizes important values

**Sidebar:**
- 3px Wells Fargo Red right border
- Branded accent line

**Links:**
- Wells Fargo Red color
- Consistent brand integration

---

## Add New Customer Branding

### Step 1: Define Branding Profile

Edit `branding_config.py` and add new customer:

```python
BRANDING_PROFILES['your_customer'] = {
    'name': 'Your Customer Name',
    'primary_color': '#1E40AF',  # Main brand color
    'secondary_color': '#1E3A8A',  # Darker shade for hover
    'header_text': '🚀 Your Company',  # Header display text
    'logo_emoji': '🚀',  # Emoji logo (or use real logo path)
    'tagline': 'Your company tagline',
    'accent_border': '#1E40AF'  # Border accent color
}
```

### Step 2: Activate Branding

```python
ACTIVE_BRANDING = 'your_customer'
```

### Step 3: Restart Dashboard

```bash
daemon.bat restart dashboard
```

---

## Available Branding Profiles

### 1. NordIQ (Default)

**Colors:**
- Primary: NordIQ Blue (`#0EA5E9`)
- Secondary: Darker Blue (`#3B82F6`)

**Header:** 🧭 ArgusAI

**Tagline:** "Predictive System Monitoring"

**Activate:**
```python
ACTIVE_BRANDING = 'nordiq'
```

---

### 2. Wells Fargo (Active)

**Colors:**
- Primary: Wells Fargo Red (`#D71E28`)
- Secondary: Dark Red (`#B71C1C`)

**Header:** 🏛️ Wells Fargo

**Tagline:** "Established 1852"

**Activate:**
```python
ACTIVE_BRANDING = 'wells_fargo'
```

---

### 3. Generic Enterprise

**Colors:**
- Primary: Professional Gray (`#4B5563`)
- Secondary: Dark Gray (`#374151`)

**Header:** 📊 Enterprise Dashboard

**Tagline:** "Predictive Infrastructure Monitoring"

**Activate:**
```python
ACTIVE_BRANDING = 'generic'
```

---

## Enterprise Color Reference

The branding config includes common enterprise colors for quick reference:

### Financial Institutions

| Company | Color Name | Hex Code |
|---------|-----------|----------|
| Wells Fargo | Red | `#D71E28` |
| Bank of America | Red | `#E31837` |
| Chase | Blue | `#117ACA` |
| Citibank | Blue | `#056DAE` |
| Capital One | Red | `#DB0011` |

### Tech Companies

| Company | Color Name | Hex Code |
|---------|-----------|----------|
| Microsoft | Blue | `#00A4EF` |
| Google | Blue | `#4285F4` |
| Amazon | Orange | `#FF9900` |
| IBM | Blue | `#0F62FE` |
| Oracle | Red | `#F80000` |

### Healthcare

| Company | Color Name | Hex Code |
|---------|-----------|----------|
| UnitedHealthcare | Blue | `#002677` |
| Anthem | Blue | `#003DA5` |
| Humana | Green | `#00A758` |

---

## Advanced Customization

### Add Real Logo Image

**1. Add logo file:**
```bash
# Place logo in assets folder
NordIQ/src/dashboard/Dashboard/assets/customer_logo.png
```

**2. Update branding config:**
```python
BRANDING_PROFILES['your_customer'] = {
    ...
    'logo_path': 'Dashboard/assets/customer_logo.png',
    'logo_width': '150px',
    ...
}
```

**3. Update CSS generator:**
```python
# In get_custom_css() function, modify header styling:
header[data-testid="stHeader"]::before {{
    content: url({branding['logo_path']});
    width: {branding['logo_width']};
    ...
}}
```

---

### Custom Fonts

**1. Add font to config:**
```python
BRANDING_PROFILES['your_customer'] = {
    ...
    'font_family': 'Arial, Helvetica, sans-serif',
    ...
}
```

**2. Update Streamlit theme:**

Edit `NordIQ/.streamlit/config.toml`:
```toml
[theme]
font = "Arial"  # Change to customer font
```

---

### Multiple Accent Colors

**1. Define color palette:**
```python
BRANDING_PROFILES['your_customer'] = {
    ...
    'colors': {
        'critical': '#DC2626',
        'warning': '#F59E0B',
        'success': '#10B981',
        'info': '#3B82F6'
    },
    ...
}
```

**2. Use in CSS:**
```python
def get_custom_css():
    branding = get_active_branding()
    colors = branding.get('colors', {})

    css = f"""
    /* Critical alerts */
    .alert-critical {{
        background-color: {colors['critical']} !important;
    }}
    ...
    """
```

---

## Configuration Files

### 1. branding_config.py (Primary)

**Location:** `NordIQ/src/dashboard/Dashboard/config/branding_config.py`

**Purpose:**
- Define branding profiles
- Set active customer
- Generate custom CSS
- Helper functions

**Edit this file to:**
- Add new customers
- Change active branding
- Modify color schemes

---

### 2. config.toml (Secondary)

**Location:** `NordIQ/.streamlit/config.toml`

**Purpose:**
- Streamlit theme settings
- Server configuration
- Global dashboard config

**Edit this file to:**
- Change default theme colors
- Adjust fonts
- Configure server settings

**Note:** `branding_config.py` overrides these settings with CSS.

---

## Testing Branding Changes

### Visual Checklist

After changing branding:

- [ ] **Header bar** - Correct color and logo
- [ ] **Buttons** - Branded color, hover works
- [ ] **Sidebar** - Accent border visible
- [ ] **Metrics** - Numbers in brand color
- [ ] **Links** - Clickable, brand color
- [ ] **Tab indicators** - Active tab highlighted
- [ ] **Charts** - Colors readable on new background

### Quick Test Script

```python
# test_branding.py
from Dashboard.config.branding_config import get_active_branding, get_custom_css

branding = get_active_branding()
print(f"Active Branding: {branding['name']}")
print(f"Primary Color: {branding['primary_color']}")
print(f"Header Text: {branding['header_text']}")

css = get_custom_css()
print(f"CSS Generated: {len(css)} characters")
```

---

## Troubleshooting

### Issue: Branding doesn't change after restart

**Solution:**
```bash
# 1. Clear browser cache (Ctrl+Shift+R)
# 2. Hard restart dashboard
daemon.bat stop
daemon.bat start

# 3. Check branding config syntax
python -m py_compile NordIQ/src/dashboard/Dashboard/config/branding_config.py
```

---

### Issue: Colors look wrong

**Cause:** CSS specificity issues

**Solution:**
- Add `!important` to CSS rules
- Check browser console for CSS errors (F12)
- Verify hex color codes are valid

---

### Issue: Header logo doesn't show

**Cause:** CSS `::before` content not rendering

**Solution:**
```python
# Use emoji instead of image for now
'header_text': '🏛️ Company Name',

# Or add logo via HTML
st.markdown('<img src="logo.png" width="150">', unsafe_allow_html=True)
```

---

## Best Practices

### 1. Test on Multiple Browsers

- ✅ Chrome/Edge (most common)
- ✅ Firefox
- ✅ Safari (if deploying to Mac users)

### 2. Use Official Brand Colors

- Get hex codes from customer brand guidelines
- Don't guess or approximate
- Test readability (white text on brand color)

### 3. Keep It Simple

- One primary brand color
- One secondary (darker for hover)
- Don't overdo customization

### 4. Document Customer Branding

```python
# Add comments to branding config
BRANDING_PROFILES['customer'] = {
    # Brand Guidelines: https://customer.com/brand
    # Approved: 2025-10-29
    # Contact: marketing@customer.com
    ...
}
```

### 5. Version Control

```bash
# Commit branding changes
git add NordIQ/src/dashboard/Dashboard/config/branding_config.py
git commit -m "feat: add [Customer] branding"
git push
```

---

## Multi-Tenant Deployment

For serving multiple customers from one installation:

### Option A: Environment Variable

```python
# branding_config.py
import os

ACTIVE_BRANDING = os.getenv('CUSTOMER_BRANDING', 'nordiq')
```

**Deploy:**
```bash
# Customer 1
export CUSTOMER_BRANDING=wells_fargo
streamlit run dashboard.py --server.port 8501

# Customer 2
export CUSTOMER_BRANDING=chase
streamlit run dashboard.py --server.port 8502
```

---

### Option B: URL Parameter

```python
# tft_dashboard_web.py
import streamlit as st

# Get customer from URL: ?customer=wells_fargo
params = st.query_params
customer = params.get('customer', 'nordiq')

# Load branding
from Dashboard.config.branding_config import BRANDING_PROFILES
branding = BRANDING_PROFILES.get(customer, BRANDING_PROFILES['nordiq'])
```

**Access:**
```
http://localhost:8501?customer=wells_fargo
http://localhost:8501?customer=chase
```

---

### Option C: Subdomain Routing

```nginx
# nginx.conf
server {
    server_name wellsfargo.dashboard.com;
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header X-Customer "wells_fargo";
    }
}

server {
    server_name chase.dashboard.com;
    location / {
        proxy_pass http://localhost:8502;
        proxy_set_header X-Customer "chase";
    }
}
```

---

## Examples

### Example 1: Simple Color Change

```python
# Add to branding_config.py
BRANDING_PROFILES['acme'] = {
    'name': 'ACME Corp',
    'primary_color': '#FF6B35',  # Bright orange
    'secondary_color': '#E55527',  # Darker orange
    'header_text': '🚀 ACME Corporation',
    'logo_emoji': '🚀',
    'tagline': 'Innovation Delivered',
    'accent_border': '#FF6B35'
}

# Activate
ACTIVE_BRANDING = 'acme'
```

**Result:** Orange-branded dashboard with rocket emoji

---

### Example 2: Professional Banking

```python
BRANDING_PROFILES['first_national'] = {
    'name': 'First National Bank',
    'primary_color': '#003366',  # Deep navy blue
    'secondary_color': '#002244',  # Darker navy
    'header_text': '🏦 First National Bank',
    'logo_emoji': '🏦',
    'tagline': 'Banking Since 1890',
    'accent_border': '#003366'
}

ACTIVE_BRANDING = 'first_national'
```

**Result:** Conservative, trustworthy blue branding

---

### Example 3: Tech Startup

```python
BRANDING_PROFILES['techstartup'] = {
    'name': 'CloudScale',
    'primary_color': '#8B5CF6',  # Purple
    'secondary_color': '#7C3AED',  # Darker purple
    'header_text': '☁️ CloudScale',
    'logo_emoji': '☁️',
    'tagline': 'Scale Without Limits',
    'accent_border': '#8B5CF6'
}

ACTIVE_BRANDING = 'techstartup'
```

**Result:** Modern, vibrant purple branding

---

## Support

### Documentation
- [branding_config.py](../NordIQ/src/dashboard/Dashboard/config/branding_config.py) - Source code with examples
- [config.toml](../NordIQ/.streamlit/config.toml) - Streamlit theme settings

### Quick Reference

**Change branding:**
1. Edit `ACTIVE_BRANDING` in `branding_config.py`
2. Run `daemon.bat restart dashboard`
3. Refresh browser (Ctrl+Shift+R)

**Add new customer:**
1. Add profile to `BRANDING_PROFILES` dict
2. Set `ACTIVE_BRANDING` to new profile name
3. Restart dashboard

**Test changes:**
```bash
python -c "from Dashboard.config.branding_config import get_active_branding; print(get_active_branding())"
```

---

## Appendix: CSS Classes Reference

### Streamlit Elements

| Element | CSS Selector | Purpose |
|---------|-------------|----------|
| Header bar | `header[data-testid="stHeader"]` | Top navigation |
| Sidebar | `section[data-testid="stSidebar"]` | Left panel |
| Metrics | `div[data-testid="stMetricValue"]` | KPI numbers |
| Buttons | `button[kind="primary"]` | Action buttons |
| Tabs | `button[data-baseweb="tab"]` | Tab navigation |
| Links | `a` | All hyperlinks |

### Custom Classes

Add custom classes in your dashboard code:

```python
st.markdown('<div class="custom-card">Content</div>', unsafe_allow_html=True)
```

Then style in branding config:

```python
css = f"""
.custom-card {{
    background-color: {branding['primary_color']};
    padding: 20px;
    border-radius: 10px;
}}
"""
```

---

**Document Version:** 1.0.0
**Last Updated:** October 29, 2025
**Company:** ArgusAI, LLC
