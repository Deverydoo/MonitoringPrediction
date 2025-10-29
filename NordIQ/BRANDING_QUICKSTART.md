# Customer Branding Quick Start

**One-page guide to changing dashboard branding**

---

## Current Branding: Wells Fargo Red 🏛️

The dashboard now displays **Wells Fargo red branding** with:
- Red header bar (#D71E28)
- Wells Fargo logo in header
- Red buttons and links
- Red sidebar accent

---

## Change to Different Customer

### Step 1: Edit Config File

```bash
notepad NordIQ\src\dashboard\Dashboard\config\branding_config.py
```

### Step 2: Change Active Branding

Find line ~45 and change:

```python
ACTIVE_BRANDING = 'wells_fargo'

# Change to:
ACTIVE_BRANDING = 'nordiq'  # NordIQ blue
# Or:
ACTIVE_BRANDING = 'generic'  # Professional gray
```

### Step 3: Restart Dashboard

```bash
daemon.bat restart dashboard
```

### Step 4: Refresh Browser

Press `Ctrl + Shift + R` to hard refresh.

---

## Available Brandings

| Name | Colors | Logo |
|------|--------|------|
| **wells_fargo** | Red (#D71E28) | 🏛️ Wells Fargo |
| **nordiq** | Blue (#0EA5E9) | 🧭 NordIQ AI |
| **generic** | Gray (#4B5563) | 📊 Enterprise |

---

## Add New Customer

### Quick Template

Edit `branding_config.py` and add:

```python
BRANDING_PROFILES['your_company'] = {
    'name': 'Your Company',
    'primary_color': '#1E40AF',  # Your brand color
    'secondary_color': '#1E3A8A',  # Darker shade
    'header_text': '🚀 Your Company',
    'logo_emoji': '🚀',
    'tagline': 'Your tagline',
    'accent_border': '#1E40AF'
}

# Then activate it:
ACTIVE_BRANDING = 'your_company'
```

Restart dashboard and done!

---

## Common Brand Colors

**Financial:**
- Wells Fargo: `#D71E28`
- Chase: `#117ACA`
- Citi: `#056DAE`
- Bank of America: `#E31837`

**Tech:**
- Microsoft: `#00A4EF`
- Google: `#4285F4`
- Amazon: `#FF9900`
- IBM: `#0F62FE`

---

## Troubleshooting

**Branding doesn't change?**
1. Hard refresh browser (Ctrl+Shift+R)
2. Check file saved correctly
3. Restart: `daemon.bat stop` then `daemon.bat start`

**Colors look wrong?**
- Verify hex color code (must start with #)
- Check browser console for errors (F12)

---

**Full Documentation:** [CUSTOMER_BRANDING_GUIDE.md](../Docs/CUSTOMER_BRANDING_GUIDE.md)
