# NordIQ.io Website

**Nordic precision, AI intelligence**

Static website for NordIQ AI Systems - Predictive Infrastructure Monitoring SaaS

---

## 🚀 Quick Deployment

This is a static HTML/CSS/JavaScript website designed for Apache deployment.

### Local Testing

```bash
# Option 1: Python HTTP server
cd NordIQ-Website
python -m http.server 8000
# Visit: http://localhost:8000

# Option 2: PHP built-in server
cd NordIQ-Website
php -S localhost:8000

# Option 3: Use Live Server extension in VS Code
```

### Apache Deployment

1. **Copy files to Apache document root:**

```bash
# SSH into your server
ssh user@your-server.com

# Navigate to web root (adjust path as needed)
cd /var/www/

# Create directory for nordiqai.io
sudo mkdir -p nordiqai.io

# Copy files from local machine
# On your local machine:
scp -r NordIQ-Website/* user@your-server.com:/var/www/nordiqai.io/

# Or use git:
cd /var/www/nordiqai.io
git clone <your-repo-url> .
```

2. **Set correct permissions:**

```bash
sudo chown -R www-data:www-data /var/www/nordiqai.io
sudo chmod -R 755 /var/www/nordiqai.io
```

3. **Configure Apache virtual host:**

Create `/etc/apache2/sites-available/nordiqai.io.conf`:

```apache
<VirtualHost *:80>
    ServerName nordiqai.io
    ServerAlias www.nordiqai.io
    DocumentRoot /var/www/nordiqai.io

    <Directory /var/www/nordiqai.io>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>

    ErrorLog ${APACHE_LOG_DIR}/nordiqai-error.log
    CustomLog ${APACHE_LOG_DIR}/nordiqai-access.log combined
</VirtualHost>
```

4. **Enable site and SSL:**

```bash
# Enable site
sudo a2ensite nordiqai.io.conf

# Install Certbot (if not already installed)
sudo apt-get install certbot python3-certbot-apache

# Get SSL certificate (Let's Encrypt)
sudo certbot --apache -d nordiqai.io -d www.nordiqai.io

# Reload Apache
sudo systemctl reload apache2
```

5. **Verify deployment:**

Visit https://nordiqai.io in your browser

---

## 📁 File Structure

```
NordIQ-Website/
├── index.html                 # Homepage ✅
├── product.html               # Product features & dashboard walkthrough ✅
├── how-it-works.html          # Technical explainer (5-step process) ✅
├── pricing.html               # Pricing tiers ✅
├── about.html                 # About Craig/NordIQ (founder story) ✅
├── contact.html               # Contact/Demo request ✅
├── css/
│   └── main.css               # Main stylesheet (Nordic theme) ✅
├── js/
│   └── main.js                # Mobile menu, animations ✅
├── images/                    # Product screenshots, logos
│   ├── favicon.png            # (TODO: Add favicon)
│   ├── logo.png               # (TODO: Add NordIQ logo)
│   ├── og-image.png           # (TODO: Add Open Graph image)
│   └── dashboard-preview.webp # (TODO: Add dashboard screenshot)
├── README.md                  # This file
└── DEPLOYMENT_CHECKLIST.md    # Complete deployment guide ✅
```

---

## 🎨 Design System

### Colors

```css
--navy-blue: #0F172A      /* Primary - trust, depth */
--ice-blue: #0EA5E9       /* Secondary - clarity, precision */
--aurora-green: #10B981   /* Accent - Nordic lights */
--white: #FFFFFF          /* Clean backgrounds */
--charcoal: #1E293B       /* Body text */
--light-gray: #F1F5F9     /* Backgrounds */
```

### Typography

- **Headers**: System fonts (fast loading)
- **Body**: -apple-system, Segoe UI, Roboto
- **Code**: JetBrains Mono, Fira Code, Consolas

### Responsive Breakpoints

- Mobile: <768px
- Tablet: 768-1024px
- Desktop: >1024px

---

## ✅ Completed Pages (6/6 Core Pages - 100%)

- ✅ **Homepage** (`index.html`) - Hero, value prop, stats, pricing preview, use cases
- ✅ **Product** (`product.html`) - Feature deep-dive, 10 dashboard tabs, technical capabilities
- ✅ **How It Works** (`how-it-works.html`) - 5-step process, TFT AI, transfer learning
- ✅ **Pricing** (`pricing.html`) - Transparent tiers, ROI calculator, FAQ, comparison table
- ✅ **About** (`about.html`) - Craig's story, Saturday Ritual, philosophy, expertise
- ✅ **Contact** (`contact.html`) - Email-only contact (craig@nordiqai.io), no forms

**Status**: All core pages complete! Ready for images and deployment.

---

## 📝 TODO: Pre-Launch Tasks

### Priority 1 (Before Launch)

- [ ] **Images** (4 critical images - see DEPLOYMENT_CHECKLIST.md)
  - favicon.png (compass icon)
  - logo.png (NordIQ wordmark)
  - dashboard-preview.webp (screenshot of dashboard)
  - og-image.png (social media preview, 1200x630)

- [ ] **Testing** (local + responsive)
  - Test all navigation links
  - Test mobile menu
  - Test on mobile/tablet/desktop
  - Verify email links work (craig@nordiqai.io)

- [ ] **Deployment** (Apache + SSL)
  - Copy files to server
  - Configure Apache virtual host
  - Set up SSL certificate (Let's Encrypt)
  - Verify HTTPS works

- [ ] **Email Setup**
  - Configure craig@nordiqai.io
  - Test send/receive

### Priority 2 (Post-Launch - Week 1)

- [ ] **Blog Landing** (`blog/index.html`) - Blog post index
- [ ] **Blog Posts** (5-10 initial posts):
  - "Why Predictive Monitoring Beats Traditional Alerts"
  - "The TFT Advantage: How AI Predicts Server Failures"
  - "Building NordIQ: 158 Hours with Claude Code"
  - "ROI Math: Why One Prevented Outage Pays for NordIQ"
  - "How We Built an AI Company with Zero Employees"

### Priority 3 (Week 3+)

- [ ] **Case Studies** - Customer success stories
- [ ] **Documentation** - Public product docs
- [ ] **ROI Calculator** - Interactive JavaScript widget

---

## 🖼️ Images Needed

### High Priority

1. **Favicon** (`images/favicon.png`)
   - 32x32px, 64x64px
   - NordIQ compass icon (🧭)
   - Use tool like [favicon.io](https://favicon.io/)

2. **Logo** (`images/logo.png`)
   - Full NordIQ logo with text
   - Transparent background
   - SVG preferred, PNG fallback

3. **Dashboard Screenshot** (`images/dashboard-preview.webp`)
   - Take screenshot of actual dashboard
   - Full-screen, clean data
   - Compress with WebP format
   - Dimensions: 1920x1080 or 2560x1440

4. **Open Graph Image** (`images/og-image.png`)
   - For social media sharing (LinkedIn, Twitter)
   - 1200x630px
   - Include: NordIQ logo + tagline + key stats
   - Use tool like [Canva](https://www.canva.com/)

### Medium Priority

5. **Hero Background** (`images/hero-background.webp`)
   - Nordic landscape or abstract tech
   - 2560x1440px
   - Dark overlay for text readability

6. **Product Screenshots** (`images/screenshots/`)
   - Dashboard tabs (10 screenshots)
   - Fleet heatmap
   - Alert table
   - Historical trends
   - Compress all with WebP

---

## 🔧 Configuration

### Email Setup

Contact email: **craig@nordiqai.io**

To set up email account:

```bash
# Option 1: Google Workspace ($6/user/month)
# - Professional email (craig@nordiqai.io)
# - Gmail interface
# - Setup: https://workspace.google.com/

# Option 2: Zoho Mail ($1/user/month)
# - Budget-friendly alternative
# - Setup: https://www.zoho.com/mail/

# Option 3: Self-hosted (if you have mail server)
# - Configure MX records for nordiqai.io
# - Point to your mail server
```

### Google Analytics (Optional)

Add before `</head>`:

```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_MEASUREMENT_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_MEASUREMENT_ID');
</script>
```

### SEO Optimization

Already included in all pages:
- ✅ Meta descriptions
- ✅ Semantic HTML5
- ✅ Mobile-responsive
- ✅ Fast loading (<1s target)

To improve:
- [ ] Add sitemap.xml
- [ ] Add robots.txt
- [ ] Submit to Google Search Console
- [ ] Submit to Bing Webmaster Tools

---

## 🚀 Performance Optimization

### Current Performance

- No external dependencies (except system fonts)
- Minimal JavaScript (< 5KB)
- CSS-only animations
- Fast loading (<1s on good connection)

### Further Optimization (Optional)

1. **Image Compression:**
   ```bash
   # Install WebP tools
   sudo apt-get install webp

   # Convert images to WebP
   cwebp dashboard.png -q 80 -o dashboard.webp
   ```

2. **Minify CSS/JS (for production):**
   ```bash
   # Install minifiers
   npm install -g csso-cli uglify-js

   # Minify CSS
   csso css/main.css -o css/main.min.css

   # Minify JS
   uglifyjs js/main.js -o js/main.min.js
   ```

3. **Enable Gzip in Apache:**
   ```apache
   # Add to .htaccess
   <IfModule mod_deflate.c>
     AddOutputFilterByType DEFLATE text/html text/css text/javascript application/javascript
   </IfModule>
   ```

4. **Browser Caching:**
   ```apache
   # Add to .htaccess
   <IfModule mod_expires.c>
     ExpiresActive On
     ExpiresByType image/jpg "access plus 1 year"
     ExpiresByType image/jpeg "access plus 1 year"
     ExpiresByType image/gif "access plus 1 year"
     ExpiresByType image/png "access plus 1 year"
     ExpiresByType image/webp "access plus 1 year"
     ExpiresByType text/css "access plus 1 month"
     ExpiresByType application/javascript "access plus 1 month"
   </IfModule>
   ```

---

## 🔒 Security

### Recommended .htaccess Rules

```apache
# Disable directory browsing
Options -Indexes

# Prevent access to sensitive files
<FilesMatch "^\.">
    Require all denied
</FilesMatch>

# Disable PHP execution (not needed for static site)
<FilesMatch "\.php$">
    Require all denied
</FilesMatch>

# Security headers
<IfModule mod_headers.c>
    Header always set X-Frame-Options "SAMEORIGIN"
    Header always set X-Content-Type-Options "nosniff"
    Header always set X-XSS-Protection "1; mode=block"
    Header always set Referrer-Policy "strict-origin-when-cross-origin"
</IfModule>
```

---

## 📊 Analytics & Tracking

### What to Track

1. **Traffic**:
   - Total visitors
   - Page views
   - Bounce rate
   - Session duration

2. **Conversions**:
   - Email clicks (mailto:craig@nordiqai.io)
   - Demo request interest
   - Pricing page views

3. **Sources**:
   - LinkedIn traffic
   - Google search
   - Direct visits
   - Referrals

### Tools

- **Google Analytics** (free)
- **Hotjar** (heatmaps, free tier)
- **Google Search Console** (SEO, free)

---

## 🔄 Git Integration (Recommended)

```bash
# Initialize git in NordIQ-Website/
cd NordIQ-Website
git init

# Add files
git add .

# Commit
git commit -m "Initial NordIQ website - homepage, contact, pricing"

# Add remote (your repo)
git remote add origin <your-repo-url>

# Push
git push -u origin main

# On server, pull updates:
cd /var/www/nordiqai.io
git pull origin main
sudo systemctl reload apache2
```

---

## 📞 Support

**Questions about deployment?**
- Check Apache logs: `sudo tail -f /var/log/apache2/nordiqai-error.log`
- Verify permissions: `ls -la /var/www/nordiqai.io`
- Test configuration: `sudo apache2ctl configtest`

**Need help?**
Email craig@nordiqai.io (that's you!) 😊

---

## 📝 License

Copyright © 2025 NordIQ AI Systems, LLC
All rights reserved.

Built by Craig Giannelli

---

**Version**: 1.0 (Initial Launch)
**Last Updated**: October 18, 2025
