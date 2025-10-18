# NordIQ Website - Deployment Checklist

**Status**: 6/6 Core Pages Complete ✅
**Ready for**: Images → Testing → Deployment

---

## ✅ Completed (100%)

### Pages Built (6/6)
- [x] **index.html** - Homepage with hero, value prop, use cases
- [x] **product.html** - Feature deep-dive, dashboard walkthrough, capabilities
- [x] **how-it-works.html** - Technical explainer (5-step process, TFT AI)
- [x] **pricing.html** - Transparent pricing tiers, ROI calculator
- [x] **about.html** - Founder story, Saturday Ritual, philosophy
- [x] **contact.html** - Email-only contact (craig@nordiqai.io)

### Design System
- [x] **css/main.css** (600+ lines) - Nordic minimalist design
- [x] **js/main.js** - Mobile menu, smooth scroll, animations
- [x] Responsive (mobile/tablet/desktop)
- [x] Fast loading target (<1s)

### Documentation
- [x] **README.md** - Deployment guide (Apache setup)
- [x] **images/README.md** - Image requirements
- [x] **DEPLOYMENT_CHECKLIST.md** - This file

---

## 📸 Images Needed (0/11)

### Critical for Launch (4 images)

#### 1. Favicon (favicon.png + favicon.ico)
**What**: NordIQ compass icon 🧭
**Sizes**: 32x32, 64x64, favicon.ico
**How to Create**:
1. Go to https://favicon.io/favicon-converter/
2. Upload a simple compass icon (blue/white)
3. Download package with all sizes
4. Copy `favicon.ico` and `favicon-32x32.png` to `images/`

**Quick Option**: Use emoji-to-favicon generator
- https://favicon.io/emoji-favicons/compass/
- Download and rename to `favicon.png` and `favicon.ico`

#### 2. Logo (logo.png or logo.svg)
**What**: NordIQ wordmark + compass icon
**Recommended Size**: 200x50px (transparent PNG or SVG)
**How to Create**:
1. Go to https://www.canva.com/ (free account)
2. Create 200x50px image
3. Add compass icon 🧭 + "NordIQ" text (navy blue #0F172A)
4. Font: Bold sans-serif (Inter, Roboto, or Arial)
5. Download as transparent PNG

**Alternative**: Use text + emoji
- Just "🧭 NordIQ" in the header (current approach)
- No logo file needed (defer to later)

#### 3. Dashboard Screenshot (dashboard-preview.webp)
**What**: Screenshot of the actual NordIQ dashboard
**Size**: 1920x1080 or larger
**How to Create**:
1. Start NordIQ dashboard locally:
   ```bash
   cd NordIQ
   ./start_all.sh  # or start_all.bat on Windows
   ```
2. Open http://localhost:8501 in browser
3. Go to "Fleet Overview" tab
4. Set scenario to "degrading" (to show some alerts)
5. Take full-screen screenshot (F11 fullscreen mode first)
6. Crop to just the dashboard content (no browser chrome)
7. Convert to WebP:
   ```bash
   # Option 1: Online converter
   https://cloudconvert.com/png-to-webp

   # Option 2: Command line (if you have ImageMagick)
   convert dashboard.png -quality 80 dashboard-preview.webp
   ```
8. Save to `images/dashboard-preview.webp`

**Where Used**: Homepage, Product page

#### 4. Open Graph Image (og-image.png)
**What**: Social media preview (LinkedIn, Twitter, Slack)
**Size**: 1200x630px
**How to Create**:
1. Go to https://www.canva.com/
2. Create custom size: 1200x630px
3. Add:
   - NordIQ logo/wordmark (top left)
   - Tagline: "Nordic precision, AI intelligence"
   - Key stat: "Predict server failures 30-60 minutes before they happen"
   - Background: Navy blue (#0F172A) with subtle gradient
4. Download as PNG
5. Save to `images/og-image.png`

**Alternative**: Use existing screenshot with text overlay
- Take dashboard-preview.webp
- Add text overlay with tagline
- Resize to 1200x630

**Where Used**: All pages (meta tags for social sharing)

---

### Nice to Have (7 images)

#### 5. Hero Background (hero-background.webp)
**What**: Background image for hero sections
**Size**: 2560x1440px
**Options**:
- Nordic landscape (mountains, fjords, aurora)
- Abstract tech pattern (circuits, networks)
- Dark gradient (navy to ice blue)

**Free Sources**:
- https://unsplash.com/s/photos/norway-landscape
- https://unsplash.com/s/photos/data-visualization
- https://www.pexels.com/search/technology/

**How to Create**:
1. Download high-res image
2. Apply dark overlay (50% opacity) in Photoshop/GIMP
3. Compress to WebP
4. Save to `images/hero-background.webp`

**Where Used**: Homepage hero (optional - gradient is fine)

#### 6-11. Dashboard Tab Screenshots (screenshots/*.webp)
**What**: Screenshots of each dashboard tab
**Size**: 1920x1080 each

Screenshots needed:
- `screenshots/fleet-overview.webp` - Fleet Overview tab
- `screenshots/heatmap.webp` - Server Heatmap
- `screenshots/top-risks.webp` - Top 5 Problem Servers
- `screenshots/historical.webp` - Historical Trends
- `screenshots/cost-avoidance.webp` - Cost Avoidance Calculator
- `screenshots/alerting.webp` - Alert Routing

**How to Create**:
1. Start dashboard (see #3 above)
2. Navigate to each tab
3. Take full-screen screenshot
4. Crop and convert to WebP
5. Save to `screenshots/` folder

**Where Used**: Product page (dashboard walkthrough section)

---

## 🧪 Testing Checklist (0/8)

### Local Testing
- [ ] **Start local HTTP server**:
  ```bash
  cd NordIQ-Website
  python -m http.server 8000
  # Open http://localhost:8000 in browser
  ```

### Navigation Testing
- [ ] All navigation links work (6 pages)
- [ ] Mobile menu works (hamburger icon)
- [ ] Footer links work
- [ ] Smooth scroll to anchors works

### Responsive Testing
- [ ] Desktop (1920x1080)
- [ ] Tablet (768x1024)
- [ ] Mobile (375x667)
- [ ] Test in Chrome, Firefox, Safari

### Content Verification
- [ ] No typos or broken text
- [ ] All images load (or placeholders shown)
- [ ] Contact email is correct: craig@nordiqai.io
- [ ] Pricing is accurate ($5K-150K)

---

## 🚀 Apache Deployment (0/10)

### Pre-Deployment
- [ ] **Register domain**: nordiqai.io (already secured ✅)
- [ ] **Set up email**: craig@nordiqai.io
  - Option 1: Google Workspace ($6/month)
  - Option 2: Zoho Mail (free for 1 user)
  - Option 3: Forward to personal email

### Server Setup
- [ ] **Apache installed**: `sudo apt-get install apache2`
- [ ] **Create site directory**:
  ```bash
  sudo mkdir -p /var/www/nordiqai.io
  sudo chown -R $USER:$USER /var/www/nordiqai.io
  ```

### File Deployment
- [ ] **Copy website files**:
  ```bash
  # From your dev machine
  rsync -avz NordIQ-Website/ user@server:/var/www/nordiqai.io/

  # Or use git
  cd /var/www/nordiqai.io
  git clone <repo-url> .
  ```

### Apache Configuration
- [ ] **Create virtual host**: `/etc/apache2/sites-available/nordiqai.io.conf`
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

- [ ] **Enable site and rewrite module**:
  ```bash
  sudo a2ensite nordiqai.io.conf
  sudo a2enmod rewrite
  sudo systemctl reload apache2
  ```

### SSL Certificate (Let's Encrypt)
- [ ] **Install Certbot**:
  ```bash
  sudo apt-get install certbot python3-certbot-apache
  ```

- [ ] **Get certificate**:
  ```bash
  sudo certbot --apache -d nordiqai.io -d www.nordiqai.io
  # Follow prompts (provide email: craig@nordiqai.io)
  ```

- [ ] **Test auto-renewal**:
  ```bash
  sudo certbot renew --dry-run
  ```

### Security & Performance
- [ ] **Create .htaccess** (already in repo):
  ```apache
  # Force HTTPS
  RewriteEngine On
  RewriteCond %{HTTPS} off
  RewriteRule ^(.*)$ https://%{HTTP_HOST}%{REQUEST_URI} [L,R=301]

  # Security headers
  Header set X-Content-Type-Options "nosniff"
  Header set X-Frame-Options "SAMEORIGIN"
  Header set X-XSS-Protection "1; mode=block"

  # Enable compression
  <IfModule mod_deflate.c>
      AddOutputFilterByType DEFLATE text/html text/css text/javascript application/javascript
  </IfModule>

  # Browser caching
  <IfModule mod_expires.c>
      ExpiresActive On
      ExpiresByType image/webp "access plus 1 year"
      ExpiresByType text/css "access plus 1 month"
      ExpiresByType application/javascript "access plus 1 month"
  </IfModule>
  ```

- [ ] **Verify HTTPS**: https://nordiqai.io should load with green lock

### Final Checks
- [ ] Website loads at https://nordiqai.io
- [ ] All pages accessible
- [ ] Images load correctly
- [ ] Mobile responsive
- [ ] SSL certificate valid
- [ ] No console errors

---

## 📊 Analytics & Monitoring (0/4)

### Google Analytics
- [ ] Create GA4 property
- [ ] Add tracking code to all pages (before `</head>`)
- [ ] Verify tracking in real-time reports

### Google Search Console
- [ ] Add property for nordiqai.io
- [ ] Submit sitemap.xml
- [ ] Verify ownership

### Monitoring
- [ ] Set up uptime monitoring (UptimeRobot, Pingdom)
- [ ] Set up broken link checker
- [ ] Monitor SSL expiration (Certbot auto-renews)

### Email Verification
- [ ] Send test email to craig@nordiqai.io
- [ ] Verify receipt
- [ ] Set up email signature

---

## 📣 Launch Marketing (0/6)

### Pre-Launch
- [ ] Update LinkedIn profile (add NordIQ AI Systems, LLC)
- [ ] Prepare announcement post
- [ ] Identify 5-10 relevant communities (Hacker News, Reddit, LinkedIn groups)

### Launch Day
- [ ] **LinkedIn post** announcing NordIQ.io launch
  - Include dashboard screenshot
  - Link to website
  - Call-to-action: Request demo
- [ ] Share on Twitter/X (if applicable)
- [ ] Post to relevant communities (with value, not spam)

### Week 1
- [ ] Monitor website analytics
- [ ] Respond to any demo requests
- [ ] Follow up with personal network

---

## 📝 Content Improvements (Future)

### Blog Posts (SEO)
- [ ] "How Temporal Fusion Transformers Predict Server Failures"
- [ ] "Why Traditional Monitoring Fails at Scale"
- [ ] "The Saturday Ritual: Shipping Production AI in 6 Hours"
- [ ] "Profile-Based Transfer Learning for Infrastructure Monitoring"
- [ ] "Context-Aware Alerting: Beyond Dumb Thresholds"

### Case Studies
- [ ] Financial services company (regulatory compliance)
- [ ] SaaS company (uptime SLA protection)
- [ ] E-commerce (revenue protection)

### Interactive Tools
- [ ] ROI calculator (JavaScript)
- [ ] Server profile quiz ("Which profile are you?")
- [ ] Cost savings estimator

---

## 🎯 Success Metrics (Track After Launch)

### Website Traffic (Month 1)
- Target: 500-1,000 unique visitors
- Bounce rate: <60%
- Avg session duration: >2 minutes

### Lead Generation (Month 1)
- Target: 5-10 demo requests
- Response time: <24 hours
- Conversion to pilot: 20-30%

### SEO (Month 3)
- Google ranking for "predictive infrastructure monitoring"
- Google ranking for "AI server monitoring"
- Backlinks from relevant sites

---

## 📚 Resources

### Design Tools
- **Canva**: https://www.canva.com/ (logo, og-image)
- **Favicon Generator**: https://favicon.io/
- **Image Compression**: https://tinypng.com/, https://squoosh.app/

### Stock Images
- **Unsplash**: https://unsplash.com/ (free high-quality photos)
- **Pexels**: https://www.pexels.com/ (free stock photos)

### Testing Tools
- **Responsive Test**: https://responsivedesignchecker.com/
- **Page Speed**: https://pagespeed.web.dev/
- **SSL Test**: https://www.ssllabs.com/ssltest/

### SEO Tools
- **Google Search Console**: https://search.google.com/search-console
- **Google Analytics**: https://analytics.google.com/

---

## 🚧 Known Issues / Future Enhancements

### Minor Issues
- None currently - all pages complete

### Future Enhancements
1. Add favicon and logo (placeholder text/emoji works for now)
2. Add actual dashboard screenshots (requires running app)
3. Add hero background images (optional - gradients work)
4. Add blog section (phase 2)
5. Add live chat widget (phase 2)
6. Add video demo (phase 2)

---

## ✅ Current Status

**Website Completion**: 100% (6/6 pages)
**Images**: 0% (0/11 images)
**Testing**: 0% (not started)
**Deployment**: 0% (not started)

**Next Steps**:
1. Create 4 critical images (favicon, logo, dashboard screenshot, og-image)
2. Test locally (30 minutes)
3. Deploy to Apache server (2-3 hours)
4. Launch marketing (LinkedIn announcement)

**Estimated Time to Launch**: 4-6 hours (with images)

---

**Last Updated**: October 18, 2025
**Maintained By**: Craig Giannelli / NordIQ AI Systems, LLC
