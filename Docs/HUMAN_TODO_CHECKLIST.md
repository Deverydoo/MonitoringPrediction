# Human TODO Checklist - NordIQ AI Systems

**Last Updated:** October 18, 2025
**Status:** System complete, ready for deployment & sales

---

## 🎯 Immediate Actions (This Week)

### 1. Test XAI Integration ⏱️ 15 minutes
- [ ] Start NordIQ system: `cd NordIQ && start_all.bat`
- [ ] Open dashboard: http://localhost:8501
- [ ] Navigate to "🧠 Insights (XAI)" tab (tab #8)
- [ ] Select a high-risk server from dropdown
- [ ] Verify all 3 sub-tabs work:
  - [ ] 📊 Feature Importance (SHAP) - shows metric names properly
  - [ ] ⏱️ Temporal Attention - shows time period focus
  - [ ] 🎯 What-If Scenarios - shows recommendations with emojis
- [ ] Check for errors in console
- [ ] Test with 2-3 different servers

**Expected Result:** Clean visualizations, professional metric names, scenario icons, fast performance

---

### 2. Website Local Testing ⏱️ 10 minutes
- [ ] Open `NordIQ-Website/index.html` in browser
- [ ] Verify hero image loads (Tron blue/red network mesh)
- [ ] Navigate to Product page
- [ ] Scroll to screenshot gallery
- [ ] Verify all 11 screenshots load properly
- [ ] Test on mobile view (resize browser)
- [ ] Check that images are sharp and clear

**Files Changed:**
- `css/main.css` - Hero image background
- `product.html` - Screenshot gallery added

---

## 📋 Short-Term (Next 2 Weeks)

### 3. Website Deployment 🌐 ⏱️ 2-4 hours
- [ ] **Get Apache server access** (hosting provider login)
- [ ] **Upload website files** to server
  - Copy entire `NordIQ-Website/` directory
  - Ensure `images/` folder uploads (1.7MB hero + 11 screenshots)
- [ ] **Point domain** nordiqai.io to server IP
- [ ] **Configure SSL certificate** (Let's Encrypt free)
- [ ] **Test deployment:**
  - [ ] https://nordiqai.io loads
  - [ ] Hero image displays
  - [ ] All pages work (6 pages)
  - [ ] Screenshots load on Product page
  - [ ] Mobile responsive (test on phone)

**Resources Needed:**
- Apache server credentials
- Domain registrar access (for DNS)
- SSL certificate (Let's Encrypt or hosting provider)

---

### 4. Marketing & Sales Launch 📢 ⏱️ 1-2 days

#### LinkedIn Announcement
- [ ] Write LinkedIn post announcing NordIQ
- [ ] Include hero image (images/hero.png)
- [ ] Key talking points:
  - Predictive monitoring (30-60 min early warning)
  - 88% accuracy
  - One prevented outage = ROI
  - XAI explanations (unique differentiator!)
- [ ] Link to nordiqai.io
- [ ] Post from personal profile + create company page

#### Sales Materials
- [ ] Create 1-page PDF overview (use screenshots)
- [ ] Create PowerPoint sales deck (10-15 slides)
  - Problem (outages cost $50K-100K)
  - Solution (NordIQ predicts 30-60 min early)
  - How it works (TFT model, 88% accuracy)
  - Screenshots (dashboard, XAI insights)
  - Pricing (starter, professional, enterprise)
  - Contact info
- [ ] Practice demo walkthrough (15 minutes):
  1. Show healthy system
  2. Switch to degrading scenario
  3. Point out prediction vs current
  4. Show XAI tab - "Here's WHY this is happening"
  5. Show recommended action
  6. Discuss ROI (one outage prevented = paid for)

---

### 5. Customer Outreach 🎯 ⏱️ Ongoing
- [ ] **Warm leads** - Reach out to:
  - Former colleagues in SRE/DevOps
  - Companies with 20+ servers
  - Industries: FinTech, E-commerce, SaaS
- [ ] **Cold outreach** - LinkedIn/email:
  - "I built an AI system that predicts server failures 30-60 min early..."
  - Offer free demo (30 minutes)
  - Emphasize: Self-hosted, works with your data
- [ ] **Track prospects** in spreadsheet:
  - Company name
  - Contact name
  - Status (cold, warm, demo scheduled, trial, customer)
  - Next action

**Goal:** 5 demos scheduled in next 2 weeks

---

## 🔧 Technical Improvements (When You Have Revenue)

### 6. Model Training Improvement ⏱️ 4-8 hours
**When:** After first paying customer

- [ ] Collect production data (1 week minimum)
- [ ] Run training script: `python NordIQ/src/training/main.py`
- [ ] Train for 20+ epochs (vs current 3 epochs)
- [ ] Evaluate on validation set
- [ ] Replace 3-epoch model if accuracy improves
- [ ] Document improvement in performance

**Expected Improvement:** 3-5% accuracy gain (from 88% to 91-93%)

---

### 7. Dashboard Performance (Optional) ⏱️ 1-2 hours
**When:** Customer complains about speed OR after 10 customers

Quick wins already done (Oct 18):
- ✅ XAI caching (30s TTL)
- ✅ Professional polish
- ✅ Streamlit optimizations

Additional optimizations (if needed):
- [ ] Add `@st.fragment` to expensive components
- [ ] Increase caching TTL to 60 seconds
- [ ] Lazy-load hidden tabs
- [ ] Reduce auto-refresh from 5s to 10s

**Expected Result:** 50% faster performance

---

### 8. Production Data Integration ⏱️ 2-4 days
**When:** First paying customer wants real server integration

Options:
- [ ] **MongoDB adapter** (if customer uses MongoDB)
- [ ] **Prometheus adapter** (if customer uses Prometheus)
- [ ] **REST API** (push metrics from customer's monitoring)
- [ ] **Custom adapter** (if exotic data source)

**Deliverable:** Customer's live servers feeding into NordIQ

---

## 🚀 Long-Term (Q1-Q2 2026)

### 9. Dashboard Migration to Plotly Dash ⏱️ 3-5 days
**When:** After 5-10 paying customers OR $5K MRR

**Trigger Points:**
- Customer complaints about Streamlit performance
- Enterprise customer requirement
- Dashboard used >30 min/day

**See:** [Docs/FUTURE_DASHBOARD_MIGRATION.md](FUTURE_DASHBOARD_MIGRATION.md)

**Cost:** $5K-8K (1 week developer time)

---

### 10. Dashboard Migration to React ⏱️ 2-3 weeks
**When:** After 20+ paying customers OR $25K MRR

**Trigger Points:**
- White-label partnership opportunity
- Series A fundraising
- Mobile app requirement
- Competitor pressure

**See:** [Docs/FUTURE_DASHBOARD_MIGRATION.md](FUTURE_DASHBOARD_MIGRATION.md)

**Cost:** $20K-30K (contractor or 3 weeks internal)

---

## 📊 Success Metrics

### Week 1
- [ ] XAI tested and working
- [ ] Website deployed to nordiqai.io
- [ ] LinkedIn post published
- [ ] 3 demo meetings scheduled

### Month 1
- [ ] 10+ demos completed
- [ ] 2-3 trial customers
- [ ] 1 paying customer ($200-500/month)

### Month 3
- [ ] 5+ paying customers
- [ ] $2K-3K MRR
- [ ] Improved model deployed
- [ ] 1-2 case studies written

### Month 6
- [ ] 10-15 paying customers
- [ ] $5K-10K MRR
- [ ] Dash migration (if needed)
- [ ] 1-2 white-label discussions

---

## 🎯 Current Priorities (Focus Here!)

**This Week:**
1. ✅ XAI integration (DONE!)
2. ✅ Website polish (DONE!)
3. 🎯 **Test XAI tab** (15 min)
4. 🎯 **Deploy website** (2-4 hours)
5. 🎯 **LinkedIn post** (1 hour)

**This Month:**
1. 🎯 **Schedule 10 demos** (ongoing outreach)
2. 🎯 **Close first customer** (revenue!)
3. Improve model (after production data)

**Don't Do Yet:**
- ❌ Rewrite dashboard (not until revenue)
- ❌ Build mobile app (not until 20+ customers)
- ❌ Hire engineers (not until $10K MRR)
- ❌ Raise funding (not until product-market fit)

---

## 🔑 Key Files & Resources

### System Startup
- **Windows:** `NordIQ/start_all.bat`
- **Linux:** `NordIQ/start_all.sh`
- **Dashboard:** http://localhost:8501
- **API:** http://localhost:8000

### Documentation
- **Quick Start:** `Docs/RAG/QUICK_START_NEXT_SESSION.md`
- **Current State:** `Docs/RAG/CURRENT_STATE.md`
- **XAI Polish:** `Docs/XAI_POLISH_CHECKLIST.md`
- **Migration Plan:** `Docs/FUTURE_DASHBOARD_MIGRATION.md`

### Website
- **Location:** `NordIQ-Website/`
- **Hero Image:** `NordIQ-Website/images/hero.png` (1.7MB, Tron theme)
- **Screenshots:** `NordIQ-Website/images/Screenshot_*.png` (11 images)

### Sales Materials
- **Value Prop:** Predict failures 30-60 min early, 88% accuracy
- **ROI:** One prevented outage ($50K-100K) = entire year paid
- **Differentiator:** XAI explanations (SHAP, Attention, What-If scenarios)
- **Target:** Companies with 20-100 servers, SRE/DevOps teams

---

## 📞 Get Help

**Questions?**
- Review: `Docs/RAG/README.md` (start here for AI assistant context)
- Technical: `Docs/RAG/PROJECT_CODEX.md` (development rules)
- Troubleshooting: Check daemon logs, dashboard errors

**Common Issues:**
- System won't start → Check .env file has API key
- Dashboard slow → Already optimized (Oct 18)
- XAI errors → Check daemon is running, API key correct

---

## ✅ What's Already Done (October 18, 2025)

**System (v1.1.0):**
- ✅ TFT model (111K parameters, 88% accuracy)
- ✅ 20-server demo (healthy/degrading/critical scenarios)
- ✅ 11-tab dashboard (modular architecture)
- ✅ XAI integration (SHAP, Attention, Counterfactuals)
- ✅ All bugs fixed (import paths, API auth)
- ✅ Documentation complete (184KB RAG docs)

**Website (6/6 pages):**
- ✅ Index (home page)
- ✅ Product (with screenshot gallery!)
- ✅ How It Works (technology)
- ✅ Company (about)
- ✅ Contact
- ✅ Documentation
- ✅ Hero image (Tron theme, stunning!)
- ✅ 11 dashboard screenshots

**Marketing:**
- ✅ Brand identity (🧭 NordIQ, blue/red Tron colors)
- ✅ Value proposition ("predict 30-60 min early")
- ✅ Technical credibility (88%, TFT model)
- ✅ Visual assets (hero, screenshots)

**You have everything you need to launch!** 🚀

---

**Next Step:** Test XAI tab (15 min), then deploy website (2-4 hours), then LinkedIn post (1 hour).

**Remember:** Perfect is the enemy of shipped. The system works. Get it in front of customers NOW!

---

**Maintained By:** Craig Giannelli / NordIQ AI Systems, LLC
**Last Review:** October 18, 2025
**Next Review:** After first paying customer
