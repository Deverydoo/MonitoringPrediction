# Managed Hosting Economics - ArgusAI

**Version:** 1.2.1
**Updated:** 2025-10-18
**Status:** Ready for Sales (Manual Deployment)

---

## Overview

NordIQ offers **Managed Hosting** as a premium tier where we deploy and manage the ArgusAI system on AWS infrastructure on behalf of the customer. This provides the "SaaS experience" without requiring automated multi-tenant infrastructure.

---

## Pricing Structure

### Tier: Managed Hosting
- **Price Range:** $25K-50K/year
- **Server Coverage:** 500-2000 monitored servers
- **Midpoint:** $37.5K/year for ~1,250 servers
- **Per-Server Cost:** ~$30/year or $2.50/month per server

---

## Unit Economics (Per Customer)

### Revenue
- **Annual Contract Value (ACV):** $37,500 (midpoint)
- **Monthly Recurring Revenue (MRR):** $3,125

### AWS Infrastructure Costs

**For 1,250 monitored servers:**

| Component | Specification | Monthly Cost |
|-----------|---------------|--------------|
| **Compute (HA)** | 2x EC2 t3.large (2 vCPU, 8GB RAM) | $120 |
| **Storage** | EBS gp3 100GB + S3 backups | $13 |
| **Database** | RDS PostgreSQL db.t3.medium | $50 |
| **Load Balancer** | Application Load Balancer | $20 |
| **Networking** | Data transfer (minimal - predictions are tiny) | $10 |
| **Monitoring** | CloudWatch logs & metrics | $10 |
| **TOTAL** | | **$223/month** |

**Annual AWS Cost:** $2,676

### Gross Margin Analysis

- **Annual Revenue:** $37,500
- **Annual AWS Costs:** $2,676
- **Annual Support/Ops Time:** ~$5,000 (10 hours @ $500/hr for setup + quarterly reviews)
- **Gross Profit:** $29,824
- **Gross Margin:** **79.5%**

### Break-Even Analysis

- **Fixed Costs per Customer:** $7,676 ($2,676 AWS + $5,000 ops)
- **Break-Even Price:** ~$8,000/year
- **Safety Margin:** 4.7x over break-even

---

## Deployment Model: White-Glove (Manual)

### Initial Setup (1-2 days)

**Day 1: Infrastructure Provisioning**
1. Create dedicated AWS account (or VPC) for customer
2. Provision EC2 instances (2x t3.large for HA)
3. Set up RDS PostgreSQL database
4. Configure ALB with SSL certificate
5. Set up CloudWatch monitoring

**Day 2: Software Deployment**
1. Clone NordIQ codebase to EC2 instances
2. Install dependencies (Python, PyTorch, etc.)
3. Configure daemons (inference, metrics generator)
4. Deploy dashboard (Streamlit)
5. Configure API keys and authentication
6. Test end-to-end (metrics ingestion → predictions → dashboard)

**Customer Handoff:**
- Provide API endpoint for metrics ingestion
- Provide dashboard URL with login credentials
- Schedule onboarding call (1 hour)
- Deliver integration documentation

### Ongoing Operations (2-4 hours/quarter)

**Monthly:**
- Monitor AWS costs (CloudWatch billing alerts)
- Review system health (uptime, prediction accuracy)
- Check for software updates

**Quarterly:**
- Customer business review call (1 hour)
- Model retraining with production data (1 hour)
- Performance optimization if needed (1-2 hours)

---

## Scalability Without Automation

### Manual Process Capacity
- **1 customer:** Easy (2 days setup, 1 hour/month maintenance)
- **5 customers:** Manageable (10 days setup, 5 hours/month maintenance)
- **10 customers:** Approaching limits (need some automation)
- **20+ customers:** Requires full automation

### Revenue Thresholds for Automation Investment

| Milestone | Revenue | Action |
|-----------|---------|--------|
| **1-3 customers** | $75K-150K ARR | Manual deployment OK |
| **5 customers** | $187K ARR | Build Terraform templates |
| **10 customers** | $375K ARR | Build self-service onboarding |
| **20+ customers** | $750K ARR | Full multi-tenant SaaS platform |

**Recommendation:** Stay manual until 5 customers, then invest $25K-50K in automation (Terraform + onboarding flow). Use revenue to fund development.

---

## Sales Positioning

### Key Messaging

**"White-Glove Managed Hosting"**
- We deploy NordIQ on dedicated AWS infrastructure for you
- You get the SaaS experience without sharing infrastructure
- Your data is isolated in your own AWS environment
- We handle all maintenance, updates, and monitoring
- Fixed annual cost, no surprise bills

### Ideal Customer Profile

**Best Fit:**
- 500-2000 servers (managed hosting sweet spot)
- Prefer not to manage infrastructure themselves
- Want predictive monitoring but lack in-house ML expertise
- Budget for premium service ($25K-50K/year is acceptable)
- Industries: FinTech, E-Commerce, Healthcare, SaaS companies

**Not a Good Fit:**
- <200 servers (self-hosted is cheaper)
- >3000 servers (enterprise tier with on-prem preferred)
- Strict data sovereignty requirements (must use self-hosted)
- Budget-constrained (<$20K/year total IT budget)

---

## Technical Requirements

### Customer Side
- Ability to send metrics to HTTPS API endpoint (NordIQ provides)
- Firewall rules to allow outbound HTTPS to NordIQ endpoints
- VPN access if they want to access dashboard from private network
- (Optional) VPC peering if they want extra security

### NordIQ Side (Per Customer)
- **AWS Account:** Dedicated or isolated VPC
- **Compute:** 2x EC2 t3.large (HA)
- **Storage:** 100GB EBS + S3
- **Database:** RDS PostgreSQL db.t3.medium
- **Networking:** ALB + Route53 DNS
- **Monitoring:** CloudWatch + SNS alerts

### Security & Compliance
- SSL/TLS encryption (Let's Encrypt or AWS ACM)
- API key authentication for metrics ingestion
- VPC isolation per customer
- Regular automated backups (daily snapshots to S3)
- CloudWatch logs retained for 90 days
- No PII stored (only infrastructure metrics)

---

## Comparison: Manual vs. Automated SaaS

| Aspect | Manual (Now) | Automated (Future) |
|--------|--------------|-------------------|
| **Setup Time** | 1-2 days | 5 minutes (self-service) |
| **Cost per Customer** | $2,676/year AWS | $500/year AWS (shared) |
| **Gross Margin** | 79.5% | 95%+ |
| **Ops Time** | 4 hours/quarter | <1 hour/quarter |
| **Max Customers** | 10 comfortably | Unlimited |
| **Infrastructure** | Dedicated per customer | Multi-tenant shared |
| **Development Cost** | $0 | $50K-100K |
| **Time to Build** | 0 weeks | 6-9 weeks |

**Insight:** Manual approach is **highly profitable** ($29.8K profit per customer) with **zero upfront development cost**. Perfect for validating market demand before investing in automation.

---

## Customer Onboarding Checklist

### Pre-Sale
- [ ] Qualify: 500-2000 servers?
- [ ] Qualify: $25K-50K budget approved?
- [ ] Demo call scheduled (show dashboard, XAI features)
- [ ] Technical discovery (what monitoring tools they use now)
- [ ] Contract signed, payment received

### Deployment (Day 1-2)
- [ ] AWS account created (or VPC provisioned)
- [ ] EC2 instances launched (2x t3.large)
- [ ] RDS database provisioned
- [ ] ALB + SSL certificate configured
- [ ] Route53 DNS configured (e.g., acme-corp.nordiq.cloud)
- [ ] Software deployed (inference daemon, dashboard)
- [ ] Test prediction with sample data
- [ ] API keys generated for customer

### Handoff (Day 3)
- [ ] Send customer API endpoint + documentation
- [ ] Send dashboard URL + login credentials
- [ ] Schedule 1-hour onboarding call
- [ ] Customer sends first metrics successfully
- [ ] Predictions visible in dashboard
- [ ] Customer confirms everything working

### Ongoing
- [ ] Monthly: Check AWS costs (should be ~$223/month)
- [ ] Monthly: Review prediction accuracy
- [ ] Quarterly: Business review call with customer
- [ ] Quarterly: Model retraining with production data
- [ ] Annually: Renewal negotiation

---

## Risk Mitigation

### Customer Churn Risk
- **Mitigation:** Quarterly business reviews show value (outages prevented)
- **Mitigation:** Annual contracts with early termination penalties
- **Mitigation:** Sticky product (model trains on their data = switching cost)

### AWS Cost Overruns
- **Mitigation:** CloudWatch billing alerts at $250/month
- **Mitigation:** Reserved instances (save 30-40% after 5 customers)
- **Mitigation:** Right-sizing (monitor actual usage, downgrade if possible)

### Operational Overhead
- **Mitigation:** Runbooks for common tasks (backups, restarts, etc.)
- **Mitigation:** Monitoring automation (CloudWatch alarms → SNS → PagerDuty)
- **Mitigation:** Terraform templates (reduce setup time from 2 days → 4 hours)

---

## Pricing Tiers (Detailed)

### Tier 1: Managed Hosting - Standard
- **Price:** $25K/year
- **Server Coverage:** 500-1000 servers
- **AWS Infrastructure:** Single region, standard support
- **SLA:** 99.5% uptime
- **Support:** Email support (24-hour response)

### Tier 2: Managed Hosting - Premium
- **Price:** $37.5K/year
- **Server Coverage:** 1000-1500 servers
- **AWS Infrastructure:** Multi-AZ deployment
- **SLA:** 99.9% uptime
- **Support:** Priority email + monthly calls

### Tier 3: Managed Hosting - Enterprise
- **Price:** $50K/year
- **Server Coverage:** 1500-2000 servers
- **AWS Infrastructure:** Multi-region (optional)
- **SLA:** 99.95% uptime
- **Support:** Dedicated Slack channel + weekly calls

---

## Next Steps for Automation (Future)

When you hit 5+ managed hosting customers ($187K+ ARR), invest in:

### Phase 1: Terraform Automation (2 weeks, $10K)
- Terraform modules for complete stack
- Reduces setup from 2 days → 4 hours
- Consistent deployments, fewer errors

### Phase 2: Self-Service Onboarding (4 weeks, $20K)
- Customer signup flow (Stripe integration)
- Automated infrastructure provisioning
- Self-service API key generation
- Reduces setup from 4 hours → 5 minutes

### Phase 3: Multi-Tenant SaaS (8 weeks, $50K+)
- Shared infrastructure (1 EC2 cluster serves all customers)
- Database per customer (or schema isolation)
- Tenant-aware authentication & authorization
- Reduces AWS cost from $2,676/customer → $500/customer

**ROI Calculation:**
- 10 customers at $37.5K = $375K ARR
- Automation investment: $80K total
- Payback period: 3-4 months
- Post-automation margin: 95%+ (vs 79.5% manual)

---

## Conclusion

**Managed Hosting is highly profitable right now:**
- 79.5% gross margin per customer
- Zero upfront development cost
- Scalable to 5-10 customers manually
- Validates market demand before building automation

**Recommendation:**
- Keep "Managed Hosting" on website as available now
- Position as "White-Glove Setup" (premium, hands-on)
- Use manual deployment until 5 customers
- Invest first $100K revenue in automation (Terraform + onboarding)
- Transition to true multi-tenant SaaS after 10+ customers

**Bottom Line:** You can start selling this **today** and make excellent margins while validating the market!

---

**Document Owner:** Craig Giannelli (ArgusAI, LLC)
**Last Updated:** 2025-10-18
**Next Review:** After first managed hosting customer or Q1 2026
