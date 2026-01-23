# 🚀 2ndBrain Production Deployment Guide (Render)

**Last Updated**: January 23, 2026
**Platform**: Render.com
**Estimated Time**: 30 minutes
**Monthly Cost**: $14-24

---

## 📋 Pre-Deployment Checklist

### Required Items

- [ ] GitHub account with 2ndBrain repo
- [ ] Render.com account (free to create)
- [ ] OpenAI API key
- [ ] Gmail OAuth credentials (optional, for Gmail integration)
- [ ] Domain name (optional)

### API Keys Needed

| Service | Required? | Where to Get | Cost |
|---------|-----------|--------------|------|
| OpenAI API Key | ✅ Yes | https://platform.openai.com/api-keys | Pay per use (~$50/mo) |
| Google OAuth | ⚠️ For Gmail | https://console.cloud.google.com/ | Free |
| GitHub Token | ⚠️ For GitHub | https://github.com/settings/tokens | Free |
| Slack Bot Token | ⚠️ For Slack | https://api.slack.com/apps | Free |

---

## 🎯 Step-by-Step Deployment

### Step 1: Push Code to GitHub

```bash
cd /Users/pranavreddymogathala/2ndbrainRepo

# Replace the old render.yaml
mv render.yaml.updated render.yaml

# Commit the updated config
git add render.yaml
git commit -m "Update render.yaml for production deployment"
git push origin main
```

---

### Step 2: Create Render Account

1. Go to https://render.com/
2. Click **"Get Started"**
3. Sign up with GitHub (recommended for auto-deploy)
4. Authorize Render to access your repositories

---

### Step 3: Deploy from Blueprint (Easiest Method)

**Option A: One-Click Deploy from render.yaml**

1. In Render Dashboard, click **"New +"** → **"Blueprint"**
2. Connect your GitHub repository: `MrIDK-crypto/2ndBrain`
3. Render will detect `render.yaml` automatically
4. Click **"Apply"**
5. Render will create:
   - Web service (Flask API)
   - PostgreSQL database
   - Redis instance
   - Persistent disk for ChromaDB

**Option B: Manual Setup**

<details>
<summary>Click to expand manual setup steps</summary>

#### 3.1 Create PostgreSQL Database

1. Click **"New +"** → **"PostgreSQL"**
2. Settings:
   - **Name**: `2ndbrain-db`
   - **Database**: `secondbrain`
   - **User**: `secondbrain_user`
   - **Region**: Oregon (or closest to you)
   - **Plan**: Starter ($7/mo)
3. Click **"Create Database"**
4. Wait 2-3 minutes for provisioning
5. Copy the **Internal Database URL** (for next step)

#### 3.2 Create Redis Instance

1. Click **"New +"** → **"Redis"**
2. Settings:
   - **Name**: `2ndbrain-redis`
   - **Region**: Oregon (same as database)
   - **Plan**: Starter (free tier or $10/mo)
3. Click **"Create Redis"**
4. Copy the **Internal Redis URL**

#### 3.3 Create Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect GitHub repository: `MrIDK-crypto/2ndBrain`
3. Settings:
   - **Name**: `2ndbrain-api`
   - **Region**: Oregon
   - **Branch**: `main`
   - **Root Directory**: Leave blank
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn backend.api.app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120`
   - **Plan**: Starter ($7/mo)

4. **Add Disk**:
   - Click **"Add Disk"**
   - **Name**: `chromadb-data`
   - **Mount Path**: `/app/data`
   - **Size**: 2 GB

5. Click **"Create Web Service"**

</details>

---

### Step 4: Configure Environment Variables

Once services are created, add environment variables:

1. Go to your web service → **"Environment"** tab
2. Click **"Add Environment Variable"**
3. Add these variables:

#### Required Variables

```bash
# Database (auto-filled if linked)
DATABASE_URL=<internal-connection-string-from-postgres>

# Redis (auto-filled if linked)
REDIS_URL=<internal-connection-string-from-redis>

# OpenAI (REQUIRED - add manually)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx

# Flask config
FLASK_ENV=production
FLASK_DEBUG=0
PYTHONUNBUFFERED=1

# Security (auto-generate in Render)
JWT_SECRET_KEY=<click "Generate" button>
MASTER_ENCRYPTION_KEY=<click "Generate" button>

# CORS (update after deployment)
CORS_ORIGINS=https://2ndbrain-api.onrender.com
```

#### Optional Variables (for integrations)

```bash
# Gmail OAuth
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-secret
GOOGLE_REDIRECT_URI=https://2ndbrain-api.onrender.com/api/connectors/gmail/callback

# GitHub
GITHUB_ACCESS_TOKEN=ghp_xxxxxxxxxxxxx

# Slack
SLACK_BOT_TOKEN=xoxb-xxxxxxxxxxxxx

# Disable Neo4j (unless you have a Neo4j instance)
NEO4J_ENABLED=false
```

4. Click **"Save Changes"**
5. Service will automatically redeploy

---

### Step 5: Wait for Deployment

**First deployment takes ~5-10 minutes**:

1. Watch the build logs in real-time
2. Common issues:
   - Missing environment variables → Add them
   - Build timeout → Increase build timeout in settings
   - Out of memory → Upgrade plan or reduce dependencies

**Successful deployment shows**:
```
==> Starting service...
✓ Build successful
✓ Health check passed
✓ Deployed to https://2ndbrain-api.onrender.com
```

---

### Step 6: Verify Deployment

#### Test Health Endpoint

```bash
curl https://2ndbrain-api.onrender.com/api/health
```

**Expected response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-23T...",
  "version": "1.0.0"
}
```

#### Test Database Connection

```bash
curl https://2ndbrain-api.onrender.com/api/status
```

Should return database and Redis connection status.

#### Access API Documentation

Open in browser:
```
https://2ndbrain-api.onrender.com/api/docs
```

---

## 🔐 Post-Deployment Security

### 1. Enable HTTPS (Automatic)

Render provides free SSL certificates automatically. Your app will be available at:
```
https://2ndbrain-api.onrender.com
```

### 2. Set Up Custom Domain (Optional)

1. In Render Dashboard → Your service → **"Settings"**
2. Scroll to **"Custom Domains"**
3. Click **"Add Custom Domain"**
4. Enter your domain: `api.yourdomain.com`
5. Add the CNAME record to your DNS provider:
   ```
   CNAME api.yourdomain.com → 2ndbrain-api.onrender.com
   ```
6. Wait for DNS propagation (5-30 minutes)
7. Render will auto-provision SSL certificate

### 3. Update OAuth Redirect URIs

If using Gmail integration, update redirect URI in Google Cloud Console:
```
OLD: http://localhost:5000/api/connectors/gmail/callback
NEW: https://2ndbrain-api.onrender.com/api/connectors/gmail/callback
```

---

## 📊 Monitoring & Maintenance

### View Logs

1. Render Dashboard → Your service → **"Logs"** tab
2. Real-time logs show:
   - HTTP requests
   - Errors
   - Database queries
   - API calls

### Set Up Alerts

1. Render Dashboard → Your service → **"Settings"**
2. Scroll to **"Notifications"**
3. Add email for:
   - Deploy failures
   - Health check failures
   - High error rates

### Database Backups

**Automatic backups** (on Starter plan):
- Daily backups retained for 7 days
- Restore from Render Dashboard → Database → **"Backups"**

**Manual backup**:
```bash
# Download database dump
render db:backup download 2ndbrain-db --output backup.sql
```

---

## 💰 Cost Breakdown

### Minimal Setup (Basic Production)

| Service | Plan | Cost | Notes |
|---------|------|------|-------|
| Web Service | Starter | $7/mo | 512 MB RAM, auto-sleep after 15 min idle |
| PostgreSQL | Starter | $7/mo | 256 MB RAM, 1 GB storage |
| Redis | Free | $0/mo | 25 MB, enough for caching |
| Persistent Disk | Included | $0 | 1 GB free |
| **Total** | | **$14/mo** | Good for testing/MVP |

### Recommended Setup (No Auto-Sleep)

| Service | Plan | Cost | Notes |
|---------|------|------|-------|
| Web Service | Standard | $25/mo | 2 GB RAM, no auto-sleep |
| PostgreSQL | Starter | $7/mo | 256 MB RAM, 1 GB storage |
| Redis | Starter | $10/mo | 256 MB, for Celery jobs |
| Persistent Disk | 2 GB | $0.25/mo | For ChromaDB |
| **Total** | | **$42/mo** | Production-ready |

### Enterprise Setup (High Traffic)

| Service | Plan | Cost | Notes |
|---------|------|------|-------|
| Web Service | Pro | $85/mo | 8 GB RAM, auto-scaling |
| PostgreSQL | Standard | $50/mo | 4 GB RAM, 50 GB storage |
| Redis | Standard | $50/mo | 2 GB RAM |
| Persistent Disk | 10 GB | $1.25/mo | More embeddings |
| **Total** | | **$186/mo** | 1000+ users |

---

## 🐛 Troubleshooting

### Build Fails with "Out of Memory"

**Solution 1**: Reduce dependencies
```bash
# Remove heavy ML libraries temporarily
# Comment out in requirements.txt:
# torch>=2.0.0
# transformers>=4.30.0
```

**Solution 2**: Upgrade plan to Standard ($25/mo)

---

### App Crashes with "Application Timeout"

**Cause**: Gunicorn timeout too short for LLM queries

**Solution**: Increase timeout in `startCommand`:
```bash
gunicorn ... --timeout 180  # 3 minutes instead of 120
```

---

### Database Connection Refused

**Cause**: DATABASE_URL not set or incorrect

**Solution**:
1. Go to PostgreSQL service → **"Info"** tab
2. Copy **"Internal Connection String"**
3. Add to web service environment variables as `DATABASE_URL`

---

### ChromaDB Data Lost After Redeploy

**Cause**: Persistent disk not mounted

**Solution**:
1. Web service → **"Settings"** → **"Disks"**
2. Add disk with mount path `/app/data`
3. Update code to use `/app/data` for ChromaDB storage

---

## 🔄 Auto-Deploy from GitHub

### Enable Auto-Deploy

1. Web service → **"Settings"** → **"Build & Deploy"**
2. Toggle **"Auto-Deploy"** → ON
3. Select branch: `main`

Now every push to `main` triggers automatic deployment.

### Manual Deploy

```bash
# In Render Dashboard
Web Service → "Manual Deploy" → "Deploy latest commit"
```

---

## 📈 Scaling Tips

### Horizontal Scaling (More Instances)

**Render Pro plan** ($85/mo) supports auto-scaling:
- Automatically adds instances during high traffic
- Scales down during low traffic
- Load balancer included

### Vertical Scaling (More RAM)

Upgrade plan:
- Starter (512 MB) → Standard (2 GB) → Pro (8 GB)

### Database Scaling

When you hit limits:
1. PostgreSQL Starter (1 GB) → Standard (50 GB)
2. Consider external managed database:
   - AWS RDS
   - Supabase
   - Neon.tech (serverless Postgres)

### Move ChromaDB to Pinecone

When embeddings exceed 2 GB disk:
1. Sign up for Pinecone ($70/mo starter)
2. Migrate embeddings (see OPTIMIZATION_STRATEGY_2026.md)
3. Remove persistent disk from Render

---

## ✅ Deployment Checklist

**Pre-Deploy**:
- [ ] Updated `render.yaml` with correct config
- [ ] Pushed code to GitHub
- [ ] Obtained OpenAI API key
- [ ] Created Render account

**During Deploy**:
- [ ] Created PostgreSQL database
- [ ] Created Redis instance
- [ ] Created web service
- [ ] Linked database and Redis
- [ ] Added all environment variables
- [ ] Added persistent disk for ChromaDB

**Post-Deploy**:
- [ ] Verified health endpoint works
- [ ] Tested API documentation page
- [ ] Updated OAuth redirect URIs
- [ ] Set up monitoring alerts
- [ ] Configured custom domain (optional)
- [ ] Enabled auto-deploy from GitHub

**Production Readiness**:
- [ ] Changed from Starter to Standard plan (no auto-sleep)
- [ ] Set up database backups
- [ ] Added error tracking (Sentry)
- [ ] Load tested with 100 concurrent requests
- [ ] Reviewed logs for errors

---

## 🆘 Support

- **Render Docs**: https://render.com/docs
- **Render Status**: https://status.render.com/
- **Community**: https://community.render.com/

---

## 🎉 Next Steps After Deployment

1. **Test all integrations** (Gmail, Slack, GitHub)
2. **Upload sample documents** and test RAG queries
3. **Invite beta users** to test the system
4. **Monitor costs** in Render Dashboard → Billing
5. **Implement Celery** for background jobs (Phase 1 of optimization plan)

---

**Your app will be live at**: `https://2ndbrain-api.onrender.com`

**Estimated deployment time**: 30 minutes
**Monthly cost**: $14-42 depending on plan
