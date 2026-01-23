# 🌐 Ngrok Deployment Plan - UCLA 2nd Brain

Complete guide to expose your 2nd Brain application to the web using ngrok.

---

## 📋 Prerequisites

- ✅ Your UCLA data is already imported
- ✅ Backend server is working locally
- ⏳ ngrok account (free tier works)
- ⏳ Backend server running

---

## 🚀 Step-by-Step Deployment

### Step 1: Install ngrok

**Option A: Using Homebrew (Recommended for Mac)**
```bash
brew install ngrok/ngrok/ngrok
```

**Option B: Download Manually**
1. Go to https://ngrok.com/download
2. Download the macOS version
3. Unzip and move to `/usr/local/bin/`:
   ```bash
   unzip ~/Downloads/ngrok-*.zip
   sudo mv ngrok /usr/local/bin/ngrok
   ```

**Verify Installation**
```bash
ngrok version
# Should show: ngrok version 3.x.x
```

---

### Step 2: Sign Up for ngrok (Free)

1. Go to https://dashboard.ngrok.com/signup
2. Sign up with email or Google
3. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken

**Add Your Authtoken**
```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

---

### Step 3: Prepare the Backend

**Update CORS Settings**

The backend needs to accept requests from ngrok URLs. Edit your `.env` file:

```bash
# Add to /Users/pranavreddymogathala/2ndbrainRepo/.env

# Allow ngrok URLs
CORS_ORIGINS=*

# OR be more specific (recommended):
# CORS_ORIGINS=https://your-ngrok-subdomain.ngrok-free.app
```

---

### Step 4: Start the Backend Server

**Terminal 1: Start Backend**
```bash
cd /Users/pranavreddymogathala/2ndbrainRepo
./start_server.sh
```

Wait for:
```
* Running on http://127.0.0.1:5000
* Running on http://0.0.0.0:5000
```

**Keep this terminal running!**

---

### Step 5: Start ngrok Tunnel

**Terminal 2: Start ngrok**
```bash
ngrok http 5000
```

You'll see output like:
```
ngrok

Session Status                online
Account                       your-email@example.com
Version                       3.x.x
Region                        United States (us)
Latency                       -
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://abc123xyz.ngrok-free.app -> http://localhost:5000

Connections                   ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

**Your public URL**: `https://abc123xyz.ngrok-free.app`

**Keep this terminal running too!**

---

### Step 6: Test the Public URL

**Test Health Endpoint**
```bash
curl https://YOUR-NGROK-URL.ngrok-free.app/api/health
```

Should return:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-22T..."
}
```

---

### Step 7: Share Access Instructions

Send your collaborator:

**Public URL**: `https://YOUR-NGROK-URL.ngrok-free.app`

**Login Credentials**:
- Email: `admin@2ndbrain.local`
- Password: `admin123`

**API Examples**:

1. **Login to get access token:**
```bash
curl -X POST https://YOUR-NGROK-URL.ngrok-free.app/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@2ndbrain.local",
    "password": "admin123"
  }'
```

2. **Ask a question:**
```bash
curl -X POST https://YOUR-NGROK-URL.ngrok-free.app/api/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ACCESS_TOKEN_HERE" \
  -d '{
    "query": "What are the three tiers in the concierge medicine model?",
    "tenant_id": "TENANT_ID_FROM_LOGIN"
  }'
```

---

## 🔒 Security Considerations

### 1. Free Tier Limitations
- ⚠️ **Warning Banner**: Free ngrok URLs show a warning page before accessing
- ⏱️ **Session Timeout**: Free tunnels expire after 2 hours
- 🔄 **URL Changes**: New random URL each time you restart ngrok

### 2. Upgrade to Paid (Optional)
- ✅ No warning banner
- ✅ Custom subdomain (e.g., `ucla-2ndbrain.ngrok.app`)
- ✅ No session timeouts
- 💰 Cost: $8/month (Basic plan)

### 3. Protect Sensitive Data
The current setup exposes your database. Consider:

**Option A: Create a Demo User (Recommended)**
```bash
# Run this Python script to create a demo user:
cd /Users/pranavreddymogathala/2ndbrainRepo

DATABASE_URL="sqlite:///./2ndbrain_ucla.db" ./venv/bin/python << EOF
from backend.database.database import SessionLocal
from backend.database.models import User, UserRole
import bcrypt

db = SessionLocal()

# Check if demo user exists
demo_user = db.query(User).filter_by(email="demo@ucla.beat").first()

if not demo_user:
    # Get tenant ID (UCLA BEAT Healthcare)
    from backend.database.models import Tenant
    tenant = db.query(Tenant).filter_by(slug="ucla-beat-healthcare").first()

    # Hash password
    password = "DemoUCLA2024"
    salt = bcrypt.gensalt(rounds=12)
    password_hash = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    # Create demo user with viewer role
    demo_user = User(
        tenant_id=tenant.id,
        email="demo@ucla.beat",
        name="Demo User",
        role=UserRole.VIEWER,
        is_active=True,
        password_hash=password_hash
    )

    db.add(demo_user)
    db.commit()
    print("✅ Demo user created: demo@ucla.beat / DemoUCLA2024")
else:
    print("ℹ️  Demo user already exists")

db.close()
EOF
```

**Share these credentials instead**:
- Email: `demo@ucla.beat`
- Password: `DemoUCLA2024`

**Option B: IP Whitelist (ngrok Paid)**
Only allow specific IP addresses to access your tunnel.

**Option C: Add Basic Auth (ngrok Built-in)**
```bash
ngrok http 5000 --basic-auth "username:password"
```

Users must enter credentials to access the ngrok URL.

---

## 🎨 Optional: Custom Domain (Paid Plan)

If you upgrade to ngrok Basic ($8/month):

```bash
ngrok http 5000 --domain=ucla-2ndbrain.ngrok.app
```

Your URL stays the same every time!

---

## 📊 Monitoring & Logs

### ngrok Web Interface
- **URL**: http://localhost:4040
- **Features**:
  - See all HTTP requests in real-time
  - Inspect request/response details
  - Replay requests for debugging

### Backend Logs
Watch the Terminal 1 (backend) for:
- API requests
- Database queries
- Errors and warnings

---

## 🛑 Stopping the Service

1. **Stop ngrok**: Press `Ctrl+C` in Terminal 2
2. **Stop backend**: Press `Ctrl+C` in Terminal 1

Your public URL will immediately stop working.

---

## 🔄 Restarting After Reboot

**Quick Restart Script** (saves typing):

```bash
#!/bin/bash
# Save as: /Users/pranavreddymogathala/2ndbrainRepo/start_with_ngrok.sh

# Start backend in background
cd /Users/pranavreddymogathala/2ndbrainRepo
DATABASE_URL="sqlite:///./2ndbrain_ucla.db" ./venv/bin/python -m backend.api.app &
BACKEND_PID=$!

echo "✅ Backend started (PID: $BACKEND_PID)"
echo "⏳ Waiting 5 seconds for backend to initialize..."
sleep 5

# Start ngrok
echo "🌐 Starting ngrok tunnel..."
ngrok http 5000

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
```

Then just run:
```bash
chmod +x start_with_ngrok.sh
./start_with_ngrok.sh
```

---

## 🎯 Quick Reference

| Action | Command |
|--------|---------|
| Install ngrok | `brew install ngrok/ngrok/ngrok` |
| Add authtoken | `ngrok config add-authtoken TOKEN` |
| Start backend | `./start_server.sh` |
| Start ngrok | `ngrok http 5000` |
| View requests | Open http://localhost:4040 |
| Stop everything | `Ctrl+C` in both terminals |

---

## 🐛 Troubleshooting

### "ERR_NGROK_108: Session expired"
**Problem**: Free tier tunnels expire after 2 hours.
**Solution**: Restart ngrok (you'll get a new URL).

### "Failed to dial backend: connection refused"
**Problem**: Backend server isn't running.
**Solution**: Start backend first, then ngrok.

### "CORS error" in browser
**Problem**: Backend not configured for ngrok URL.
**Solution**: Add ngrok URL to CORS_ORIGINS in `.env`.

### Someone sees "ngrok warning page"
**Problem**: Free tier shows interstitial page.
**Solution**: Upgrade to Basic plan ($8/mo) or ask them to click "Visit Site".

### URL keeps changing
**Problem**: Free tier gives random URL each time.
**Solution**: Upgrade to Basic plan for custom subdomain.

---

## 💡 Tips for Demo

1. **Prepare Sample Questions** - Send your collaborator a list of good questions to ask
2. **Test Before Sharing** - Verify everything works from another device/network
3. **Use Screen Sharing** - For live demos, consider Zoom/Meet instead of ngrok
4. **Document Everything** - Send clear instructions with screenshots
5. **Set Expectations** - Warn about the ngrok warning page on free tier

---

## 📞 Support

- ngrok Docs: https://ngrok.com/docs
- ngrok Dashboard: https://dashboard.ngrok.com
- ngrok Status: https://status.ngrok.com

---

## 🔐 Security Checklist

Before sharing:
- [ ] Create demo user with limited permissions
- [ ] Test public URL yourself
- [ ] Check CORS settings
- [ ] Review backend logs for errors
- [ ] Consider IP whitelist if ngrok paid
- [ ] Don't share admin credentials
- [ ] Set up basic auth if needed
- [ ] Plan to stop tunnel after demo

---

**Next Steps**: Follow Step 1 to install ngrok!
