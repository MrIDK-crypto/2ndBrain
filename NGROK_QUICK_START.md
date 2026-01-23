# 🚀 Ngrok Deployment - Quick Start Guide

The fastest way to share your UCLA 2nd Brain application on the web.

---

## ⚡ One-Command Setup (Recommended)

```bash
cd /Users/pranavreddymogathala/2ndbrainRepo
./setup_ngrok.sh
```

This will:
1. ✅ Install ngrok (if needed)
2. ✅ Configure your authtoken
3. ✅ Create a demo user
4. ✅ Update CORS settings
5. ✅ Test the backend

**Time**: ~3 minutes

---

## 🌐 Start Sharing

After setup, run:

```bash
./start_with_ngrok.sh
```

You'll see:
```
🚀 UCLA 2nd Brain - Ngrok Deployment
======================================

✅ ngrok is installed
✅ Backend started (PID: 12345)
✅ Backend is healthy

🌐 Starting ngrok tunnel...

Session Status                online
Forwarding                    https://abc-123-xyz.ngrok-free.app -> http://localhost:5000
```

**Your public URL**: Copy the `https://abc-123-xyz.ngrok-free.app` part

---

## 📤 Share With Collaborators

### What to Send

1. **The ngrok URL** (e.g., `https://abc-123-xyz.ngrok-free.app`)

2. **Login Credentials**:
   ```
   Email: demo@ucla.beat
   Password: DemoUCLA2024
   ```

3. **Instructions**: Send them `COLLABORATOR_GUIDE.md`

### Security Notes

- ✅ Demo user has **read-only** access
- ✅ Cannot delete or modify data
- ✅ Can only search and view documents
- ⚠️ Free ngrok shows warning page (users click "Visit Site")

---

## 🎯 Example Demo Flow

**You (Host)**:
```bash
# Terminal 1: Start everything
./start_with_ngrok.sh

# Copy the ngrok URL from output
# Send to collaborator along with credentials
```

**Collaborator**:
```bash
# 1. Login
curl -X POST https://YOUR-URL.ngrok-free.app/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"demo@ucla.beat","password":"DemoUCLA2024"}'

# 2. Save access_token and tenant_id from response

# 3. Ask a question
curl -X POST https://YOUR-URL.ngrok-free.app/api/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ACCESS_TOKEN" \
  -d '{"query":"What are the three tiers?","tenant_id":"TENANT_ID"}'
```

---

## 📊 Monitoring

While ngrok is running:

1. **ngrok Web Interface**: http://localhost:4040
   - See all HTTP requests
   - Inspect request/response
   - Replay requests

2. **Backend Logs**: `tail -f backend.log`
   - API requests
   - Database queries
   - Errors

---

## 🛑 Stop Sharing

Press `Ctrl+C` in the terminal running ngrok.

Everything stops immediately:
- ❌ ngrok tunnel closes
- ❌ Public URL stops working
- ✅ Backend remains safe on your machine

---

## 💰 Cost & Limits

### Free Tier (Default)
- ✅ Unlimited bandwidth
- ✅ HTTPS included
- ⚠️ Shows warning page
- ⚠️ Random URL each time
- ⚠️ 2-hour session limit

### Basic Tier ($8/month) - Optional
- ✅ No warning page
- ✅ Custom subdomain (e.g., `ucla-2ndbrain.ngrok.app`)
- ✅ No session timeouts
- ✅ IP whitelisting

**Upgrade**: https://dashboard.ngrok.com/billing/subscription

---

## 🐛 Common Issues

### "ngrok: command not found"
```bash
# Install ngrok
brew install ngrok/ngrok/ngrok
```

### "Session expired"
Free tier tunnels expire after 2 hours. Just restart:
```bash
./start_with_ngrok.sh
```
(You'll get a new URL)

### "Connection refused"
Backend isn't running. The script should start it automatically, but check:
```bash
# Test backend directly
curl http://localhost:5000/api/health
```

### "CORS error"
Make sure `.env` has:
```
CORS_ORIGINS=*
```

---

## 📖 Full Documentation

For detailed information:

- **Complete Setup**: `NGROK_DEPLOYMENT_PLAN.md`
- **Collaborator Guide**: `COLLABORATOR_GUIDE.md`
- **Security Tips**: See "Security Considerations" in NGROK_DEPLOYMENT_PLAN.md

---

## ✅ Checklist

Before sharing:

- [ ] Ran `./setup_ngrok.sh` successfully
- [ ] Demo user created (demo@ucla.beat)
- [ ] Started ngrok with `./start_with_ngrok.sh`
- [ ] Tested ngrok URL yourself
- [ ] Prepared COLLABORATOR_GUIDE.md to send
- [ ] Noted the ngrok URL to share
- [ ] Set expectation about ngrok warning page (free tier)

---

## 🎓 Sample Questions for Collaborators

Suggest they try:

1. "What are the three tiers in the concierge medicine model?"
2. "What is the pricing for Tier 2?"
3. "Who are the team members of BEAT Healthcare Consulting?"
4. "What demographics are targeted for concierge medicine?"
5. "Explain the pilot model compensation structure"

---

## 💡 Pro Tips

1. **Screen Record**: Record a quick demo video showing how to use it
2. **Zoom Demo**: Consider screen sharing instead of ngrok for live presentations
3. **Test First**: Always test the public URL yourself before sharing
4. **Time It**: Schedule demos for specific times, then shut down ngrok after
5. **Upgrade**: If doing frequent demos, $8/month for Basic plan is worth it

---

**Ready to go?** Run `./setup_ngrok.sh` to get started! 🚀
