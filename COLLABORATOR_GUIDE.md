# 🎓 UCLA BEAT Healthcare - 2nd Brain Access Guide

Welcome! You've been invited to access the UCLA BEAT Healthcare knowledge system.

---

## 🌐 Access Information

**Public URL**: `https://YOUR-NGROK-URL.ngrok-free.app`
*(The person who shared this with you will provide the actual URL)*

**Login Credentials**:
- Email: `demo@ucla.beat`
- Password: `DemoUCLA2024`

---

## 🚀 Quick Start

### Step 1: Access the URL

1. Open the provided ngrok URL in your browser
2. You may see an ngrok warning page (free tier)
3. Click **"Visit Site"** to continue

### Step 2: Get Your Access Token

Use this command in your terminal (replace `YOUR-NGROK-URL`):

```bash
curl -X POST https://YOUR-NGROK-URL.ngrok-free.app/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "demo@ucla.beat",
    "password": "DemoUCLA2024"
  }'
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "user": {
    "id": "...",
    "email": "demo@ucla.beat",
    "role": "viewer"
  },
  "tenant_id": "c1eaca5c-4bbf-4d75-bd4b-b0ae4e0cc11a"
}
```

**Save these**:
- `access_token` - Use this for API requests
- `tenant_id` - Include in search queries

---

## 📋 Available Data

The system contains information about:

- **BEAT Healthcare Consulting** company overview
- **Concierge Medicine Business Plan**
- **Three-Tiered Service Model** with pricing
- **Demographics Analysis** (older vs younger populations)
- **Pilot Model** compensation structure
- **Team Members** and roles

---

## 💬 How to Ask Questions

### Using API

Replace placeholders with your actual values:

```bash
curl -X POST https://YOUR-NGROK-URL.ngrok-free.app/api/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -d '{
    "query": "What are the three tiers in the concierge medicine model?",
    "tenant_id": "YOUR_TENANT_ID"
  }'
```

### Sample Questions to Try

1. **Pricing Information**:
   - "What is the pricing for each tier?"
   - "How much does Tier 2 cost?"

2. **Service Details**:
   - "What benefits are included in Tier 3?"
   - "What's the difference between Tier 1 and Tier 2?"

3. **Company Information**:
   - "Who founded BEAT Healthcare Consulting?"
   - "Who are the team members?"

4. **Demographics**:
   - "What demographics are targeted for concierge medicine?"
   - "What did the study show about older vs younger populations?"

5. **Business Model**:
   - "Explain the pilot model compensation structure"
   - "How does the concierge premium work?"

---

## 🔧 API Endpoints

### Authentication
- **Login**: `POST /api/auth/login`
- **Refresh Token**: `POST /api/auth/refresh`
- **Logout**: `POST /api/auth/logout`
- **Get User Info**: `GET /api/auth/me`

### Search & Chat
- **Search Documents**: `POST /api/search`
- **List Documents**: `GET /api/documents`
- **Get Document**: `GET /api/documents/{id}`

### Health Check
- **System Health**: `GET /api/health`

---

## 📊 Response Format

### Successful Search Response

```json
{
  "answer": "The three-tiered concierge medicine model includes...",
  "sources": [
    {
      "document_id": "...",
      "title": "BEAT Concierge Medicine Meeting",
      "relevance_score": 0.95,
      "excerpt": "..."
    }
  ],
  "query": "What are the three tiers...",
  "processing_time": 1.234
}
```

---

## 🔒 Your Permissions

As a **VIEWER**, you can:
- ✅ Search and read all documents
- ✅ Ask questions via the chatbot
- ✅ View document metadata

You **cannot**:
- ❌ Upload or delete documents
- ❌ Modify system settings
- ❌ Create new users
- ❌ Access admin features

---

## 🐛 Troubleshooting

### "401 Unauthorized" Error
**Problem**: Access token expired (15 minutes).
**Solution**: Login again to get a new token.

### "ngrok warning page"
**Problem**: Free ngrok tier shows interstitial page.
**Solution**: Click "Visit Site" button.

### "Connection refused"
**Problem**: The backend server is offline.
**Solution**: Contact the person who shared the link.

### "CORS error"
**Problem**: Browser blocking cross-origin request.
**Solution**: Use curl/Postman instead of browser for API calls.

---

## 💡 Tips

1. **Save Your Tokens**: Access tokens expire in 15 minutes, refresh tokens last 7 days
2. **Use Postman**: Easier than curl for testing APIs
3. **Check Response Times**: First query may be slow (cold start)
4. **Be Specific**: More detailed questions get better answers
5. **Cite Sources**: Responses include source documents for verification

---

## 📞 Need Help?

- **Technical Issues**: Contact the person who shared this link
- **Questions About Data**: Refer to the BEAT Healthcare Consulting presentation
- **API Documentation**: See the main README in the project repository

---

## ⏰ Session Duration

- Free ngrok tunnels expire after **2 hours**
- You'll get a new URL when the tunnel restarts
- Your access token expires after **15 minutes** (renewable)

---

## 🎯 Example Workflow

1. **Login** to get access token
2. **Ask a question** about concierge medicine pricing
3. **Review sources** to verify the answer
4. **Refine query** for more specific information
5. **Export findings** (copy-paste responses)

---

**Enjoy exploring the UCLA BEAT Healthcare knowledge base!** 🎓
