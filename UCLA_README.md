# 🎓 UCLA BEAT Healthcare - 2nd Brain Setup

Your UCLA BEAT Concierge Medicine data has been successfully imported!

## ✅ What's Imported

- **1 Document**: BEAT Concierge Medicine Meeting (PowerPoint presentation)
- **Organization**: UCLA BEAT Healthcare
- **Admin User**: `admin@2ndbrain.local` (password: `admin123`)
- **Database**: `2ndbrain_ucla.db` (SQLite)
- **Vector Store**: ChromaDB (local, no Pinecone needed for this version)

## 🚀 How to Use

### Step 1: Start the Backend Server

```bash
cd /Users/pranavreddymogathala/2ndbrainRepo
./start_server.sh
```

The server will start on **http://localhost:5000**

### Step 2: Access the Application

Open your browser and navigate to:
- **API Health Check**: http://localhost:5000/api/health
- **Login Endpoint**: http://localhost:5000/api/auth/login

### Step 3: Get Access Token

Use this curl command to login and get your access token:

```bash
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@2ndbrain.local",
    "password": "admin123"
  }'
```

You'll receive a response with an `access_token`. Save it!

### Step 4: Ask Questions About Your Data

Use the chatbot to ask questions about the BEAT Concierge Medicine presentation:

```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN_HERE" \
  -d '{
    "query": "What is the three-tiered service model for concierge medicine?",
    "tenant_id": "TENANT_ID_FROM_LOGIN"
  }'
```

## 📋 Example Questions You Can Ask

Based on your imported document, try asking:

1. "What are the three tiers in the concierge medicine model?"
2. "What is the pricing for each tier?"
3. "Who founded BEAT Healthcare Consulting?"
4. "What demographics are targeted for concierge medicine?"
5. "What benefits are included in Tier 2?"
6. "What is the pilot model compensation structure?"

## 🔧 Next Steps

### Option A: Build the Search Index (Recommended)

To enable AI-powered semantic search with embeddings:

1. Make sure you have your OpenAI API key in `.env`
2. Run the index rebuild:

```bash
cd /Users/pranavreddymogathala/2ndbrainRepo
DATABASE_URL="sqlite:///./2ndbrain_ucla.db" ./venv/bin/python -c "
from backend.knowledge_graph.vector_database import VectorDatabaseBuilder
from backend.database.database import SessionLocal
from backend.database.models import Document

db = SessionLocal()
docs = db.query(Document).all()

vdb = VectorDatabaseBuilder(
    persist_directory='./chroma_db',
    use_openai_embeddings=True,
    openai_api_key='YOUR_OPENAI_API_KEY'
)

for doc in docs:
    vdb.add_document({
        'id': str(doc.id),
        'title': doc.title,
        'content': doc.content,
        'metadata': doc.doc_metadata
    })

print(f'✅ Indexed {len(docs)} documents')
"
```

### Option B: Add More Data

Import more JSONL files using the same script:

```bash
./venv/bin/python import_jsonl.py "/path/to/your/data.jsonl" "UCLA BEAT Healthcare"
```

### Option C: Use Pinecone (Already Configured!)

Your Pinecone credentials are already in `.env`:
- API Key: `pcsk_7QD7v6_aMkk...`
- Index: `2ndbrain`
- Host: `https://2ndbrain-nqzj8ny.svc.aped-4627-b74a.pinecone.io`

To switch from ChromaDB to Pinecone, you'd need to install the Pinecone library and update the RAG implementation (this version uses ChromaDB by default).

## 📁 File Locations

- **Database**: `/Users/pranavreddymogathala/2ndbrainRepo/2ndbrain_ucla.db`
- **Environment**: `/Users/pranavreddymogathala/2ndbrainRepo/.env`
- **Backend Code**: `/Users/pranavreddymogathala/2ndbrainRepo/backend/`
- **Import Script**: `/Users/pranavreddymogathala/2ndbrainRepo/import_jsonl.py`

## 🐛 Troubleshooting

### Server Won't Start
```bash
# Check if port 5000 is in use
lsof -i :5000

# Kill the process if needed
kill -9 PID_FROM_ABOVE
```

### Database Issues
```bash
# Reinitialize the database
rm 2ndbrain_ucla.db
./venv/bin/python import_jsonl.py "/path/to/your.jsonl"
```

### Python Environment Issues
```bash
# Reinstall dependencies
./venv/bin/pip install -r requirements.txt
```

## 📞 Support

For issues or questions, check the main project documentation:
- `/Users/pranavreddymogathala/2ndbrainRepo/README.md`

---

**Note**: This is the simpler ChromaDB version (2ndbrainRepo). The Pinecone version (2ndBrainFinal) requires database schema migrations to match the Supabase PostgreSQL setup.
