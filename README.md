# KnowledgeVault

A comprehensive enterprise knowledge management system that captures, organizes, and makes searchable tacit and explicit knowledge from employee documents using a hierarchical RAG (Retrieval-Augmented Generation) framework.

## Overview

KnowledgeVault processes unstructured employee data (emails, documents) through multiple stages:

1. **Data Unclustering** - Flattens organized data into a single corpus
2. **Employee Clustering** - Groups documents by employee
3. **Project Clustering** - Uses BERTopic to semantically cluster into projects
4. **Work/Personal Classification** - Uses GPT-4o-mini to filter personal content
5. **Gap Analysis** - Identifies missing knowledge and information gaps
6. **Question Generation** - Creates targeted questions to extract tacit knowledge
7. **Knowledge Graph** - Builds Neo4j graph of relationships
8. **Vector Database** - Creates ChromaDB embeddings for semantic search
9. **Hierarchical RAG** - Combines graph + vector search for intelligent querying
10. **Content Generation** - Creates PowerPoint presentations and training videos

## Project Structure

```
2ndbrainRepo/
├── backend/                    # Backend Python modules
│   ├── api/                    # Flask web application
│   │   └── app.py              # Main Flask app (unified)
│   ├── classification/         # Work/personal classification
│   │   ├── work_personal_classifier.py
│   │   ├── project_classifier.py
│   │   └── global_project_classifier.py
│   ├── clustering/             # Document clustering
│   │   ├── employee_clustering.py
│   │   ├── project_clustering.py
│   │   ├── intelligent_project_clustering.py
│   │   └── llm_first_clusterer.py
│   ├── content_generation/     # PowerPoint/video generation
│   │   ├── powerpoint_generator.py
│   │   ├── video_generator.py
│   │   └── gamma_presentation.py
│   ├── data_processing/        # Data parsing
│   │   └── enron_parser.py
│   ├── gap_analysis/           # Knowledge gap detection
│   │   ├── gap_analyzer.py
│   │   └── question_generator.py
│   ├── integrations/           # External API connectors
│   │   ├── gmail_connector.py
│   │   ├── slack_connector.py
│   │   ├── github_connector.py
│   │   ├── base_connector.py
│   │   └── connector_manager.py
│   ├── knowledge_graph/        # Graph & vector databases
│   │   ├── knowledge_graph.py  # Neo4j builder
│   │   └── vector_database.py  # ChromaDB builder
│   ├── parsing/                # Document parsers
│   │   ├── document_parser.py
│   │   └── llamaparse_parser.py
│   ├── rag/                    # RAG implementations
│   │   ├── hierarchical_rag.py
│   │   ├── enhanced_rag.py
│   │   ├── enhanced_rag_v2.py
│   │   ├── semantic_chunker.py
│   │   ├── stakeholder_graph.py
│   │   └── multimodal.py
│   ├── document_manager.py     # Document lifecycle management
│   ├── message_filter_v2.py    # Message filtering
│   └── incremental_indexer.py  # Incremental indexing
├── frontend/                   # Frontend assets
│   ├── templates/              # HTML templates
│   └── static/                 # CSS/JS files
├── config/                     # Configuration
│   └── config.py
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md
│   ├── QUICKSTART.md
│   └── SECURITY_ANALYSIS.md
├── scripts/                    # Utility scripts
│   ├── test_setup.py
│   └── verify_setup.py
├── data/                       # Processed data (runtime)
├── club_data/                  # Personal knowledge base
├── output/                     # Generated outputs
├── main.py                     # Master orchestration script
├── requirements.txt            # Python dependencies
├── .env.template               # Environment template
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- Neo4j (optional, for knowledge graph)
- FFmpeg (for video generation)

### Setup

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Configure environment**

```bash
cp .env.template .env
# Edit .env and add your API keys
```

Required environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `ENRON_MAILDIR` - Path to Enron dataset
- `NEO4J_URI` - Neo4j connection URI (optional)
- `NEO4J_USER` - Neo4j username (optional)
- `NEO4J_PASSWORD` - Neo4j password (optional)

3. **Install FFmpeg** (for video generation)

```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg
```

## Usage

### Full Pipeline

Run the complete pipeline:

```bash
python main.py
```

### Pipeline Options

```bash
# Test with limited data
python main.py --limit 1000

# Skip work/personal classification (saves API costs)
python main.py --skip-classification

# Skip video generation (faster processing)
python main.py --skip-videos

# Run with interactive RAG chatbot
python main.py --interactive-rag

# Combine options
python main.py --limit 500 --skip-videos --interactive-rag
```

### Web Application

Run the Flask web application:

```bash
python -m backend.api.app
```

Then open http://localhost:5000 in your browser.

### Individual Components

Run components separately:

```bash
# Data processing
python -m backend.data_processing.enron_parser

# Clustering
python -m backend.clustering.employee_clustering
python -m backend.clustering.project_clustering

# Classification
python -m backend.classification.work_personal_classifier

# Gap analysis
python -m backend.gap_analysis.gap_analyzer
python -m backend.gap_analysis.question_generator

# Knowledge graph & vector database
python -m backend.knowledge_graph.knowledge_graph
python -m backend.knowledge_graph.vector_database

# RAG system
python -m backend.rag.hierarchical_rag

# Content generation
python -m backend.content_generation.powerpoint_generator
python -m backend.content_generation.video_generator
```

## Technology Stack

### Core ML/NLP
- **BERTopic** - Advanced topic modeling with HDBSCAN clustering
- **Sentence Transformers** - Document embeddings (all-mpnet-base-v2)
- **OpenAI GPT-4o-mini** - Classification, gap analysis, question generation, RAG

### Databases
- **ChromaDB** - Vector database for semantic search
- **Neo4j** - Graph database for knowledge relationships (optional)
- **Pinecone** - Scalable vector database (optional)

### Web Framework
- **Flask** - Web application server
- **Flask-CORS** - Cross-origin support

### Content Generation
- **python-pptx** - PowerPoint generation
- **gTTS** - Text-to-speech for narration
- **MoviePy** - Video assembly and rendering

### Integrations
- **Gmail API** - Email ingestion
- **Slack API** - Message collection
- **GitHub API** - Repository analysis

## Key Features

### 1. BERTopic Clustering
- Automatically discovers project clusters without predefined categories
- Uses UMAP for dimensionality reduction
- HDBSCAN for density-based clustering
- Generates interpretable topic labels

### 2. Hierarchical RAG
- **Entity Extraction** - Identifies employees, projects, topics from queries
- **Graph Traversal** - Finds relevant clusters via knowledge graph
- **Scoped Retrieval** - Searches only within relevant clusters
- **Context-Aware Generation** - Synthesizes answers with citations

### 3. Gap Analysis
- Identifies missing document types
- Detects knowledge gaps and context gaps
- Generates targeted questions to fill gaps
- Creates employee questionnaires

### 4. Privacy-First Classification
- Distinguishes work from personal content
- Confidence-based filtering (>0.85 threshold)
- Flags uncertain content for human review
- Removes personal data before indexing

### 5. Multi-Source Integration
- Gmail inbox integration
- Slack message collection
- GitHub repository analysis
- File upload support (PDF, DOCX, TXT, etc.)

## Output Artifacts

After running the pipeline, you'll find:

### Data Outputs (`/data`)
- `unclustered/` - Flattened JSONL documents
- `employee_clusters/` - Documents grouped by employee
- `project_clusters/` - Documents clustered by project
- `classified/` - Work vs personal classification results

### Analysis Outputs (`/output`)
- `gap_analysis/` - Knowledge gap reports (JSON)
- `questionnaires/` - Employee questionnaires (JSON + TXT)
- `powerpoints/` - Training presentations (PPTX)
- `videos/` - Training videos (MP4)
- `reports/` - Various statistics and summaries

## API Costs

Approximate OpenAI API costs (GPT-4o-mini):

- **Classification** (50 docs): ~$0.05
- **Gap Analysis** (10 projects): ~$0.10
- **Question Generation** (10 projects): ~$0.10
- **RAG Queries** (10 queries): ~$0.05

Total estimated cost for 1000 documents: **~$0.50 - $1.00**

Use `--skip-classification` and `--limit` flags to reduce costs during testing.

## Troubleshooting

### Neo4j Connection Failed
If Neo4j is not installed or not running, the system will continue without graph functionality.

### FFmpeg Not Found
Video generation requires FFmpeg. Install it or use `--skip-videos` flag.

### Out of Memory
For large datasets, use the `--limit` flag to process fewer documents.

### API Rate Limits
Add delays or use smaller batches if hitting OpenAI rate limits.

## License

This project is for educational and research purposes.

---

Built with KnowledgeVault - Enterprise Knowledge Continuity Platform
