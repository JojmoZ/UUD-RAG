# UUD-RAG: Indonesian Legal Document RAG System

A Retrieval-Augmented Generation system for Indonesian legal documents with multiple vector database options.

## Quick Start

### Create Environment
```bash
conda create -n rag-uud45 python=3.12 -y
conda activate rag-uud45
pip install -r requirements.txt
```

### Start Qdrant (Optional - for Qdrant database)
```bash
docker-compose up -d
```

## Vector Database Options

This system supports two vector database backends:

### 1. Qdrant (Advanced)
- Multi-vector search (dense + sparse + late interaction)
- Hybrid search with RRF fusion
- Requires Docker container
- Best for production deployments

### 2. FAISS (Simple & Local)
- Dense vector search only
- Local file-based storage
- No external dependencies
- Best for development and simple deployments

## Usage

### Using FAISS
```python
# Set USE_FAISS = True in main.py
python main.py
```

### Using Qdrant
```python
# Set USE_FAISS = False in main.py
python main.py
```