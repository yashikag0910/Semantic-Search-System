# Semantic Search System with Fuzzy Clustering and Semantic Cache
This project implements a lightweight **semantic search system** using the **20 Newsgroups dataset** (~20,000 discussion posts).  
Instead of relying on keyword matching, the system retrieves documents based on **semantic similarity using vector embeddings**.
The system combines:
- SentenceTransformer embeddings
- FAISS vector search
- fuzzy clustering using Gaussian Mixture Models
- a semantic caching layer
- a FastAPI service
- Docker containerization
The main goal is to demonstrate how semantic representations, clustering, and caching can work together to build a scalable search system.
---
# Dataset
Dataset used:
**20 Newsgroups Dataset**
Source:
https://archive.ics.uci.edu/dataset/113/twenty+newsgroups
The dataset contains roughly **20,000 documents across 20 topics**, including:
- politics
- religion
- technology
- sports
- science
- firearms
The dataset contains noise such as email headers and quoted replies, so basic preprocessing is applied before generating embeddings.
---
# System Architecture
The overall pipeline works as follows:
User Query
│
▼
SentenceTransformer Embedding
│
▼
Semantic Cache (FAISS)
│
┌─┴────────────┐
│              │
Cache Hit    Cache Miss
│              │
▼              ▼
Return result   Vector Database Search (FAISS)
│
▼
Retrieve Documents
│
▼
Store Result in Cache
│
▼
Response
---
# Key Components
## 1. Text Preprocessing
File: `data_loader.py`
The dataset contains many artifacts that are not useful for semantic analysis, including:
- email headers
- quoted replies
- signatures
- excessive whitespace
These are removed during preprocessing.
Very short documents are also filtered out since they produce poor embeddings.
---
# 2. Sentence Embeddings
File: `embedder.py`
The system uses the following embedding model:
sentence-transformers/all-MiniLM-L6-v2
Reasons for choosing this model:
- strong semantic similarity performance
- small embedding dimension (384)
- fast inference time
- widely used in semantic search systems
Each document and query is converted into a dense vector representation.
---
# 3. Vector Database
File: `vector_store.py`
Vector search is implemented using **FAISS (Facebook AI Similarity Search)**.
FAISS enables fast nearest-neighbor search in high dimensional vector spaces.
Document embeddings are stored inside a FAISS index:
IndexFlatIP
Using normalized embeddings allows inner product similarity to behave like cosine similarity.
---
# 4. Fuzzy Clustering
File: `clusterer.py`
Instead of assigning documents to a single topic, the system uses **Gaussian Mixture Models (GMM)**.
This allows **probabilistic cluster assignments**.
This reflects the real structure of discussions where topics often overlap.
---
# Automatic Cluster Selection
Instead of manually choosing the number of clusters, the system evaluates multiple cluster counts using **BIC (Bayesian Information Criterion)**.
The cluster count with the lowest BIC score is selected automatically.
This avoids arbitrary cluster selection and provides evidence-based clustering.
---
# 5. Semantic Cache
File: `semantic_cache.py`
Traditional caching relies on exact query matching.
Example:
“What is gun control?”
“Explain firearm regulation”
Even though these queries mean the same thing, a traditional cache would treat them as different.
The semantic cache solves this by:
1. embedding the query
2. comparing it to previously seen query embeddings
3. returning cached results if similarity exceeds a threshold
Similarity threshold used:
0.82
---
# FAISS Cache Optimization
Instead of scanning all cached queries, cached embeddings are stored in a **FAISS index**.
This allows fast similarity lookup even when the cache grows large.
---
# 6. Search Engine
File: `search_engine.py`
The search engine coordinates the entire pipeline:
1. embed query
2. determine dominant cluster
3. check semantic cache
4. if cache hit → return stored result
5. if cache miss → perform vector search
6. store result in cache
7. return response
---
# 7. FastAPI Service
File: `app.py`
The system exposes a REST API using FastAPI.
### Query Endpoint
POST /query
GET /cache/stats
DELETE /cache
##Cluster Visualization
File: visualize_clusters.py
UMAP is used to reduce high dimensional embeddings into 2D space.
This allows visualizing how documents cluster semantically across topics.
#######################################################
Running the Project Locally
################################################
Clone repository
git clone https://github.com/yashikag0910/Semantic-Search-System.git
cd Semantic-Search-System

Create virtual environment
python -m venv venv
source venv/bin/activate

Install dependencies
pip install -r requirements.txt

Start API
python -m uvicorn app:app --reload
Open:
http://127.0.0.1:8000/docs

Running with Docker
Build container
docker build -t semantic-search .

Run container
docker run -p 8000:8000 semantic-search
Open:
http://127.0.0.1:8000/docs
⸻
Possible Future Improvements
	•	precompute embeddings and load from disk
	•	persistent cache storage
	•	LRU cache eviction policy
	•	hybrid keyword + semantic search
	•	improved visualization tools
⸻
Conclusion

This project demonstrates how semantic embeddings, probabilistic clustering, and intelligent caching can be combined to build a practical semantic search system.
It highlights the interaction between machine learning models, vector databases, caching systems, and API services in a unified architecture.
