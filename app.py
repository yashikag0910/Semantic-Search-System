from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from data_loader import load_dataset
from embedder import Embedder
from vector_store import VectorStore
from clusterer import Clusterer
from semantic_cache import SemanticCache
from search_engine import SearchEngine


# request schema for the /query endpoint
class QueryRequest(BaseModel):
    query: str


engine = None
cache = None

# using startup event so heavy initialization happens once
# this loads dataset, embeddings, clustering etc
# lifespan event handles startup and shutdown cleanly
# this replaces the old @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):

    global engine, cache

    # loading dataset
    docs = load_dataset()

    # initialize embedding model
    embedder = Embedder()

    # compute document embeddings
    doc_embeddings = embedder.embed(docs)

    # build FAISS vector index
    vector_store = VectorStore(doc_embeddings)

    # fit fuzzy clustering model
    clusterer = Clusterer()
    clusterer.fit(doc_embeddings)

    # initialize semantic cache
    cache = SemanticCache()

    # create search engine
    engine = SearchEngine(
        docs,
        embedder,
        vector_store,
        clusterer,
        cache
    )

    yield


# create fastapi app with lifespan lifecycle
app = FastAPI(
    title="Semantic Search API",
    lifespan=lifespan
)


# simple health check route
@app.get("/")
def home():
    return {"message": "Semantic Search API running"}


# main semantic search endpoint
@app.post("/query")
def query_api(request: QueryRequest):

    return engine.query(request.query)


# return cache statistics
@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


# clear the entire cache
@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "cache cleared"}