import faiss
import numpy as np


class VectorStore:

    def __init__(self, embeddings):

        # FAISS requires float32 vectors
        self.embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]

        # using inner product index (with normalized vectors this becomes cosine similarity)
        self.index = faiss.IndexFlatIP(dim)

        # store document embeddings
        self.index.add(self.embeddings)

    # retrieve k nearest documents
    def search(self, query_embedding, k=5):

        query = np.array([query_embedding]).astype("float32")

        scores, indices = self.index.search(query, k)

        return scores[0], indices[0]