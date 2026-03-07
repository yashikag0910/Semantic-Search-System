import numpy as np
import faiss


class CacheEntry:

    def __init__(self, query, embedding, result):

        self.query = query
        self.embedding = embedding
        self.result = result


class SemanticCache:

    def __init__(self, threshold=0.82):

        # similarity threshold determines when two queries are "close enough"
        self.threshold = threshold

        self.entries = []

        # statistics
        self.hit_count = 0
        self.miss_count = 0

        # FAISS index used for fast similarity lookup
        self.index = None
        self.dim = None


    def initialize_index(self, embedding_dim):

        self.dim = embedding_dim

        # using cosine similarity via inner product
        self.index = faiss.IndexFlatIP(self.dim)


    def lookup(self, query_embedding):

        if self.index is None or len(self.entries) == 0:
            self.miss_count += 1
            return None, None, False

        query = np.array([query_embedding]).astype("float32")

        scores, indices = self.index.search(query, 1)

        best_score = scores[0][0]
        best_index = indices[0][0]

        if best_score >= self.threshold:

            self.hit_count += 1
            entry = self.entries[best_index]

            return entry, best_score, True

        self.miss_count += 1
        return None, best_score, False


    def add(self, query, embedding, result):

        entry = CacheEntry(query, embedding, result)

        self.entries.append(entry)

        emb = np.array([embedding]).astype("float32")

        if self.index is None:
            self.initialize_index(len(embedding))

        self.index.add(emb)


    def stats(self):

        total = self.hit_count + self.miss_count

        return {
            "total_entries": len(self.entries),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total > 0 else 0
        }


    def clear(self):

        self.entries = []
        self.index = None
        self.hit_count = 0
        self.miss_count = 0