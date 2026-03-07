import numpy as np


# simple cosine similarity function
def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class CacheEntry:

    def __init__(self, query, embedding, result, cluster):

        self.query = query
        self.embedding = embedding
        self.result = result
        self.cluster = cluster


class SemanticCache:

    def __init__(self, threshold=0.82):

        # threshold determines when two queries are "similar enough"
        self.threshold = threshold

        # cluster -> list of cache entries
        self.cache = {}

        self.hit_count = 0
        self.miss_count = 0

    # check if query already exists semantically
    def lookup(self, query_embedding, cluster):

        if cluster not in self.cache:
            self.miss_count += 1
            return None, None, None

        best_score = 0
        best_entry = None

        # search only entries within the same cluster
        for entry in self.cache[cluster]:

            score = cosine_similarity(query_embedding, entry.embedding)

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold:
            self.hit_count += 1
            return best_entry, best_score, True

        self.miss_count += 1
        return None, best_score, False

    # store new cache entry
    def add(self, query, embedding, result, cluster):

        entry = CacheEntry(query, embedding, result, cluster)

        if cluster not in self.cache:
            self.cache[cluster] = []

        self.cache[cluster].append(entry)

    # return cache statistics
    def stats(self):

        total = self.hit_count + self.miss_count

        return {
            "total_entries": sum(len(v) for v in self.cache.values()),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.hit_count / total if total > 0 else 0
        }

    # clear the cache completely
    def clear(self):

        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0