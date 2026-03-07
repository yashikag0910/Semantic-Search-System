class SearchEngine:

    def __init__(self, docs, embedder, vector_store, clusterer, cache):

        self.docs = docs
        self.embedder = embedder
        self.vector_store = vector_store
        self.clusterer = clusterer
        self.cache = cache

    def query(self, text):

        # convert query into embedding
        query_emb = self.embedder.embed_query(text)

        # determine which cluster this query belongs to
        cluster = self.clusterer.dominant_cluster(query_emb)

        self.cache.add(text, query_emb, result_text)

        # if cache miss perform vector search
        scores, indices = self.vector_store.search(query_emb)

        results = [self.docs[i] for i in indices[:3]]

        result_text = " ".join(results)

        # store result in cache
        self.cache.add(text, query_emb, result_text, cluster)

        return {
            "query": text,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": result_text,
            "dominant_cluster": cluster
        }