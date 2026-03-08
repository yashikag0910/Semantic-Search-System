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

        # determine dominant cluster
        cluster = self.clusterer.dominant_cluster(query_emb)

        # check semantic cache first
        entry, score, hit = self.cache.lookup(query_emb)

        if hit and entry is not None:
            return {
                "query": text,
                "cache_hit": True,
                "matched_query": entry.query,
                "similarity_score": float(score),
                "result": entry.result,
                "dominant_cluster": cluster
            }

        # if cache miss → perform vector search
        scores, indices = self.vector_store.search(query_emb)

        results = [self.docs[i] for i in indices[:3]]

        # combine results into one text block
        result_text = " ".join(results)

        # store in semantic cache
        self.cache.add(text, query_emb, result_text)

        return {
            "query": text,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
            "result": result_text,
            "dominant_cluster": cluster
        }