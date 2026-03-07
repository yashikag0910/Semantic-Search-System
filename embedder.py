from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:

    def __init__(self):

        # this model is small but performs surprisingly well
        # it generates 384 dimensional sentence embeddings
        # also loads fast which matters for this assignment
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # embed a list of documents
    def embed(self, texts):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return np.array(embeddings)

    # embed a single query
    def embed_query(self, text):

        emb = self.model.encode(
            [text],
            normalize_embeddings=True
        )

        return emb[0]