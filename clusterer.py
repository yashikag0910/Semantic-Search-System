from sklearn.mixture import GaussianMixture
import numpy as np


class Clusterer:

    def __init__(self, n_clusters=27):

        # gaussian mixture model allows soft clustering
        # each document can belong to multiple clusters with probabilities
        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=42
        )

    def fit(self, embeddings):

        # learn cluster distributions
        self.model.fit(embeddings)

    # returns probability distribution across clusters
    def get_distribution(self, embedding):

        probs = self.model.predict_proba([embedding])[0]

        return probs

    # return the most likely cluster
    def dominant_cluster(self, embedding):

        probs = self.get_distribution(embedding)

        return int(np.argmax(probs))