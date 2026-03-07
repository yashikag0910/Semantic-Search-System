from sklearn.mixture import GaussianMixture
import numpy as np


class Clusterer:

    def __init__(self, min_clusters=10, max_clusters=40):

        # instead of fixing number of clusters manually
        # we search for the best number using BIC score
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.model = None
        self.n_clusters = None


    def find_optimal_clusters(self, embeddings):

        # BIC penalizes overly complex models
        # lower score means better balance between fit and complexity
        best_bic = float("inf")
        best_k = None

        for k in range(self.min_clusters, self.max_clusters + 1):

            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42
            )

            gmm.fit(embeddings)

            bic = gmm.bic(embeddings)

            if bic < best_bic:
                best_bic = bic
                best_k = k

        return best_k


    def fit(self, embeddings):

        # determine best cluster count automatically
        self.n_clusters = self.find_optimal_clusters(embeddings)

        print(f"Selected number of clusters: {self.n_clusters}")

        self.model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",
            random_state=42
        )

        self.model.fit(embeddings)


    def get_distribution(self, embedding):

        probs = self.model.predict_proba([embedding])[0]
        return probs


    def dominant_cluster(self, embedding):

        probs = self.get_distribution(embedding)
        return int(np.argmax(probs))