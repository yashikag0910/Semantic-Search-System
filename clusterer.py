from sklearn.mixture import GaussianMixture
import numpy as np


class Clusterer:

    def __init__(self, min_clusters=10, max_clusters=40):

        # instead of choosing cluster count manually,
        # we evaluate several possibilities and select the best one using BIC
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        self.model = None
        self.n_clusters = None


    def find_optimal_clusters(self, embeddings):

        # BIC (Bayesian Information Criterion) helps balance
        # model complexity vs how well the model fits the data
        # lower BIC = better tradeoff

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

            # keep track of the best performing cluster count
            if bic < best_bic:
                best_bic = bic
                best_k = k

        return best_k


    def fit(self, embeddings):

        # automatically determine best cluster number
        self.n_clusters = self.find_optimal_clusters(embeddings)

        print("Selected number of clusters:", self.n_clusters)

        self.model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",
            random_state=42
        )

        self.model.fit(embeddings)


    def get_distribution(self, embedding):

        # returns probability distribution across clusters
        # this is what gives us fuzzy clustering
        probs = self.model.predict_proba([embedding])[0]

        return probs


    def dominant_cluster(self, embedding):

        # choose the cluster with highest probability
        probs = self.get_distribution(embedding)

        return int(np.argmax(probs))