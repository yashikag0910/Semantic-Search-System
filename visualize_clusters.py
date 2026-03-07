import umap
import matplotlib.pyplot as plt
from data_loader import load_dataset
from embedder import Embedder
from clusterer import Clusterer


# this script helps visualize the semantic structure
# of the dataset in 2D space

docs = load_dataset()

embedder = Embedder()

embeddings = embedder.embed(docs)

clusterer = Clusterer()
clusterer.fit(embeddings)

labels = clusterer.model.predict(embeddings)


# reduce 384 dimensional embeddings into 2D
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1
)

embedding_2d = reducer.fit_transform(embeddings)


plt.figure(figsize=(10,7))

plt.scatter(
    embedding_2d[:,0],
    embedding_2d[:,1],
    c=labels,
    cmap="tab20",
    s=5
)

plt.title("Semantic Clusters of the 20 Newsgroups Dataset")

plt.xlabel("UMAP dimension 1")
plt.ylabel("UMAP dimension 2")

plt.show()