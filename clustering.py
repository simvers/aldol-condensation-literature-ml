import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Import clean data
data = pd.read_excel('data/data_clean.xlsx')
composition = pd.read_excel('data/composition_clean.xlsx')
print(data.head(10))

# Cluster data
n_cluster = np.arange(1, 16, 1)
k_score = []
for i in n_cluster:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(composition)
    k_score.append(kmeans.score(composition))

# Plot
plt.plot(n_cluster, np.gradient(k_score), 'DarkBlue')
plt.xlabel('Number of clusters')
plt.ylabel('K-means score')
plt.show()

# Optimal clustering
n_cluster = 3
opt_cluster = KMeans(n_clusters=n_cluster)
clustering = opt_cluster.fit_predict(composition)
print(clustering)

# Centroids
centroids = pd.DataFrame(opt_cluster.cluster_centers_, columns=composition.columns)
centroids = centroids.apply(lambda row: row.sort_values(ascending=False).index.values[:5], axis=1)
print(centroids)
# Cluster 1: [P, V, Ti, Si, W]
# Cluster 2: [Si, Al, Cs, P, Na]
# Clsuter 3: [Al, Cs, Ti, P, Ba]

# --------------------------------------

# Not sure what PCA and TSNE can be used for

# PCA
pca = PCA(n_components=2)
pca_reduced_data = pca.fit_transform(composition)
plt.scatter(pca_reduced_data[:, 0], pca_reduced_data[:, 1], c=clustering, cmap=plt.get_cmap('plasma', n_cluster), label=centroids)
plt.show()

# TSNE
tsne = TSNE(n_components=2, init='pca')
tsne_reduced_data = tsne.fit_transform(composition)
plt.scatter(tsne_reduced_data[:, 0], tsne_reduced_data[:, 1], c=clustering, cmap=plt.get_cmap('plasma', n_cluster), label=centroids)
plt.legend()
plt.show()
