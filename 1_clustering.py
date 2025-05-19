import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys


# Import clean data
data = pd.read_excel('data/data_clean.xlsx')
composition = pd.read_excel('data/composition_clean.xlsx')
print(data.head(10))

# Control randomness
np.random.seed(4321)

# Cluster data

for _ in range(10):
        
    n_cluster = np.arange(1, 16, 1)
    k_score = []
    for i in n_cluster:
        kmeans = KMeans(n_clusters=i, n_init=20)
        kmeans.fit(composition)
        k_score.append(kmeans.score(composition))

    # Plot
    plt.plot(n_cluster, np.gradient(k_score), 'DarkBlue')
    plt.xlabel('Number of clusters')
    plt.ylabel('K-means score')

plt.show()

sys.exit()

# Optimal clustering
n_cluster = 3
opt_cluster = KMeans(n_clusters=n_cluster)
clustering = opt_cluster.fit_predict(composition)
# print(clustering)

# Centroids
centroids = pd.DataFrame(opt_cluster.cluster_centers_, columns=composition.columns)
centroids = centroids.apply(lambda row: row.sort_values(ascending=False).index.values[:5], axis=1)
centroids = pd.DataFrame({'Cluster composition': centroids, 'Cluster title': ['-'.join(centroid[:3]) for centroid in centroids]})
print(centroids)
# Cluster 1: [P, V, Ti, Si, W]
# Cluster 2: [Si, Al, Cs, P, Na]
# Cluster 3: [Al, Cs, Ti, P, Ba]

# Save clusters
data_clustered = pd.concat(
    [data, pd.DataFrame(clustering, columns=['Cluster_n'])], axis=1
)
for i in range(n_cluster):
    data_clustered.loc[data_clustered['Cluster_n'] == i, 'Cluster title'] = centroids.loc[i, 'Cluster title']
data_clustered.to_excel('data/data_clustered.xlsx', index=False)
centroids.to_csv('data/centroids.csv', index=True)

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
