import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# Import clean data
data = pd.read_csv('data/data_processed.csv', na_values=[''], keep_default_na=False)
elements = pd.read_csv('data/elements.csv', header=None).squeeze('columns').to_list()
composition = data.loc[:, elements]
print(data.head(10), '\n', composition.head(10))
assert composition.notna().all(axis=None)

# Dummy composition
# composition[:] = np.where(composition < 1e-10, 0, 1)
print(composition)

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

# Optimal clustering
n_cluster = 4
opt_cluster = KMeans(n_clusters=n_cluster)
clustering = opt_cluster.fit_predict(composition)
# print(clustering)

# Centroids
centroids = pd.DataFrame(opt_cluster.cluster_centers_, columns=composition.columns)
centroids = centroids.apply(lambda row: row.sort_values(ascending=False).index.values[:5], axis=1)
centroids = pd.DataFrame({'Cluster_composition': centroids, 
                          'Cluster_title': ['-'.join(centroid[:3]) for centroid in centroids],
                          'Counts': np.unique(clustering, return_counts=True)[1]})
print(centroids)
# Cluster 1: [P, V, Ti, Si, W]
# Cluster 2: [Si, Al, Cs, P, Na]
# Cluster 3: [Al, Cs, Ti, P, Ba]

# Save clusters
# Concat cluster number
data_clustered = pd.concat(
    [data, pd.DataFrame(clustering, columns=['Cluster_n'])], axis=1
)
# Concat cluster title
for i in range(n_cluster):
    data_clustered.loc[data_clustered['Cluster_n'] == i, 'Cluster_title'] = centroids.loc[i, 'Cluster_title']

# Save clustered data
data_clustered.to_csv('data/data_clustered.csv', index=False)
centroids.to_csv('data/centroids.csv', index=True)
print(data_clustered.head(10))

# --------------------------------------

# Not sure what PCA and TSNE can be used for

exit()

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
