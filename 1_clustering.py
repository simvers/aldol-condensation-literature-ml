import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Import processed data and elements
data = pd.read_csv('data/data_processed.csv', na_values=[''], keep_default_na=False)
elements = pd.read_csv('data/elements.csv', header=None).squeeze('columns').to_list()
composition = data.loc[:, elements]
assert composition.notna().all(axis=None)

# Binarize composition
binarize = True
if binarize:
    composition[:] = np.where(composition < 1e-10, 0, 1)

# Control randomness
np.random.seed(4321)

# Cluster data
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for _ in range(10):
        
    n_cluster = np.arange(1, 10, 1)
    k_score = []
    for i in n_cluster:
        kmeans = KMeans(n_clusters=i, n_init=20)
        kmeans.fit(composition)
        k_score.append(kmeans.score(composition))

    # Plot
    ax.plot(n_cluster, np.gradient(k_score), 'DarkBlue')
ax.set(xlabel='Number of clusters', ylabel='K-means score', xlim=(1, 9), xticks=[1, 3, 5, 7, 9], yticks=[])
fig.savefig(f'figures/clustering/clustering_score{"_binarized"*binarize}.png', dpi=300)

# Optimal clustering
n_cluster = 4 if binarize else 4
opt_cluster = KMeans(n_clusters=n_cluster)
clustering = opt_cluster.fit_predict(composition)

# Centroids
centroids = pd.DataFrame(opt_cluster.cluster_centers_, columns=composition.columns)
centroids = centroids.apply(lambda row: row.sort_values(ascending=False).index.values[:5], axis=1)
centroids = pd.DataFrame({'Cluster_composition': centroids, 
                          'Cluster_title': ['-'.join(centroid[:3]) for centroid in centroids],
                          'Counts': np.unique(clustering, return_counts=True)[1]})
print(centroids)

# --------------------------------------

# Save clusters

# Concat cluster number
data_clustered = pd.concat([data, pd.DataFrame(clustering, columns=['Cluster_n'])], axis=1)

# Concat cluster title
for i in range(n_cluster):
    data_clustered.loc[data_clustered['Cluster_n'] == i, 'Cluster_title'] = centroids.loc[i, 'Cluster_title']

# Save clustered data
data_clustered.to_csv('data/data_clustered.csv', index=False)
centroids.to_csv('data/centroids.csv', index=True)
print(data_clustered.head(10))
