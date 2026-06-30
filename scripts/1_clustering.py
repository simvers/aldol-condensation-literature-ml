import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8

BINARIZE = True
N_CLUSTER = 3 if BINARIZE else 4
RANDOM_STATE = 54321


def load_composition():
    data = pd.read_csv('data/catalysts/data_processed.csv', na_values=[''], keep_default_na=False)
    elements = pd.read_csv('data/catalysts/elements.csv', header=None).squeeze('columns').to_list()
    composition = data.loc[:, elements]
    if not composition.notna().all(axis=None):
        raise ValueError("Composition contains missing values")

    if BINARIZE:
        composition[:] = np.where(composition < 1e-10, 0, 1)

    return data, composition


def plot_elbow_scores(composition, cluster_range=np.arange(1, 10, 1), n_reps=10):
    # Repeat the elbow curve n_reps times (without a fixed seed per fit) to visualize how
    # sensitive the K-means score is to random initialization at each candidate cluster count.
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
    for _ in range(n_reps):
        k_score = []
        for i in cluster_range:
            kmeans = KMeans(n_clusters=i, n_init=20)
            kmeans.fit(composition)
            k_score.append(kmeans.score(composition))
        ax.plot(cluster_range, np.gradient(k_score), 'DarkBlue')
    ax.set(xlabel='Number of clusters', ylabel='K-means score', xlim=(1, 9), xticks=[1, 3, 5, 7, 9], yticks=[])
    fig.savefig(f'figures/clustering/clustering_score{"_binarized"*BINARIZE}.svg', dpi=300, format='svg', bbox_inches='tight')


def cluster_catalysts(composition, n_cluster):
    # random_state is fixed here (unlike the elbow loop above) since this clustering is saved
    # and consumed by downstream scripts, and should be reproducible run-to-run.
    opt_cluster = KMeans(n_clusters=n_cluster, n_init=20, random_state=RANDOM_STATE)
    clustering = opt_cluster.fit_predict(composition)
    print(f'K-means score: {opt_cluster.score(composition)}')
    return opt_cluster, clustering


def extract_centroids(opt_cluster, composition, clustering):
    centroids = pd.DataFrame(opt_cluster.cluster_centers_, columns=composition.columns)
    centroids = centroids.apply(lambda row: row.sort_values(ascending=False).index.values[:5], axis=1)
    centroids = pd.DataFrame({'Cluster_composition': centroids,
                              'Cluster_title': ['-'.join(centroid[:3]) for centroid in centroids],
                              'Counts': np.unique(clustering, return_counts=True)[1]})
    print(centroids)
    return centroids


def assign_clusters(data, clustering, centroids, n_cluster):
    # Cluster number
    data_clustered = pd.concat([data, pd.DataFrame(clustering, columns=['Cluster_n'])], axis=1)
    # Cluster title
    for i in range(n_cluster):
        data_clustered.loc[data_clustered['Cluster_n'] == i, 'Cluster_title'] = centroids.loc[i, 'Cluster_title']
    return data_clustered


def main():
    np.random.seed(RANDOM_STATE)

    # Import preprocessed data
    data, composition = load_composition()

    # Plot clustering scores
    plot_elbow_scores(composition)

    # Optimal clustering and centroid extraction
    opt_cluster, clustering = cluster_catalysts(composition, N_CLUSTER)
    centroids = extract_centroids(opt_cluster, composition, clustering)
    data_clustered = assign_clusters(data, clustering, centroids, N_CLUSTER)

    # Save data
    data_clustered.to_csv('data/catalysts/data_clustered.csv', index=False)
    centroids.to_csv('data/catalysts/centroids.csv', index=True)
    print(data_clustered.head(10))


if __name__ == "__main__":
    main()
