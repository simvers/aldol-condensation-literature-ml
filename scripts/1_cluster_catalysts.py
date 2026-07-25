import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src import utils


# Format plot
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8

ROOT = Path(__file__).resolve().parents[1]

PALETTE = ["#009688", "#1565C0", "#AD1457"]
BINARIZE = True
N_CLUSTER = 3 if BINARIZE else 4
RS = 54321

if __name__ == "__main__":

    # Load data
    data = utils.load_data(ROOT / 'data/processed/data_processed.csv')
    elements = pd.read_csv(ROOT / 'data/processed/elements.csv', header=None).squeeze('columns').to_list()

    # Extract composition and binarize
    composition = data[elements].copy()
    if not composition.notna().all(axis=None):
        raise ValueError("Composition contains missing values")
    if BINARIZE:
        composition[:] = np.where(composition < 1e-10, 0, 1)

    # Elbow plot — repeated n_reps times to show sensitivity to random initialisation
    cluster_range = np.arange(1, 10)
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    for _ in range(10):
        scores = [KMeans(n_clusters=k, n_init=20).fit(composition).score(composition) for k in cluster_range]
        ax.plot(cluster_range, np.gradient(scores), color='DarkBlue')
    ax.set(xlabel='Number of clusters', ylabel='K-means score gradient',
           xlim=(1, 9), xticks=[1, 3, 5, 7, 9], yticks=[])
    fig.savefig(ROOT / f'figures/clustering/clustering_score{"_binarized" * BINARIZE}.svg',
                dpi=300, format='svg', bbox_inches='tight')

    # Fit optimal clustering
    opt_cluster = KMeans(n_clusters=N_CLUSTER, n_init=20, random_state=RS)
    clustering = opt_cluster.fit_predict(composition)

    # Extract centroids
    centroids = pd.DataFrame(opt_cluster.cluster_centers_, columns=composition.columns)
    centroids = centroids.apply(lambda row: row.sort_values(ascending=False).index.values[:5], axis=1)
    centroids = pd.DataFrame({
        'Cluster_composition': centroids,
        'Cluster_title': ['-'.join(c[:3]) for c in centroids],
        'Counts': np.unique(clustering, return_counts=True)[1]
    })
    print(centroids)

    # Assign clusters to data
    data_clustered = pd.concat([data, pd.DataFrame(clustering, columns=['Cluster_n'])], axis=1)
    for i in range(N_CLUSTER):
        data_clustered.loc[data_clustered['Cluster_n'] == i, 'Cluster_title'] = centroids.loc[i, 'Cluster_title']
    clusters = np.sort(data_clustered['Cluster_title'].unique())

    # Save data
    data_clustered.to_csv(ROOT / 'data/processed/data_clustered.csv', index=False)
    centroids.to_csv(ROOT / 'data/processed/centroids.csv', index=True)

    # -----------------------------------------------

    # Plot PCA and TSNE

    # Scale or binarize composition

    # PCA of composition space
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(composition)
    pca_df = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'])
    pca_df['Cluster_title'] = data_clustered['Cluster_title'].values

    # Print top element contributions per PC and build axis labels
    loadings = pd.DataFrame(pca.components_.T, index=elements, columns=['PC1', 'PC2'])
    pc_labels = {}
    for i, pc in enumerate(['PC1', 'PC2']):
        top = loadings[pc].abs().nlargest(10)
        print(f'\nTop contributions to {pc} ({pca.explained_variance_ratio_[i]*100:.1f}%):')
        for el in top.index:
            print(f'  {el:4s}  {loadings.loc[el, pc]:+.2f}')
        n_shown = 5 if pc == 'PC1' else 4
        loading_str = ',  '.join(f'{el} {loadings.loc[el, pc]:+.2f}' for el in top.index[:n_shown])
        pc_labels[pc] = f'{pc} ({pca.explained_variance_ratio_[i]*100:.0f}%)\n{loading_str}'

    fig = plt.figure(figsize=(3.5, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0)
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)

    # Scatter plot
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster_title', hue_order=clusters,
                    palette=PALETTE, ax=ax_joint, s=20, linewidth=0, alpha=0.6)
    ax_joint.set(xlabel=pc_labels['PC1'], ylabel=pc_labels['PC2'])
    ax_joint.get_legend().remove()

    # KDE plot
    sns.kdeplot(data=pca_df, x='PC1', hue='Cluster_title', hue_order=clusters, palette=PALETTE, cut=1, fill=True, ax=ax_x)
    sns.kdeplot(data=pca_df, y='PC2', hue='Cluster_title', hue_order=clusters, palette=PALETTE, cut=1, fill=True, ax=ax_y)
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()

    # Legend
    handles, labels = ax_joint.get_legend_handles_labels()
    ax_joint.legend(handles, labels, loc='lower right', frameon=False, title='Catalyst clusters')
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/clustering/composition_pca.svg', dpi=300, format='svg')

    # t-SNE of composition space
    tsne_coords = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(composition)
    tsne_df = pd.DataFrame(tsne_coords, columns=['tSNE1', 'tSNE2'])
    tsne_df['Cluster_title'] = data_clustered['Cluster_title'].values

    # Plot TSNE
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    sns.scatterplot(data=tsne_df, x='tSNE1', y='tSNE2', hue='Cluster_title', hue_order=clusters,
                    palette=PALETTE, ax=ax, s=20, linewidth=0)
    ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2')
    sns.move_legend(ax, loc='best', frameon=False, title='Catalyst clusters', handlelength=1.5)
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/clustering/composition_tsne.svg', dpi=300, format='svg')

