import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src import features_engineering, utils


# Plot formatting
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
PALETTE = ["#009688", "#1565C0", "#AD1457"]

# Context variables
ROOT = Path(__file__).resolve().parents[1]
VERBOSE = True

# Remove Si from features calculations
DROP_Si = True

# Selected features
# cov_rad or polarization for electron cloud distortion
# n_val, en_pauling for charge transfer (acid-base character), or ionenergy for electron donation (Lewis acid-base)
SELECTED_FEATURES = ['cov_rad', 'n_val']
N_FEAT = len(SELECTED_FEATURES)


if __name__ == '__main__':
    
    # Load data
    data = utils.load_data(ROOT / 'data/processed/data_clustered.csv')
    elements = pd.read_csv(ROOT / 'data/processed/elements.csv', header=None).squeeze('columns').to_list()
    composition = data.loc[:, elements]
    clusters = np.sort(data['Cluster_title'].unique())

    # ---------------------------------------------

    # Remove Si content when Si < 1.0
    if DROP_Si:
        suffix = '_noSi'
        composition['Si'] = np.where(composition['Si'] < 0.9999, 0, composition['Si'])
        composition = composition.div(composition.sum(axis=1), axis=0)
    else:
        suffix = ''

    # ---------------------------------------------

    # Extract all element features
    with open(ROOT / 'data/raw/atomic_features.json') as f:
        feature_map = json.load(f)
    feature_labels = list(feature_map.values())
    element_features = features_engineering.get_elements_features(elements, feature_map)
    if VERBOSE:
        print(element_features)

    # Correlation of mean and var features
    feature_array = StandardScaler().fit_transform(element_features)
    corr_matrix = np.abs(np.corrcoef(feature_array, rowvar=False))
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(corr_matrix, ax=ax, cmap='plasma_r', vmin=0, vmax=1, square=True, 
                xticklabels=feature_labels, yticklabels=feature_labels)  # cbar_kws={"label": "Correlation coefficient"}
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Correlation coefficient', rotation=270, labelpad=15)
    fig.savefig(ROOT / 'figures/features_engineering/corr_atomic_features.png', dpi=300)

    # ---------------------------------------------

    # Compute weighted average and variance of features
    mean_features, var_features = features_engineering.compute_engineered_features(composition, element_features.loc[:, SELECTED_FEATURES])

    # Concat engineered features with original df
    data[mean_features.columns] = mean_features
    data[var_features.columns] = var_features

    # Save feature labels
    engineered_labels = mean_features.columns.to_list() + var_features.columns.to_list()
    pd.Series(engineered_labels).to_csv(ROOT / 'data/processed/engineered_features.csv', index=False, header=False)

    # ----------------------------------------

    # Features correlation
    
    # Feature labels
    with open(ROOT / 'config/feature_labels.json') as f:
        label_mapper = json.load(f)["short"]

    # Normalize
    mean_array = StandardScaler().fit_transform(mean_features)
    var_array = StandardScaler().fit_transform(var_features)
    mean_var_array = np.concatenate([mean_array, var_array], axis=1)

    # Correlation of mean and var features
    corr_matrix = np.abs(np.corrcoef(mean_var_array, rowvar=False))

    # Plot correlation
    ax_labels = utils.list_mapper(engineered_labels, label_mapper, func=lambda x: x.split('_', maxsplit=1)[-1])
    fig, ax = plt.subplots(1, 1, figsize=(3*1.2, 3))
    sns.heatmap(corr_matrix, ax=ax, cmap='plasma_r', vmin=0, vmax=1, 
                xticklabels=ax_labels, yticklabels=ax_labels)  # cbar_kws={"label": "Correlation coefficient"}
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Correlation coefficient', rotation=270, labelpad=15)
    ax.text(0.25, -0.175, 'Mean', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
    ax.annotate('', xytext=(0+0.005, -0.12), xy=(0.5-0.005, -0.12), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
    ax.text(0.75, -0.175, 'Variance', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
    ax.annotate('', xytext=(0.5+0.005, -0.12), xy=(1-0.005, -0.12), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
    ax.text(-0.175, 0.75, 'Mean', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
    ax.annotate('', xytext=(-0.12, 0+0.005), xy=(-0.12, 0.5-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
    ax.text(-0.175, 0.25, 'Variance', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
    ax.annotate('', xytext=(-0.12, 0.5+0.005), xy=(-0.12, 1-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
    # fig.tight_layout()
    fig.savefig(ROOT / f'figures/features_engineering/corr_mean_var_features{suffix}.svg', bbox_inches='tight', dpi=300)

    # ----------------------------------------

    # PCA of mean and var features

    # PCA
    mean_pca = PCA(n_components=min([4, N_FEAT]))
    mean_pca_reduced_data = mean_pca.fit_transform(mean_array)
    var_pca = PCA(n_components=min([4, N_FEAT]))
    var_pca_reduced_data = var_pca.fit_transform(var_array)
    if VERBOSE:
        print('Mean PCA: ', mean_pca.explained_variance_ratio_)
        print('Variance PCA: ', var_pca.explained_variance_ratio_)
    
    # Plot PCA
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    features_engineering.plot_pca_heatmap(ax[0], mean_pca, utils.list_mapper(engineered_labels[:N_FEAT], label_mapper))
    features_engineering.plot_pca_heatmap(ax[1], var_pca, utils.list_mapper(engineered_labels[N_FEAT:], label_mapper))
    fig.savefig(ROOT / f'figures/features_engineering/pca_features_heatmap{suffix}.svg', dpi=300)

    # PCA - KDE plot
    fig = plt.figure(figsize=(6+1, 3))
    gs = fig.add_gridspec(1, 2, wspace=0.5, hspace=0.5)
    features_engineering.plot_pca_scatter(fig, gs[0, 0], mean_array, mean_pca, utils.list_mapper(engineered_labels[:N_FEAT], label_mapper),
                                          data['Cluster_title'], clusters, PALETTE, show_legend=False, panel_label='a')
    features_engineering.plot_pca_scatter(fig, gs[0, 1], var_array, var_pca, utils.list_mapper(engineered_labels[N_FEAT:], label_mapper),
                                          data['Cluster_title'], clusters, PALETTE, show_legend=True, panel_label='b')
    fig.savefig(ROOT / f'figures/features_engineering/pca_features_pcascatter{suffix}.svg', bbox_inches='tight', dpi=600)

    # Extract first component of var pca
    data['var_pca'] = var_pca_reduced_data[:, 0]

    # ----------------------------------------

    # Save data
    data.to_csv(ROOT / f'data/processed/data_engineered{suffix}.csv', index=False)
