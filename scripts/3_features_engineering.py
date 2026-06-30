import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from functions.features_engineering import scale_composition, get_elements_features, average_element_features


def plot_pca_scatter(fig, gs_cell, array, pca, labels, hue, hue_order, palette, show_legend):
    # PC1 vs PC2 scatter, with the PCA loadings drawn as arrows and marginal KDEs on each axis.
    # Used for both the mean-feature PCA and the variance-feature PCA below - same plot, different input.
    sub_gs = gs_cell.subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0)
    ax_joint = fig.add_subplot(sub_gs[1, 0])
    ax_x = fig.add_subplot(sub_gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(sub_gs[1, 1], sharey=ax_joint)

    sns.scatterplot(ax=ax_joint, x=array[:, 0], y=array[:, 1], hue=hue, hue_order=hue_order, palette=palette)
    for comp, var, color in zip(pca.components_, pca.explained_variance_, ['DarkRed', 'DarkBlue']):
        comp = comp * var  # scale component by its variance explanation power
        sns.lineplot(ax=ax_joint, x=[0, comp[0]], y=[0, comp[1]], color=color)
    ax_joint.set(xlabel=f'Normalized {labels[0]}', ylabel=f'Normalized {labels[1]}')
    if show_legend:
        ax_joint.legend(loc='upper right', frameon=False)
    else:
        ax_joint.get_legend().remove()

    sns.kdeplot(ax=ax_x, x=array[:, 0], hue=hue, hue_order=hue_order, palette=palette)
    sns.kdeplot(ax=ax_y, y=array[:, 1], hue=hue, hue_order=hue_order, palette=palette)
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
mycolor = 'cividis'

# ---------------------------------------------

# Import data

# Import processed data and elements
# data = pd.read_csv('data/data_deactivation.csv', na_values=[''], keep_default_na=False)
data = pd.read_csv('data/catalysts/data_clustered.csv', na_values=[''], keep_default_na=False)
elements = pd.read_csv('data/catalysts/elements.csv', header=None).squeeze('columns').to_list()
composition = data.loc[:, elements]

# -----------------------------

# Color palette
clusters = np.sort(data['Cluster_title'].unique())
cmap = plt.get_cmap('cividis')
palette = cmap(np.linspace(0, 1, len(clusters)))
palette = sns.color_palette(palette, as_cmap=False, desat=0.8)

# ---------------------------------------------

# Scale composition
# Not reversible !!
# composition = scale_composition(composition, n=0.3)

# Remove Si content when Si < 1.0
remove_Si = True
# print(np.sort(composition['Si'].unique()))
if remove_Si:
    sufix = '_noSi'
    composition['Si'] = np.where(composition['Si'] < 0.9999, 0, composition['Si'])
    composition = composition.div(composition.sum(axis=1), axis=0)
else:
    sufix = ''
# print((composition.sum(axis=1) > 1.00001).sum())
# print((composition.sum(axis=1) < 0.99999).sum())

# ---------------------------------------------

# Extract all element features. The mapping of mendeleev attribute -> short label lives in
# data/element_features.json (a dict, not two parallel lists, so the pairing can't get out of sync)
with open('data/element_features.json') as f:
    feature_map = json.load(f)
feature_labels = list(feature_map.values())
element_features = get_elements_features(elements, feature_map)
print(element_features)

# Correlation of mean and var features
feature_array = StandardScaler().fit_transform(element_features)
corr_matrix = np.abs(np.corrcoef(feature_array, rowvar=False))
fig, ax = plt.subplots(1, 1)
sns.heatmap(corr_matrix, ax=ax, cmap='plasma_r', vmin=0, vmax=1, square=True, 
            xticklabels=feature_labels, yticklabels=feature_labels)  # cbar_kws={"label": "Correlation coefficient"}
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Correlation coefficient', rotation=270, labelpad=15)
# fig.tight_layout()
fig.savefig(f'figures/features_engineering/corr_atomic_features.png', dpi=300)

# ---------------------------------------------

# Compute weighted average and variance of features
# cov_rad or polarization for electron cloud distortion
# n_val, en_pauling for charge transfer (acid-base character), or ionenergy for electron donation (Lewis acid-base)
selected_features = ['cov_rad', 'n_val']  # ionenergy correlated with en_pauling
mean_features, var_features = average_element_features(composition, element_features.loc[:, selected_features])
print(mean_features, var_features)

# Concat engineered features with original df
data[mean_features.columns] = mean_features
data[var_features.columns] = var_features
featurized_cols = pd.Series(mean_features.columns.to_list() + var_features.columns.to_list())
featurized_cols.to_csv('data/catalysts/comp_features.csv', index=False, header=False)

# Plot features distribution
fig, ax = plt.subplots(2, len(mean_features.columns), figsize=(5.5/2*len(mean_features.columns), 5.5))
ax = np.ravel(ax)
# Top row = mean features, bottom row = variance features (matches the 2-row grid above)
for i, item in enumerate(mean_features.columns.to_list() + var_features.columns.to_list()):
    sns.kdeplot(data, x=item, ax=ax[i], hue='Cluster_title', hue_order=clusters, palette=palette, clip=(0, None))
    ax[i].set(yticks=[])
    if i != 0:
        ax[i].get_legend().remove()
    if (i != 0) & (i != len(mean_features.columns)):
        ax[i].set(ylabel=None)
# sns.move_legend(ax[0], loc='center left', frameon=False, bbox_to_anchor=(0, 1.05), title=None, ncols=4)
sns.move_legend(ax[0], loc='upper right', frameon=False, title=None)
# fig.tight_layout(h_pad=5)
fig.savefig(f'figures/features_engineering/kde_features{sufix}.png', dpi=300)

# ----------------------------------------

# Features correlation and PCA

# Normalize
mean_array = StandardScaler().fit_transform(mean_features)
var_array = StandardScaler().fit_transform(var_features)
mean_var_array = np.concatenate([mean_array, var_array], axis=1)

# Correlation of mean and var features
# Note: this reuses the name "engineered_labels" rather than overwriting feature_labels above,
# which holds the 9 raw atomic feature labels, not these 4 av_/var_ engineered columns
corr_matrix = np.abs(np.corrcoef(mean_var_array, rowvar=False))
n_mean = len(mean_features.columns)
engineered_labels = mean_features.columns.to_list() + var_features.columns.to_list()
ax_labels = [item.split('_', maxsplit=1)[-1] for item in engineered_labels]
fig, ax = plt.subplots(1, 1, figsize=(3*1.2, 3))
sns.heatmap(corr_matrix, ax=ax, cmap='plasma_r', vmin=0, vmax=1, 
            xticklabels=ax_labels, yticklabels=ax_labels)  # cbar_kws={"label": "Correlation coefficient"}
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Correlation coefficient', rotation=270, labelpad=15)
ax.text(0.25, -0.175, 'Mean', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.text(0.75, -0.175, 'Variance', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.text(-0.175, 0.75, 'Mean', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
ax.text(-0.175, 0.25, 'Variance', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
# fig.tight_layout()
fig.savefig(f'figures/features_engineering/corr_mean_var_features{sufix}.svg', bbox_inches='tight', dpi=300)

# PCA of mean features
mean_pca = PCA(n_components=min([4, len(selected_features)]))
mean_pca_reduced_data = mean_pca.fit_transform(mean_array)
print('Mean PCA: ', mean_pca.explained_variance_ratio_)
fig = plt.figure()
sns.heatmap(mean_pca.components_, cmap='plasma_r', square=True, vmin=-1, vmax=1,
            xticklabels=engineered_labels[:n_mean],
            yticklabels=[f'Comp {i+1}\n{x:.02} explanation' for i, x in enumerate(mean_pca.explained_variance_ratio_)])
# fig.tight_layout()
fig.savefig(f'figures/features_engineering/pca_mean_features{sufix}.png', dpi=300)

# PCA of var features
var_pca = PCA(n_components=min([4, len(selected_features)]))
var_pca_reduced_data = var_pca.fit_transform(var_array)
print('Variance PCA: ', var_pca.explained_variance_ratio_)
fig = plt.figure()
sns.heatmap(var_pca.components_, cmap='plasma_r', square=True, vmin=-1, vmax=1,
            xticklabels=engineered_labels[n_mean:],
            yticklabels=[f'Comp {i+1}\n{x:.02} explanation' for i, x in enumerate(var_pca.explained_variance_ratio_)])
fig.tight_layout()
fig.savefig(f'figures/features_engineering/pca_var_features{sufix}.png', dpi=300)

# PCA - KDE plot
# Mean and variance PCA get the same plot treatment (see plot_pca_scatter at the top of this file)
fig = plt.figure(figsize=(6, 3))
gs = fig.add_gridspec(1, 2)
plot_pca_scatter(fig, gs[0, 0], mean_array, mean_pca, engineered_labels[:n_mean],
                  data['Cluster_title'], clusters, palette, show_legend=False)
plot_pca_scatter(fig, gs[0, 1], var_array, var_pca, engineered_labels[n_mean:],
                  data['Cluster_title'], clusters, palette, show_legend=True)
fig.tight_layout()
fig.savefig(f'figures/features_engineering/pca_features_pcascatter{sufix}.svg', dpi=300)

# Extract first component of var pca
data['var_pca'] = var_pca_reduced_data[:, 0]

# ----------------------------------------

# Save data
data.to_csv(f'data/catalysts/data_engineered{sufix}.csv', index=False)
