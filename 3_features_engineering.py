import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from functions.functions_features_engineering import scale_composition, get_elements_features, average_element_features

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

# -----------------------------

# Composition plot
# fig, ax = plt.subplots(1, clusters)
# for i, cluster in enumerate(clusters):
#     tmp = data.loc[data['Cluster_title'] == cluster, elements]
#     sns.scatterplot(tmp, x=tmp.columns)
fig = sns.clustermap(composition, cmap='viridis', col_cluster=False, norm=SymLogNorm(linthresh=0.01))
fig.ax_row_dendrogram.set_visible(False)
fig.savefig(f'figures/features_engineering/composition{sufix}.png', dpi=300)

# ---------------------------------------------

# Extract all element features
features = ['covalent_radius_cordero', 'nvalence_', 'en_pauling', 'lattice_constant', 'electron_affinity', 'electrophilicity_', 'vdw_radius', 'ionenergy', 'dipole_polarizability']  #,'proton_affinity'
feature_labels = ['cov_rad', 'n_val', 'en_pauling', 'lattice_cst', 'e_aff', 'e_phil', 'vdw_rad', 'ionenergy', 'polarization']
element_features = get_elements_features(elements, features, feature_labels=feature_labels)
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
selected_features = ['cov_rad', 'n_val', 'en_pauling']  # ionenergy correlated with en_pauling
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
for i, item in enumerate(mean_features + var_features):
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
corr_matrix = np.abs(np.corrcoef(mean_var_array, rowvar=False))
feature_labels = mean_features.columns.to_list() + var_features.columns.to_list()
fig, ax = plt.subplots(1, 1)
sns.heatmap(corr_matrix, ax=ax, cmap='plasma_r', vmin=0, vmax=1, square=True, 
            xticklabels=feature_labels, yticklabels=feature_labels)  # cbar_kws={"label": "Correlation coefficient"}
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Correlation coefficient', rotation=270, labelpad=15)
# fig.tight_layout()
fig.savefig(f'figures/features_engineering/corr_mean_var_features{sufix}.png', dpi=300)

# PCA of mean features
pca = PCA(n_components=min([4, len(selected_features)]))
pca_reduced_data = pca.fit_transform(mean_array)
print('Mean PCA: ', pca.explained_variance_ratio_)
fig = plt.figure()
sns.heatmap(pca.components_, cmap='plasma_r', square=True, vmin=-1, vmax=1,
            xticklabels=feature_labels[0:int(len(feature_labels)/2)],  # ['_'.join(feat.split('_')[0:2]) for feat in features] 
            yticklabels=[f'Comp {i+1}\n{x:.02} explanation' for i, x in enumerate(pca.explained_variance_ratio_)])
# fig.tight_layout()
fig.savefig(f'figures/features_engineering/pca_mean_features{sufix}.png', dpi=300)

# PCA components
fig = plt.figure(figsize=(3, 3))
gs = fig.add_gridspec(1, 1)
ax = []
ax.append(gs[0, 0].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
ax_joint = fig.add_subplot(ax[0][1, 0])
ax_x = fig.add_subplot(ax[0][0, 0], sharex=ax_joint)
ax_y = fig.add_subplot(ax[0][1, 1], sharey=ax_joint)
sns.scatterplot(ax=ax_joint, x=mean_array[:, 0], y=mean_array[:, 1], hue=data['Cluster_title'], hue_order=clusters, palette=palette)
for i, (comp, var, color) in enumerate(zip(pca.components_, pca.explained_variance_, ['DarkRed', 'DarkBlue'])):
    comp = comp * var  # scale component by its variance explanation power
    sns.lineplot(ax=ax_joint, x=[0, comp[0]], y=[0, comp[1]], color=color)
ax_joint.set(xlabel=feature_labels[0], ylabel=feature_labels[1])
ax_joint.legend(loc='upper right', frameon=False)
sns.kdeplot(ax=ax_x, x=mean_array[:, 0], hue=data['Cluster_title'], hue_order=clusters, palette=palette)
sns.kdeplot(ax=ax_y, y=mean_array[:, 1], hue=data['Cluster_title'], hue_order=clusters, palette=palette)
for item in [ax_x, ax_y]:
    item.set_axis_off()
    item.get_legend().remove()
fig.savefig(f'figures/features_engineering/pca_mean_features_pcascatter{sufix}.png', dpi=300)

# PCA of var features
pca = PCA(n_components=min([4, len(selected_features)]))
pca_reduced_data = pca.fit_transform(var_array)
print('Variance PCA: ', pca.explained_variance_ratio_)
fig = plt.figure()
sns.heatmap(pca.components_, cmap='plasma_r', square=True, vmin=-1, vmax=1,
            xticklabels=feature_labels[int(len(feature_labels)/2):],
            yticklabels=[f'Comp {i+1}\n{x:.02} explanation' for i, x in enumerate(pca.explained_variance_ratio_)])
fig.tight_layout()
fig.savefig(f'figures/features_engineering/pca_var_features{sufix}.png', dpi=300)

# Extract first component of var pca
data['var_pca'] = pca_reduced_data[:, 0]

# ----------------------------------------

# Save data
data.to_csv(f'data/catalysts/data_engineered{sufix}.csv', index=False)
