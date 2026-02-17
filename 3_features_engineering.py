import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# ---------------------------------------------

# Scale composition such that small values are larger, larger values are smaller, keeping order, and sum = 1
# Not reversible !!
# composition = scale_composition(composition, n=0.3)

# Extract element features
features = ['covalent_radius_cordero', 'nvalence_']  # , 'en_pauling', 'lattice_constant', 'electron_affinity', 'electrophilicity_','proton_affinity', 'vdw_radius', 'ionenergy'
feature_labels = ['cov_rad', 'n_val']
element_features = get_elements_features(elements, features, feature_labels=feature_labels)

# Compute weighted average and variance of features
mean_features, var_features = average_element_features(composition, element_features)
print(mean_features, var_features)

# Concat engineered features with original df
data[mean_features.columns] = mean_features
data[var_features.columns] = var_features
featurized_cols = pd.Series(mean_features.columns.to_list() + var_features.columns.to_list())
featurized_cols.to_csv('data/catalysts/comp_features.csv', index=False, header=False)

# Plot features distribution
fig, ax = plt.subplots(2, len(mean_features.columns), figsize=(5.5, 5.5))
ax = np.ravel(ax)
for i, item in enumerate(mean_features + var_features):
    sns.kdeplot(data, x=item, ax=ax[i], hue='Cluster_title', palette=mycolor, clip=(0, None))
    ax[i].set(yticks=[])
    if i != 0:
        ax[i].get_legend().remove()
    if (i != 0) & (i != len(mean_features.columns)):
        ax[i].set(ylabel=None)
# sns.move_legend(ax[0], loc='center left', frameon=False, bbox_to_anchor=(0, 1.05), title=None, ncols=4)
sns.move_legend(ax[0], loc='upper right', frameon=False, title=None)
# fig.tight_layout(h_pad=5)
fig.savefig('figures/features_engineering/kde_features.png', dpi=300)

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
fig.savefig('figures/features_engineering/corr_mean_var_features.png', dpi=300)

# PCA of mean features
pca = PCA(n_components=min([4, len(features)]))
pca_reduced_data = pca.fit_transform(mean_array)
print('Mean PCA: ', pca.explained_variance_ratio_)
fig = plt.figure()
sns.heatmap(pca.components_, cmap='plasma_r', square=True, vmin=-1, vmax=1,
            xticklabels=feature_labels[0:int(len(feature_labels)/2)],  # ['_'.join(feat.split('_')[0:2]) for feat in features] 
            yticklabels=[f'Comp {i+1}\n{x:.02} explanation' for i, x in enumerate(pca.explained_variance_ratio_)])
# fig.tight_layout()
fig.savefig('figures/features_engineering/pca_mean_features.png', dpi=300)

# PCA of var features
pca = PCA(n_components=min([4, len(features)]))
pca_reduced_data = pca.fit_transform(var_array)
print('Variance PCA: ', pca.explained_variance_ratio_)
fig = plt.figure()
sns.heatmap(pca.components_, cmap='plasma_r', square=True, vmin=-1, vmax=1,
            xticklabels=feature_labels[int(len(feature_labels)/2):-1],
            yticklabels=[f'Comp {i+1}\n{x:.02} explanation' for i, x in enumerate(pca.explained_variance_ratio_)])
fig.tight_layout()
fig.savefig('figures/features_engineering/pca_var_features.png', dpi=300)

# Extract first component of var pca
data['var_pca'] = pca_reduced_data[:, 0]

# ----------------------------------------

# Save data
data.to_csv('data/catalysts/data_engineered.csv', index=False)
