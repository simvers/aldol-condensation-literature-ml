import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mendeleev import element
from mendeleev.models import Element
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# ---------------------------------------------

# Import data

# Import processed data and elements
# data = pd.read_csv('data/data_deactivation.csv', na_values=[''], keep_default_na=False)
data = pd.read_csv('data/data_clustered.csv', na_values=[''], keep_default_na=False)
elements = pd.read_csv('data/elements.csv', header=None).squeeze('columns').to_list()
composition = data.loc[:, elements]

# ---------------------------------------------

# Scale content such that small values are larger, larger values are smaller, keeping order, and sum = 1

# Define scaler function
power_scaler = lambda df, n: (df**n).div((df**n).sum(axis=1), axis=0)  # Is not reversible

# Transform
transformed_composition = power_scaler(composition, n=0.3)

# Plot distribution of mol%
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(composition.to_numpy().flatten(), transformed_composition.to_numpy().flatten())
ax.set(xlabel='Original mol content', ylabel='Scaled mol content', xlim=(0, 1), ylim=(0, 1))
fig.savefig('figures/features_engineering/content_scaler.png', dpi=300)

# ----------------------------------------

# Modify class method into attribute
Element.electrophilicity_ = property(lambda self: self.electrophilicity())
Element.nvalence_ = property(lambda self: self.nvalence())
Element.ionenergy = property(lambda self: self.ionenergies.get(1))

# Create df of element features
def get_elements_features(elements, features):
    
    # Initialize dictionary
    element_dir = {}

    # Loop through elements
    for elem in elements:
        elem_feat = element(elem)
        element_dir[elem] = {}

        # Loop through features
        for feat in features:
            element_dir[elem][feat] = getattr(elem_feat, feat)
    
    # Return df
    return pd.DataFrame.from_dict(element_dir, orient='index')


# Average element features
def average_element_features(composition_df, element_features_df):

    # Safety checks
    assert composition_df.notna().all(axis=None)
    assert element_features_df.notna().all(axis=None)
    assert set(composition_df.columns) == set(element_features_df.index)

    # pd dot product
    # mult_df = composition_df.dot(element_features_df)

    # Np mean and variance
    cols = element_features_df.columns
    rows = composition_df.index
    comp = composition_df.values  # (n_obs, n_elem)
    feat = element_features_df.values  # (n_elem, n_feat)
    mean = comp @ feat  # (n_obs, n_feat)
    diff = feat[None, :, :] - mean[:, None, :]  # (n_obs, n_elem, n_feat)
    var = (comp[:, :, None] * diff**2).sum(axis=1)  # (n_obs, n_elem, n_feat) summed over n_elem
    return pd.DataFrame(mean, columns='av_'+cols, index=rows), pd.DataFrame(var, columns='var_'+cols, index=rows)

# Extract element features
features = ['covalent_radius_cordero', 'nvalence_']  # , 'en_pauling', 'lattice_constant', 'electron_affinity', 'electrophilicity_','proton_affinity', 'vdw_radius', 'ionenergy'
element_features = get_elements_features(elements, features)

# Print missing values and fill with mean of feature
print(element_features.isna().sum(axis=0))
element_features.fillna(element_features.mean(), inplace=True)

# Compute weighted average of features
mean_features, var_features = average_element_features(transformed_composition, element_features)
print(mean_features, var_features)

# Concat engineered features with original df
data[mean_features.columns] = mean_features
data[var_features.columns] = var_features
featurized_cols = pd.Series(mean_features.columns.to_list() + var_features.columns.to_list())
featurized_cols.to_csv('data/comp_features.csv', index=False, header=False)
data.to_csv('data/data_engineered.csv', index=False)

# Plot features distribution
fig, ax = plt.subplots(2, len(mean_features.columns), figsize=(10, 8))
ax = np.ravel(ax)
for i, item in enumerate(mean_features + var_features):
    sns.kdeplot(data, x=item, ax=ax[i], hue='Cluster_title', palette='plasma', clip=(0, None))
    ax[i].set(yticks=[])
    if i != 0:
        ax[i].get_legend().remove()
    if (i != 0) & (i != len(mean_features.columns)):
        ax[i].set(ylabel=None)
sns.move_legend(ax[0], loc='center left', frameon=False, bbox_to_anchor=(0, 1.05), title=None, ncols=4)
# fig.tight_layout(h_pad=5)
fig.savefig('figures/features_engineering/kde_features.png', dpi=300)

# ----------------------------------------

# Check for correlation

# Normalize
mean_array = StandardScaler().fit_transform(mean_features)
var_array = StandardScaler().fit_transform(var_features)

# Correlation
corr_matrix = np.abs(np.corrcoef(mean_array, rowvar=False))
fig = plt.figure()
sns.heatmap(corr_matrix, cmap='plasma_r', xticklabels=features, yticklabels=features, vmin=0, vmax=1, square=True)
fig.savefig('figures/features_engineering/corr_mean_features.png', dpi=300)

# PCA
pca = PCA(n_components=min([4, len(features)]))
pca_reduced_data = pca.fit_transform(mean_array)
print(pca.explained_variance_ratio_)
fig = plt.figure()
sns.heatmap(pca.components_, cmap='plasma_r', square=True,  # vmin=-0.5, vmax=0.7,
            xticklabels=['_'.join(feat.split('_')[0:2]) for feat in features], 
            yticklabels=[f'Comp {i+1}\n{x:.02} expl' for i, x in enumerate(pca.explained_variance_ratio_)])
fig.tight_layout()
fig.savefig('figures/features_engineering/components_weights_mean_features.png', dpi=300)

# Correlation
corr_matrix = np.abs(np.corrcoef(var_array, rowvar=False))
fig = plt.figure()
sns.heatmap(corr_matrix, cmap='plasma_r', xticklabels=features, yticklabels=features, vmin=0, vmax=1, square=True)
fig.savefig('figures/features_engineering/corr_var_features.png', dpi=300)

# PCA
pca = PCA(n_components=min([4, len(features)]))
pca_reduced_data = pca.fit_transform(var_array)
print(pca.explained_variance_ratio_)
fig = plt.figure()
sns.heatmap(pca.components_, cmap='plasma_r', square=True,  # vmin=-0.5, vmax=0.7,
            xticklabels=['_'.join(feat.split('_')[0:2]) for feat in features], 
            yticklabels=[f'Comp {i+1}\n{x:.02} expl' for i, x in enumerate(pca.explained_variance_ratio_)])
fig.tight_layout()
fig.savefig('figures/features_engineering/components_weights_var_features.png', dpi=300)
