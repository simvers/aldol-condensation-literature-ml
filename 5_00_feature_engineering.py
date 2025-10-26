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
data = pd.read_csv('data/data_deactivation.csv', na_values=[''], keep_default_na=False)
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

# Plot distribution of mol%
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))  #, constrained_layout=True)
# sns.kdeplot(composition.replace(0, pd.NA), ax=ax, warn_singular=False)
# ax.set(xlim=(0, 1))  #, xscale='symlog', yscale='symlog')
# sns.move_legend(ax, loc='lower left', ncols=10, frameon=False, bbox_to_anchor=(0, 1.02))
# fig.savefig('figures/features_engineering/content_kde.png', dpi=300)

# Plot distribution of mol%
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))  #, constrained_layout=True)
# sns.kdeplot(transformed_composition.replace(0, pd.NA), ax=ax, warn_singular=False)
# ax.set(xlim=(0, 1))  #, xscale='symlog', yscale='symlog')
# sns.move_legend(ax, loc='lower left', ncols=10, frameon=False, bbox_to_anchor=(0, 1.02))
# fig.savefig('figures/features_engineering/trans_content_kde.png', dpi=300)

# ----------------------------------------

# Modify class method into attribute
Element.electrophilicity_ = property(lambda self: self.electrophilicity())

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

    mult_df = composition_df.dot(element_features_df)
    return mult_df


# Extract element features
features = ['covalent_radius_cordero', 'en_pauling', 'electron_affinity', 'electrophilicity_',
            'lattice_constant', 'proton_affinity', 'vdw_radius']  # 'covalent_radius_pyykko'
element_features = get_elements_features(elements, features)

# Print missing values and fill with mean of feature
print(element_features.isna().sum(axis=0))
element_features.fillna(element_features.mean(), inplace=True)

# Compute weighted average of features
engineered_features = average_element_features(transformed_composition, element_features)
print(engineered_features)

# Plot features distribution
fig, ax = plt.subplots(2, 4)
ax = np.ravel(ax)
for i, item in enumerate(engineered_features):
    sns.kdeplot(engineered_features, x=item, ax=ax[i])
fig.savefig('figures/features_engineering/kde_features.png', dpi=300)

# Concat engineered features with original df
data[engineered_features.columns] = engineered_features
data.to_csv('data/data_engineered.csv', index=False)

# ----------------------------------------

# Check for correlation

# Normalize
array = StandardScaler().fit_transform(engineered_features)

# Correlation
corr_matrix = np.abs(np.corrcoef(array, rowvar=False))
fig = plt.figure()
sns.heatmap(corr_matrix, cmap='plasma_r', xticklabels=features, yticklabels=features, vmin=0, vmax=1, square=True)
fig.savefig('figures/features_engineering/corr_features.png', dpi=300)

# PCA
pca = PCA(n_components=4)
pca_reduced_data = pca.fit_transform(array)
print(pca.explained_variance_ratio_)
fig = plt.figure()
sns.heatmap(pca.components_, cmap='plasma_r', square=True,  # vmin=-0.5, vmax=0.7,
            xticklabels=['_'.join(feat.split('_')[0:2]) for feat in features], 
            yticklabels=[f'Comp {i+1}\n{x:.02} expl' for i, x in enumerate(pca.explained_variance_ratio_)])
fig.tight_layout()
fig.savefig('figures/features_engineering/components_weights.png', dpi=300)
