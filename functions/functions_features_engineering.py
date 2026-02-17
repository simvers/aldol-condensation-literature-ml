import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mendeleev import element
from mendeleev.models import Element
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def scale_composition(composition: pd.DataFrame, n=0.3):

    # Scale content such that small values are larger, larger values are smaller, keeping order, and sum = 1

    # Define scaler function
    power_scaler = lambda df, n: (df**n).div((df**n).sum(axis=1), axis=0)  # Is not reversible

    # Transform
    transformed_composition = power_scaler(composition, n=n)

    # Plot distribution of mol%
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(composition.to_numpy().flatten(), transformed_composition.to_numpy().flatten())
    ax.set(xlabel='Original mol content', ylabel='Scaled mol content', xlim=(0, 1), ylim=(0, 1))
    fig.savefig('figures/features_engineering/content_scaler.png', dpi=300)

    return transformed_composition


# Create df of element features
def get_elements_features(elements, features, feature_labels=None):
    
    # Modify class method into attribute
    Element.electrophilicity_ = property(lambda self: self.electrophilicity())
    Element.nvalence_ = property(lambda self: self.nvalence())
    Element.ionenergy = property(lambda self: self.ionenergies.get(1))

    # Initialize dictionary
    element_dir = {}
    if feature_labels is None:
        feature_labels = features

    # Loop through elements
    for elem in elements:
        elem_feat = element(elem)
        element_dir[elem] = {}

        # Loop through features
        for feat, label in zip(features, feature_labels):
            element_dir[elem][label] = getattr(elem_feat, feat)
    
    # Make df
    element_df = pd.DataFrame.from_dict(element_dir, orient='index')

    # Print missing values and fill with mean of feature
    print('Features missing:\n', element_df.isna().sum(axis=0))
    element_df.fillna(element_df.mean(), inplace=True)

    return element_df


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
