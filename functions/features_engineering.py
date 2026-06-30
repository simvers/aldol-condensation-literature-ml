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
def get_elements_features(elements, feature_map):
    # feature_map: {mendeleev attribute name: short output label}, e.g. from data/element_features.json.
    # A dict (instead of two parallel lists) keeps each attribute paired with its label, so
    # re-ordering or editing one of them can't silently misalign with the other.

    # Some mendeleev properties are methods, not attributes - expose them as properties so
    # getattr(elem_feat, feat) below works uniformly for all features
    Element.electrophilicity_ = property(lambda self: self.electrophilicity())
    Element.nvalence_ = property(lambda self: self.nvalence())
    Element.ionenergy = property(lambda self: self.ionenergies.get(1))
    Element.zeff = property(lambda self: self.zeff())

    # Loop through elements and requested features
    element_dir = {}
    for elem in elements:
        elem_feat = element(elem)
        element_dir[elem] = {label: getattr(elem_feat, feat) for feat, label in feature_map.items()}

    # Make df
    element_df = pd.DataFrame.from_dict(element_dir, orient='index')

    # Some elements don't have a value for every property (e.g. no measured electron affinity) -
    # report how many are missing, then fill with the column mean rather than dropping the element
    print('Features missing:\n', element_df.isna().sum(axis=0))
    element_df.fillna(element_df.mean(), inplace=True)

    return element_df


# Average element features
def average_element_features(composition_df, element_features_df):

    # Safety checks
    if not composition_df.notna().all(axis=None):
        raise ValueError("composition_df contains missing values")
    if not element_features_df.notna().all(axis=None):
        raise ValueError("element_features_df contains missing values")
    if set(composition_df.columns) != set(element_features_df.index):
        raise ValueError("composition_df columns and element_features_df index must contain the same elements")
    # Same elements, but also in the same order - required since the matrix multiplication below relies on positional alignment
    if not (composition_df.columns == element_features_df.index).all(axis=None):
        raise ValueError("composition_df columns and element_features_df index must be in the same order")

    # Composition-weighted mean of each feature, per catalyst: a plain matrix product, since
    # mean = sum over elements of (molar fraction * feature value)
    cols = element_features_df.columns
    rows = composition_df.index
    comp = composition_df.values  # (n_obs, n_elem)
    feat = element_features_df.values  # (n_elem, n_feat)
    mean = comp @ feat  # (n_obs, n_feat)

    # Composition-weighted variance of each feature, per catalyst: sum over elements of
    # molar fraction * (feature value - weighted mean)^2
    diff = feat[None, :, :] - mean[:, None, :]  # (n_obs, n_elem, n_feat)
    var = (comp[:, :, None] * diff**2).sum(axis=1)  # (n_obs, n_feat), summed over elements

    return pd.DataFrame(mean, columns='av_'+cols, index=rows), pd.DataFrame(var, columns='var_'+cols, index=rows)


if __name__ == '__main__':


    a = pd.DataFrame(
        [
            [1, 0, 0], 
            [0, 1, 0], 
            [0, 0, 2]
        ], 
        columns=['a', 'c', 'b'],
        index=['m', 'n', 'o']
    )
    b = pd.DataFrame(
        [
            [1, 0, 0], 
            [0, 2, 0], 
            [0, 0, 1]
        ],
        columns=['x', 'y', 'z'], 
        index=['a', 'b', 'c']
    )
    print(a @ b)

    # mean, _ = average_element_features(a, b)
    # print(mean)

    a = a.reindex(sorted(a.columns), axis=1)
    print(a)

    mean, _ = average_element_features(a, b)
    print(mean)


