import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mendeleev import element
from mendeleev.models import Element


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


def get_elements_features(elements, feature_map):
    
    # Create df of element features

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

    # Address missing values
    print('Features missing:\n', element_df.isna().sum(axis=0))
    element_df = element_df.fillna(element_df.mean())

    return element_df


def compute_engineered_features(composition_df, element_features_df):

    # Average and variance of element features

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


def plot_pca_scatter(fig, gs_cell, array, pca, labels, hue, hue_order, palette, show_legend, panel_label=''):

    # PC1 vs PC2 scatter, with the PCA loadings drawn as arrows and marginal KDEs on each axis.

    # Ax structure
    sub_gs = gs_cell.subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0)
    ax_joint = fig.add_subplot(sub_gs[1, 0])
    ax_x = fig.add_subplot(sub_gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(sub_gs[1, 1], sharey=ax_joint)

    # Scatter plot
    sns.scatterplot(ax=ax_joint, x=array[:, 0], y=array[:, 1], hue=hue, hue_order=hue_order, palette=palette)

    # PCA vectors
    for comp, var in zip(pca.components_, pca.explained_variance_):
        comp = comp * var  # scale component by its variance explanation power
        sns.lineplot(ax=ax_joint, x=[0, comp[0]], y=[0, comp[1]], color='k')

    # Format ax
    ax_joint.set(xlabel=f'Normalized {labels[0]}', ylabel=f'Normalized {labels[1]}')
    if show_legend:
        ax_joint.legend(loc='upper right', frameon=False)
    else:
        ax_joint.get_legend().remove()

    # KDE plot
    sns.kdeplot(ax=ax_x, x=array[:, 0], hue=hue, hue_order=hue_order, palette=palette)  # clip, cut
    sns.kdeplot(ax=ax_y, y=array[:, 1], hue=hue, hue_order=hue_order, palette=palette)
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()
    
    # Number plot
    ax_joint.text(-0.2/5*6, 1.1/5*6, panel_label, fontsize=10, fontweight='bold', fontfamily='arial', transform=ax_joint.transAxes)


def plot_pca_heatmap(ax, pca, labels):
    sns.heatmap(pca.components_, ax=ax, cmap='plasma_r', square=True, vmin=-1, vmax=1,
                xticklabels=labels,
                yticklabels=[f'Comp {i+1}\n{x:.02} explanation' for i, x in enumerate(pca.explained_variance_ratio_)])

