import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import SymLogNorm
import seaborn as sns
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
mycolor = 'cividis'


def load_data():
    data_clustered = pd.read_csv('data/catalysts/data_clustered.csv', na_values=[''], keep_default_na=False)
    elements = pd.read_csv('data/catalysts/elements.csv', header=None).squeeze('columns').to_list()
    composition = data_clustered.loc[:, elements]
    centroids = pd.read_csv('data/catalysts/centroids.csv', index_col=0)
    return data_clustered, composition, centroids


def build_cluster_palette(data_clustered):
    clusters = np.sort(data_clustered['Cluster_title'].unique())
    cmap = plt.get_cmap('cividis')
    palette = cmap(np.linspace(0, 1, len(clusters)))
    palette = sns.color_palette(palette, as_cmap=False, desat=0.8)
    return clusters, palette


def report_conditions(data_clustered):
    conditions = ['Ac_source', 'Fa_source', 'Stabilizer', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'O_content', 'LHSV_mlhg', 'Temperature_K', 'Pressure_bar']
    for column in conditions:
        print(f'Unique values for {column}', data_clustered[column].unique())


def plot_cluster_history(data_clustered, palette, clusters):
    # Groupby doi, year, and cluster
    grouped_data = data_clustered[['doi', 'Year', 'Cluster_title']].groupby(['doi', 'Cluster_title', 'Year'], as_index=False).count()

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    sns.histplot(grouped_data, x='Year', hue='Cluster_title', palette=palette, ax=ax, discrete=True, multiple='stack', shrink=0.8, edgecolor=None, alpha=1, hue_order=clusters)
    sns.move_legend(ax, loc='upper left', frameon=False, title='Catalyst clusters', handlelength=1.5)
    ax.set(ylabel='Publication count', xlim=(1965, 2026))
    fig.tight_layout()
    fig.savefig('figures/description/cluster_history.svg', dpi=300, format='svg')


def plot_composition_heatmap(composition):
    # Extract data from clustermap
    tmp = sns.clustermap(composition.T, cmap='viridis', row_cluster=False, norm=SymLogNorm(linthresh=0.01), cbar_pos=(1.1, 0, 0.05, 1), dendrogram_ratio=0.0001)
    tmp_data = tmp.data2d

    fig, ax = plt.subplots(1, 1, figsize=(5*1.2, 5))
    im = ax.pcolormesh(tmp_data.values, cmap='viridis', norm=SymLogNorm(linthresh=0.01))
    ax.set(xticks=[0, 100, 200, 300], xlabel='Catalysts', yticks=np.arange(len(tmp_data.index)) + 0.5, yticklabels=tmp_data.index)
    ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Atomic composition / mol%', rotation=270, verticalalignment='baseline')
    fig.savefig('figures/description/composition_v.png', dpi=300)


def plot_stacked_category(ax, data, column, palette, xlabel, panel_label, show_legend):
    # Groupby cluster_title and descriptor
    # Select any index of the groupby, and get the count of the descriptor
    # Unstack Cluster_title as columns
    # NaN value if no catalyst with this descriptor, therefore fill na with 0
    tmp_data = data.groupby([column, 'Cluster_title'])[column].count().unstack('Cluster_title').fillna(0) / len(data) * 100
    tmp_data.plot.bar(ax=ax, stacked=True, color=palette, width=0.1*len(tmp_data))
    ax.set(xlabel=xlabel, ylabel='Reported catalysts / %')
    if show_legend:
        ax.legend(frameon=False)
    else:
        ax.get_legend().remove()
    ax.text(-0.2, 1.1, panel_label, fontsize=10, fontweight='bold', fontfamily='arial', transform=ax.transAxes)


def plot_joint_kde(fig, gs_cell, data, x, y, palette, clusters, panel_label, configure_joint_axis):
    sub_gs = gs_cell.subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0)
    ax_joint = fig.add_subplot(sub_gs[1, 0])
    ax_x = fig.add_subplot(sub_gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(sub_gs[1, 1], sharey=ax_joint)

    sns.scatterplot(ax=ax_joint, data=data, x=x, y=y, hue='Cluster_title', hue_order=clusters, palette=palette)
    configure_joint_axis(ax_joint)
    ax_joint.get_legend().remove()

    sns.kdeplot(ax=ax_x, data=data, x=x, hue='Cluster_title', hue_order=clusters, palette=palette, cut=1)  # fill=True, alpha=0.25, 
    sns.kdeplot(ax=ax_y, data=data, y=y, hue='Cluster_title', hue_order=clusters, palette=palette, cut=1)  # fill=True, alpha=0.25, 
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()

    ax_joint.text(-0.2/5*6, 1.1/5*6, panel_label, fontsize=10, fontweight='bold', fontfamily='arial', transform=ax_joint.transAxes)


def configure_lhsv_temperature_axis(ax):
    ax.set(xlabel='LHSV / ml h$^{-1}$ g$^{-1}$', ylabel='T / K', xlim=(0, 8), ylim=(550, 700), xticks=[0, 2, 4, 6, 8], yticks=[550, 600, 650, 700])


def configure_ratio_axis(ax):
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=0.5, linscale=0.25)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set(xlabel='Ac/Fa', ylabel='Stab/Fa', xlim=(0.1, 10), ylim=(0, 5), xticks=[0.1, 0.5, 1, 2, 10], yticks=[0, 0.5, 1, 2, 5])


def plot_summary_panel(data_clustered, palette, clusters):
    # Initialize grid
    fig = plt.figure(figsize=(5, 9))
    gs = fig.add_gridspec(3, 2)

    # Add standard and complex subplots
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    # ax.append(gs[2, 0].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
    # ax.append(gs[2, 1].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))

    # Plot stacked categories
    plot_stacked_category(ax[0], data_clustered, 'Ac_source', palette, 'Ac precursor', 'a', show_legend=True)
    plot_stacked_category(ax[1], data_clustered, 'Fa_source', palette, 'Fa precursor', 'b', show_legend=False)
    plot_stacked_category(ax[2], data_clustered, 'Stabilizer', palette, 'Additives', 'c', show_legend=False)
    data_clustered['O_presence'] = data_clustered['O_content'] != 0
    plot_stacked_category(ax[3], data_clustered, 'O_presence', palette, 'Oxygen presence', 'd', show_legend=False)

    # Plot scatter-kde plots
    plot_joint_kde(fig, gs[2, 0], data_clustered, 'LHSV_mlhg', 'Temperature_K', palette, clusters, 'e', configure_lhsv_temperature_axis)
    plot_joint_kde(fig, gs[2, 1], data_clustered, 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', palette, clusters, 'f', configure_ratio_axis)

    fig.tight_layout()
    fig.savefig('figures/description/Fig_multi_kde.svg', dpi=300, format='svg')


def plot_feature_correlation(data_clustered):
    corr_columns = ['LHSV_mlhg', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'Temperature_K', 'O_content']
    norm_array = StandardScaler().fit_transform(data_clustered.dropna(subset=corr_columns).loc[:, corr_columns])
    corr_matrix = np.abs(np.corrcoef(norm_array, rowvar=False))
    fig = plt.figure()
    sns.heatmap(corr_matrix, cmap='plasma_r', xticklabels=corr_columns, yticklabels=corr_columns, vmin=0, vmax=1, square=True)
    fig.savefig('figures/description/feature_correlation.svg', dpi=600, bbox_inches='tight')


def main():
    data_clustered, composition, centroids = load_data()
    clusters, palette = build_cluster_palette(data_clustered)

    report_conditions(data_clustered)

    plot_cluster_history(data_clustered, palette, clusters)
    plot_composition_heatmap(composition)
    plot_summary_panel(data_clustered, palette, clusters)
    plot_feature_correlation(data_clustered)


if __name__ == "__main__":
    main()
