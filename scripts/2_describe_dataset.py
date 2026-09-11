"""
Descriptive figures of the literature dataset: catalyst publication history, composition
heatmap by cluster, and reaction condition distributions/correlations.

Reads:  data/processed/data_clustered.csv, data/processed/elements.csv, data/processed/centroids.csv
Writes: figures/description/*.svg, figures/description/*.png
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import SymLogNorm
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from src import utils


# Format plot
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
# cmap = plt.get_cmap('cividis')
# palette = cmap(np.linspace(0, 1, 3))
# palette = sns.color_palette(palette, as_cmap=False, desat=0.8)
PALETTE = ["#009688", "#1565C0", "#AD1457"]

ROOT = Path(__file__).resolve().parents[1]
VERBOSE = True

with open(ROOT / 'config/feature_labels.json') as f:
    _labels = json.load(f)
LONG_LABELS = _labels['long']
SHORT_LABELS = _labels['short']


def report_conditions(data_clustered):
    conditions = ['Ac_source', 'Fa_source', 'Stabilizer', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'O_content', 'LHSV_mlhg', 'Temperature_K', 'Pressure_bar']
    for column in conditions:
        print(f'Unique values for {column}', data_clustered[column].unique())


def plot_stacked_category(ax, data, column, palette, panel_label, show_legend):
    tmp_data = data.groupby([column, 'Cluster_title'])[column].count().unstack('Cluster_title').fillna(0) / len(data) * 100
    tmp_data.plot.bar(ax=ax, stacked=True, color=palette, width=0.1*len(tmp_data))
    ax.set(xlabel=LONG_LABELS.get(column, column), ylabel='Reported catalysts / %')
    if show_legend:
        ax.legend(frameon=False)
    else:
        ax.get_legend().remove()
    ax.text(-0.2, 1.1, panel_label, fontsize=10, fontweight='bold', fontfamily='arial', transform=ax.transAxes)


def plot_joint_kde(fig, gs_cell, data, x, y, palette, clusters, panel_label, configure_joint_axis):
    # Structure plot
    sub_gs = gs_cell.subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0)
    ax_joint = fig.add_subplot(sub_gs[1, 0])
    ax_x = fig.add_subplot(sub_gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(sub_gs[1, 1], sharey=ax_joint)

    # Scatter plot
    sns.scatterplot(ax=ax_joint, data=data, x=x, y=y, hue='Cluster_title', hue_order=clusters, palette=palette, alpha=0.6)
    configure_joint_axis(ax_joint)
    ax_joint.set(xlabel=LONG_LABELS.get(x, x), ylabel=LONG_LABELS.get(y, y))
    ax_joint.get_legend().remove()

    # KDE plots
    sns.kdeplot(ax=ax_x, data=data, x=x, hue='Cluster_title', hue_order=clusters, palette=palette, cut=1, fill=True)  # fill=True, alpha=0.25, 
    sns.kdeplot(ax=ax_y, data=data, y=y, hue='Cluster_title', hue_order=clusters, palette=palette, cut=1, fill=True)  # fill=True, alpha=0.25, 
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()

    # Number plot
    ax_joint.text(-0.2/5*6, 1.1/5*6, panel_label, fontsize=10, fontweight='bold', fontfamily='arial', transform=ax_joint.transAxes)


def configure_lhsv_temperature_axis(ax):
    ax.set(xlim=(0, 8), ylim=(550, 700), xticks=[0, 2, 4, 6, 8], yticks=[550, 600, 650, 700])


def configure_ratio_axis(ax):
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=0.5, linscale=0.25)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set(xlim=(0.1, 10), ylim=(0, 5), xticks=[0.1, 0.5, 1, 2, 10], yticks=[0, 0.5, 1, 2, 5])


if __name__ == "__main__":
    
    # Load data
    data_clustered = utils.load_data(ROOT / 'data/processed/data_clustered.csv')
    elements = pd.read_csv(ROOT / 'data/processed/elements.csv', header=None).squeeze('columns').to_list()
    composition = data_clustered.loc[:, elements]
    centroids = pd.read_csv(ROOT / 'data/processed/centroids.csv', index_col=0)
    clusters = np.sort(data_clustered['Cluster_title'].unique())

    # Print unique values
    if VERBOSE == True:
        report_conditions(data_clustered)

    # -----------------------------------------------

    # Plot catalyst history
    grouped_data = data_clustered[['doi', 'Year', 'Cluster_title']].groupby(['doi', 'Cluster_title', 'Year'], as_index=False).count()
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    sns.histplot(grouped_data, x='Year', hue='Cluster_title', palette=PALETTE, ax=ax, discrete=True, multiple='stack', shrink=0.8, edgecolor=None, alpha=1, hue_order=clusters)
    sns.move_legend(ax, loc='upper left', frameon=False, title='Catalyst clusters', handlelength=1.5)
    ax.set(ylabel='Publication count', xlim=(1965, 2026))
    fig.tight_layout()
    fig.savefig(ROOT / 'figures/description/cluster_history.svg', dpi=300, format='svg')

    # -----------------------------------------------

    # Plot composition

    # Extract data from clustermap
    tmp = sns.clustermap(composition.T, cmap='viridis', row_cluster=False, norm=SymLogNorm(linthresh=0.01), cbar_pos=(1.1, 0, 0.05, 1), dendrogram_ratio=0.0001)
    tmp_data = tmp.data2d

    fig, ax = plt.subplots(1, 1, figsize=(5*1.2, 5))
    im = ax.pcolormesh(tmp_data.values, cmap='viridis', norm=SymLogNorm(linthresh=0.01))
    ax.set(xticks=[0, 100, 200, 300], xlabel='Catalysts', yticks=np.arange(len(tmp_data.index)) + 0.5, yticklabels=tmp_data.index)
    ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Atomic composition / mol%', rotation=270, verticalalignment='baseline')
    ax.text((0.00+0.27)/2, 1.04, 'Al-Si-Cs', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
    ax.text((0.27+0.61)/2, 1.04, 'Si-Cs-P', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
    ax.text((0.61+1.00)/2, 1.10, 'P-V-Ti', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
    ax.text((0.61+0.77)/2, 1.04, 'Supp.', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
    ax.text((0.77+1.00)/2, 1.04, 'Bulk', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
    ax.annotate('', xytext=(0.00+0.000, 1.02), xy=(0.27-0.001, 1.02), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
    ax.annotate('', xytext=(0.27+0.001, 1.02), xy=(0.61-0.001, 1.02), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
    ax.annotate('', xytext=(0.61+0.001, 1.08), xy=(1.00-0.001, 1.08), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
    ax.annotate('', xytext=(0.61+0.001, 1.02), xy=(0.77-0.001, 1.02), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
    ax.annotate('', xytext=(0.77+0.001, 1.02), xy=(1.00-0.000, 1.02), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
    fig.savefig(ROOT / 'figures/description/composition_description.png', dpi=300)

    # -----------------------------------------------

    # Plot reaction conditions distribution

    # Initialize grid
    fig = plt.figure(figsize=(5, 9))
    gs = fig.add_gridspec(3, 2)

    # Add standard and complex subplots
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    # ax.append(gs[2, 0].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
    # ax.append(gs[2, 1].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))

    # Plot stacked categories
    plot_stacked_category(ax[0], data_clustered, 'Ac_source', PALETTE, 'a', show_legend=True)
    plot_stacked_category(ax[1], data_clustered, 'Fa_source', PALETTE, 'b', show_legend=False)
    plot_stacked_category(ax[2], data_clustered, 'Stabilizer', PALETTE, 'c', show_legend=False)
    data_clustered['O_presence'] = data_clustered['O_content'] != 0
    plot_stacked_category(ax[3], data_clustered, 'O_presence', PALETTE, 'd', show_legend=False)

    # Plot LHSV-T scatter-kde plot
    plot_joint_kde(fig, gs[2, 0], data_clustered, 'LHSV_mlhg', 'Temperature_K', PALETTE, clusters, 'e', configure_lhsv_temperature_axis)

    # Plot reactant ratio scatter-kde plot
    plot_joint_kde(fig, gs[2, 1], data_clustered, 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', PALETTE, clusters, 'f', configure_ratio_axis)

    fig.tight_layout()
    fig.savefig(ROOT / 'figures/description/reaction_conditions_description.svg', dpi=300, format='svg')

    # -----------------------------------------------
    
    # Plot feature correlation
    corr_columns = ['LHSV_mlhg', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'Temperature_K', 'O_content']
    with open(ROOT / 'config/feature_labels.json') as f:
        feature_labels_long = json.load(f)['short']
    corr_labels = [feature_labels_long.get(c, c) for c in corr_columns]
    norm_array = StandardScaler().fit_transform(data_clustered.dropna(subset=corr_columns).loc[:, corr_columns])
    corr_matrix = np.abs(np.corrcoef(norm_array, rowvar=False))
    fig = plt.figure()
    sns.heatmap(corr_matrix, cmap='plasma_r', xticklabels=corr_labels, yticklabels=corr_labels, vmin=0, vmax=1, square=True)
    fig.savefig(ROOT / 'figures/description/reaction_conditions_correlation.svg', dpi=600, bbox_inches='tight')
