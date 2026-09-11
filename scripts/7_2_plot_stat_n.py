"""
Partial-correlation (Spearman, covariance-controlled) analysis of deactivation drivers:
STY0-n relationships, per-feature scatter-kde plots, and univariate vs.
STY0/O_content-controlled Spearman rho per catalyst cluster.

Reads:  data/processed/data_deactivation{DATA_TYPE}.csv, data/processed/elements.csv
Writes: figures/Stats_n/*.svg
Edit before running: DATA_TYPE
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns
import string
import json
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from src import stats_analysis

ROOT = Path(__file__).resolve().parents[1]

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8

DATA_TYPE = '_noSi'  # '' to include Si in the engineered feature set

FIGURE_DIR = ROOT / 'figures/Stats_n'

PALETTE = ["#009688", "#1565C0", "#AD1457"]


def plot_sty0_n_scatter(fig, gs_cell, data, clusters, palette, show_legend, panel_label, reg=False, log_scale=True):
    # STY0 vs n scatter with marginal KDEs — called for the full dataset and the oxygen-free subset
    sub_gs = gs_cell.subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0)
    ax_joint = fig.add_subplot(sub_gs[1, 0])
    ax_x = fig.add_subplot(sub_gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(sub_gs[1, 1], sharey=ax_joint)
    # ax_joint.set_box_aspect(1)

    # Plot
    # if reg:
    #     for i, cluster in enumerate(clusters):
    #         sns.regplot(ax=ax_joint, data=data[data['Cluster_title'] == cluster], x='STY0', y='n', label=cluster, color=PALETTE[i], scatter_kws={'alpha': 0.6})
    # else:
    #     sns.scatterplot(ax=ax_joint, data=data, x='STY0', y='n', hue='Cluster_title', hue_order=clusters, palette=palette, alpha=0.6)
    sns.scatterplot(ax=ax_joint, data=data, x='STY0', y='n', hue='Cluster_title', hue_order=clusters, palette=palette, alpha=0.6, s=20)
    if reg:
        for i, cluster in enumerate(clusters):
            sns.regplot(ax=ax_joint, data=data[data['Cluster_title'] == cluster], x='STY0', y='n', color=PALETTE[i], scatter=False, robust=False)

    # Legend
    ax_joint.legend()
    if show_legend:
        sns.move_legend(ax_joint, loc='center left', bbox_to_anchor=(1.25, 0.5), frameon=False, title=None, ncols=1)
    else:
        ax_joint.legend().remove()

    # Axes
    if log_scale:
        ax_joint.set_xscale('symlog')
        ax_joint.set_yscale('symlog', linthresh=1e-2)
        ax_joint.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_joint.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_joint.set(xlim=(0, None), ylim=(0, 1),
                 xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')

    # Marginal plots
    sns.kdeplot(ax=ax_x, data=data, x='STY0', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
    sns.kdeplot(ax=ax_y, data=data, y='n', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()

    # Number plot
    ax_joint.text(-0.2/5*6, 1.1/5*6, panel_label, fontsize=10, fontweight='bold', fontfamily='arial', transform=ax_joint.transAxes)

    return ax_joint


if __name__ == "__main__":

    # Import data
    data = pd.read_csv(ROOT / f"data/processed/data_deactivation{DATA_TYPE}.csv")
    elements = pd.read_csv(ROOT / 'data/processed/elements.csv', header=None).squeeze('columns').to_list()
    with open(ROOT / "config/feature_labels.json") as f:
        feature_labels = json.load(f)["long"]

    # Drop columns
    data.drop(elements + ['Y_Acryl_Ac', 'Y_Acryl_Fa', 'doi', 'Link_to_excel', 'Year', # 'Cluster_title',
            'Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming',
            'Ac_source', 'Fa_source', 'Stabilizer',
            # 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'LHSV_mlhg',
            'STY_Acryl_mmolhg', 'Pressure_bar', # 'STY0', 'n',
            'av_cov_rad', 'av_n_val', 'var_cov_rad', 'var_n_val',
            'var_pca',
            'Cluster_n'], axis=1, inplace=True)
    print(data.columns)

    clusters = np.sort(data['Cluster_title'].unique())

    # Identify numerical and categorical features
    numerical_cols = [col for col in data.select_dtypes(include=['int64', 'float64']).columns.to_list() if col not in ['STY0', 'n']]
    print(numerical_cols)

    # Exclude exp with no activity
    data = data[data['STY0'] > 0.000001]
    # Exclude exp under oxygen
    data_no_oxygen = data.loc[data['O_content'] < 0.00001, :]
    print(data.groupby('Cluster_title').count())
    print(data_no_oxygen.groupby('Cluster_title').count())

    # ------------------------------------------------------------------------------------------------------------------

    # Scatter plot of sty0 vs n

    fig = plt.figure(figsize=(6, 2.5))
    gs = fig.add_gridspec(1, 2, wspace=0.5)
    ax = plot_sty0_n_scatter(fig, gs[0, 0], data, clusters, PALETTE, show_legend=False, panel_label='a')
    ax.text(0.05, 0.95, 'All data', ha='left', va='top', transform=ax.transAxes)
    ax = plot_sty0_n_scatter(fig, gs[0, 1], data_no_oxygen, clusters, PALETTE, show_legend=True, panel_label='b')
    ax.text(0.05, 0.95, 'No O$_2$', ha='left', va='top', transform=ax.transAxes)
    fig.savefig(FIGURE_DIR / 'sty0-n_scatterplot.svg', dpi=600, bbox_inches='tight')

    # ------------------------------------------------------------------------------------------------------------------

    # Normalization
    log1p_list = ['LHSV_mlhg', 'STY0']
    data[log1p_list] = np.log1p(data[log1p_list])
    data_no_oxygen[log1p_list] = np.log1p(data_no_oxygen[log1p_list])
    print(data.columns)

    # ------------------------------------------------------------------------------------------------------------------

    fig = plt.figure(figsize=(6, 2.5))
    gs = fig.add_gridspec(1, 2, wspace=0.5)
    ax = plot_sty0_n_scatter(fig, gs[0, 0], data, clusters, PALETTE, show_legend=False, panel_label='a', reg=True, log_scale=False)
    ax.set(xlabel='ln( 1 + STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$ )', xlim=(0, 4), ylim=(0, 0.3), yticks=[0, 0.1, 0.2, 0.3])
    ax.text(0.05, 0.95, 'All data', ha='left', va='top', transform=ax.transAxes)
    ax = plot_sty0_n_scatter(fig, gs[0, 1], data_no_oxygen, clusters, PALETTE, show_legend=True, panel_label='b', reg=True, log_scale=False)
    ax.set(xlabel='ln( 1 + STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$ )', xlim=(0, 4), ylim=(0, 0.3), yticks=[0, 0.1, 0.2, 0.3])
    ax.text(0.05, 0.95, 'No O$_2$', ha='left', va='top', transform=ax.transAxes)
    fig.savefig(FIGURE_DIR / 'sty0-n_scatterplot_norm_reg.svg', dpi=600, bbox_inches='tight')

    # ------------------------------------------------------------------------------------------------------------------

    # Feature-n scatter with marginal KDE
    letters = list(string.ascii_lowercase)
    features = numerical_cols + ['STY0']
    n_feats = len(features)
    n_rows = int(np.ceil(n_feats / 2))
    has_empty = (n_rows * 2 > n_feats)

    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[i],
                              label=cluster, markersize=6)
                      for i, cluster in enumerate(clusters)]

    fig = plt.figure(figsize=(6, n_rows * 3+1))
    gs = fig.add_gridspec(n_rows, 2, hspace=0.4, wspace=0.4)
    ax = []

    for i, feature in enumerate(features):
        row, col = divmod(i, 2)

        ax.append(gs[row, col].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
        ax_joint = fig.add_subplot(ax[i][1, 0])
        ax_x = fig.add_subplot(ax[i][0, 0], sharex=ax_joint)
        ax_y = fig.add_subplot(ax[i][1, 1], sharey=ax_joint)

        selected_df = data_no_oxygen if feature != 'O_content' else data

        for j, cluster in enumerate(clusters):
            sns.regplot(ax=ax_joint, data=selected_df[selected_df['Cluster_title'] == cluster], x=feature, y='n', label=cluster, color=PALETTE[j], robust=True, scatter_kws=dict(s=15, alpha=0.6))

        prefix, sufix = ('ln( 1 + ', ' )') if feature in log1p_list else ('', '')
        xlabel = prefix + feature_labels.get(feature, feature) + sufix
        ax_joint.set(ylim=(0, 0.2), xlabel=xlabel, ylabel='n /')
        if i == 0 and not has_empty:
            ax_joint.legend(handles=legend_handles, loc='upper left', frameon=False, fontsize=7)
        sns.kdeplot(ax=ax_x, data=selected_df, x=feature, hue='Cluster_title', hue_order=clusters, palette=PALETTE, fill=True, alpha=0.25, clip=(0, None), cut=1)
        sns.kdeplot(ax=ax_y, data=selected_df, y='n', hue='Cluster_title', hue_order=clusters, palette=PALETTE, fill=True, alpha=0.25, clip=(0, None), cut=1)
        for item in [ax_x, ax_y]:
            item.set_axis_off()
            item.get_legend().remove()
        ax_joint.text(-0.2/5*6, 1.0/5*6, letters[i], fontsize=10, fontweight='bold', fontfamily='arial', transform=ax_joint.transAxes)

    if has_empty:
        legend_ax = fig.add_subplot(gs[n_rows - 1, 1])
        legend_ax.axis('off')
        legend_ax.legend(handles=legend_handles, loc='center', frameon=False)

    fig.savefig(FIGURE_DIR / 'feature-n_kde-scatterplot.svg', dpi=600, bbox_inches='tight')

    # ------------------------------------------------------------------------------------------------------------------

    # Calculate Spearman correlation coefficients (univariate and covariance-controlled)
    spearmanr_ = np.empty((len(features), len(clusters)))
    spearmanr_covar = np.empty((len(features), len(clusters)))
    partial_spearmanr = np.empty((len(features), len(clusters)))
    pval_ = np.empty((len(features), len(clusters)))
    pval_covar = np.empty((len(features), len(clusters)))
    pval_partial_spearman = np.empty((len(features), len(clusters)))

    for i, feature in enumerate(features):
        for j, cluster in enumerate(clusters):

            # Univariate Spearman ρ
            rho, pval = spearmanr(
                a=data.loc[data['Cluster_title'] == cluster, feature].values,
                b=data.loc[data['Cluster_title'] == cluster, 'n'].values
            )
            spearmanr_[i, j] = rho
            pval_[i, j] = pval

            # Partial Spearman ρ controlling for STY0 and O_content
            # Skip when feature is itself a covariate (degenerate residuals)
            if feature in ['STY0', 'O_content']:
                spearmanr_covar[i, j], pval_covar[i, j] = np.nan, np.nan
                partial_spearmanr[i, j], pval_partial_spearman[i, j] = np.nan, np.nan
            else:
                rho, pval = stats_analysis.partial_corr_xcovar(
                    x=data.loc[data['Cluster_title'] == cluster, feature].values,
                    y=data.loc[data['Cluster_title'] == cluster, 'n'].values,
                    covars=[data.loc[data['Cluster_title'] == cluster, 'STY0'].values,
                            data.loc[data['Cluster_title'] == cluster, 'O_content'].values]
                )
                spearmanr_covar[i, j], pval_covar[i, j] = rho, pval

                rho, pval = stats_analysis.pg_partial_spearman(
                    df=data.loc[data['Cluster_title'] == cluster, :],
                    x=feature, y='n', z=['STY0', 'O_content']
                )
                partial_spearmanr[i, j], pval_partial_spearman[i, j] = rho, pval

    # ------------------------------------------------------------------------------------------------------------------

    # Plot Spearman ρ per feature and cluster
    fig, (ax, cax) = plt.subplots(
        1, 2,
        figsize=(5, 3),
        gridspec_kw={'width_ratios': [20, 1]},
        constrained_layout=True
    )

    n = len(features)
    y = np.arange(n)
    norm = mcolors.LogNorm(vmin=0.01, vmax=0.2)

    for i, cluster in enumerate(clusters):
        # Univariate Spearman ρ (circle)
        alphas = np.clip(norm(np.nan_to_num(pval_[:, i], nan=1.0)), 0.0, 0.8)
        rgbas = [mcolors.to_rgba(PALETTE[i], alpha=1-a) for a in alphas]
        ax.scatter(spearmanr_[:, i], y, color=rgbas, s=80, zorder=3, marker='o', label=cluster)

        # Covariance-controlled Spearman ρ (triangle)
        alphas = np.clip(norm(np.nan_to_num(pval_partial_spearman[:, i], nan=1.0)), 0.0, 0.8)
        rgbas = [mcolors.to_rgba(PALETTE[i], alpha=1-a) for a in alphas]
        ax.scatter(partial_spearmanr[:, i], y, color=rgbas, s=80, zorder=3, marker='^')

    ax.set_yticks(y)
    ax.set_yticklabels([feature_labels.get(feature, '').split(' /')[0] for feature in features])
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel('Spearman ρ')
    # ax.grid(axis='x', color='#D3D1C7', linewidth=0.5, zorder=1)
    ax.axvline(0, color='k', linewidth=2.0, linestyle='--', zorder=2)

    # Colorbar (p-value)
    grey_cmap = mcolors.LinearSegmentedColormap.from_list('pval_grey', ['#5A5A5A', '#F1EFE8'])

    cb = ColorbarBase(cax, cmap=grey_cmap, orientation='vertical', norm=mcolors.LogNorm(vmin=0.01, vmax=0.2))
    cb.ax.set_ylabel('p-value', rotation=270, verticalalignment='baseline')
    cb.set_ticks([0.01, 0.05, 0.1, 0.2])
    cb.set_ticklabels([0.01, 0.05, 0.10, 0.20])
    cb.ax.tick_params(labelsize=8)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=8, label='Cluster'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[0], markersize=8, label=clusters[0]),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[1], markersize=8, label=clusters[1]),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE[2], markersize=8, label=clusters[2]),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=8, label='Metric'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Univariate'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=8, label='Partial')
    ]
    leg = ax.legend(handles=legend_elements, frameon=False, loc='upper left', facecolor='w')
    for text in leg.get_texts():
        if text.get_text() in ['Cluster', 'Metric']:
            text.set_fontweight('bold')

    fig.savefig(FIGURE_DIR / 'spearman_plot.svg', dpi=600)
