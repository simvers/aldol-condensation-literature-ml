import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import string
import json
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from src import stats_deact

ROOT = Path(__file__).resolve().parents[1]

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8

DATA_TYPE = '_noSi'

FIGURE_DIR = ROOT / 'figures/Stats_n'

PALETTE = ["#009688", "#1565C0", "#AD1457"]


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
    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(1, 1)
    ax = []
    ax.append(gs[0, 0].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
    ax_joint = fig.add_subplot(ax[0][1, 0])
    ax_x = fig.add_subplot(ax[0][0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(ax[0][1, 1], sharey=ax_joint)
    ax_joint.set_box_aspect(1)
    for i, cluster in enumerate(clusters):
        sns.regplot(ax=ax_joint, data=data_no_oxygen[data_no_oxygen['Cluster_title'] == cluster], x='STY0', y='n', label=cluster, color=PALETTE[i])
    ax_joint.legend()
    sns.move_legend(ax_joint, loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncols=4, title=None)
    ax_joint.set(xlim=(0.1, 40), ylim=(0, 0.3), xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')
    sns.kdeplot(ax=ax_x, data=data_no_oxygen, x='STY0', hue='Cluster_title', hue_order=clusters, palette=PALETTE, fill=True, alpha=0.25, clip=(0, None), cut=1)
    sns.kdeplot(ax=ax_y, data=data_no_oxygen, y='n', hue='Cluster_title', hue_order=clusters, palette=PALETTE, fill=True, alpha=0.25, clip=(0, None), cut=1)
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'sty0-n_scatterplot_reg.svg', dpi=600)

    # ------------------------------------------------------------------------------------------------------------------

    # Normalization
    log1p_list = ['LHSV_mlhg', 'STY0']
    data_no_oxygen[log1p_list] = np.log1p(data_no_oxygen[log1p_list])

    # ------------------------------------------------------------------------------------------------------------------

    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(1, 1)
    ax = []
    ax.append(gs[0, 0].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
    ax_joint = fig.add_subplot(ax[0][1, 0])
    ax_x = fig.add_subplot(ax[0][0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(ax[0][1, 1], sharey=ax_joint)
    for i, cluster in enumerate(clusters):
        sns.regplot(ax=ax_joint, data=data_no_oxygen[data_no_oxygen['Cluster_title'] == cluster], x='STY0', y='n', label=cluster, color=PALETTE[i], robust=True)
    ax_joint.legend()
    sns.move_legend(ax_joint, loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=False, ncols=1, title=None)
    ax_joint.set(xlabel='ln( 1 + STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$ )', ylabel='n /', xlim=(0, 4), ylim=(0, 0.3))
    sns.kdeplot(ax=ax_x, data=data_no_oxygen, x='STY0', hue='Cluster_title', hue_order=clusters, palette=PALETTE, fill=True, alpha=0.25, clip=(0, None), cut=1)
    sns.kdeplot(ax=ax_y, data=data_no_oxygen, y='n', hue='Cluster_title', hue_order=clusters, palette=PALETTE, fill=True, alpha=0.25, clip=(0, None), cut=1)
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'sty0-n_scatterplot_norm_reg_no_oxygen.svg', dpi=600)
    print(data.columns)

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
    pval_ = np.empty((len(features), len(clusters)))
    pval_covar = np.empty((len(features), len(clusters)))

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
            else:
                rho, pval = stats_deact.partial_corr_xcovar(
                    x=data.loc[data['Cluster_title'] == cluster, feature].values,
                    y=data.loc[data['Cluster_title'] == cluster, 'n'].values,
                    covars=[data.loc[data['Cluster_title'] == cluster, 'STY0'].values,
                            data.loc[data['Cluster_title'] == cluster, 'O_content'].values]
                )
                spearmanr_covar[i, j], pval_covar[i, j] = rho, pval

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
        alphas = np.clip(norm(np.nan_to_num(pval_covar[:, i], nan=1.0)), 0.0, 0.8)
        rgbas = [mcolors.to_rgba(PALETTE[i], alpha=1-a) for a in alphas]
        ax.scatter(spearmanr_covar[:, i], y, color=rgbas, s=80, zorder=3, marker='^')

    ax.set_yticks(y)
    ax.set_yticklabels([feature_labels.get(feature, '').split(' /')[0] for feature in features])
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel('Spearman ρ')
    # ax.grid(axis='x', color='#D3D1C7', linewidth=0.5, zorder=1)
    ax.axvline(0, color='k', linewidth=2.0, linestyle='--', zorder=2)

    # Colorbar (p-value)
    grey_cmap = mcolors.LinearSegmentedColormap.from_list('pval_grey', ['#2C2C2A', '#F1EFE8'])
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
        Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=8, label='Covar-control')
    ]
    leg = ax.legend(handles=legend_elements, frameon=False, loc='upper left', facecolor='w')
    for text in leg.get_texts():
        if text.get_text() in ['Cluster', 'Metric']:
            text.set_fontweight('bold')

    fig.savefig(FIGURE_DIR / 'spearman_plot.svg', dpi=600)
