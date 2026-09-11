"""
Feature-feature and feature-target (STY, n) correlation plots: Pearson/Spearman
correlation heatmaps of the reaction-condition and catalyst features, plus
per-feature scatter-kde plots against STY and deactivation rate n.

Reads:  data/processed/data_engineered{SUFFIX}.csv, data/processed/data_deactivation{SUFFIX}.csv
Writes: figures/feature_correlation/*.svg
Edit before running: SUFFIX, REAC_INPUT, CAT_INPUT
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src import utils


# Format plot
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8

# Context variables
ROOT = Path(__file__).resolve().parents[1]
SUFFIX = '_noSi'  # '' to include Si in the engineered feature set
PALETTE = ["#009688", "#1565C0", "#AD1457"]

# Features
REAC_INPUT = [  # reaction condition features
    'LHSV_mlhg',
    'Ratio_Ac_Fa', 'Ratio_Stab_Fa',
    'Temperature_K',
    'Ac_source', 'Fa_source',
    'Stabilizer',
    'O_content',
]
CAT_INPUT = [  # catalyst (engineered) features
    'SSA_m2g',
    'av_cov_rad',
    'av_n_val',
    'var_cov_rad',
    'var_n_val',
]
INPUT = REAC_INPUT + CAT_INPUT
OUTPUT = 'STY_Acryl_mmolhg'


def plot_pub_distribution(ax, data, target, group='doi', color='k'):
    obs = data.dropna(subset=target).groupby(group).count()[target].values
    print('Data with target: ', len(data.dropna(subset=target)), 'from ', len(data), 'over ', len(data['doi'].unique()), 'max ', obs.max(), 'mean ', obs.mean())
    sns.histplot(ax=ax, x=obs, discrete=True, binwidth=1, color=color, alpha=0.6, linewidth=0, label=f'{target.split("_")[0]}: {len(data.dropna(subset=target))} obs. over {len(obs)} pub.')
    ax.axvline(x=obs.mean(), color=color, linestyle='--', linewidth=0.75)


if __name__ == '__main__':
        
    # Import data
    data_STY = utils.load_data(ROOT / f'data/processed/data_engineered{SUFFIX}.csv')
    data_n = utils.load_data(ROOT / f'data/processed/data_deactivation{SUFFIX}.csv')

    # -------------------------------------------------------------

    # Distribution of observations per publication
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_pub_distribution(ax, data_STY, target='STY_Acryl_mmolhg', color='DarkBLue')
    plot_pub_distribution(ax, data_n, target='n', color='DarkRed')
    ax.set(xlabel='Observations per publication /', ylabel='Publication count /', xlim=(0, None))
    ax.legend(loc='upper right', frameon=False, title=f'Dataset: {len(data_STY)} obs. over {len(data_STY["doi"].unique())} pub.', alignment='left')
    fig.savefig(ROOT / 'figures/feature_correlation/observations_count.svg', dpi=600)
    
    # -------------------------------------------------------------

    # Drop observations with missing output
    data_STY = data_STY.dropna(subset=OUTPUT)
    if data_STY[INPUT].isna().any().any():
        print('Dropping observations with NaNs in input feature!')
        data_STY = data_STY.dropna(subset=INPUT)
    print('Data with STY: ', len(data_STY))

    # Select features
    data = data_STY[INPUT]

    # -------------------------------------------------------------

    # Identify numerical and categorical features
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

    # Condense data
    mask = (data['Stabilizer'] == 'MeOH') | (data['Stabilizer'] == 'MeOH \n+ H$_2$O') | (data['Stabilizer'] == 'EtOH')
    data.loc[mask, 'Stabilizer'] = 'Stabilizer'

    # Preprocessing: normalize and one-hot encode
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], verbose_feature_names_out=False)
    norm_array = preprocessor.fit_transform(data)
    onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()
    df = pd.DataFrame(norm_array, columns=feature_names)

    # Select columns
    remove_col = categorical_cols.tolist() + ['Ac_source_MAc', 'Ac_source_EAc', 'Fa_source_TRX', 'Fa_source_DMM', 'Fa_source_MeOH', 'Stabilizer_Stabilizer']
    selected_col = [col for col in (REAC_INPUT + onehot_feature_names.tolist() + CAT_INPUT) if col not in remove_col]
    df_selected = df.loc[:, selected_col]

    # Rename columns
    with open(ROOT / 'config/feature_labels.json') as f:
        feature_mapper = json.load(f)['short']
    feature_mapper.update({'av_cov_rad': 'Cov. rad.', 'av_n_val': 'n val.', 'var_cov_rad': 'Cov. rad.', 'var_n_val': 'n val.'})
    df_renamed = df_selected.rename(columns=feature_mapper)

    # -------------------------------------------------------------

    # Plot correlation matrices
    metrics = ['pearson', 'spearman']

    for metric in metrics:

        # Rename and select columns
        corr_matrix = df_renamed.corr(method=metric).abs()
        # corr_df = pd.DataFrame(np.abs(np.corrcoef(norm_array, rowvar=False)), index=feature_names, columns=feature_names)

        # Correlation of features
        fig, ax = plt.subplots(1, 1)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, cmap='plasma_r', vmin=0, vmax=1, square=True, mask=mask, ax=ax)
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel(f'{metric.title()} correlation coefficient', rotation=270, labelpad=15)
        # Horizontal text
        title, subtitle = -0.3, -0.2
        line, subline = -0.25, -0.16
        ax.text(4/13, title, 'Reaction conditions', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
        # ax.text(6.5/13, title, 'Reactants', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
        ax.text(10.5/13, title, 'Catalyst', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
        ax.text(10/13, subtitle, 'Mean', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
        ax.text(12/13, subtitle, 'Variance', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
        ax.annotate('', xytext=(0+0.005, line), xy=(8/13-0.005, line), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
        # ax.annotate('', xytext=(5/13+0.005, line), xy=(8/13-0.005, line), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
        ax.annotate('', xytext=(8/13+0.005, line), xy=(13/13-0.005, line), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
        ax.annotate('', xytext=(9/13+0.005, subline), xy=(11/13-0.005, subline), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
        ax.annotate('', xytext=(11/13+0.005, subline), xy=(13/13-0.005, subline), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
        # Vertical text
        ax.text(subtitle, 1/13, 'Variance', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
        ax.text(subtitle, 3/13, 'Mean', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
        ax.text(title, 2.5/13, 'Catalyst', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
        # ax.text(title, 6.5/13, 'Reactants', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
        ax.text(title, 9/13, 'Reaction conditions', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
        ax.annotate('', xytext=(line, 0+0.005), xy=(line, 5/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
        # ax.annotate('', xytext=(line, 5/13+0.005), xy=(line, 8/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
        ax.annotate('', xytext=(line, 5/13+0.005), xy=(line, 13/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
        ax.annotate('', xytext=(subline, 0+0.005), xy=(subline, 2/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
        ax.annotate('', xytext=(subline, 2/13+0.005), xy=(subline, 4/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
        fig.savefig(ROOT / f'figures/feature_correlation/feature_correlation_{metric}.svg', dpi=600, bbox_inches='tight')

    # -------------------------------------------------------------

    # Target-feature correlation plot

    cluster_list = data_STY['Cluster_title']
    clusters = np.sort(cluster_list.unique())
    x = data.loc[:, numerical_cols]
    y = np.log1p(data_STY.loc[:, 'STY_Acryl_mmolhg'])

    fig = plt.figure(figsize=(6, 3*5+1))
    gs = fig.add_gridspec(int(np.ceil(len(x.columns)/2)), 2)
    ax = []
    for i, feature in enumerate(x.columns):
        ax.append(gs[i//2, i%2].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
        ax_joint = fig.add_subplot(ax[i][1, 0])
        ax_x = fig.add_subplot(ax[i][0, 0], sharex=ax_joint)
        ax_y = fig.add_subplot(ax[i][1, 1], sharey=ax_joint)
        sns.scatterplot(x=x.loc[:, feature], y=y, ax=ax_joint, hue=cluster_list, hue_order=clusters, palette=PALETTE)
        sns.regplot(x=x.loc[:, feature], y=y, ax=ax_joint, order=1, scatter=False, line_kws={"color": 'k'})
        ax_joint.set(xlabel=feature, ylabel='STY', ylim=(0, 4), yticks=[0, 1, 2, 3, 4])
        ax_joint.get_legend().remove()
        sns.kdeplot(ax=ax_x, x=x.loc[:, feature], hue=cluster_list, hue_order=clusters, palette=PALETTE, cut=0)
        sns.kdeplot(ax=ax_y, y=y, hue=cluster_list, hue_order=clusters, palette=PALETTE, cut=0)
        for item in [ax_x, ax_y]:
            item.set_axis_off()
            item.get_legend().remove()
    fig.savefig(ROOT / f'figures/feature_correlation/target_STY_correlation.svg', dpi=600, bbox_inches='tight')

    # -------------------------------------------------------------

    # Target-feature correlation plot for n (deactivation rate)

    N_INPUT = INPUT + ['STY_Acryl_mmolhg']
    data_n_clean = data_n.dropna(subset=['n'])
    cluster_list_n = data_n_clean['Cluster_title']
    clusters_n = np.sort(cluster_list_n.unique())
    x_n = data_n_clean[N_INPUT].select_dtypes(include=['int64', 'float64'])
    y_n = data_n_clean.loc[:, 'n']

    fig = plt.figure(figsize=(6, 3*6+1))
    gs = fig.add_gridspec(int(np.ceil(len(x_n.columns) / 2)), 2)
    ax = []
    for i, feature in enumerate(x_n.columns):
        ax.append(gs[i//2, i%2].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
        ax_joint = fig.add_subplot(ax[i][1, 0])
        ax_x = fig.add_subplot(ax[i][0, 0], sharex=ax_joint)
        ax_y = fig.add_subplot(ax[i][1, 1], sharey=ax_joint)
        sns.scatterplot(x=x_n.loc[:, feature], y=y_n, ax=ax_joint, hue=cluster_list_n, hue_order=clusters_n, palette=PALETTE)
        sns.regplot(x=x_n.loc[:, feature], y=y_n, ax=ax_joint, order=1, scatter=False, line_kws={"color": 'k'})
        ax_joint.set(xlabel=feature, ylabel='n')
        ax_joint.get_legend().remove()
        sns.kdeplot(ax=ax_x, x=x_n.loc[:, feature], hue=cluster_list_n, hue_order=clusters_n, palette=PALETTE, cut=0)
        sns.kdeplot(ax=ax_y, y=y_n, hue=cluster_list_n, hue_order=clusters_n, palette=PALETTE, cut=0)
        for item in [ax_x, ax_y]:
            item.set_axis_off()
            item.get_legend().remove()
    fig.savefig(ROOT / f'figures/feature_correlation/target_n_correlation.svg', dpi=600, bbox_inches='tight')