import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress, spearmanr, kendalltau
import string
import json
from functions.stats_deact import partial_corr_xcovar, partial_corr_1covar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D



# Configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8


# ------------------------------------------------------------------------------------------------------------------

# Import data
df = pd.read_csv("data/catalysts/data_deactivation_noSi.csv")
elements = pd.read_csv('data/catalysts/elements.csv', header=None).squeeze('columns').to_list()
with open("data/features.json") as f:
    feature_labels = json.load(f)

# Drop columns
df.drop(elements + ['Y_Acryl_Ac', 'Y_Acryl_Fa', 'doi', 'Link_to_excel', 'Year', # 'Cluster_title', 
        'Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming',
        'Ac_source', 'Fa_source', 'Stabilizer', 
        # 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'LHSV_mlhg',
        'STY_Acryl_mmolhg', 'Pressure_bar', # 'STY0', 'n',
        'av_cov_rad', 'av_n_val', 'var_cov_rad', 'var_n_val', 
        # 'av_lat_cst', 'var_lat_cst', 
        'var_pca',
        'Cluster_n'], axis=1, inplace=True)
print(df.columns)

# Make colors
mycolor = 'cividis'
clusters = np.sort(df['Cluster_title'].unique())
cmap = plt.get_cmap(mycolor)
palette = cmap(np.linspace(0, 1, len(clusters)))

# Identify numerical and categorical features
numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns.to_list() if col not in ['STY0', 'n']]
print(numerical_cols)

# Exclude exp with no activity
df = df[df['STY0'] > 0.000001]
# Exclude exp under oxygen
data_no_oxygen = df.loc[df['O_content'] < 0.00001, :]
print(df.groupby('Cluster_title').count())
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
for i, cluster in enumerate(clusters):
    sns.regplot(ax=ax_joint, data=data_no_oxygen[data_no_oxygen['Cluster_title'] == cluster], x='STY0', y='n', label=cluster, color=palette[i])
ax_joint.legend()
sns.move_legend(ax_joint, loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncols = 4, title=None)
ax_joint.set(xlim=(0.1, 40), ylim=(0, 0.3), xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')
sns.kdeplot(ax=ax_x, data=data_no_oxygen, x='STY0', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
sns.kdeplot(ax=ax_y, data=data_no_oxygen, y='n', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
for item in [ax_x, ax_y]:
    item.set_axis_off()
    item.get_legend().remove()
fig.savefig(f'figures/Stats_n/sty0-n_scatterplot_reg.png', dpi=600)

# ------------------------------------------------------------------------------------------------------------------

# Normalization
# data_no_oxygen[['STY0_norm', 'n_norm']] = StandardScaler().fit_transform(data_no_oxygen[['STY0', 'n']])
# data_no_oxygen['n_norm'] = data_no_oxygen['n']
# data_no_oxygen['STY0_norm'] = np.log1p(data_no_oxygen['STY0'])
log1p_list = ['LHSV_mlhg', 'STY0']
# df[[col + '_norm' for col in log1p_list]] = np.log1p(df[log1p_list])
# df = df.drop(labels=log1p_list, axis=1)
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
    sns.regplot(ax=ax_joint, data=data_no_oxygen[data_no_oxygen['Cluster_title'] == cluster], x='STY0', y='n', label=cluster, color=palette[i], robust=True)
ax_joint.legend()
sns.move_legend(ax_joint, loc='upper left', bbox_to_anchor=(0.0, 1.0), frameon=False, ncols = 1, title=None)
ax_joint.set(xlabel='ln( 1 + STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$ )', ylabel='n /', xlim=(0, 4), ylim=(0, 0.3))
sns.kdeplot(ax=ax_x, data=data_no_oxygen, x='STY0', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
sns.kdeplot(ax=ax_y, data=data_no_oxygen, y='n', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
for item in [ax_x, ax_y]:
    item.set_axis_off()
    item.get_legend().remove()
fig.savefig(f'figures/Stats_n/sty0-n_scatterplot_norm_reg_no_oxygen.png', dpi=600)
print(df.columns)

# ------------------------------------------------------------------------------------------------------------------

# Scatter plot of sty0 vs n
fig, ax = plt.subplots(4, 2, figsize=(10, 20))
ax = np.ravel(ax)
for i, feature in enumerate(numerical_cols):
    ax[i].set_box_aspect(1)
    sns.scatterplot(df, x='STY0', y='n', hue=feature, ax=ax[i], palette='Reds')
    # sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncols = 4, title=None)
    # ax[i].set(xscale='symlog', xlim=(0, None), ylim=(0, None), xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')
    ax[i].set(xlim=(0, None), ylim=(0, None), xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')
    ax[i].set_yscale('symlog', linthresh=1e-2)

    norm = plt.Normalize(df[feature].min(), df[feature].max())
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    ax[i].get_legend().remove()
    ax[i].figure.colorbar(sm, ax=ax[i])
fig.savefig(f'figures/Stats_n/sty0-n-feature_scatterplot.png', dpi=600, bbox_inches='tight')

# ------------------------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(6, 12))
# outer_gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.4)
ax = []
letters = list(string.ascii_lowercase)
features = numerical_cols + ['STY0']

# iterate ove columns
for i, feature in enumerate(features):
    row, col = divmod(i, 2)
    # row, col = i//gs.ncols, i%gs.ncols

    # Add inner grid
    ax.append(gs[row, col].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
    # inner_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[row, col],
    #     width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.0, hspace=0.0,)

    # Create subplot
    ax_joint = fig.add_subplot(ax[i][1, 0])
    # ax_joint.set_box_aspect(1)
    ax_x = fig.add_subplot(ax[i][0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(ax[i][1, 1], sharey=ax_joint)

    # Share axes AFTER creation (avoids the gap from box_aspect)
    ax_x.sharex(ax_joint)
    ax_y.sharey(ax_joint)
    
    # Include or exclude oxygen
    selected_df = data_no_oxygen if feature != 'O_content' else df

    for j, cluster in enumerate(clusters):
        sns.regplot(ax=ax_joint, data=selected_df[selected_df['Cluster_title'] == cluster], x=feature, y='n', label=cluster, color=palette[j], robust=True, scatter_kws=dict(s=15, alpha=0.6),)
        
    prefix, sufix = ('ln( 1 + ', ' )') if feature in log1p_list else ('', '')
    xlabel = prefix + feature_labels.get(feature, feature) + sufix
    ax_joint.set(ylim=(0, 0.2), xlabel=xlabel, ylabel='n /')
    # ax_joint.set_yscale('symlog', linthresh=1e-2)
    # if feature == 'O_content':
    #     ax_joint.set(xlim=(0., None), xscale='symlog')
    if i == 0:
        ax_joint.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.95, 0.95), title=None, ncols=1)
    # if (i != 0) & (i != len(mean_features.columns)):
    #     ax[i].set(ylabel=None)
    sns.kdeplot(ax=ax_x, data=selected_df, x=feature, hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
    sns.kdeplot(ax=ax_y, data=selected_df, y='n', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()
    ax_joint.text(-0.2/5*6, 1.0/5*6, letters[i], fontsize=10, fontweight='bold', fontfamily='arial', transform=ax_joint.transAxes)
fig.savefig(f'figures/Stats_n/feature-n_kde-scatterplot.png', dpi=600)

# ax_joint.set_xscale('log')
# ax_joint.set_yscale('symlog', linthresh=0.5, linscale=0.25)
# ax_joint.xaxis.set_major_formatter(mticker.ScalarFormatter())
# ax_joint.yaxis.set_major_formatter(mticker.ScalarFormatter())


# ──────────────────────────────────────────────────────────────────────

# Calculate correlation coefficients
spearmanr_ = np.empty((len(features), len(clusters)))
spearmanr_covar = np.empty((len(features), len(clusters)))
pval_ = np.empty((len(features), len(clusters)))
pval_covar = np.empty((len(features), len(clusters)))

for i, feature in enumerate(features):
    for j, cluster in enumerate(clusters):
        
        # Spearman r: non-parametric monotonic relationship
        rho, pval = spearmanr(
            a=df.loc[df['Cluster_title'] == cluster, feature].values,
            b=df.loc[df['Cluster_title'] == cluster, 'n'].values
        )
        # print(f'Feature {feature} for cluster {cluster}: rho {rho}, pval {pval} - no covar')
        spearmanr_[i, j] = rho
        pval_[i, j] = pval

        # Spearman r: non-parametric monotonic relationship
        # Without STY covar
        rho, pval = partial_corr_xcovar(
            x=df.loc[df['Cluster_title'] == cluster, feature].values,
            y=df.loc[df['Cluster_title'] == cluster, 'n'].values,
            covars=[df.loc[df['Cluster_title'] == cluster, 'STY0'].values,
                    df.loc[df['Cluster_title'] == cluster, 'O_content'].values]
        )
        # print(f'Feature {feature} for cluster {cluster}: rho {rho}, pval {pval} - covar')
        spearmanr_covar[i, j] = rho
        pval_covar[i, j] = pval

        # Pearson: linear relationship
        # res = linregress(select_df['STY0_norm'], select_df['n_norm'])
        # print(res)

        # Kendall tau
        # res = kendalltau(select_df['STY0_norm'], select_df['n_norm'])
        # print(res)

        # Correlation matrix
        # norm_array = select_df[['STY0_norm', 'n_norm']].values
        # corr_matrix = np.abs(np.corrcoef(norm_array, rowvar=False))
        # print(cluster, corr_matrix)

# ────────────────────────────────────────────────────────────────────

# Figure
fig, (ax, cax) = plt.subplots(
    1, 2,
    figsize=(7, 4),
    gridspec_kw={'width_ratios': [20, 1]},
    constrained_layout=True
)

# y axis
n = len(features)
y = np.arange(n)

# Colors
norm = mcolors.LogNorm(vmin=0.01, vmax=0.2)
# print('No covar: ', spearmanr_, pval_)
# print('Covar: ', spearmanr_covar, pval_covar)

for i, cluster in enumerate(clusters):
    # Colors
    # alpha: continuous from 0 (p=0, fully opaque) to 1 (p=1, fully transparent)
    # clipped 1 - alpha to keep dots visible

    # Spearman r
    alphas = np.clip(norm(np.nan_to_num(pval_[:, i], nan=1.0)), 0.0, 0.8)
    rgbas = [mcolors.to_rgba(palette[i], alpha=1-a) for a in alphas]  # invert: low p → opaque
    ax.scatter(spearmanr_[:, i], y, color=rgbas, s=80, zorder=3, marker='o', label=cluster)

    # Spearman r covar
    alphas = np.clip(norm(np.nan_to_num(pval_covar[:, i], nan=1.0)), 0.0, 0.8)
    rgbas = [mcolors.to_rgba(palette[i], alpha=1-a) for a in alphas]  # invert: low p → opaque
    ax.scatter(spearmanr_covar[:, i], y, color=rgbas, s=80, zorder=3, marker='^')

# formatting
ax.set_yticks(y)
ax.set_yticklabels([feature_labels.get(feature, '').split(' /')[0] for feature in features])
# ax.set_yticklabels(['\nmolar'.join(feature_labels.get(feature, '').split(' /')[0].split(' molar')) for feature in features])
ax.set_xlim(-1.0, 1.0)
ax.set_xlabel('Spearman ρ')
# ax.set_title(clusters[cluster_idx], fontsize=11, fontweight='bold', pad=8)
ax.grid(axis='x', color='#D3D1C7', linewidth=0.5, zorder=1)
# ax.spines[['top', 'right']].set_visible(False)

# zero reference line
ax.axvline(0, color='k', linewidth=2.0, linestyle='--', zorder=2)

# ─────────────────────────────────────────────────────────

# Colorbar

# grey colormap: opaque (p=0) → transparent (p=1), shown as dark→light grey
grey_cmap = mcolors.LinearSegmentedColormap.from_list(
    'pval_grey', ['#2C2C2A', '#F1EFE8']  # dark=significant, light=not
)
cb = ColorbarBase(cax, cmap=grey_cmap, orientation='vertical', norm=mcolors.LogNorm(vmin=0.01, vmax=0.2))
cb.ax.set_ylabel('p-value', rotation=270, verticalalignment='baseline')
cb.set_ticks([0.01, 0.05, 0.1, 0.2])
cb.set_ticklabels([0.01, 0.05, 0.10, 0.20])
cb.ax.tick_params(labelsize=8) 

# ─────────────────────────────────────────────────────────

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=8, label='Cluster'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[0], markersize=8, label=clusters[0]),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[1], markersize=8, label=clusters[1]),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[2], markersize=8, label=clusters[2]),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=8, label='Metric'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8, label='Univariate'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='k', markersize=8, label='Covariance-controlled')
]
ax.legend(handles=legend_elements, frameon=False, loc='upper left', facecolor='w')

fig.savefig(f'figures/Stats_n/spearman_dotplot.png', dpi=600)

