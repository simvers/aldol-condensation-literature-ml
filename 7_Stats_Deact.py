import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress, spearmanr, kendalltau


# Configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8


# ------------------------------------------------------------------------------------------------------------------

# Import data
df = pd.read_csv("data/data_deactivation.csv")
elements = pd.read_csv('data/elements.csv', header=None).squeeze('columns').to_list()

# Drop columns
df.drop(elements + ['Y_Acryl_Ac', 'Y_Acryl_Fa', 'doi', 'Link_to_excel', # 'Cluster_title', 
        # 'Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming',
        'Ac_source', 'Fa_source', 'Stabilizer', 
        'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'LHSV_mlhg',
        'STY_Acryl_mmolhg', 'Pressure_bar', # 'STY0', 'n',
        'Cluster_n'], axis=1, inplace=True)
print(df.columns)

# Identify numerical and categorical features
numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns.to_list() if col not in ['STY0', 'n']]

# ------------------------------------------------------------------------------------------------------------------
print(numerical_cols)
df = df[df['STY0'] > 0.000001]
df = df[df['n'] < 1]
# exit()

# Scatter plot of sty0 vs n
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for cluster in df['Cluster_title'].unique():

    # Select data
    select_df = df[df['Cluster_title'] == cluster]
    ax.scatter(select_df['STY0'], select_df['n'], label=cluster)
    # sns.regplot(df[df['Cluster_title'] == cluster], x='STY0', y='n', ax=ax, label=cluster)

    x_array = np.linspace(select_df['STY0'].min(), select_df['STY0'].max(), 100)
    # Linear regression in normal space
    # reg = HuberRegressor().fit(select_df[['STY0']], select_df['n'])
    # ax.plot(x_array, reg.coef_*x_array+reg.intercept_)
ax.legend()
sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncols = 4, title=None)
ax.set(xlim=(0, None), ylim=(0, None), xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')
ax.set_xscale('symlog', linthresh=1e-2)
ax.set_yscale('symlog', linthresh=1e-2)
fig.savefig(f'figures/Stats_n/sty0-n_scatterplot_reg.png', dpi=600)
plt.show()


df[['STY0_norm', 'n_norm']] = StandardScaler().fit_transform(df[['STY0', 'n']])

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for cluster in df['Cluster_title'].unique():

    # Select data
    select_df = df.loc[df['Cluster_title'] == cluster, ['STY0_norm', 'n_norm']]
    print(len(select_df))
    # norm_array = StandardScaler().fit_transform(select_df)

    # ax.scatter(select_df['STY0_norm'], select_df['n_norm'], label=cluster)
    sns.regplot(select_df, x='STY0_norm', y='n_norm', ax=ax, label=cluster)
    res = linregress(select_df['STY0_norm'], select_df['n_norm'])
    print(res)

    res = spearmanr(select_df['STY0_norm'], select_df['n_norm'])
    print(res)

    # res = kendalltau(select_df['STY0_norm'], select_df['n_norm'])
    # print(res)

    boot_sr = []
    index = select_df.index.to_list()
    for _ in range(1000):
        idx = np.random.choice(index, len(index), replace=True)
        res = spearmanr(select_df.loc[idx, 'STY0_norm'], select_df.loc[idx, 'n_norm'])
        boot_sr.append(res.statistic)
    print(np.percentile(boot_sr, [2.5, 97.5]))

    # Correlation matrix
    norm_array = select_df[['STY0_norm', 'n_norm']].values
    corr_matrix = np.abs(np.corrcoef(norm_array, rowvar=False))
    print(cluster, corr_matrix)

ax.legend()
sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncols = 4, title=None)
ax.set(xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')
plt.show()

print(df.columns)
plt.scatter(df['Ac_mmolming'], df['STY0'], c='DarkBlue')
plt.scatter(df['Fa_mmolming'], df['STY0'], c='DarkRed')
plt.show()

fig, ax = plt.subplots(1, 2)
sns.scatterplot(df, x='Ac_mmolming', y='STY0', hue='Cluster_title', ax=ax[0])
sns.scatterplot(df, x='Fa_mmolming', y='STY0', hue='Cluster_title', ax=ax[1])
plt.show()

exit()

# ------------------------------------------------------------------------------------------------------------------

# Scatter plot of sty0 vs n
fig, ax = plt.subplots(3, 4, figsize=(10, 10))
ax = np.ravel(ax)
for i, feature in enumerate(numerical_cols + ['STY0']):
    sns.scatterplot(df, x=feature, y='n', hue='Cluster_title', ax=ax[i], palette='plasma')
    # sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncols = 4, title=None)
    ax[i].set(ylim=(0, None), xlabel=feature, ylabel='n /')
    ax[i].set_yscale('symlog', linthresh=1e-2)
    if feature == 'STY0':
        ax[i].set(xlim=(0, None), xscale='symlog')
    if i != 0:
        ax[i].get_legend().remove()
    # if (i != 0) & (i != len(mean_features.columns)):
    #     ax[i].set(ylabel=None)
sns.move_legend(ax[0], loc='center left', frameon=False, bbox_to_anchor=(0, 1.05), title=None, ncols=4)
# fig.savefig(f'figures/deactivation_modelling/{best_model}_sty0-n_scatterplot.png', dpi=600)
plt.show()

# Scatter plot of sty0 vs n
fig, ax = plt.subplots(3, 4, figsize=(10, 10))
ax = np.ravel(ax)
for i, feature in enumerate(numerical_cols):
    sns.scatterplot(df, x='STY0', y='n', hue=feature, ax=ax[i], palette='Reds')
    # sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncols = 4, title=None)
    ax[i].set(xscale='symlog', xlim=(0, None), ylim=(0, None), xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')
    ax[i].set_yscale('symlog', linthresh=1e-2)

    norm = plt.Normalize(df[feature].min(), df[feature].max())
    sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
    ax[i].get_legend().remove()
    ax[i].figure.colorbar(sm, ax=ax[i])
# fig.savefig(f'figures/deactivation_modelling/{best_model}_sty0-n_scatterplot.png', dpi=600)
plt.show()

