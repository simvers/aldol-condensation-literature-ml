import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
mycolor = 'cividis'

# Import clustered data and cluster centroids
data_clustered = pd.read_csv('data/catalysts/data_clustered.csv', na_values=[''], keep_default_na=False)
centroids = pd.read_csv('data/catalysts/centroids.csv', index_col=0)
print(data_clustered.head(5))
centroids.head(5)

# Select data without any nan
# temp_data = data_clustered.loc[~data_clustered['Ac_source'].isna()]

# -----------------------------

# Print unique values of reaction conditions
conditions = ['Ac_source', 'Fa_source', 'Stabilizer', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'O_content', 'LHSV_mlhg', 'Temperature_K', 'Pressure_bar']
for column in conditions:
    print(f'Unique values for {column}', data_clustered[column].unique())

# -----------------------------

# History graph
grouped_data = data_clustered[['doi', 'Year', 'Cluster_title']].groupby(['doi', 'Cluster_title', 'Year'], as_index=False).count()
print(grouped_data)
print(data_clustered[['doi', 'Year', 'Cluster_title']].groupby(['doi', 'Cluster_title', 'Year'], as_index=False)['doi'].count().to_string())
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
sns.histplot(grouped_data, x='Year', hue='Cluster_title', palette=mycolor, ax=ax, discrete=True, multiple='stack', shrink=0.8, edgecolor=None, alpha=1)  # multiple 'dodge'
sns.move_legend(ax, loc='upper left', frameon=False, title='Catalyst clusters', handlelength=1.5)
ax.set(ylabel='Publication count', xlim=(1965, 2026))
fig.tight_layout()
fig.savefig('figures/description/cluster_history.svg', dpi=300, format='svg')

# -----------------------------

fig = plt.figure(figsize=(5, 7.5))
gs = fig.add_gridspec(3, 2)

# Normal subplots
ax = []
ax.append(fig.add_subplot(gs[0, 0]))
ax.append(fig.add_subplot(gs[0, 1]))
ax.append(fig.add_subplot(gs[1, 0]))
ax.append(fig.add_subplot(gs[1, 1]))

# Describe Ac_source
temp_data = data_clustered.groupby(['Ac_source', 'Cluster_title'])['Ac_source'].count().unstack('Cluster_title').fillna(0)/len(data_clustered)*100
temp_data.plot.bar(ax=ax[0], stacked=True, colormap=mycolor, width=0.1*len(temp_data))
ax[0].set(xlabel='Ac source', ylabel='Reported catalysts / %')
ax[0].legend(frameon=False)

# Describe Fa_source
temp_data = data_clustered.groupby(['Fa_source', 'Cluster_title'])['Fa_source'].count().unstack('Cluster_title').fillna(0)/len(data_clustered)*100
temp_data.plot.bar(ax=ax[1], stacked=True, colormap=mycolor, width=0.1*len(temp_data))
ax[1].set(xlabel='Fa source', ylabel='Reported catalysts / %')
ax[1].get_legend().remove()

# Describe Stab_source
temp_data = data_clustered.groupby(['Stabilizer', 'Cluster_title'])['Stabilizer'].count().unstack('Cluster_title').fillna(0)/len(data_clustered)*100
temp_data.plot.bar(ax=ax[2], stacked=True, colormap=mycolor, width=0.1*len(temp_data))
ax[2].set(xlabel='Additives', ylabel='Reported catalysts / %')
ax[2].get_legend().remove()

# Describe O_content
data_clustered['O_presence'] = data_clustered['O_content'] != 0
temp_data = data_clustered.groupby(['O_presence', 'Cluster_title'])['O_presence'].count().unstack('Cluster_title').fillna(0)/len(data_clustered)*100
temp_data.plot.bar(ax=ax[3], stacked=True, colormap=mycolor, width=0.1*len(temp_data))
ax[3].set(xlabel='Oxygen presence', ylabel='Reported catalysts / %')
ax[3].get_legend().remove()

# Subsubplots
ax.append(gs[2, 0].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
ax.append(gs[2, 1].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))

# Describe LHSV - temperature
clusters = np.sort(data_clustered['Cluster_title'].unique())
cmap = plt.get_cmap(mycolor)
palette = cmap(np.linspace(0, 1, len(clusters)))
ax_joint = fig.add_subplot(ax[4][1, 0])
ax_x = fig.add_subplot(ax[4][0, 0], sharex=ax_joint)
ax_y = fig.add_subplot(ax[4][1, 1], sharey=ax_joint)
# sns.kdeplot(ax=ax_joint, data=data_clustered, x='LHSV_mlhg', y='Temperature_K', hue='Cluster_title', hue_order=clusters, fill=False, alpha=.5, palette=palette)
sns.scatterplot(ax=ax_joint, data=data_clustered, x='LHSV_mlhg', y='Temperature_K', hue='Cluster_title', hue_order=clusters, palette=palette)
# # ax_joint.minorticks_on()
ax_joint.set(xlabel='LHSV / ml h$^{-1}$ g$^{-1}$', ylabel='T / K', xlim=(0, 8), ylim=(550, 700), xticks=[0, 2, 4, 6, 8], yticks=[550, 600, 650, 700])
ax_joint.get_legend().remove()
sns.kdeplot(ax=ax_x, data=data_clustered, x='LHSV_mlhg', hue='Cluster_title', hue_order=clusters, palette=palette)
sns.kdeplot(ax=ax_y, data=data_clustered, y='Temperature_K', hue='Cluster_title', hue_order=clusters, palette=palette)
for item in [ax_x, ax_y]:
    item.set_axis_off()
    item.get_legend().remove()

# Describe ratios
ax_joint = fig.add_subplot(ax[5][1, 0])
ax_x = fig.add_subplot(ax[5][0, 0], sharex=ax_joint)
ax_y = fig.add_subplot(ax[5][1, 1], sharey=ax_joint)
# sns.kdeplot(ax=ax_joint, data=data_clustered, x='Ratio_Ac_Fa', y='Ratio_Stab_Fa', hue='Cluster_title', hue_order=clusters, fill=False, alpha=.5, palette=palette)
sns.scatterplot(ax=ax_joint, data=data_clustered, x='Ratio_Ac_Fa', y='Ratio_Stab_Fa', hue='Cluster_title', hue_order=clusters, palette=palette)
# # ax_joint.minorticks_on()
ax_joint.set_xscale('log')
ax_joint.set_yscale('symlog', linthresh=0.5, linscale=0.25)
ax_joint.xaxis.set_major_formatter(mticker.ScalarFormatter())
ax_joint.yaxis.set_major_formatter(mticker.ScalarFormatter())
ax_joint.set(xlabel='Ac/Fa', ylabel='Stab/Fa', xlim=(0.1, 10), ylim=(0, 5), xticks=[0.1, 0.5, 1, 2, 10], yticks=[0, 0.5, 1, 2, 5])
ax_joint.get_legend().remove()
sns.kdeplot(ax=ax_x, data=data_clustered, x='Ratio_Ac_Fa', hue='Cluster_title', hue_order=clusters, palette=palette)
sns.kdeplot(ax=ax_y, data=data_clustered, y='Ratio_Stab_Fa', hue='Cluster_title', hue_order=clusters, palette=palette)
for item in [ax_x, ax_y]:
    item.set_axis_off()
    item.get_legend().remove()

fig.tight_layout()
plt.savefig('figures/description/Fig_multi_kde.svg', dpi=300, format='svg')
