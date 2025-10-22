import pandas as pd
import matplotlib.pyplot as plt


# Import clustered data and cluster centroids
data_clustered = pd.read_csv('data/data_clustered.csv', na_values=[''], keep_default_na=False)
centroids = pd.read_csv('data/centroids.csv', index_col=0)
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

# Describe reactants

# Set bar width and scale by number of containers for each graph to have consistent width
bar_width = 0.1

# Groupby cluster_title and descriptor
# Select any index of the groupby, and get the count of the descriptor
# Unstack Cluster_title as columns
# NaN value if no catalyst with this descriptor, therefore fill na with 0

# Describe Ac_source
temp_data = data_clustered.groupby(['Ac_source', 'Cluster_title'])['Ac_source'].count().unstack('Cluster_title').fillna(0)/len(data_clustered)*100
print(temp_data)
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
ax = temp_data.plot.bar(ax=ax, stacked=True, colormap='plasma', width=0.1*len(temp_data))
ax.set_ylabel('Reported catalysts / %')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/description/Fig_Ac-source.png', dpi=300)

# Describe Fa_source
temp_data = data_clustered.groupby(['Fa_source', 'Cluster_title'])['Fa_source'].count().unstack('Cluster_title').fillna(0)/len(data_clustered)*100
print(temp_data)
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
ax = temp_data.plot.bar(ax=ax, stacked=True, colormap='plasma', width=0.1*len(temp_data))
ax.set_ylabel('Reported catalysts / %')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/description/Fig_Fa-source.png', dpi=300)

# Describe Stab_source
temp_data = data_clustered.groupby(['Stabilizer', 'Cluster_title'])['Stabilizer'].count().unstack('Cluster_title').fillna(0)/len(data_clustered)*100
print(temp_data)
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
ax = temp_data.plot.bar(ax=ax, stacked=True, colormap='plasma', width=0.1*len(temp_data))
ax.set_xlabel('Additives')
ax.set_ylabel('Reported catalysts / %')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/description/Fig_Additives.png', dpi=300)

# Describe O_content
data_clustered['O_presence'] = data_clustered['O_content'] != 0
temp_data = data_clustered.groupby(['O_presence', 'Cluster_title'])['O_presence'].count().unstack('Cluster_title').fillna(0)/len(data_clustered)*100
print(temp_data)
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
ax = temp_data.plot.bar(ax=ax, stacked=True, colormap='plasma', width=0.1*len(temp_data))
ax.set_xlabel('Oxygen presence')
ax.set_ylabel('Reported catalysts / %')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/description/Fig_O_content.png', dpi=300)

# -----------------------------

# Describe conditions

# Describe LHSV - temperature
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
n = len(data_clustered['Cluster_title'].unique())
cmap = plt.get_cmap('plasma')
for i, (category, group) in enumerate(data_clustered.groupby(['Cluster_title'])):
    ax.scatter(group['LHSV_mlhg'], group['Temperature_K'], color=cmap(i/(n-1)), label=category[0])
# ax = sns.scatterplot(data_clustered, x='LHSV [ml/h/g]', y='Temperature [K]', hue='Cluster_title', palette='viridis', ax=ax)
ax.set_xlabel('LHSV [ml/h/g]')
ax.set_ylabel('T [K]')
#ax.minorticks_on()
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([550, 600, 650, 700])
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/description/Fig_LHSV-T.png', dpi=300)

# Describe ratios
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
n = len(data_clustered['Cluster_title'].unique())
cmap = plt.get_cmap('plasma')
for i, (category, group) in enumerate(data_clustered.groupby(['Cluster_title'])):
    ax.scatter(group['Ratio_Ac_Fa'], group['Ratio_Stab_Fa'], color=cmap(i/(n-1)), label=category[0])
ax.set_xlabel('Ac/Fa')
ax.set_ylabel('Stab/Fa')
#ax.minorticks_on()
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/description/Fig_Ratios.png', dpi=300)

