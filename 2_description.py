import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions import calculate_molarflowrates


# Import clustered data and cluster centroids
data_clustered = pd.read_excel('data/data_clustered.xlsx')
centroids = pd.read_csv('data/centroids.csv', index_col=0)
print(data_clustered.head(5))
centroids.head(5)

# Calculate molar flowrates
data_clustered = calculate_molarflowrates(data_clustered)

# Overwrite stabilizer when formalin is used as Fa source
data_clustered.loc[data_clustered['Fa source'] == 'FORM', 'Stabilizer'] = 'MeOH \n + H$_2$O'
data_clustered = data_clustered.fillna({'Stabilizer': 'None'})

# Select data without any nan
# temp_data = data_clustered.loc[~data_clustered['Ac source'].isna()]

# -----------------------------

# Describe reactants

# Set bar width and scale by number of containers for each graph to have consistent width
bar_width = 0.1

# Describe Ac source
temp_data = data_clustered.groupby(['Ac source', 'Cluster title'])['Ac source'].count().unstack('Cluster title').fillna(0)/len(data_clustered)*100
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
ax = temp_data.plot.bar(ax=ax, stacked=True, colormap='plasma', width=0.1*len(temp_data))
ax.set_ylabel('Reported catalysts / %')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/Fig_Ac-source.png', dpi=300)

# Describe Fa source
temp_data = data_clustered.groupby(['Fa source', 'Cluster title'])['Fa source'].count().unstack('Cluster title').fillna(0)/len(data_clustered)*100
print(temp_data)
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
ax = temp_data.plot.bar(ax=ax, stacked=True, colormap='plasma', width=0.1*len(temp_data))
ax.set_ylabel('Reported catalysts / %')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/Fig_Fa-source.png', dpi=300)

# Describe Stab source
temp_data = data_clustered.groupby(['Stabilizer', 'Cluster title'])['Stabilizer'].count().unstack('Cluster title').fillna(0)/len(data_clustered)*100
print(temp_data)
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
ax = temp_data.plot.bar(ax=ax, stacked=True, colormap='plasma', width=0.1*len(temp_data))
ax.set_xlabel('Additives')
ax.set_ylabel('Reported catalysts / %')
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/Fig_Additives.png', dpi=300)

# -----------------------------

# Describe conditions

# Describe temperature
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
n = len(data_clustered['Cluster title'].unique())
cmap = plt.get_cmap('plasma')
for i, (category, group) in enumerate(data_clustered.groupby(['Cluster title'])):
    ax.scatter(group['LHSV [ml/h/g]'], group['Temperature [K]'], color=cmap(i/(n-1)), label=category[0])
# ax = sns.scatterplot(data_clustered, x='LHSV [ml/h/g]', y='Temperature [K]', hue='Cluster title', palette='viridis', ax=ax)
ax.set_xlabel('LHSV [ml/h/g]')
ax.set_ylabel('T [K]')
#ax.minorticks_on()
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([550, 600, 650, 700])
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/Fig_LHSV-T.png', dpi=300)


# -----------------------------

# Describe performance
exit()
fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
n = len(data_clustered['Cluster title'].unique())
cmap = plt.get_cmap('plasma')
for i, (category, group) in enumerate(data_clustered.groupby(['Cluster title'])):
    ax.scatter(group[''], group['Temperature [K]'], color=cmap(i/(n-1)), label=category[0])
# ax = sns.scatterplot(data_clustered, x='LHSV [ml/h/g]', y='Temperature [K]', hue='Cluster title', palette='viridis', ax=ax)
ax.set_xlabel('LHSV [ml/h/g]')
ax.set_ylabel('T [K]')
#ax.minorticks_on()
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_yticks([550, 600, 650, 700])
ax.legend(frameon=False)
fig.tight_layout()
plt.savefig('figures/Fig_LHSV-T.png', dpi=300)



