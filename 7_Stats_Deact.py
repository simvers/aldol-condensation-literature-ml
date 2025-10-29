import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8


# ------------------------------------------------------------------------------------------------------------------

# Import data
df = pd.read_csv("data/data_engineered.csv")
elements = pd.read_csv('data/elements.csv', header=None).squeeze('columns').to_list()

# Drop columns
df.drop(elements + ['Y_Acryl_Ac', 'Y_Acryl_Fa', 'doi', 'Link_to_excel', 'Cluster_title', 
         'Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming', 
         'STY_Acryl_mmolhg', 'Pressure_bar',
         'Cluster_n'], axis=1, inplace=True)

# Identify numerical and categorical features
numerical_cols = [col for col in df.select_dtypes(include=['int64', 'float64']).columns.to_list() if col not in ['STY0', 'n']]

# ------------------------------------------------------------------------------------------------------------------
print(numerical_cols)

# Scatter plot of sty0 vs n
fig, ax = plt.subplots(4, 4, figsize=(10, 10))
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

