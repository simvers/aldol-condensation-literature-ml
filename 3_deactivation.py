from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions.utils import filter_type, process_doi
from functions.functions_deactivation import merge_conditions_deactivation, extract_df_deactivation_from_excel, deactivation_modelling
from functions.deactivation_models import *

# ------------------------------------------------------------------------------------------------

# Import digitized data

# Directory with digitized data in xlsx files
path = "/mnt/c/Users/u0156112/OneDrive - KU Leuven/Shared_AC2GEN/Review/DigitizedData/"
# path = 'C:\\Users\\u0156112\\OneDrive - KU Leuven\\Shared_AC2GEN\\Review\\DigitizedData\\'

# Import all excel files as directory, and expand the directory to a list of df
doi_digitized, sheet_digitized, df_digitized = extract_df_deactivation_from_excel(path)
print('Number of deactivation files: ', len(df_digitized))

# ------------------------------------------------------------------------------------------------

# Import clustered data
data_clustered = pd.read_csv('data/data_clustered.csv')
data_clustered['doi'] = process_doi(data_clustered['doi'])
initial_columns = data_clustered.columns.to_list()

# ------------------------------------------------------------------------------------------------

# Merge data_clustered with list of df_deactivation
# Creating time-dependent performance columns, containing numpy arrays
# Drop observations without STY deactivation data
data_deactivation = merge_conditions_deactivation(data_clustered, doi_digitized, sheet_digitized, df_digitized).dropna(subset=['STY_Acryl_mmolhg_t']).copy()
print('Number of rows with deactivation data: ', len(data_deactivation))

# Plot all deactivation curves
fig, ax = plt.subplots(figsize=(6, 4))
for i, row in data_deactivation.iterrows():
    # Extract time and STY as numpy arrays
    time = row['Time_h_t']  
    sty = row['STY_Acryl_mmolhg_t']

    # PLot sty deactivation
    ax.plot(time, sty, marker='o')
ax.set(xlim=(0, 20), ylim=(0, None), xlabel='Time-on-stream / h', ylabel='STY$_{Acryl}$ / mmol h$^{-1}$ g$^{-1}$')
fig.savefig('figures/deactivation_modelling/sty-tos_plot.png', dpi=600)

# ------------------------------------------------------------------------------------------------

# Different models
n_pow_exp, n_lan = 1, 2
models = {
    'pow3': {'func': power_law_model_3, 'p0': [0, 0, 0], 'bounds': ([0, 0, 0], [100, n_pow_exp, 50]), 
             'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))},
    'pow2': {'func': power_law_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n_pow_exp]), 
             'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
    'exp3': {'func': exp_model_3, 'p0': [0, 0, 0], 'bounds': ([0, 0, 0], [100, n_pow_exp, 50]), 
             'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))}, 
    'exp2': {'func': exp_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n_pow_exp]), 
             'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
    'lan3': {'func': langmuir_model_3, 'p0': [0, 0, 0], 'bounds': ([0, 0, 0], [100, n_lan, 50]), 
             'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))},
    'lan2': {'func': langmuir_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n_lan]), 
             'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))}
}

# Fit all models
data_deactivation, models = deactivation_modelling(data_deactivation, models)

# ------------------------------------------------------------------------------------------------

# Save data
best_model = 'lan2'
data_deactivation.rename(columns={best_model + '_STY0': 'STY0', best_model + '_n': 'n'}, inplace=True)
data_deactivation.loc[:, initial_columns + ['STY0', 'n']].to_csv('data/data_deactivation.csv', index=False)

# Scatter plot of sty0 vs n
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.scatterplot(data_deactivation, x='STY0', y='n', hue='Cluster_title', ax=ax)
sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncols = 4, title=None)
ax.set(xscale='symlog', xlim=(0, None), ylim=(0, None), xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')
ax.set_yscale('symlog', linthresh=1e-2)
fig.savefig(f'figures/deactivation_modelling/{best_model}_sty0-n_scatterplot.png', dpi=600)
