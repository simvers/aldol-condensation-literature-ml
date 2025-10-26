from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions.utils import filter_type, process_doi
from functions.functions_deactivation import merge_conditions_deactivation, extract_df_deactivation_from_excel
from functions.deactivation_models import *

# ------------------------------------------------------------------------------------------------

# Import digitized data

# Directory with digitized data in xlsx files
path = 'C:\\Users\\u0156112\\OneDrive - KU Leuven\\Shared_AC2GEN\\Review\\DigitizedData\\'
# path = "/mnt/c/Users/u0156112/OneDrive - KU Leuven/Shared_AC2GEN/Review/DigitizedData/"

# Import all excel files as directory, and expand the directory to a list of df
doi_digitized, sheet_digitized, df_digitized = extract_df_deactivation_from_excel(path)

print('Number of deactivation files: ', len(df_digitized))

# ------------------------------------------------------------------------------------------------

# Import clustered data
data_clustered = pd.read_csv('data/data_clustered.csv')
data_clustered['doi'] = process_doi(data_clustered['doi'])
initial_columns = data_clustered.columns.to_list()

# ------------------------------------------------------------------------------------------------

# Identify publications that still need to be processed

# print(set(doi_digitized).issubset(doi))
doi_temp = []
for doi_i in set(doi_digitized):
    if doi_i not in data_clustered['doi'].unique():
        doi_temp.append(doi_i)
print('DOI to process in Mendeley: ', len(doi_temp), doi_temp)

doi_temp = []
for doi_i in data_clustered['doi'].unique():
    if doi_i not in doi_digitized:
        doi_temp.append(doi_i)
print('DOI to process digitally', len(doi_temp), doi_temp)

# ------------------------------------------------------------------------------------------------

# Merge data_clustered with list of df_deactivation
# Creating time-dependent performance columns, containing numpy arrays
data_clustered = merge_conditions_deactivation(data_clustered, doi_digitized, sheet_digitized, df_digitized)

# ------------------------------------------------------------------------------------------------

# Statistics on deactivation data

# Select rows with STY deactivation data
data_deactivation = data_clustered[~data_clustered['STY_Acryl_mmolhg_t'].isna()].copy()
data_deactivation.to_csv('data/data_deactivation.csv', index=False)
print('Number of rows with deactivation data: ', len(data_deactivation))

# Basic statistics
basic_statistics = False
if basic_statistics:

    # Statistics on sty and deactivation
    max_values = []

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, row in data_deactivation.iterrows():
        # Extract time and STY as numpy arrays
        time = row['Time_h_t']  
        sty = row['STY_Acryl_mmolhg_t']

        # Extract statistics
        max_values.append(sty.max())

        # PLot sty deactivation
        ax.plot(time, sty, marker='o')
    plt.show()

    # Bar plot of max STY values
    plt.figure(figsize=(6, 4))
    sns.histplot(max_values, bins=30)
    plt.show()

    # Bar plot of max STY values
    plt.figure(figsize=(6, 4))
    sns.kdeplot(max_values, cut=0)
    plt.show()

# ------------------------------------------------------------------------------------------------

# Different models
n, coef = 1, 10
models = {
    'pow3': {'func': power_law_model_3, 'p0': [0, 0, 0], 'bounds': ([0, 0, 0], [100, n, 50]), 
             'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))},
    'pow2': {'func': power_law_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n]), 
             'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
    'exp3': {'func': exp_model_3, 'p0': [0, 0, 0], 'bounds': ([0, 0, 0], [100, n, 50]), 
             'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))}, 
    'exp2': {'func': exp_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n]), 
             'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
    'lan3': {'func': langmuir_model_3, 'p0': [0, 0, 0], 'bounds': ([0, 0, 0], [100, coef, 50]), 
             'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))},
    'lan2': {'func': langmuir_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, coef]), 
             'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))}
}

for name, spec in models.items():

    # Initialize columns
    # data_deactivation.loc[:, [name + '_r2', name + '_nrmse']] = np.nan
    data_deactivation.loc[:, name + '_nrmse'] = np.nan
    # data_deactivation.loc[:, [name + '_param', name + '_perr', name + '_cv', name + '_pcorr']] = pd.NA
    data_deactivation.loc[:, [name + '_param', name + '_STY0', name + '_n']] = pd.NA

    # Fit deactivation models
    for i, row in data_deactivation.iterrows():

        # Initialize fitting variables
        t = row['Time_h_t']
        sty = row['STY_Acryl_mmolhg_t']

        # Fit model
        popt, perr, cv, pcorr, r2, nrmse = fit_model(t, sty, model=spec['func'], p0=spec['p0'], bounds=spec['bounds'])

        # Add fitted model to deactivation data
        data_deactivation.at[i, f'{name}_param'] = popt
        data_deactivation.loc[i, [f'{name}_STY0', f'{name}_n']] = popt[0:2]
        # data_deactivation.at[i, f'{name}_perr'] = perr
        # data_deactivation.at[i, f'{name}_cv'] = cv
        # data_deactivation.at[i, f'{name}_pcorr'] = pcorr
        # data_deactivation.at[i, f'{name}_r2'] = r2
        data_deactivation.at[i, f'{name}_nrmse'] = nrmse

        # Add fitted model to model data
        models[name]['param'] = np.vstack((models[name]['param'], popt))
        models[name]['perr'] = np.vstack((models[name]['perr'], perr))
        models[name]['cv'] = np.vstack((models[name]['cv'], cv))
        models[name]['pcorr'] = np.vstack((models[name]['pcorr'], np.expand_dims(pcorr, axis=0)))
    
    # Print statistics
    print(f'Model {name}:\nNRMSE: {data_deactivation.loc[:, name+"_nrmse"].mean()}\nMax n: {models[name]["param"][:, 1].max()}\nAv n: {models[name]["param"][:, 1].mean()}')

# ------------------------------------------------------------------------------------------------

# Plot
# colors = ["#448AFF", "#1565C0", "#009688", "#8BC34A", "#FFC107", "#FF9800", "#F44336", "#AD1457"]
colors = ["#009688", "#009688", "#1565C0", "#1565C0", "#AD1457", "#AD1457", "#FF9800"]
alpha = [0.5, 1, 0.5, 1, 0.5, 1]

plot = True
if plot:

    # KDE plot of NRMSE values
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, name in enumerate(models.keys()):
        sns.kdeplot(ax=ax, data=data_deactivation[f'{name}_nrmse'], color=colors[i], alpha=alpha[i], label=name, cut=0)
    ax.legend(frameon=False)
    ax.set(xlabel='NRMSE', xlim=(0, None))
    fig.savefig('figures/deactivation_modelling/NRMSE_kdeplot.png', dpi=600)

    # NRMSE violinplot 
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.violinplot(ax=ax, data=[data_deactivation[f'{name}_nrmse'] for name in models.keys()], palette=colors, cut=0)
    for i, patch in enumerate(ax.collections):
        patch.set_alpha(alpha[i])
    ax.set(xlabel='Deactivation models', ylabel='NRMSE', xticklabels=models.keys())
    fig.savefig('figures/deactivation_modelling/NRMSE_violinplot.png', dpi=600)

    # KDE plot of param values
    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    for i, name in enumerate(models.keys()):
        for j in range(models.get(name).get('param').shape[1]):
            if j == 1:
                print(name, ' n > 2.5', sum(models.get(name).get('param')[:, j] > 1))
            sns.kdeplot(ax=ax[j], x=models.get(name).get('param')[:, j], color=colors[i], alpha=alpha[i], label=name, cut=0)
    ax[0].legend(frameon=False)
    ax[0].set(xlabel='STY$_{\mathdefault{0}}$')
    ax[1].set(xlabel='n')
    ax[2].set(xlabel='STY$\mathdefault{_{inf}}$')
    fig.savefig('figures/deactivation_modelling/Param_kdeplot.png', dpi=600)

    # Error plot of param values
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i, name in enumerate(models.keys()):
        for j in range(models.get(name).get('param').shape[1]):
            sns.scatterplot(ax=ax[j], x=models.get(name).get('param')[:, j], y=models.get(name).get('perr')[:, j], color=colors[i], alpha=alpha[i], label=name)
    ax[0].legend(frameon=False, bbox_to_anchor=(0, 1.02), loc='lower left', ncols=6), ax[1].legend().remove(), ax[2].legend().remove()
    ax[0].set(xlabel='STY_0', ylabel='Error', xscale='symlog', yscale='symlog', xlim=(0, None), ylim=(0, None))
    ax[1].set(xlabel='n', ylabel='Error', xscale='symlog', yscale='symlog', xlim=(0, None), ylim=(0, None))
    ax[2].set(xlabel='STY_inf', ylabel='Error', xscale='symlog', yscale='symlog', xlim=(0, None), ylim=(0, None))
    fig.savefig('figures/deactivation_modelling/Error_scatterplot.png', dpi=600)

    # Correlation plot 
    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    coo = [(0, 1), (0, 2), (1, 2)]
    for i, name in enumerate(models.keys()):
        n = models.get(name).get('param').shape[1]
        for j in range(int(n*(n-1)/2)):
            row, col = coo[j]
            sns.kdeplot(ax=ax[j], x=models.get(name).get('pcorr')[:, row, col], color=colors[i], alpha=alpha[i], label=name, cut=0)
    ax[0].legend() 
    ax[0].set(xlabel=r'Correlation STY$\mathdefault{_{0}}$ - n')
    ax[1].set(xlabel=r'Correlation STY$\mathdefault{_{0}}$ - STY$\mathdefault{_{inf}}$')
    ax[2].set(xlabel=r'Correlation n - STY$\mathdefault{_{inf}}$')
    fig.savefig('figures/deactivation_modelling/Corr_kdeplot.png', dpi=600)

    # Correlation violinplot 
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    coo = [(0, 1), (0, 2), (1, 2)]
    j = 0
    row, col = coo[j]
    sns.violinplot(ax=ax, data=[models.get(name).get('pcorr')[:, row, col] for name in models.keys()], palette=colors, cut=0)
    for i, patch in enumerate(ax.collections):
        patch.set_alpha(alpha[i])
    ax.set(ylabel=r'Correlation STY$\mathdefault{_{0}}$ - n', xticklabels=models.keys())
    fig.savefig('figures/deactivation_modelling/Corr_violinplot.png', dpi=600)

# ------------------------------------------------------------------------------------------------

# Save data
best_model = 'lan2'
data_deactivation.rename(columns={best_model + '_STY0': 'STY0', best_model + '_n': 'n'}).loc[:, initial_columns + ['STY0', 'n']].to_csv('data/data_deactivation.csv', index=False)

# ------------------------------------------------------------------------------------------------

exit()

# Find index to plot
my_array = models.get('lan2').get('param')[:, 1]
index = np.where(my_array > 1e10)[0]
index = np.where(my_array < 0.0001)[0]

# Get rows
selected_data = data_deactivation.iloc[index, :]
# print(selected_data.to_string())

# Plot all deactivation curves
for i in range(len(selected_data)):
    series = selected_data.iloc[i, :]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(series['Time_h_t'], series['STY_Acryl_mmolhg_t'], marker='o')
    for name, spec in models.items():
        t = np.linspace(0, series['Time_h_t'].max(), 100)
        ax.plot(t, spec['func'](t, *series[name + '_param']), label=name)
    ax.set(ylim=[0, series['STY_Acryl_mmolhg_t'].max()*1.1])
    ax.legend()
    plt.show()

