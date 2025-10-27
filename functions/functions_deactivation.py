import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions.utils import filter_type, process_doi, clean_df_deactivation
from functions.deactivation_models import fit_model


def extract_df_deactivation_from_excel(path_xlsx_directory):

    """

    Input: path to directory with xlsx files to read
    Transform:
        - Import each xlsx file as a directory of each worksheet and store it in a list
        - Expend the list of file names to account for xlsx files with multiple worksheet
        - Expend the list of directory to a list of df
    Output: list of file names, list of df

    """

    # List xlsx files in target directory
    files_xlsx = filter_type(os.listdir(path_xlsx_directory), delim='.', type='xlsx')

    # Initialize lists
    files_xlsx_expanded, sheet_digitized, df_digitized = [], [], []

    # Read all files, returns a list of dictionary containing all excel sheets of each file
    list_dict = list(map(lambda x: pd.read_excel(path_xlsx_directory + x, sheet_name=None, skiprows=2), files_xlsx))

    # Expend the list of dict in a list of df and extend the list of files accordingly
    for i, dict in enumerate(list_dict):
        files_xlsx_expanded.extend([files_xlsx[i]] * len(dict))
        sheet_digitized.extend(list(dict.keys()))
        df_digitized.extend(list(dict.values()))

    # Clean dataframe: remove NaN columns and rename columns as _t
    df_digitized = list(map(clean_df_deactivation, df_digitized))
    
    # Extract doi from file name
    doi_digitized = process_doi(files_xlsx_expanded)

    return doi_digitized, sheet_digitized, df_digitized


def merge_conditions_deactivation(data_conditions, doi_list, sheet_list, df_deactivation_list):

    """

    Input:
        - data_conditions: dataframe with reaction conditions and metadata with doi and xlsx sheet name
        - doi_list: list of doi corresponding to each df in df_deactivation_list
        - sheet_list: list of sheet names corresponding to each df in df_deactivation_list
        - df_deactivation_list: list of dataframe with time-dependent deactivation data
    Transform: merger the time-dependent data as numpy arrays in the reaction conditions df
    Output: reaction conditions df with deactivaiton data

    """

    # Process column names
    time_columns = set()
    for i, df in enumerate(df_deactivation_list):

        # Update set
        time_columns.update(df.columns)
        
        # Check if problematic column
        # if 'Redo!' in df.columns: print(doi_list[i], df)

    # Initialize columns
    data_conditions[sorted(time_columns)] = pd.NA
    print(data_conditions.columns)

    # Merge each df as np.arrays in data_conditions
    for doi, sheet, df in zip(doi_list, sheet_list, df_deactivation_list):

        # Find index of doi and sheet in data_conditions
        # Check that there is only one match
        index = ((data_conditions['doi'] == doi) & (data_conditions['Link_to_excel'] == sheet))
        assert index.sum() == 1
        index = np.where(index)[0][0]  # Get integer index

        # Fill df with X, S, and Y values if missing
        # Fill S_Ac_Acryl
        if (('X_Ac_t' in df.columns) & ('S_Ac_Acryl_t' not in df.columns) & ('Y_Ac_Acryl_t' in df.columns)):
            df['S_Ac_Acryl_t'] = df['Y_Ac_Acryl_t'] / df['X_Ac_t'] * 100
        # Fill X_Ac
        if (('X_Ac_t' not in df.columns) & ('S_Ac_Acryl_t' in df.columns) & ('Y_Ac_Acryl_t' in df.columns)):
            df['X_Ac_t'] = df['Y_Ac_Acryl_t'] / df['S_Ac_Acryl_t'] * 100
        # Fill Y_Ac_Acryl
        if (('X_Ac_t' in df.columns) & ('S_Ac_Acryl_t' in df.columns) & ('Y_Ac_Acryl_t' not in df.columns)):
            df['Y_Ac_Acryl_t'] = df['S_Ac_Acryl_t'] * df['X_Ac_t'] / 100
        # Fill S_Fa_Acryl
        if (('X_Fa_t' in df.columns) & ('S_Fa_Acryl_t' not in df.columns) & ('Y_Fa_Acryl_t' in df.columns)):
            df['S_Fa_Acryl_t'] = df['Y_Fa_Acryl_t'] / df['X_Fa_t'] * 100
        # Fill X_Ac
        if (('X_Fa_t' not in df.columns) & ('S_Fa_Acryl_t' in df.columns) & ('Y_Fa_Acryl_t' in df.columns)):
            df['X_Fa_t'] = df['Y_Fa_Acryl_t'] / df['S_Fa_Acryl_t'] * 100
        # Fill Y_Fa_Acryl
        if (('X_Fa_t' in df.columns) & ('S_Fa_Acryl_t' in df.columns) & ('Y_Fa_Acryl_t' not in df.columns)):
            df['Y_Fa_Acryl_t'] = df['S_Fa_Acryl_t'] * df['X_Fa_t'] / 100
        
        # Fill Y_Ac from Y_Fa or viceversa
        if (('Y_Ac_Acryl_t' in df.columns) & ('Y_Fa_Acryl_t' not in df.columns)):
            df['Y_Fa_Acryl_t'] = df['Y_Ac_Acryl_t'] * data_conditions.loc[index, 'Ratio_Ac_Fa']
        if (('Y_Ac_Acryl_t' not in df.columns) & ('Y_Fa_Acryl_t' in df.columns)):
            df['Y_Ac_Acryl_t'] = df['Y_Fa_Acryl_t'] / data_conditions.loc[index, 'Ratio_Ac_Fa']

        # Identify missing lhsv
        if data_conditions.loc[index, ['Ac_mmolming', 'Fa_mmolming']].isna().any():
            print(doi, sheet)  #, '\n', data_conditions.loc[index, ['Ac_mmolming', 'Ac_mmolmin', 'Fa_mmolming', 'Fa_mmolmin']])
        
        # Calculate STY if missing and check it is the same from Ac and Fa
        elif 'STY_Acryl_mmolhg_t' not in df.columns:
            df['STY_Acryl_mmolhg_t'] = df['Y_Ac_Acryl_t']/100 * data_conditions.loc[index, 'Ac_mmolming'] * 60
            assert (abs(df['STY_Acryl_mmolhg_t'] - df['Y_Fa_Acryl_t']/100 * data_conditions.loc[index, 'Fa_mmolming'] * 60) < 1e-10).all()

        # Assign each time variable as a numpy array in the dataframe
        for col in df.columns:
            data_conditions.at[index, col] = df[col].to_numpy()
        
    return data_conditions


def deactivation_modelling(data_df, models, target='STY_Acryl_mmolhg_t'):

    # Loop over models
    for name, spec in models.items():

        # Initialize columns
        data_df.loc[:, name + '_nrmse'] = np.nan
        # data_df.loc[:, [name + '_param', name + '_perr', name + '_cv', name + '_pcorr']] = pd.NA
        data_df.loc[:, [name + '_param', name + '_STY0', name + '_n']] = pd.NA

        # Fit deactivation models
        for i, row in data_df.iterrows():

            # Initialize fitting variables
            t = row['Time_h_t']
            sty = row[target]

            # Fit model
            popt, perr, cv, pcorr, r2, nrmse = fit_model(t, sty, model=spec['func'], p0=spec['p0'], bounds=spec['bounds'])

            # Add fitted model to deactivation data
            data_df.at[i, f'{name}_param'] = popt
            data_df.loc[i, [f'{name}_STY0', f'{name}_n']] = popt[0:2]
            # data_df.at[i, f'{name}_perr'] = perr
            # data_df.at[i, f'{name}_cv'] = cv
            data_df.at[i, f'{name}_nrmse'] = nrmse

            # Add fitted model to model data
            models[name]['param'] = np.vstack((models[name]['param'], popt))
            models[name]['perr'] = np.vstack((models[name]['perr'], perr))
            models[name]['cv'] = np.vstack((models[name]['cv'], cv))
            models[name]['pcorr'] = np.vstack((models[name]['pcorr'], np.expand_dims(pcorr, axis=0)))
        
        # Print statistics
        models[name]['NRMSE_mean'] = data_df.loc[:, name+"_nrmse"].mean()
        models[name]['n_mean'] = models[name]["param"][:, 1].mean()
        models[name]['n_max'] = models[name]["param"][:, 1].max()
        # print(f'Model {name}:\nNRMSE: {data_df.loc[:, name+"_nrmse"].mean()}\nMax n: {models[name]["param"][:, 1].max()}\nAv n: {models[name]["param"][:, 1].mean()}')
    
    # Save models dictionary
    peformance_dict = {model: {k:v for k, v in att.items() if k not in ['func', 'p0', 'bounds', 'param', 'perr', 'cv', 'pcorr']} for model, att in models.items()}
    # peformance_dict.to_json('data/fit_model.json')
    print(json.dumps(peformance_dict, indent=4))

    # ------------------------------------------------------------------------------------------------

    # Plot
    # colors = ["#448AFF", "#1565C0", "#009688", "#8BC34A", "#FFC107", "#FF9800", "#F44336", "#AD1457"]
    colors = ["#009688", "#009688", "#1565C0", "#1565C0", "#AD1457", "#AD1457", "#FF9800"]
    alpha = [0.5, 1, 0.5, 1, 0.5, 1]

    # KDE plot of NRMSE values
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, name in enumerate(models.keys()):
        sns.kdeplot(ax=ax, data=data_df[f'{name}_nrmse'], color=colors[i], alpha=alpha[i], label=name, cut=0)
    ax.legend(frameon=False)
    ax.set(xlabel='NRMSE', xlim=(0, None))
    fig.savefig('figures/deactivation_modelling/NRMSE_kdeplot.png', dpi=600)

    # NRMSE violinplot 
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    sns.violinplot(ax=ax, data=[data_df[f'{name}_nrmse'] for name in models.keys()], palette=colors, cut=0)
    for i, patch in enumerate(ax.collections):
        patch.set_alpha(alpha[i])
    ax.set(xlabel='Deactivation models', ylabel='NRMSE', xticklabels=models.keys())
    fig.savefig('figures/deactivation_modelling/NRMSE_violinplot.png', dpi=600)

    # KDE plot of param values
    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    for i, name in enumerate(models.keys()):
        for j in range(models.get(name).get('param').shape[1]):
            if j == 1:
                print(name, ' n > 1', sum(models.get(name).get('param')[:, j] > 1))
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
    ax[0].set(xlabel='Correlation STY$\mathdefault{_{0}}$ - n')
    ax[1].set(xlabel='Correlation STY$\mathdefault{_{0}}$ - STY$\mathdefault{_{inf}}$')
    ax[2].set(xlabel='Correlation n - STY$\mathdefault{_{inf}}$')
    fig.savefig('figures/deactivation_modelling/Corr_kdeplot.png', dpi=600)

    # Correlation violinplot 
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    coo = [(0, 1), (0, 2), (1, 2)]
    j = 0
    row, col = coo[j]
    sns.violinplot(ax=ax, data=[models.get(name).get('pcorr')[:, row, col] for name in models.keys()], palette=colors, cut=0)
    for i, patch in enumerate(ax.collections):
        patch.set_alpha(alpha[i])
    ax.set(ylabel='Correlation STY$\mathdefault{_{0}}$ - n', xticklabels=models.keys())
    fig.savefig('figures/deactivation_modelling/Corr_violinplot.png', dpi=600)

    # ------------------------------------------------------------------------------------------------

    # Debug model fitting

    # Find index to plot
    my_array = models.get('lan2').get('param')[:, 1]
    index = np.where(my_array < 0.0001)[0]
    index = []

    if len(index) > 0:

        # Get rows
        selected_data = data_df.iloc[index, :]
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

    # ------------------------------------------------------------------------------------------------

    return data_df, models

