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

    # Read all files, returns a list of dictionary {sheet_name: df} for each file
    list_dict = [pd.read_excel(os.path.join(path_xlsx_directory, f), sheet_name=None, skiprows=2) for f in files_xlsx]

    # Expand each file's dict of sheets into flat, parallel lists (one entry per sheet)
    files_xlsx_expanded, sheet_digitized, df_digitized = [], [], []
    for file_name, sheet_dict in zip(files_xlsx, list_dict):
        files_xlsx_expanded.extend([file_name] * len(sheet_dict))
        sheet_digitized.extend(list(sheet_dict.keys()))
        df_digitized.extend(list(sheet_dict.values()))

    # Clean dataframe: remove NaN columns and rename columns as _t
    df_digitized = list(map(clean_df_deactivation, df_digitized))

    # Extract doi from file name
    doi_digitized = process_doi(files_xlsx_expanded)

    return doi_digitized, sheet_digitized, df_digitized


def fill_missing_xsy(df, reactant):
    # Conversion (X), selectivity (S), and yield (Y) are related by Y = X*S/100.
    # If exactly one of the three is missing for a reactant, derive it from the other two.
    x_col, s_col, y_col = f'X_{reactant}_t', f'S_{reactant}_Acryl_t', f'Y_{reactant}_Acryl_t'

    if x_col in df.columns and s_col not in df.columns and y_col in df.columns:
        df[s_col] = df[y_col] / df[x_col] * 100
    if x_col not in df.columns and s_col in df.columns and y_col in df.columns:
        df[x_col] = df[y_col] / df[s_col] * 100
    if x_col in df.columns and s_col in df.columns and y_col not in df.columns:
        df[y_col] = df[s_col] * df[x_col] / 100


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

    # Collect every time-dependent column across all digitized sheets, so data_conditions can
    # hold the union of them (most sheets only report a subset)
    time_columns = set()
    for df in df_deactivation_list:
        time_columns.update(df.columns)

    # Initialize columns
    data_conditions[sorted(time_columns)] = pd.NA
    print(data_conditions.columns)

    # Merge each df as np.arrays in data_conditions
    for doi, sheet, df in zip(doi_list, sheet_list, df_deactivation_list):

        # Find index of doi and sheet in data_conditions - must match exactly one row
        index = ((data_conditions['doi'] == doi) & (data_conditions['Link_to_excel'] == sheet))
        if index.sum() != 1:
            raise ValueError(f"Expected exactly one row matching doi={doi!r}, sheet={sheet!r}, found {index.sum()}")
        index = np.where(index)[0][0]  # Get integer index

        # Fill in conversion/selectivity/yield wherever only two of the three were digitized
        fill_missing_xsy(df, 'Ac')
        fill_missing_xsy(df, 'Fa')

        # Fill Y_Ac from Y_Fa or vice-versa, via the reported Ac/Fa molar ratio
        if (('Y_Ac_Acryl_t' in df.columns) & ('Y_Fa_Acryl_t' not in df.columns)):
            df['Y_Fa_Acryl_t'] = df['Y_Ac_Acryl_t'] * data_conditions.loc[index, 'Ratio_Ac_Fa']
        if (('Y_Ac_Acryl_t' not in df.columns) & ('Y_Fa_Acryl_t' in df.columns)):
            df['Y_Ac_Acryl_t'] = df['Y_Fa_Acryl_t'] / data_conditions.loc[index, 'Ratio_Ac_Fa']

        # Identify missing lhsv
        if data_conditions.loc[index, ['Ac_mmolming', 'Fa_mmolming']].isna().any():
            print('Missing LHSV for deactivation modelling: ', doi, sheet)

        # Calculate STY if missing and check it is the same from Ac and Fa
        elif 'STY_Acryl_mmolhg_t' not in df.columns:
            df['STY_Acryl_mmolhg_t'] = df['Y_Ac_Acryl_t']/100 * data_conditions.loc[index, 'Ac_mmolming'] * 60
            sty_from_fa = df['Y_Fa_Acryl_t']/100 * data_conditions.loc[index, 'Fa_mmolming'] * 60
            if not (abs(df['STY_Acryl_mmolhg_t'] - sty_from_fa) < 1e-10).all():
                raise ValueError(f"STY computed from Ac and from Fa disagree for doi={doi!r}, sheet={sheet!r}")

        # Assign each time variable as a numpy array in the dataframe
        for col in df.columns:
            data_conditions.at[index, col] = df[col].to_numpy()

    return data_conditions


def fit_all_models(data_df, models, target='STY_Acryl_mmolhg_t'):
    # Fit every model in `models` to each row's deactivation curve. Stores per-row results
    # (param/STY0/n/nrmse) directly in data_df, and accumulates per-model parameter arrays
    # (param/perr/cv/pcorr) in `models`, used by the diagnostic plots below.

    for name, spec in models.items():

        # Initialize columns
        data_df.loc[:, name + '_nrmse'] = np.nan
        data_df.loc[:, [name + '_param', name + '_STY0', name + '_n']] = pd.NA

        # Fit deactivation models, one catalyst (row) at a time
        for i, row in data_df.iterrows():
            t = row['Time_h_t']
            sty = row[target]

            popt, perr, cv, pcorr, r2, nrmse = fit_model(t, sty, model=spec['func'], p0=spec['p0'], bounds=spec['bounds'])

            # Add fitted model to deactivation data
            data_df.at[i, f'{name}_param'] = popt
            data_df.loc[i, [f'{name}_STY0', f'{name}_n']] = popt[0:2]
            data_df.at[i, f'{name}_nrmse'] = nrmse

            # Accumulate per-model parameter arrays
            models[name]['param'] = np.vstack((models[name]['param'], popt))
            models[name]['perr'] = np.vstack((models[name]['perr'], perr))
            models[name]['cv'] = np.vstack((models[name]['cv'], cv))
            models[name]['pcorr'] = np.vstack((models[name]['pcorr'], np.expand_dims(pcorr, axis=0)))

        # Summary statistics per model
        models[name]['NRMSE_mean'] = data_df.loc[:, name+"_nrmse"].mean()
        models[name]['n_mean'] = models[name]["param"][:, 1].mean()
        models[name]['n_max'] = models[name]["param"][:, 1].max()
        models[name]['sty0_max'] = models[name]["param"][:, 0].max()

    # Report performance, excluding the raw fitted arrays
    performance_dict = {model: {k: v for k, v in att.items() if k not in ['func', 'p0', 'bounds', 'param', 'perr', 'cv', 'pcorr']} for model, att in models.items()}
    print(json.dumps(performance_dict, indent=4))

    return data_df, models


def plot_nrmse_distribution(data_df, models, colors):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    for i, name in enumerate(models.keys()):
        sns.kdeplot(ax=ax, data=data_df[f'{name}_nrmse'], color=colors[i], label=models.get(name).get('name'), cut=0)
        ax.axvline(x=models.get(name).get('NRMSE_mean'), color=colors[i], linestyle='--', linewidth=0.75)
    ax.legend(frameon=False)
    ax.set(xlabel='NRMSE', xlim=(0, 0.2), ylim=(0, 14), xticks=[0, 0.05, 0.1, 0.15, 0.2])
    fig.savefig('figures/deactivation_modelling/NRMSE_kdeplot.svg', dpi=300, format='svg', bbox_inches='tight')


def plot_param_error(models, colors):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i, name in enumerate(models.keys()):
        for j in range(models.get(name).get('param').shape[1]):
            sns.scatterplot(ax=ax[j], x=models.get(name).get('param')[:, j], y=models.get(name).get('perr')[:, j], color=colors[i], label=models.get(name).get('name'))
    ax[0].legend(frameon=False, bbox_to_anchor=(0, 1.02), loc='lower left', ncols=6)
    ax[1].legend().remove()
    ax[0].set(xlabel='STY_0', ylabel='Error', xscale='symlog', yscale='symlog', xlim=(0, None), ylim=(0, 100))
    ax[1].set(xlabel='n', ylabel='Error', xscale='symlog', yscale='symlog', xlim=(0, 1), ylim=(0, 1))
    ax[2].set(xlabel='STY_inf', ylabel='Error', xscale='symlog', yscale='symlog', xlim=(0, None), ylim=(0, 100))
    fig.savefig('figures/deactivation_modelling/Error_scatterplot.svg', dpi=300, format='svg')


def plot_param_cv(models, colors):
    # Coefficient-of-variation per fitted parameter (STY_0, n), with marginal distributions
    n_param = models.get(list(models.keys())[0]).get('param').shape[1]
    fig = plt.figure(figsize=(3*n_param, 3))
    gs = fig.add_gridspec(1, n_param)
    ax, ax_joint, ax_x, ax_y = [], [], [], []
    for i in range(n_param):
        ax.append(gs[0, i].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
        ax_joint.append(fig.add_subplot(ax[i][1, 0]))
        ax_x.append(fig.add_subplot(ax[i][0, 0], sharex=ax_joint[i]))
        ax_y.append(fig.add_subplot(ax[i][1, 1], sharey=ax_joint[i]))
        for item in [ax_x[i], ax_y[i]]:
            item.set_axis_off()

    for i, name in enumerate(models.keys()):
        for j in range(models.get(name).get('param').shape[1]):
            sns.scatterplot(ax=ax_joint[j], x=models.get(name).get('param')[:, j], y=models.get(name).get('cv')[:, j], color=colors[i], label=models.get(name).get('name'))
            sns.kdeplot(ax=ax_x[j], x=models.get(name).get('param')[:, j], color=colors[i], label=models.get(name).get('name'))
            sns.kdeplot(ax=ax_y[j], y=models.get(name).get('cv')[:, j], color=colors[i], label=models.get(name).get('name'))

    ax_joint[0].set(xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='Coefficient of variation /', xscale='symlog', yscale='symlog', xlim=(0, None), ylim=(0, 2))
    ax_joint[0].set_yscale('symlog', linthresh=0.05, linscale=0.3)
    ax_joint[0].legend().remove()
    ax_joint[1].set(xlabel='n /', ylabel='Coefficient of variation /', xscale='symlog', yscale='symlog', xlim=(0, None), ylim=(0, 10))
    ax_joint[1].set_xscale('symlog', linthresh=0.01, linscale=0.3)
    ax_joint[1].set_yscale('symlog', linthresh=0.1, linscale=0.3)
    ax_joint[1].legend(frameon=False, bbox_to_anchor=(1.25, 0.5), loc='center left')
    fig.savefig('figures/deactivation_modelling/CV_scatterplot.svg', dpi=300, format='svg', bbox_inches='tight')


def debug_plot_outlier_fits(data_df, models, reference_model='lan2', cv_threshold=100):
    # Manual inspection tool: plot the raw deactivation data plus every fitted curve, for
    # catalysts whose reference_model 'n' coefficient-of-variation exceeds cv_threshold -
    # a sign that fit is unstable and worth a visual sanity check.
    cv_n = models.get(reference_model).get('cv')[:, 1]
    index = np.where(cv_n > cv_threshold)[0]
    if len(index) == 0:
        return

    selected_data = data_df.iloc[index, :]
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


def deactivation_modelling(data_df, models, target='STY_Acryl_mmolhg_t', debug=False):
    # Fit every deactivation model to every catalyst's curve, report performance, and plot
    # NRMSE / parameter-error / parameter-CV diagnostics across models. Set debug=True to also
    # interactively inspect curves with a poor reference-model fit (high coefficient of variation).

    data_df, models = fit_all_models(data_df, models, target=target)

    colors = ["#009688", "#1565C0", "#AD1457"]
    plot_nrmse_distribution(data_df, models, colors)
    plot_param_error(models, colors)
    plot_param_cv(models, colors)

    if debug:
        debug_plot_outlier_fits(data_df, models)

    return data_df, models
