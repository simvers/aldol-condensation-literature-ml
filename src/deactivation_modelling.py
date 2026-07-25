import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src import utils, function_fitting


def extract_deact_data(directory_path: Path):

    # Load all xlsx deactivation files from directory; return parallel lists of doi, sheet name, df

    # Import all excel files as list of dict, sheet as keys, df as values
    xlsx_files = sorted(directory_path.glob('*.xlsx'))
    list_dict = [pd.read_excel(f, sheet_name=None, skiprows=2) for f in xlsx_files]

    # Extract df from dictionary and expand the doi list accordingly
    xlsx_files_expanded, sheet_digitized, df_digitized = [], [], []
    for file_path, sheet_dict in zip(xlsx_files, list_dict):
        xlsx_files_expanded.extend([file_path.name] * len(sheet_dict))
        sheet_digitized.extend(list(sheet_dict.keys()))
        df_digitized.extend(list(sheet_dict.values()))

    # Clean df and doi
    df_digitized = list(map(utils.clean_df, df_digitized))
    doi_digitized = utils.process_doi(xlsx_files_expanded)
    return doi_digitized, sheet_digitized, df_digitized


def fill_missing_xsy(df, reactant):
    # Derive the missing one of X/S/Y from the other two (Y = X*S/100)
    x_col, s_col, y_col = f'X_{reactant}_t', f'S_{reactant}_Acryl_t', f'Y_{reactant}_Acryl_t'
    if x_col in df.columns and s_col not in df.columns and y_col in df.columns:
        df[s_col] = df[y_col] / df[x_col] * 100
    if x_col not in df.columns and s_col in df.columns and y_col in df.columns:
        df[x_col] = df[y_col] / df[s_col] * 100
    if x_col in df.columns and s_col in df.columns and y_col not in df.columns:
        df[y_col] = df[s_col] * df[x_col] / 100


def merge_deact_data(data_conditions, doi_list, sheet_list, df_deactivation_list):
    # Merge time-dependent deactivation arrays into the reaction-conditions dataframe
    time_columns = set()
    for df in df_deactivation_list:
        time_columns.update(df.columns)
    data_conditions[sorted(time_columns)] = pd.NA
    print(data_conditions.columns)

    for doi, sheet, df in zip(doi_list, sheet_list, df_deactivation_list):
        index = (data_conditions['doi'] == doi) & (data_conditions['Link_to_excel'] == sheet)
        if index.sum() != 1:
            raise ValueError(f"Expected exactly one row matching doi={doi!r}, sheet={sheet!r}, found {index.sum()}")
        index = np.where(index)[0][0]

        # Conversion, selectivity, yield relationship
        fill_missing_xsy(df, 'Ac')
        fill_missing_xsy(df, 'Fa')

        # Yield_Ac, yield_Fa relationship
        if 'Y_Ac_Acryl_t' in df.columns and 'Y_Fa_Acryl_t' not in df.columns:
            df['Y_Fa_Acryl_t'] = df['Y_Ac_Acryl_t'] * data_conditions.loc[index, 'Ratio_Ac_Fa']
        if 'Y_Ac_Acryl_t' not in df.columns and 'Y_Fa_Acryl_t' in df.columns:
            df['Y_Ac_Acryl_t'] = df['Y_Fa_Acryl_t'] / data_conditions.loc[index, 'Ratio_Ac_Fa']

        # Calculate STY(t) if LHSV is specified
        if data_conditions.loc[index, ['Ac_mmolming', 'Fa_mmolming']].isna().any():
            print('Missing LHSV for deactivation modelling: ', doi, sheet)
        elif 'STY_Acryl_mmolhg_t' not in df.columns:
            df['STY_Acryl_mmolhg_t'] = df['Y_Ac_Acryl_t'] / 100 * data_conditions.loc[index, 'Ac_mmolming'] * 60
            sty_from_fa = df['Y_Fa_Acryl_t'] / 100 * data_conditions.loc[index, 'Fa_mmolming'] * 60
            if not (abs(df['STY_Acryl_mmolhg_t'] - sty_from_fa) < 1e-10).all():
                raise ValueError(f"STY computed from Ac and from Fa disagree for doi={doi!r}, sheet={sheet!r}")

        # Fill output df
        for col in df.columns:
            data_conditions.at[index, col] = df[col].to_numpy()

    return data_conditions


def fit_all_models(data_df, models, target='STY_Acryl_mmolhg_t', debug=False):
    # Fit every model in `models` to each row's deactivation curve
    for name, spec in models.items():
        data_df.loc[:, name + '_nrmse'] = np.nan
        data_df.loc[:, [name + '_param', name + '_STY0', name + '_n']] = pd.NA

        for i, row in data_df.iterrows():
            popt, perr, cv, pcorr, r2, nrmse = function_fitting.fit_model(
                row['Time_h_t'], row[target], model=spec['func'], p0=spec['p0'], bounds=spec['bounds']
            )
            data_df.at[i, f'{name}_param'] = popt
            data_df.loc[i, [f'{name}_STY0', f'{name}_n']] = popt[0:2]
            data_df.at[i, f'{name}_nrmse'] = nrmse

            models[name]['param'] = np.vstack((models[name]['param'], popt))
            models[name]['perr'] = np.vstack((models[name]['perr'], perr))
            models[name]['cv'] = np.vstack((models[name]['cv'], cv))
            models[name]['pcorr'] = np.vstack((models[name]['pcorr'], np.expand_dims(pcorr, axis=0)))

        models[name]['NRMSE_mean'] = data_df.loc[:, name + '_nrmse'].mean()
        models[name]['n_mean'] = models[name]['param'][:, 1].mean()
        models[name]['n_max'] = models[name]['param'][:, 1].max()
        models[name]['sty0_max'] = models[name]['param'][:, 0].max()

    performance_dict = {
        model: {k: v for k, v in att.items() if k not in ['func', 'p0', 'bounds', 'param', 'perr', 'cv', 'pcorr']}
        for model, att in models.items()
    }
    print(json.dumps(performance_dict, indent=4))

    if debug:
        cv_n = models['lan2']['cv'][:, 1]
        for i in np.where(cv_n > 100)[0]:
            series = data_df.iloc[i]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(series['Time_h_t'], series['STY_Acryl_mmolhg_t'], marker='o')
            for name, spec in models.items():
                t = np.linspace(0, series['Time_h_t'].max(), 100)
                ax.plot(t, spec['func'](t, *series[name + '_param']), label=name)
            ax.set(ylim=[0, series['STY_Acryl_mmolhg_t'].max() * 1.1])
            ax.legend()
            plt.show()

    return data_df, models
