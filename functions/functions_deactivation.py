import numpy as np
import pandas as pd
import os
from functions.utils import filter_type, process_doi, clean_df_deactivation


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
            print(doi, sheet, '\n', data_conditions.loc[index, ['Ac_mmolming', 'Ac_mmolmin', 'Fa_mmolming', 'Fa_mmolmin']])
        
        # Calculate STY if missing and check it is the same from Ac and Fa
        elif 'STY_Acryl_mmolhg_t' not in df.columns:
            df['STY_Acryl_mmolhg_t'] = df['Y_Ac_Acryl_t']/100 * data_conditions.loc[index, 'Ac_mmolming'] * 60
            assert (abs(df['STY_Acryl_mmolhg_t'] - df['Y_Fa_Acryl_t']/100 * data_conditions.loc[index, 'Fa_mmolming'] * 60) < 1e-10).all()

        # Assign each time variable as a numpy array in the dataframe
        for col in df.columns:
            data_conditions.at[index, col] = df[col].to_numpy()
        
    return data_conditions

