from pathlib import Path
import pandas as pd


def filter_type(list_files, delim='.', type='csv'):
    return [f for f in list_files if f.lower().endswith(f"{delim}{type}")]


def list_mapper(list_, dict_, func=lambda x: x):
    return [dict_.get(func(item), func(item)) for item in list_]


def load_data(path, columns='all'):

    # Load data
    path = Path(path)
    if path.suffix.lower() == '.csv':
        data = pd.read_csv(path, na_values=['', ' '], keep_default_na=False)
    else:
        data = pd.read_excel(path, na_values=['', ' '], keep_default_na=False)

    # Return
    if columns == 'all':
        return data
    else:
        return data[columns]


def process_doi(doi_list):

    # Extract a normalized doi key from a digitized-data file name, e.g.
    # "2015_10_1016_j_jiec_2014_11_014.xlsx" -> "101016jjiec20141114". Used to match a
    # digitized excel file back to its row in the reaction-conditions table by doi.
    return [doi.split('_', 1)[-1]  # Remove date
            .replace('.xlsx', '')  # Remove xlsx extension
            .replace('.', '')  # Remove symbols
            .replace('/', '')
            .replace('_', '')
            .replace('-', '')
            .replace('(', '')
            .replace(')', '')
            .lower()  # Unify lowercase
            for doi in doi_list]


def clean_df(df: pd.DataFrame):

    # Drop columns that are entirely NaN (e.g. unused time-series columns in a digitized sheet)
    df = df.dropna(axis=1, how='all')

    # Rename columns, e.g. "Y (%)" -> "Y_t", to mark them as time-dependent series
    df.columns = df.columns.str.replace(' (', '_').str.strip(' )') + '_t'

    return df
