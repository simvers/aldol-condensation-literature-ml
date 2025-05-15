import pandas as pd
import os
from utils import filter_type, process_doi


def extract_df(list_files):

    """

    Input: list of xlsx files to read
    - Import each xlsx file as a directory of each worksheet and store it in a list
    - Expend the list of file names to account for xlsx files with multiple worksheet
    - Expend the list of directory to a list of df
    Output: list of file names, list of df

    """

    # Initialize lists
    list_files_expanded, list_df = [], []

    # Read all files, returns a list of dictionary containing all excel sheets of each file
    list_dict = list(map(lambda x: pd.read_excel(path + x, sheet_name=None, skiprows=5), list_files))

    # Expend the list of dict in a list of df and extend the list of files accordingly
    for i, dict in enumerate(list_dict):
        list_files_expanded.extend([list_files[i]] * len(dict))
        list_df.extend(list(dict.values()))
    return list_files_expanded, list_df


def clean_df(df: pd.DataFrame):
    return df.dropna(axis=1, how='all')

# ------------------------------------------------------------------------------------------------

# Import digitized data

# List files in target directory
path = 'C:\\Users\\u0156112\\OneDrive - KU Leuven\\Shared_AC2GEN\\Review\\DigitizedData\\'
list_files = filter_type(os.listdir(path), delim='.', type='xlsx')

# Import all excel files as directory, and expand the directory to a list of df
list_files, list_df = extract_df(list_files)

# Extract doi from file name
list_doi = process_doi(list_files)

# ------------------------------------------------------------------------------------------------

# Import clustered data
data_clustered = pd.read_excel('data/data_clustered.xlsx')
doi_data = process_doi(data_clustered['doi'].unique())

# ------------------------------------------------------------------------------------------------

# Identify publications that still need to be processed

# print(set(list_doi).issubset(doi))
doi_mendeley = []
for doi_i in list_doi:
    if doi_i not in doi_data:
        if doi_i not in doi_mendeley:
            doi_mendeley.append(doi_i)
print('DOI to process in Mendeley: ', len(doi_mendeley))

doi_digital = []
for doi_i in doi_data:
    if doi_i not in list_doi:
        if doi_i not in doi_digital:
            doi_digital.append(doi_i)
print('DOI to process digitally', len(doi_digital))

# ------------------------------------------------------------------------------------------------

# Clean dataframe
list_df = list(map(clean_df, list_df))

# Process column names
column_list = []
for i, df in enumerate(list_df):
    for column in df.columns:
        if column not in column_list:
            column_list.append(column)
        if column == 'Conv': print(list_files[i], list_df[i])
print(column_list)
