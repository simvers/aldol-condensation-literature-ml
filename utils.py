import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def filter_type(list_files, delim='.', type='csv'):
    return [f for f in list_files if f.lower().endswith(f"{delim}{type}")]


def process_doi(doi_list):

    # Extract doi from file name
    # list_doi = ['/'.join(file.split('_', 1)[-1].split('_')) for file in list_files]
    # list_doi = [file.split('_', 1)[-1]  # Remove date
    #             .replace('_', '.', 1)   # Replace first underscore with . as in 10.xxxx
    #             .replace('_', '/', 1)   # Replace second underscore with / as in 10.xxxx/
    #             .replace('_', '.')      # Replace all remaining underscores with .
    #             .replace('.xlsx', '')   # Remove .xlsx
    #             for file in doi_list]

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