import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from functions.preprocessing import calculate_composition, calculate_molarflowrates, impute_ssa

RAW_DATA_PATH = Path("/mnt/c/Users/u0156112/OneDrive - KU Leuven/Shared_AC2GEN/Review/Data/Catalysts.xlsx")
# RAW_DATA_PATH = Path("C:/Users/u0156112/OneDrive - KU Leuven/Shared_AC2GEN/Review/Data/Catalysts.xlsx")

ELEMENT_COLUMNS = ['Supp_1', 'Supp_2', 'Atom_1', 'Atom_2', 'Atom_3', 'Atom_4']
COMP_COLUMNS = ['Supp_Mass', 'Supp_Mol_1', 'Supp_Mass_Oxide_1', 'Supp_Mol_2', 'Supp_Mass_Oxide_2',
                'Atom_Mol_1', 'Atom_Mass_Elem_1', 'Atom_Mass_Oxide_1', 'Atom_Mol_2', 'Atom_Mass_Elem_2', 'Atom_Mass_Oxide_2',
                'Atom_Mol_3', 'Atom_Mass_Elem_3', 'Atom_Mass_Oxide_3', 'Atom_Mol_4', 'Atom_Mass_Elem_4', 'Atom_Mass_Oxide_4']
GET_COLUMNS = ['Ac_source', 'Fa_source', 'Stabilizer', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa',
               'O_content', 'LHSV_mlhg', 'Temperature_K', 'Pressure_bar',
               'Y_Acryl_Ac', 'Y_Acryl_Fa',
               'doi', 'Link_to_excel', 'Year', 'SSA_m2g', 'SSA_Supp_m2g']


def load_raw_data(path, verbose=True):
    data = pd.read_excel(path, na_values=['', ' '], keep_default_na=False)
    if verbose:
        print(data.columns.to_list())
    return data[ELEMENT_COLUMNS + COMP_COLUMNS + GET_COLUMNS]


def build_processed_dataset(data, verbose=True):

    # Calculate elements molar composition
    molar_composition = calculate_composition(data[ELEMENT_COLUMNS + COMP_COLUMNS])

    # Encode atom composition and save elements to csv
    elements = molar_composition.columns.to_series()
    elements.to_csv('data/catalysts/elements.csv', index=False, header=False)
    if verbose:
        print('Elements in catalysts: \n', elements.to_list())

    # Most popular elements
    major_elements = molar_composition.mean(axis=0).sort_values(ascending=False)
    if verbose:
        print('Major elements in catalysts: \n', major_elements)

    # Concat encoded atom composition and rest of data
    data_comp = pd.concat([molar_composition, data[GET_COLUMNS]], axis=1)

    # Calculate molar flowrates
    data_processed = calculate_molarflowrates(data_comp)
    if verbose:
        print(data_processed.columns)

    # Impute SSA
    data_processed.loc[:, 'SSA_m2g'] = impute_ssa(data_processed, elements)
    data_processed.drop(['SSA_Supp_m2g'], axis=1, inplace=True)

    return data_processed


def main():
    data = load_raw_data(RAW_DATA_PATH)
    data_processed = build_processed_dataset(data)
    data_processed.to_csv('data/catalysts/data_processed.csv', index=False)


if __name__ == "__main__":
    main()
