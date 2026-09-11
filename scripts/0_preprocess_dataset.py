"""
Preprocess the raw literature dataset: encode catalyst composition, compute molar
flow rates, and impute missing SSA values.

Reads:  data/raw/data_raw.xlsx
Writes: data/processed/data_processed.csv, data/processed/elements.csv
"""

import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src import preprocessing, utils

ROOT = Path(__file__).resolve().parents[1]
os.makedirs(ROOT / 'data/processed', exist_ok=True)

ELEMENT_COLUMNS = ['Supp_1', 'Supp_2', 'Atom_1', 'Atom_2', 'Atom_3', 'Atom_4']

COMP_COLUMNS = ['Supp_Mass', 'Supp_Mol_1', 'Supp_Mass_Oxide_1', 'Supp_Mol_2', 'Supp_Mass_Oxide_2',
                'Atom_Mol_1', 'Atom_Mass_Elem_1', 'Atom_Mass_Oxide_1', 'Atom_Mol_2', 'Atom_Mass_Elem_2', 'Atom_Mass_Oxide_2',
                'Atom_Mol_3', 'Atom_Mass_Elem_3', 'Atom_Mass_Oxide_3', 'Atom_Mol_4', 'Atom_Mass_Elem_4', 'Atom_Mass_Oxide_4']
                
GET_COLUMNS = ['Ac_source', 'Fa_source', 'Stabilizer', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa',
               'O_content', 'LHSV_mlhg', 'Temperature_K', 'Pressure_bar',
               'Y_Acryl_Ac', 'Y_Acryl_Fa',
               'doi', 'Link_to_excel', 'Year', 'SSA_m2g', 'SSA_Supp_m2g']

VERBOSE = True


if __name__ == "__main__":

    # Load raw data
    data = utils.load_data(ROOT / "data/raw/data_raw.xlsx", columns=ELEMENT_COLUMNS+COMP_COLUMNS+GET_COLUMNS)

    # Calculate elements molar composition
    molar_composition = preprocessing.calculate_composition(data[ELEMENT_COLUMNS + COMP_COLUMNS])

    # Encode atom composition and save elements to csv
    elements = molar_composition.columns.to_series()
    elements.to_csv(ROOT / 'data/processed/elements.csv', index=False, header=False)
    if VERBOSE:
        print('Elements in catalysts: \n', elements.to_list())

    # Most popular elements
    major_elements = molar_composition.mean(axis=0).sort_values(ascending=False)
    if VERBOSE:
        print('Major elements in catalysts: \n', major_elements)

    # Concat encoded atom composition and rest of data
    data_merged = pd.concat([molar_composition, data[GET_COLUMNS]], axis=1)

    # Calculate molar flowrates
    data_processed = preprocessing.calculate_molarflowrates(data_merged)
    if VERBOSE:
        print(data_processed.columns)

    # Impute SSA
    data_processed.loc[:, 'SSA_m2g'] = preprocessing.impute_ssa(data_processed, elements)
    data_processed.drop(['SSA_Supp_m2g'], axis=1, inplace=True)

    # Save processed data
    data_processed.to_csv('data/processed/data_processed.csv', index=False)
