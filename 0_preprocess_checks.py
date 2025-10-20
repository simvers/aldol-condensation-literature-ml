from pathlib import Path
import numpy as np
import pandas as pd
from functions.functions_preprocessing import calculate_composition, calculate_molarflowrates


# Load data and extract columns
# path = Path("/mnt/c/Users/u0156112/OneDrive - KU Leuven/Shared_AC2GEN/Review/Catalysts.xlsx")
path = Path("C:/Users/u0156112/OneDrive - KU Leuven/Shared_AC2GEN/Review/Catalysts.xlsx")

data = pd.read_excel(path, na_values=['', ' '], keep_default_na=False)
print(data.columns)

# Different columns
element_columns = ['Supp_1', 'Supp_2', 'Atom_1', 'Atom_2', 'Atom_3', 'Atom_4']
comp_columns = ['Supp_Mass', 'Supp_Mol_1', 'Supp_Mass_Oxide_1', 'Supp_Mol_2', 'Supp_Mass_Oxide_2', 
                'Atom_Mol_1', 'Atom_Mass_Elem_1', 'Atom_Mass_Oxide_1', 'Atom_Mol_2', 'Atom_Mass_Elem_2', 'Atom_Mass_Oxide_2', 
                'Atom_Mol_3', 'Atom_Mass_Elem_3', 'Atom_Mass_Oxide_3', 'Atom_Mol_4', 'Atom_Mass_Elem_4', 'Atom_Mass_Oxide_4']
get_columns = ['Ac_source', 'Fa_source', 'Stabilizer', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 
               'O_content', 'LHSV_mlhg', 'Temperature_K', 'Pressure_bar', 'g_cat', 'Fa_mmolmin', 'Ac_mmolmin',
               'Y_Acryl_Ac', 'Y_Acryl_Fa', 'STY_Acryl_mmolhg_manual',
               'doi', 'Link_to_excel']
data = data[element_columns + comp_columns + get_columns]

# ------------------------------------------------------------------------------------------------

# Calculate elements molar composition
molar_composition = calculate_composition(data[element_columns + comp_columns])
# print(molar_composition.to_string())
# print(molar_composition.loc[molar_composition.isna().all(axis=1), :])

# Encode atom composition and save elements to csv
elements = molar_composition.columns.to_series()
elements.to_csv('data/elements.csv', index=False, header=False)
print('Elements in catalysts: ', elements.to_list())

# Concat encoded atom composition and rest of data
data_clean = pd.concat(
    [molar_composition, 
     data[get_columns]], 
    axis=1
)

# Most popular elements
major_elements = molar_composition.mean(axis=0).sort_values(ascending=False)
print(major_elements)

# ------------------------------------------------------------------------------------------------

# Calculate molar flowrates
print(data_clean.columns)
data_clean = calculate_molarflowrates(data_clean)

# ------------------------------------------------------------------------------------------------

# Save clean data
# molar_composition.to_csv('data/molar_composition.csv', index=False)
data_clean.to_csv('data/data_clean.csv', index=False)

print(data_clean.head(10))
