from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from functions.functions_preprocessing import calculate_composition, calculate_molarflowrates, impute_ssa


# Load data and extract columns
path = Path("/mnt/c/Users/u0156112/OneDrive - KU Leuven/Shared_AC2GEN/Review/Data/Catalysts.xlsx")
# path = Path("C:/Users/u0156112/OneDrive - KU Leuven/Shared_AC2GEN/Review/Data/Catalysts.xlsx")
data = pd.read_excel(path, na_values=['', ' '], keep_default_na=False)
print(data.columns.to_list())

# Different columns
element_columns = ['Supp_1', 'Supp_2', 'Atom_1', 'Atom_2', 'Atom_3', 'Atom_4']
comp_columns = ['Supp_Mass', 'Supp_Mol_1', 'Supp_Mass_Oxide_1', 'Supp_Mol_2', 'Supp_Mass_Oxide_2', 
                'Atom_Mol_1', 'Atom_Mass_Elem_1', 'Atom_Mass_Oxide_1', 'Atom_Mol_2', 'Atom_Mass_Elem_2', 'Atom_Mass_Oxide_2', 
                'Atom_Mol_3', 'Atom_Mass_Elem_3', 'Atom_Mass_Oxide_3', 'Atom_Mol_4', 'Atom_Mass_Elem_4', 'Atom_Mass_Oxide_4']
get_columns = ['Ac_source', 'Fa_source', 'Stabilizer', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 
               'O_content', 'LHSV_mlhg', 'Temperature_K', 'Pressure_bar',
               'Y_Acryl_Ac', 'Y_Acryl_Fa',
               'doi', 'Link_to_excel', 'Year', 'SSA_m2g', 'SSA_Supp_m2g']
data = data[element_columns + comp_columns + get_columns]

# ------------------------------------------------------------------------------------------------

# Calculate elements molar composition
molar_composition = calculate_composition(data[element_columns + comp_columns])

# Encode atom composition and save elements to csv
elements = molar_composition.columns.to_series()
elements.to_csv('data/catalysts/elements.csv', index=False, header=False)
print('Elements in catalysts: \n', elements.to_list())

# Most popular elements
major_elements = molar_composition.mean(axis=0).sort_values(ascending=False)
print('Major elements in catalysts: \n', major_elements)

# Concat encoded atom composition and rest of data
data_comp = pd.concat([molar_composition, data[get_columns]], axis=1)

# ------------------------------------------------------------------------------------------------

# Calculate molar flowrates
data_processed = calculate_molarflowrates(data_comp)
print(data_processed.columns)

# ------------------------------------------------------------------------------------------------

# Impute SSA
data_processed.loc[:, 'SSA_m2g'] = impute_ssa(data_processed.copy(), elements)
data_processed.drop(['SSA_Supp_m2g'], axis=1, inplace=True)

# ------------------------------------------------------------------------------------------------

# Save clean data
# molar_composition.to_csv('data/molar_composition.csv', index=False)
data_processed.to_csv('data/catalysts/data_processed.csv', index=False)
# print(data_processed.head(10))
