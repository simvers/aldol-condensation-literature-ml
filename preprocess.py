import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt


# Load data and extract columns
data = pd.read_excel(r"C:\Users\u0156112\OneDrive - KU Leuven\Shared_AC2GEN\Review\Review_V001\Catalysts.xlsx")
print(data.columns)
get_columns = ['Atom 1', 'Atom 2', 'Atom 3', 'Atom 4', 'Atom 5', 'Atom 6',
               'Ac source', 'Fa source', 'Stabilizer', 'Ratio Ac/Fa', 'Ratio MeOH/Fa', 'N2:O2',
               'LHSV [ml/h/g]', 'Temperature [K]', 'Pressure [bar]', 'g cat', 'Fa mmol/min', 'Ac mmol/min',
               'STY MA+AA (mmol/h/g)']
data = data[get_columns]

# Encode atom composition
atom_columns = ['Atom 1', 'Atom 2', 'Atom 3', 'Atom 4', 'Atom 5', 'Atom 6']
data['Dummy'] = np.ones(len(data))
pivoted_tables = [data.pivot_table(columns=column_i, index=data.index, values='Dummy', fill_value=0) for column_i in atom_columns]
composition_clean = pd.concat(pivoted_tables, axis=1, sort=True).fillna(0).T.groupby(level=0).sum().T
print(composition_clean.columns)

# Concat encoded atom composition and rest of data
data_clean = pd.concat(
    [composition_clean, 
     data[[column for column in get_columns if column not in atom_columns]]], 
    axis=1
)
print(data_clean.head(10))

# Save clean data
composition_clean.to_excel('data/composition_clean.xlsx', index=False)
data_clean.to_excel('data/data_clean.xlsx', index=False)

# Most popular elements
major_elements = composition_clean.sum(axis=0).sort_values(ascending=False)
print(major_elements)
