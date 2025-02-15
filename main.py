import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt


# Load data and extract columns
data = pd.read_excel(r"C:\Users\u0156112\OneDrive - KU Leuven\Shared_AC2GEN\Review\Review_V001\Catalysts.xlsx")
print(data.columns)
get_columns = ['Atom 1', 'Atom 2', 'Atom 3', 'Atom 4', 'Atom 5', 'Atom 6',
               'LHSV [ml/h/g]', 'Temperature [K]', 'STY MA+AA (mmol/h/g)']
data = data[get_columns]
data['Dummy'] = np.ones(len(data))
print(data.head(10))

columns = ['Atom 1', 'Atom 2', 'Atom 3', 'Atom 4', 'Atom 5', 'Atom 6']
table = data[columns]
pivoted_tables = [table.pivot_table(columns=column_i, index=table.index, values='Dummy', fill_value=0) for column_i in columns]
pivoted_table = pd.concat(pivoted_tables, axis=1, sort=True).fillna(0).groupby(level=0, axis=1).sum()
print(pivoted_table.columns)

