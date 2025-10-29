import numpy as np
import pandas as pd
from mendeleev import element 
from rdkit.Chem.rdchem import GetPeriodicTable

a = ['a_', 'b', 'd_', 'e']
b = ['_']
print(a-b)

exit()

period_table = GetPeriodicTable()

elements = pd.read_csv('data/elements.csv', header=None).squeeze('columns').to_list()
print(elements)
for elem in elements:
    elem_obj = element(elem)
    print(elem, elem_obj.covalent_radius_cordero)

    atomic_number = period_table.GetAtomicNumber(elem)
    print(elem, period_table.GetRcovalent(atomic_number)*100)

