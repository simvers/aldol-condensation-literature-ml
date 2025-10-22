import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mendeleev


# Import processed data and elements
data = pd.read_csv('data/data_processed.csv', na_values=[''], keep_default_na=False)
elements = pd.read_csv('data/elements.csv', header=None).squeeze('columns').to_list()
composition = data.loc[:, elements]

composition.replace(0, pd.NA, inplace=True)
print(composition)

sns.kdeplot(composition, cut=0)
plt.show()
