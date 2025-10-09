import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mplt
import ast
import re
from functions.utils import str_to_list


# Load processed deactivation data
data = pd.read_csv('data/data_deactivation.csv')

# Choose a DOI to visualize
doi = '101016jcattod201806019'
selected_data = data.loc[data['doi'] == doi, :].reset_index(drop=True)

# Colors
colors = ["#009688", "#AD1457", 'DarkBlue', 'DarkRed']

# Plot deactivation curves for the selected DOI
fig, ax = plt.subplots(figsize=(3.5, 3.5))
fig.tight_layout(pad=2.5, h_pad=4, w_pad=4)
for i, row in selected_data.iterrows():
    time = str_to_list(row['Time_h_t'])
    sty = str_to_list(row['STY_Acryl_mmolhg_t'])
    ax.plot(time, sty, marker='o', label=row['Link to excel'], c=colors[i])
ax.set(xlabel='Time / h', ylabel='STY$\mathdefault{_{Acryl}}$ / $\mathdefault{mmol \/ h^{-1} \/ g_{cat}^{-1}}$', xlim=(0, 25), ylim=(0, 2))
ax.legend(loc='upper right', frameon=False)
ax.xaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))
ax.xaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
ax.yaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=5))
fig.savefig('figures/deactivation_example.png', format='png', dpi=300)
plt.show()

