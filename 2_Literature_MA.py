import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
from Graph_YY_XS_YSTY import plot_xs_ysty, plot_yy, plot_ssty

# --------------------------------------------------
source = 'F'
target = 'MA'
save = True

columns = ['Catalyst type', 'Ac source', 'F source', 'Stabilizer', 'Conversion MAc', f'Selectivity {target} wrt MAc',
           f'Conversion {source}', f'Selectivity {target} wrt {source}', f'Yield {target} wrt MAc',
           f'Yield {target} wrt {source}', f'STY {target} (mmol/h/g)', 'Best']
df = pd.read_excel('I:/PhD/Literature/Catalysts.xlsx', sheet_name=1, usecols=columns)
print(df.shape)
print(df.head(5))

filter_type = 'Catalyst type'  # 'Catalyst type', 'Ac source', 'F source'
filter_data = sorted(df[filter_type].unique())  # cat_type.sort()
print(filter_data)

# Color
color = 'plasma'  # viridis, plasma, inferno, magma, cividis, twilight, gist_heat

# --------------------------------------------------

# Graph MAc vs Form yield

# Select data for yield graph
df_yy = df[df[f'Yield {target} wrt MAc'].notna() & df[f'Yield {target} wrt {source}'].notna()][columns]

# Plot
plot_yy(df_yy, filter_type, filter_data, color, source, save)

# --------------------------------------------------

# Graph on Formaldehyde performance
molecule = 'F'

# Select data for XS graph
df_xs_form = df[df[f'Conversion {molecule}'].notna() & df[f'Selectivity {target} wrt {molecule}'].notna()][columns]

# Select data for STY graph
df_ysty_form = df[df[f'Yield {target} wrt {molecule}'].notna() & df[f'STY {target} (mmol/h/g)'].notna()][columns]

# Plot
plot_xs_ysty(df_xs_form, df_ysty_form, filter_type, filter_data, color, molecule, target, source, save)

# --------------------------------------------------

# Graph on Methyl acetate performance
molecule = 'MAc'

# Select data for XS graph
df_xs_mac = df[df[f'Conversion {molecule}'].notna() & df[f'Selectivity {target} wrt {molecule}'].notna()][columns]

# Select data for STY graph
df_ysty_mac = df[df[f'Yield {target} wrt {molecule}'].notna() & df[f'STY {target} (mmol/h/g)'].notna()][columns]

# Plot
plot_xs_ysty(df_xs_mac, df_ysty_mac, filter_type, filter_data, color, molecule, target, source, save)

# --------------------------------------------------

# Graph on S vs STY
molecule = ['MAc', 'F']

# Select data for SSTY graph
df_ssty_mac = df[df[f'Selectivity {target} wrt {molecule[0]}'].notna() & df[f'STY {target} (mmol/h/g)'].notna()][columns]

# Select data for SSTY graph
df_ssty_form = df[df[f'Selectivity {target} wrt {molecule[1]}'].notna() & df[f'STY {target} (mmol/h/g)'].notna()][columns]


# Plot
plot_ssty(df_ssty_mac, df_ssty_form, filter_type, filter_data, color, molecule, target, source, save)

