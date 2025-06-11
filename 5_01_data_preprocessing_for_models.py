import pandas as pd
import warnings
import os


#configurations
warnings.filterwarnings("ignore") # to get rid of warning while using pandas inplace methods


df = pd.read_excel("data/data_clean.xlsx")
df.drop(["doi"], axis=1, inplace=True) # drop the doi column

# Remove empty STYs, empty LSHVs and empty temperatures (these may be included later with little investigation of the corresponding papers. But there is just 3 of them.)
df.dropna(subset=["STY MA+AA (mmol/h/g)", 'LHSV [ml/h/g]','Temperature [K]'], inplace=True) 

# Change empty values of stabilizers to "No Stabilizer" and empty values of ratios to 0
na_dict = {
    "Stabilizer" : "No Stabilizer",
    "Ratio Stab/Fa" : 0
}

for item in na_dict.items():
    df[item[0]] = df[item[0]].fillna(item[1])

# Change the "N2:O2" to percentage of oxygen
for value in df["N2:O2"].unique():
    if not isinstance(value, int):
        if ":" in value:
            O_percentage = float(value.split(":")[1])*100/(float(value.split(":")[0]) + float(value.split(":")[1]))
            df["N2:O2"].replace(to_replace=value, value=O_percentage, inplace=True)
        else:
            df["N2:O2"].replace(to_replace=value, value=0, inplace=True)

df.rename(columns = {"N2:O2": "O_Percentage"}, inplace=True)

# Get rid of special characters and space from the columns names as some of the algorithms complain about it.
special_chars = ["[", "]", " ", ","]
for col in df.columns:
    if any(char in col for char in special_chars):
        cleaned_name = col
        for char in special_chars:
            cleaned_name = cleaned_name.replace(char, "_")
        df.rename(columns={col: cleaned_name}, inplace=True)

# Save the result to temp directory which will be read by the algorithms later
path = './data/tmp/'
if not os.path.isdir(path):
    os.mkdir(path)

df.to_csv(f"{path}processed_data.csv", index=False)