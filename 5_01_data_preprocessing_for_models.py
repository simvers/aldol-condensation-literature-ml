import pandas as pd
import warnings
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pprint import pprint #This is temp remove this before commiting

#configurations
warnings.filterwarnings("ignore") # to get rid of warning while using pandas inplace methods

# Temppath to save resulsts
path = './data/tmp/'
if not os.path.isdir(path):
    os.mkdir(path)

df = pd.read_excel("data/data_clean.xlsx")
df.drop(["doi"], axis=1, inplace=True) # drop the doi column

# Remove empty STYs, empty LSHVs and empty temperatures (these may be included later with little investigation of the corresponding papers. But there is just 3 of them.)
df.dropna(subset=["STY MA+AA (mmol/h/g)", 'LHSV [ml/h/g]','Temperature [K]'], inplace=True) 

# Change empty values of stabilizers to "No Stabilizer" and empty values of ratios to 0
na_dict = {
    "Stabilizer" : 0,
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

df.to_csv(f"{path}processed_data.csv", index=False)

# Litttle bit more finetuning to get rid of few columns (date: 25/09/2025)
# Drop g_cat, Fa_mmol/min and Ac_mmol/min, Pressure_bar (The first three are captured by LHSV and Ratio_Stab/Fa)
df.drop(['Pressure__bar_', 'g_cat', 'Fa_mmol/min', 'Ac_mmol/min'], axis=1, inplace=True)
# Deal with Ac Source
molecules = {
    "MAc": {"smiles": "CC(=O)OC"},
    "HAc": {"smiles": "CC(=O)O"},
    "TRX": {"smiles": "C1OCOCO1"},
    "FORM": {"smiles": "C=O"},
    "DMM": {"smiles": "COCOC"},
    "MeOH": {"smiles": "CO"},
    }

for molecule in molecules.keys():
    mol = Chem.MolFromSmiles(molecules[molecule]["smiles"])
    vals = Descriptors.CalcMolDescriptors(mol)
    for prop in [
        "TPSA",
        # "MolWt",
        "MolLogP",
        'MinAbsPartialCharge',
        'MaxAbsPartialCharge',
        # "NumHDonors",
        # "NumHAcceptors"
        ]:
        molecules[molecule][prop] = vals[prop]

descriptor_df = pd.DataFrame(molecules).T
descriptor_df.drop(["smiles"], axis = 1, inplace=True)
scaler = MinMaxScaler(feature_range=(0,1)) # Scale the data before pca
descriptor_df_scaled = scaler.fit_transform(descriptor_df)

pca = PCA(n_components=1)
pca_scores = pca.fit_transform(descriptor_df_scaled)
molecul_pca_scores =dict(zip(molecules.keys(), pca_scores.flatten()))
df = df.map(lambda x: molecul_pca_scores[x] if x in molecul_pca_scores.keys() else x)
df.to_csv(f"{path}processed_data_mols_as_num.csv", index=False)


# Add reactants column
n1 = df["Ratio_Ac/Fa"]
n2 = df["Ratio_Stab/Fa"]
x_Ac = n1/(n1+n2+1)
x_Fa = 1/(n1+n2+1)
x_stab = n2/(n1+n2+1)

df["reactant_score"] = x_Ac * df["Ac_source"] + x_Fa * df["Fa_source"] + x_stab * df["Stabilizer"]
df.to_csv(f"{path}processed_data_with_reactant_score.csv", index=False)
