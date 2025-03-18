import pandas as pd


def calculate_molarflowrates(conditions):

    # Import mol properties
    ac_properties = pd.read_json('data/mol_properties/ac_properties.json')
    fa_properties = pd.read_json('data/mol_properties/fa_properties.json')
    stab_properties = pd.read_json('data/mol_properties/stab_properties.json')

    # Initialize new columns
    conditions = conditions.reindex(conditions.columns.tolist() + ['Density','Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming'], axis=1)

    for i, condition in conditions.iterrows():
        
        # Check if required info is reported
        if condition[['Ac source', 'Fa source', 'Stabilizer', 'Ratio Ac/Fa', 'Ratio Stab/Fa', 'LHSV [ml/h/g]']].isna().any():
            conditions.loc[i, ['Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming']] = pd.NA
        else:
            # Extract reactants properties
            ac = ac_properties.loc[ac_properties['Compound'] == condition['Ac source']].iloc[0]
            fa = fa_properties.loc[fa_properties['Compound'] == condition['Fa source']].iloc[0]
            stab = stab_properties.loc[stab_properties['Compound'] == condition['Stabilizer']].iloc[0]

            # Calculate mass and volume of reactant mixture that contains 1 mol of Ac
            mass_molac = ac['MW']/ac['Purity'] + fa['MW']/fa['Purity']/condition['Ratio Ac/Fa'] + stab['MW']/stab['Purity']*condition['Ratio Stab/Fa']/condition['Ratio Ac/Fa']
            vol_molac = ac['MW']/ac['Purity']/ac['Density'] + fa['MW']/fa['Purity']/condition['Ratio Ac/Fa']/fa['Density'] + stab['MW']/stab['Purity']*condition['Ratio Stab/Fa']/condition['Ratio Ac/Fa']/stab['Density']
            conditions.loc[i, 'Density'] = mass_molac/vol_molac

            # Calculate the molar flowrate in mmol/min/g
            ac_mmolming = condition['LHSV [ml/h/g]']/vol_molac/60*1000
            fa_mmolming = ac_mmolming/condition['Ratio Ac/Fa']
            meoh_mmolming = fa_mmolming*condition['Ratio Stab/Fa'] if condition['Stabilizer'] == 'MeOH' else 0
            water_mmolming = fa_mmolming*condition['Ratio Stab/Fa'] if condition['Stabilizer'] == 'Water' else 0

            # If formalin is used, consider the water (13wt.%) and methanol (100-37-13wt.%) present in it
            if condition['Fa source'] == 'Formalin':
                meoh_mmolming += fa_mmolming*fa['MW']/fa['Purity']*0.13/stab_properties.loc[stab_properties['Compound'] == 'Methanol', 'MW'].iloc[0]
                water_mmolming += fa_mmolming*fa['MW']/fa['Purity']*(1-0.37-0.13)/stab_properties.loc[stab_properties['Compound'] == 'Water', 'MW'].iloc[0]
            
            conditions.loc[i, ['Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming']] = ac_mmolming, fa_mmolming, meoh_mmolming, water_mmolming

    return conditions


if __name__ == "__main__":

    # Define Ac properties
    ac_properties = pd.DataFrame({
        "Compound": ['HAc', 'MAc', 'EAc'],
        "MW": [60.05, 74.08, 88.11],
        "Density": [1.05, 0.93, 0.90],
        "Purity": [1, 1, 1]
    })
    ac_properties.to_json('data/mol_properties/ac_properties.json')

    # Define Fa properties
    fa_properties = pd.DataFrame({
        "Compound": ['DMM', 'Formalin', 'Trioxane'],
        "MW": [76.09, 30.03, 30.03],
        "Density": [0.86, 1.09, 1000],
        "Purity": [1, 0.37, 1]
    })
    fa_properties.to_json('data/mol_properties/fa_properties.json')

    # Define stab proeprties
    stab_properties = pd.DataFrame({
        "Compound": ['Water', 'Methanol', 'Ethanol', 'None'],
        "MW": [18.02, 32.04, 46.07, 1],
        "Density": [1, 0.79, 0.79, 1],
        "Purity": [1, 1, 1, 1]
    })
    stab_properties.to_json('data/mol_properties/stab_properties.json')

    # Define mock conditions
    conditions = pd.DataFrame({
        "Ac source": ['MAc', 'HAc', 'HAc'],
        "Fa source": ['Trioxane', 'DMM', 'DMM'],
        "Stabilizer": ['Methanol', 'None', 'None'],
        "Ratio Ac/Fa": [1, 2.5, 0],
        "Ratio Stab/Fa": [2, 0, 0],
        "LHSV [ml/h/g]": [3.08, 0.44, pd.NA],
        "mass_cat": [1.8, 3, pd.NA]
    })

    # MAc	Trioxane	Methanol	1	2	/	3.08	633	1	1.8	0.575714004	0.575714004	0.699607657
    # HAc	DMM	/	2.5	0	97.75:2.25	0.44	633	1	3	0.094995781	0.237489453	0.847362367

    # Calculate molar flowrates
    conditions = calculate_molarflowrates(conditions=conditions)

    # Print result
    print(conditions.loc[0, ['Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming']]*1.8)
    print(conditions.loc[1, ['Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming']]*3)
    print(conditions.loc[2, ['Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming']]*1)



