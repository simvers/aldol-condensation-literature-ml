import numpy as np
import pandas as pd
from mendeleev import element


def calculate_composition(compositions):

    # Columns
    supps = ['Supp_1', 'Supp_2']
    atoms = ['Atom_1', 'Atom_2', 'Atom_3', 'Atom_4']
    supp_mols = ['Supp_Mol_1', 'Supp_Mol_2']
    atom_mols = ['Atom_Mol_1', 'Atom_Mol_2', 'Atom_Mol_3', 'Atom_Mol_4']
    supp_oxs = ['Supp_Mass_Oxide_1', 'Supp_Mass_Oxide_2']
    atom_oxs = ['Atom_Mass_Oxide_1', 'Atom_Mass_Oxide_2', 'Atom_Mass_Oxide_3', 'Atom_Mass_Oxide_4']

    # Extract elements
    elements = sorted(set(x for x in compositions[supps + atoms].to_numpy().flatten() if pd.notna(x)))

    # One-hot encode elements
    # compositions['Dummy'] = np.ones(len(compositions))
    # pivoted_tables = [compositions.pivot_table(columns=column_i, index=compositions.index, values='Dummy', fill_value=0) for column_i in atom_columns]
    # composition_clean = pd.concat(pivoted_tables, axis=1, sort=True).fillna(0).T.groupby(level=0).sum().T

    # Extract metal and oxide mw
    mw_dict = {"mw_element": {}, "mw_oxide": {}, "oxi_state": {}}
    oxygen = element('O')
    for sym in elements:
        elem = element(sym)
        mw_dict["mw_element"][sym] = elem.mass
        mw_dict["mw_oxide"][sym] = elem.mass + (elem.oxistates[-1]/2)*oxygen.mass
        mw_dict["oxi_state"][sym] = elem.oxistates[-1]

    # Initialize final df
    supp_content = ['Supp_%_1', 'Supp_%_2']
    atom_content = ['Atom_%_1', 'Atom_%_2', 'Atom_%_3', 'Atom_%_4']
    mass_composition = compositions.loc[:, supps + atoms].copy()
    mass_composition[supp_content + atom_content] = 0.0
    molar_composition = mass_composition.copy()
    # print(molar_composition)

    # ------------------------------------------------------------------------------------------------

    # Transform mass percentage of element to mass percentage of oxide, relative to support oxide mass
    # All these catalysts have a support
    for i in range(1, len(atoms)+1):

        # Find indexes
        indices = (compositions['Atom_' + str(i)].notna() & compositions['Atom_Mass_Elem_' + str(i)].notna())
        assert compositions.loc[indices, 'Atom_Mass_Oxide_' + str(i)].isna().all()

        # Compute oxide content
        compositions.loc[indices, 'Atom_Mass_Oxide_' + str(i)] = (compositions.loc[indices, 'Atom_Mass_Elem_' + str(i)] / compositions.loc[indices, 'Atom_' + str(i)].map(mw_dict['mw_element'].get) * compositions.loc[indices, 'Atom_' + str(i)].map(mw_dict['mw_oxide'].get)).astype(float)
    
    # ------------------------------------------------------------------------------------------------

    # Transform mass percentage of oxide, relative to support oxide mass, to relative to total oxide mass

    # Identify rows
    indices = (compositions['Atom_1'].notna() & compositions['Atom_Mass_Oxide_1'].notna())
    assert compositions.loc[indices, 'Supp_Mass'].isna().all()  # Check that support mass content is not specified

    # Compute support oxide content, relative to total catalyst oxide mass
    compositions.loc[indices, 'Supp_Mass'] = (1/(1 + compositions.loc[indices, atom_oxs].sum(axis=1))).astype(float)

    # Mass percentage of oxide, relative to total catalyst oxide mass
    mass_composition.loc[indices, atom_content] += compositions.loc[indices, atom_oxs].mul(compositions.loc[indices, 'Supp_Mass'], axis=0).fillna(0).values

    # ------------------------------------------------------------------------------------------------

    # Transform molar ratio of support to mass percentage of oxide, relative to total support oxide mass

    # Find indexes where molar ratio is present
    indices = compositions['Supp_Mol_1'].notna()
    assert compositions.loc[indices, supp_oxs].isna().all(axis=None)

    # Compute oxide content
    supp_oxides = compositions.loc[indices, supp_mols].mul(compositions.loc[indices, supps].map(mw_dict['mw_oxide'].get).values)
    compositions.loc[indices, supp_oxs] = supp_oxides.div(supp_oxides.sum(axis=1), axis=0).values
    
    # ------------------------------------------------------------------------------------------------

    # Transform mass percentage of support oxide, relative to support oxide mass, to relative to total oxide mass

    # Identify rows
    indices = (compositions['Supp_Mass'].notna() & compositions['Supp_Mass_Oxide_1'].notna())
    # indices = compositions['Supp_Mass_Oxide_1'].notna()
    # print(compositions.loc[compositions['Supp_Mass_Oxide_1'].notna() & compositions['Supp_Mass'].isna(), ['Supp_Mass', 'Supp_1', 'Supp_Mass_Oxide_1']])
    # assert compositions.loc[indices, 'Supp_Mass'].notna().all()  # Some Supp_Mass are not specified

    # Mass percentage of support oxide, relative to total catalyst oxide mass
    mass_composition.loc[indices, supp_content] += compositions.loc[indices, supp_oxs].mul(compositions.loc[indices, 'Supp_Mass'], axis=0).fillna(0).values
    
    # ------------------------------------------------------------------------------------------------

    # Transform molar ratio of element to mass percentage of oxide, relative to total catalyst oxide mass
    # Not all of these catalysts have a suport

    # Identify rows
    indices = compositions.loc[:, 'Atom_Mol_1'].notna()
    assert compositions.loc[indices, 'Supp_Mass'].notna().all()  # Check that support mass is always reported

    # Mass percentage of oxide, relative to total catalyst oxide mass (from mass input)
    spec_oxides = compositions.loc[indices, atom_oxs].mul(compositions.loc[indices, 'Supp_Mass'], axis=0)

    # Mass of oxide (from mol input)
    mass_oxides = compositions.loc[indices, atom_mols].mul(compositions.loc[indices, atoms].map(mw_dict['mw_oxide'].get).values)

    # Mass percentage of oxide, relative to total catalyst oxide mass (from mol input)
    content_oxides = (mass_oxides.div(mass_oxides.sum(axis=1), axis=0)  # Divide by total oxide mass
                      .mul(1 - (spec_oxides.sum(axis=1) + compositions.loc[indices, 'Supp_Mass']), axis=0)  # Account for other specified oxides and support
    )

    # Sum mass percentage of oxides
    # content_oxides.loc[:, :] = np.nansum(np.dstack((spec_oxides.values, content_oxides.values)), -1)
    mass_composition.loc[indices, atom_content] += content_oxides.fillna(0).values
    mass_composition.loc[indices, atom_content] += spec_oxides.fillna(0).values
    
    # ------------------------------------------------------------------------------------------------

    # Show oxide mass content
    # print(mass_composition.to_string())

    # Safety check: sum of oxide content must be 1.0
    sum = mass_composition[supp_content + atom_content].sum(axis=1)
    assert (((sum > 0.99999) & (sum < 1.00001)) == compositions['Supp_Mass'].notna()).all()

    # Calculate molar content
    mol_elements = mass_composition[supp_content + atom_content].div(mass_composition[supps + atoms].map(lambda x: mw_dict['mw_oxide'].get(x, 1)).values)
    molar_composition[supp_content + atom_content] = mol_elements.div(mol_elements.sum(axis=1).values, axis=0)

    # Safety check: sum of molar content must be 1.0
    sum = molar_composition[supp_content + atom_content].sum(axis=1)
    assert (((sum > 0.99999) & (sum < 1.00001)) == compositions['Supp_Mass'].notna()).all()
    
    # ------------------------------------------------------------------------------------------------

    # Pivot composition

    # Pivot each element in catalysts and its corresponding molar content
    pivoted_tables = [molar_composition.pivot_table(columns=elem_i, index=molar_composition.index, values=values_i, fill_value=0) 
                      for elem_i, values_i in zip(supps + atoms, supp_content + atom_content)]
    # Concatenate all pivoted tables
    molar_composition_clean = pd.concat(pivoted_tables, axis=1, sort=True).fillna(0).T.groupby(level=0).sum().T

    return molar_composition_clean
    

def calculate_molarflowrates(conditions):

    # Import mol properties
    ac_properties = pd.read_json('data/mol_properties/ac_properties.json')
    fa_properties = pd.read_json('data/mol_properties/fa_properties.json')
    stab_properties = pd.read_json('data/mol_properties/stab_properties.json')

    # Initialize new columns
    conditions = conditions.reindex(conditions.columns.tolist() + ['Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming'], axis=1)

    # Initialize missing values
    missing_values, missing_source, missing_ratio, missing_lhsv = 0, 0, 0, 0

    for i, condition in conditions.iterrows():
        
        # Check if required info is reported
        if condition[['Ac_source', 'Fa_source', 'Stabilizer', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'LHSV_mlhg']].isna().any():
            missing_values += 1
            if condition[['Ac_source', 'Fa_source', 'Stabilizer']].isna().any():
                missing_source += 1
            if condition[['Ratio_Ac_Fa', 'Ratio_Stab_Fa']].isna().any():
                missing_ratio += 1
            if condition[['LHSV_mlhg']].isna().any():
                missing_lhsv += 1
            conditions.loc[i, ['Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming']] = pd.NA
        else:
            # Extract reactants properties
            ac = ac_properties.loc[ac_properties['Compound'] == condition['Ac_source']].iloc[0]
            fa = fa_properties.loc[fa_properties['Compound'] == condition['Fa_source']].iloc[0]
            stab = stab_properties.loc[stab_properties['Compound'] == condition['Stabilizer']].iloc[0]

            # Calculate volume of reactant mixture that contains 1 mol of Ac
            vol_molac = ac['MW']/ac['Purity']/ac['Density'] + fa['MW']/fa['Purity']/condition['Ratio_Ac_Fa']/fa['Density'] + stab['MW']/stab['Purity']*condition['Ratio_Stab_Fa']/condition['Ratio_Ac_Fa']/stab['Density']
            # mass_molac = ac['MW']/ac['Purity'] + fa['MW']/fa['Purity']/condition['Ratio_Ac_Fa'] + stab['MW']/stab['Purity']*condition['Ratio_Stab_Fa']/condition['Ratio_Ac_Fa']
            # conditions.loc[i, 'Density'] = mass_molac/vol_molac

            # Calculate the molar flowrate in mmol/min/g
            ac_mmolming = condition['LHSV_mlhg']/vol_molac/60*1000
            fa_mmolming = ac_mmolming/condition['Ratio_Ac_Fa']
            meoh_mmolming = fa_mmolming*condition['Ratio_Stab_Fa'] if condition['Stabilizer'] == 'MeOH' else 0
            water_mmolming = fa_mmolming*condition['Ratio_Stab_Fa'] if condition['Stabilizer'] == 'Water' else 0

            # If formalin is used, consider the methanol (13wt.%) and water (100-37-13wt.%) present in it
            if condition['Fa_source'] == 'FORM':
                meoh_mmolming += fa_mmolming*fa['MW']/fa['Purity']*0.13/stab_properties.loc[stab_properties['Compound'] == 'MeOH', 'MW'].iloc[0]
                water_mmolming += fa_mmolming*fa['MW']/fa['Purity']*(1-0.37-0.13)/stab_properties.loc[stab_properties['Compound'] == 'Water', 'MW'].iloc[0]
            
            conditions.loc[i, ['Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming']] = ac_mmolming, fa_mmolming, meoh_mmolming, water_mmolming


    # Overwrite stabilizer when formalin is used as Fa source
    conditions.loc[conditions['Fa_source'] == 'FORM', 'Stabilizer'] = 'MeOH + H$_2$O'
    # conditions = conditions.fillna({'Stabilizer': 'None', "Ratio_Stab_Fa" : 0})

    # Calculate STY
    conditions['STY_Acryl_mmolhg'] = conditions['Ac_mmolming'] * conditions['Y_Acryl_Ac'] * 60

    # ------------------------------------------------------------------------------------------------
    
    # Check molarflowrates
    if all(col in conditions.columns for col in ['Fa_mmolmin', 'g_cat']):
        Fa_ratio = conditions['Fa_mmolming'] / (conditions['Fa_mmolmin']/conditions['g_cat'])
        Ac_ratio = conditions['Ac_mmolming'] / (conditions['Ac_mmolmin']/conditions['g_cat'])
        assert (Fa_ratio.min() > 0.99) & (Fa_ratio.max() < 1.01)
        assert (Ac_ratio.min() > 0.99) & (Ac_ratio.max() < 1.01)

    # Check STY and yield ratio
    sty_acryl_mmolhg = conditions['Fa_mmolming'] * conditions['Y_Acryl_Fa'] * 60
    ratio_STY_Ac_Fa = conditions['STY_Acryl_mmolhg']/sty_acryl_mmolhg
    assert (ratio_STY_Ac_Fa.min() > 0.99) & (ratio_STY_Ac_Fa.max() < 1.01)

    # Check STY
    if 'STY_Acryl_mmolhg_manual' in conditions.columns:
        # Check values
        ratio_STY = conditions['STY_Acryl_mmolhg']/conditions['STY_Acryl_mmolhg_manual']
        assert ((ratio_STY.max() < 1.01) & (ratio_STY.min() > 0.99))

        # Check number of missing STY
        assert (conditions['STY_Acryl_mmolhg'].isna() == conditions['STY_Acryl_mmolhg_manual'].isna()).all  # no_STY == 82

    # Print report of missing values
    print('Number of missing values: ', missing_values)
    print(' - Missing source: ', missing_source)
    print(' - Missing ratio: ', missing_ratio)
    print(' - Missing LHSV: ', missing_lhsv)

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
        "Compound": ['DMM', 'FORM', 'TRX', 'MeOH'],
        "MW": [76.09, 30.03, 30.03, 32.04],
        "Density": [0.86, 1.09, 1000, 0.79],
        "Purity": [1, 0.37, 1, 1]
    })
    fa_properties.to_json('data/mol_properties/fa_properties.json')

    # Define stab proeprties
    stab_properties = pd.DataFrame({
        "Compound": ['Water', 'MeOH', 'EtOH', 'None'],
        "MW": [18.02, 32.04, 46.07, 1],
        "Density": [1, 0.79, 0.79, 1],
        "Purity": [1, 1, 1, 1]
    })
    stab_properties.to_json('data/mol_properties/stab_properties.json')
