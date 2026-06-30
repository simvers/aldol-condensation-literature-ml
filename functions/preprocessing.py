import pandas as pd
from mendeleev import element
from sklearn.impute import KNNImputer


def calculate_composition(compositions):

    compositions = compositions.copy()

    # Columns
    supps = ['Supp_1', 'Supp_2']
    atoms = ['Atom_1', 'Atom_2', 'Atom_3', 'Atom_4']
    supp_mols = ['Supp_Mol_1', 'Supp_Mol_2']
    atom_mols = ['Atom_Mol_1', 'Atom_Mol_2', 'Atom_Mol_3', 'Atom_Mol_4']
    supp_oxs = ['Supp_Mass_Oxide_1', 'Supp_Mass_Oxide_2']
    atom_oxs = ['Atom_Mass_Oxide_1', 'Atom_Mass_Oxide_2', 'Atom_Mass_Oxide_3', 'Atom_Mass_Oxide_4']

    # Extract elements
    elements = sorted(set(x for x in compositions[supps + atoms].to_numpy().flatten() if pd.notna(x)))

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

    # ------------------------------------------------------------------------------------------------

    # Transform mass percentage of element to mass percentage of oxide, relative to support oxide mass
    # All these catalysts have a support
    for i in range(1, len(atoms)+1):

        # Find indexes
        indices = (compositions['Atom_' + str(i)].notna() & compositions['Atom_Mass_Elem_' + str(i)].notna())
        if not compositions.loc[indices, 'Atom_Mass_Oxide_' + str(i)].isna().all():
            raise ValueError(f"Atom_Mass_Oxide_{i} should be missing wherever Atom_Mass_Elem_{i} is reported")

        # Compute oxide content
        compositions.loc[indices, 'Atom_Mass_Oxide_' + str(i)] = (compositions.loc[indices, 'Atom_Mass_Elem_' + str(i)] / compositions.loc[indices, 'Atom_' + str(i)].map(mw_dict['mw_element'].get) * compositions.loc[indices, 'Atom_' + str(i)].map(mw_dict['mw_oxide'].get)).astype(float)

    # ------------------------------------------------------------------------------------------------

    # Transform mass percentage of oxide, relative to support oxide mass, to relative to total oxide mass

    # Identify rows
    indices = (compositions['Atom_1'].notna() & compositions['Atom_Mass_Oxide_1'].notna())
    if not compositions.loc[indices, 'Supp_Mass'].isna().all():
        raise ValueError("Supp_Mass should not be specified when Atom_Mass_Oxide_1 is derived from Atom_Mass_Elem_1")

    # Compute support oxide content, relative to total catalyst oxide mass
    compositions.loc[indices, 'Supp_Mass'] = (1/(1 + compositions.loc[indices, atom_oxs].sum(axis=1))).astype(float)

    # Mass percentage of oxide, relative to total catalyst oxide mass
    mass_composition.loc[indices, atom_content] += compositions.loc[indices, atom_oxs].mul(compositions.loc[indices, 'Supp_Mass'], axis=0).fillna(0).values

    # ------------------------------------------------------------------------------------------------

    # Transform molar ratio of support to mass percentage of oxide, relative to total support oxide mass

    # Find indexes where molar ratio is present
    indices = compositions['Supp_Mol_1'].notna()
    if not compositions.loc[indices, supp_oxs].isna().all(axis=None):
        raise ValueError("Supp_Mass_Oxide columns should be missing wherever Supp_Mol_1 is reported")

    # Compute oxide content
    supp_oxides = compositions.loc[indices, supp_mols].mul(compositions.loc[indices, supps].map(mw_dict['mw_oxide'].get).values)
    compositions.loc[indices, supp_oxs] = supp_oxides.div(supp_oxides.sum(axis=1), axis=0).values

    # ------------------------------------------------------------------------------------------------

    # Transform mass percentage of support oxide, relative to support oxide mass, to relative to total oxide mass

    # Identify rows
    indices = (compositions['Supp_Mass'].notna() & compositions['Supp_Mass_Oxide_1'].notna())

    # Mass percentage of support oxide, relative to total catalyst oxide mass
    mass_composition.loc[indices, supp_content] += compositions.loc[indices, supp_oxs].mul(compositions.loc[indices, 'Supp_Mass'], axis=0).fillna(0).values

    # ------------------------------------------------------------------------------------------------

    # Transform molar ratio of element to mass percentage of oxide, relative to total catalyst oxide mass
    # Not all of these catalysts have a support

    # Identify rows
    indices = compositions.loc[:, 'Atom_Mol_1'].notna()
    if not compositions.loc[indices, 'Supp_Mass'].notna().all():
        raise ValueError("Supp_Mass must be reported wherever Atom_Mol_1 is reported")

    # Mass percentage of oxide, relative to total catalyst oxide mass (from mass input)
    spec_oxides = compositions.loc[indices, atom_oxs].mul(compositions.loc[indices, 'Supp_Mass'], axis=0)

    # Mass of oxide (from mol input)
    mass_oxides = compositions.loc[indices, atom_mols].mul(compositions.loc[indices, atoms].map(mw_dict['mw_oxide'].get).values)

    # Mass percentage of oxide, relative to total catalyst oxide mass (from mol input)
    content_oxides = (mass_oxides.div(mass_oxides.sum(axis=1), axis=0)  # Divide by total oxide mass
                      .mul(1 - (spec_oxides.sum(axis=1) + compositions.loc[indices, 'Supp_Mass']), axis=0)  # Account for other specified oxides and support
    )

    # Sum mass percentage of oxides
    mass_composition.loc[indices, atom_content] += content_oxides.fillna(0).values
    mass_composition.loc[indices, atom_content] += spec_oxides.fillna(0).values

    # ------------------------------------------------------------------------------------------------

    # Safety check: sum of oxide content must be 1.0
    oxide_total = mass_composition[supp_content + atom_content].sum(axis=1)
    if not (((oxide_total > 0.99999) & (oxide_total < 1.00001)) == compositions['Supp_Mass'].notna()).all():
        raise ValueError("Oxide mass content does not sum to 1.0 for all catalysts with a reported Supp_Mass")

    # Calculate molar content
    mol_elements = mass_composition[supp_content + atom_content].div(mass_composition[supps + atoms].map(lambda x: mw_dict['mw_oxide'].get(x, 1)).values)
    molar_composition[supp_content + atom_content] = mol_elements.div(mol_elements.sum(axis=1).values, axis=0)

    # Safety check: sum of molar content must be 1.0
    oxide_total = molar_composition[supp_content + atom_content].sum(axis=1)
    if not (((oxide_total > 0.99999) & (oxide_total < 1.00001)) == compositions['Supp_Mass'].notna()).all():
        raise ValueError("Molar content does not sum to 1.0 for all catalysts with a reported Supp_Mass")

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

            # Calculate the molar flowrate in mmol/min/g
            ac_mmolming = condition['LHSV_mlhg']/vol_molac/60*1000
            fa_mmolming = ac_mmolming/condition['Ratio_Ac_Fa']
            meoh_mmolming = fa_mmolming*condition['Ratio_Stab_Fa'] if (condition['Stabilizer'] == 'MeOH') | (condition['Stabilizer'] == 'EtOH') else 0
            water_mmolming = fa_mmolming*condition['Ratio_Stab_Fa'] if condition['Stabilizer'] == 'Water' else 0

            # If formalin is used, consider the methanol (13wt.%) and water (100-37-13wt.%) present in it
            if condition['Fa_source'] == 'FORM':
                meoh_mmolming += fa_mmolming*fa['MW']/fa['Purity']*0.13/stab_properties.loc[stab_properties['Compound'] == 'MeOH', 'MW'].iloc[0]
                water_mmolming += fa_mmolming*fa['MW']/fa['Purity']*(1-0.37-0.13)/stab_properties.loc[stab_properties['Compound'] == 'Water', 'MW'].iloc[0]

            # Save molar flowrates
            conditions.loc[i, ['Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming']] = ac_mmolming, fa_mmolming, meoh_mmolming, water_mmolming

            # Fix Stab_Fa ratio
            conditions.loc[i, 'Ratio_Stab_Fa'] = (meoh_mmolming + water_mmolming) / fa_mmolming

    # Overwrite stabilizer when formalin is used as Fa source
    conditions.loc[conditions['Fa_source'] == 'FORM', 'Stabilizer'] = 'MeOH \n+ H$_2$O'

    # Calculate STY
    conditions['STY_Acryl_mmolhg'] = conditions['Ac_mmolming'] * conditions['Y_Acryl_Ac'] * 60

    # ------------------------------------------------------------------------------------------------

    # Check molarflowrates
    if all(col in conditions.columns for col in ['Fa_mmolmin', 'g_cat']):
        Fa_ratio = conditions['Fa_mmolming'] / (conditions['Fa_mmolmin']/conditions['g_cat'])
        Ac_ratio = conditions['Ac_mmolming'] / (conditions['Ac_mmolmin']/conditions['g_cat'])
        if not ((Fa_ratio.min() > 0.99) & (Fa_ratio.max() < 1.01)):
            raise ValueError("Computed Fa molar flowrate does not match the reported Fa_mmolmin/g_cat")
        if not ((Ac_ratio.min() > 0.99) & (Ac_ratio.max() < 1.01)):
            raise ValueError("Computed Ac molar flowrate does not match the reported Ac_mmolmin/g_cat")

    # Check STY and yield ratio
    sty_acryl_mmolhg = conditions['Fa_mmolming'] * conditions['Y_Acryl_Fa'] * 60
    ratio_STY_Ac_Fa = conditions['STY_Acryl_mmolhg']/sty_acryl_mmolhg
    if not ((ratio_STY_Ac_Fa.min() > 0.99) & (ratio_STY_Ac_Fa.max() < 1.01)):
        raise ValueError("STY computed from Ac and from Fa yields disagree")

    # Check STY
    if 'STY_Acryl_mmolhg_manual' in conditions.columns:
        # Check values
        ratio_STY = conditions['STY_Acryl_mmolhg']/conditions['STY_Acryl_mmolhg_manual']
        if not ((ratio_STY.max() < 1.01) & (ratio_STY.min() > 0.99)):
            raise ValueError("Computed STY does not match STY_Acryl_mmolhg_manual")

        # Check number of missing STY
        if not (conditions['STY_Acryl_mmolhg'].isna() == conditions['STY_Acryl_mmolhg_manual'].isna()).all:
            raise ValueError("Missing STY values do not align with STY_Acryl_mmolhg_manual")

    # Print report of missing values
    print('Number of observations with missing values: ', missing_values)
    print(' - Missing LHSV: ', missing_lhsv)

    return conditions


def impute_ssa(composition, elements):

    composition = composition.copy()

    # Compute ratio, and identify missing values
    composition.loc[:, 'SSA_ratio'] = composition.loc[:, 'SSA_m2g'] / composition.loc[:, 'SSA_Supp_m2g']
    missing_index = composition['SSA_m2g'].isna()

    # Fit imputer
    imputer = KNNImputer(n_neighbors=20, weights='distance')
    result = pd.DataFrame(imputer.fit_transform(composition[elements.to_list() + ['SSA_ratio', 'SSA_Supp_m2g']]),
                          columns=imputer.get_feature_names_out(), index=composition.index)

    # Impute values
    if not composition.loc[missing_index, 'SSA_m2g'].isna().all():
        raise ValueError("Expected all SSA_m2g values at missing_index to be NaN before imputation")
    composition.loc[missing_index, 'SSA_ratio'] = result.loc[missing_index, 'SSA_ratio']
    composition.loc[missing_index, 'SSA_m2g'] = composition.loc[missing_index, 'SSA_ratio'] * composition.loc[missing_index, 'SSA_Supp_m2g']

    return composition['SSA_m2g']


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

    # Define stab properties
    stab_properties = pd.DataFrame({
        "Compound": ['Water', 'MeOH', 'EtOH', 'None'],
        "MW": [18.02, 32.04, 46.07, 1],
        "Density": [1, 0.79, 0.79, 1],
        "Purity": [1, 1, 1, 1]
    })
    stab_properties.to_json('data/mol_properties/stab_properties.json')
