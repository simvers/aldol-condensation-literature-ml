import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from functions.utils import filter_type, process_doi
from functions.deactivation_modelling import merge_conditions_deactivation, extract_df_deactivation_from_excel, deactivation_modelling
from functions.deactivation_models import *

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8
mycolor = 'cividis'

# Directory with digitized deactivation data, one xlsx per literature source
DIGITIZED_DATA_PATH = "/mnt/c/Users/u0156112/OneDrive - KU Leuven/Shared_AC2GEN/Review/Data/DigitizedData/"
# DIGITIZED_DATA_PATH = 'C:\\Users\\u0156112\\OneDrive - KU Leuven\\Shared_AC2GEN\\Review\\Data\\DigitizedData\\'

REMOVE_SI = True  # exclude Si content < 1.0, matches 3_features_engineering.py
BEST_MODEL = 'exp2'  # deactivation model used downstream (Stats/ML stages) as STY0/n


def load_engineered_data(sufix):
    data = pd.read_csv(f'data/catalysts/data_engineered{sufix}.csv')
    data['doi'] = process_doi(data['doi'])
    return data


def plot_all_deactivation_curves(data_deactivation):
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, row in data_deactivation.iterrows():
        ax.plot(row['Time_h_t'], row['STY_Acryl_mmolhg_t'], marker='o')
    ax.set(xlim=(0, 20), ylim=(0, None), xlabel='Time-on-stream / h', ylabel='STY$_{Acryl}$ / mmol h$^{-1}$ g$^{-1}$')
    fig.savefig('figures/deactivation_modelling/sty-tos_plot.png', dpi=600)


def build_model_configs():
    # 2-parameter variants (STY_inf fixed at 0, i.e. full deactivation as t -> inf)
    n_pow_exp, n_lan = 2, 2
    return {
        'pow2': {'name': 'Power eq.', 'func': power_law_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n_pow_exp]),
                 'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
        'exp2': {'name': 'Exponential eq.', 'func': exp_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n_pow_exp]),
                 'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
        'lan2': {'name': 'Langmuir eq.', 'func': langmuir_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n_lan]),
                 'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))}
    }
    # 3-parameter alternative (STY_inf free, i.e. deactivation plateaus above 0) - kept here for
    # reference, swap in if a model needs a non-zero steady-state STY:
    # n_pow_exp, n_lan = 2, 2
    # return {
    #     'pow3': {'func': power_law_model_3, 'p0': [0, 0, 0], 'bounds': ([0, 0, 0], [100, n_pow_exp, 50]),
    #              'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))},
    #     'pow2': {'func': power_law_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n_pow_exp]),
    #              'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
    #     'exp3': {'func': exp_model_3, 'p0': [0, 0, 0], 'bounds': ([0, 0, 0], [100, n_pow_exp, 50]),
    #              'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))},
    #     'exp2': {'func': exp_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n_pow_exp]),
    #              'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
    #     'lan3': {'func': langmuir_model_3, 'p0': [0, 0, 0], 'bounds': ([0, 0, 0], [100, n_lan, 50]),
    #              'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))},
    #     'lan2': {'func': langmuir_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, n_lan]),
    #              'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))}
    # }


def save_best_model_data(data_deactivation, initial_columns, best_model, sufix):
    # Rename the chosen model's fitted STY0/n so downstream scripts (Stats/ML) don't need
    # to know which deactivation model was used to compute them
    data_deactivation = data_deactivation.rename(columns={best_model + '_STY0': 'STY0', best_model + '_n': 'n'})
    data_deactivation.loc[:, initial_columns + ['STY0', 'n']].to_csv(f'data/catalysts/data_deactivation{sufix}.csv', index=False)
    return data_deactivation


def plot_sty0_n_scatter(data, clusters, palette, filename, show_legend):
    # STY0 vs n scatter with marginal KDEs - used both for the full dataset and for the
    # oxygen-free subset below, hence the helper.
    fig = plt.figure(figsize=(3, 3))
    gs = fig.add_gridspec(1, 1)
    sub_gs = gs[0, 0].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0)
    ax_joint = fig.add_subplot(sub_gs[1, 0])
    ax_x = fig.add_subplot(sub_gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(sub_gs[1, 1], sharey=ax_joint)

    sns.scatterplot(ax=ax_joint, data=data, x='STY0', y='n', hue='Cluster_title', hue_order=clusters, palette=palette)
    if show_legend:
        sns.move_legend(ax_joint, loc='center left', bbox_to_anchor=(1.25, 0.5), frameon=False, title=None)
    else:
        ax_joint.legend().remove()
    ax_joint.set(xscale='symlog', xlim=(0, None), ylim=(0, 1), xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')
    ax_joint.set_yscale('symlog', linthresh=1e-2)
    ax_joint.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_joint.yaxis.set_major_formatter(mticker.ScalarFormatter())

    sns.kdeplot(ax=ax_x, data=data, x='STY0', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
    sns.kdeplot(ax=ax_y, data=data, y='n', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()

    fig.savefig(filename, dpi=300, format='svg', bbox_inches='tight')


def plot_sty0_sty_parity(data_deactivation, clusters, palette, best_model):
    # Parity plot: STY0 predicted by the deactivation model fit vs the directly reported STY
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.scatterplot(ax=ax, data=data_deactivation, x='STY0', y='STY_Acryl_mmolhg', hue='Cluster_title', hue_order=clusters, palette=palette)
    ax.plot([0, 50], [0, 50], '-k', linewidth=0.5)
    sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncols=4, title=None)
    ax.set(xlim=(0, 50), ylim=(0, 50), xlabel='STY$_{\\mathdefault{0, model}}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='STY$_{\\mathdefault{0, reported}}$ / mmol h$^{-1}$ g$^{-1}$')
    fig.savefig(f'figures/deactivation_modelling/{best_model}_sty0-sty_scatterplot.svg', dpi=300, format='svg')


def main():
    sufix = '_noSi' if REMOVE_SI else ''

    # Import digitized data
    doi_digitized, sheet_digitized, df_digitized = extract_df_deactivation_from_excel(DIGITIZED_DATA_PATH)
    print('Number of deactivation files: ', len(df_digitized))

    # Import engineered/clustered data
    data_clustered = load_engineered_data(sufix)
    initial_columns = data_clustered.columns.to_list()

    # Merge data_clustered with list of df_deactivation (creates time-dependent performance
    # columns containing numpy arrays), and drop observations without STY deactivation data
    data_deactivation = merge_conditions_deactivation(data_clustered, doi_digitized, sheet_digitized, df_digitized).dropna(subset=['STY_Acryl_mmolhg_t']).copy()
    print('Number of rows with deactivation data: ', len(data_deactivation))

    plot_all_deactivation_curves(data_deactivation)

    # Fit all deactivation models
    models = build_model_configs()
    data_deactivation, models = deactivation_modelling(data_deactivation, models)

    # Save data using the chosen model's fitted STY0/n
    data_deactivation = save_best_model_data(data_deactivation, initial_columns, BEST_MODEL, sufix)

    # Make colors
    clusters = np.sort(data_clustered['Cluster_title'].unique())
    cmap = plt.get_cmap(mycolor)
    palette = cmap(np.linspace(0, 1, len(clusters)))
    palette = sns.color_palette(palette, as_cmap=False, desat=0.8)

    plot_sty0_n_scatter(data_deactivation, clusters, palette,
                         f'figures/deactivation_modelling/{BEST_MODEL}_sty0-n_scatterplot.svg', show_legend=False)

    data_no_oxygen = data_deactivation.loc[data_deactivation['O_content'] < 0.00001, :]
    plot_sty0_n_scatter(data_no_oxygen, clusters, palette,
                         f'figures/deactivation_modelling/{BEST_MODEL}_sty0-n_scatterplot_no_oxygen.svg', show_legend=True)

    plot_sty0_sty_parity(data_deactivation, clusters, palette, BEST_MODEL)


if __name__ == "__main__":
    main()
