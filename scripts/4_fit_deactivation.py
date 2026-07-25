import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from src import utils, function_fitting, deactivation_modelling


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8

ROOT = Path(__file__).resolve().parents[1]
VERBOSE = True

DROP_Si = True
BEST_MODEL = 'exp2'
PALETTE = ["#009688", "#1565C0", "#AD1457"]
FIT_CMAP = 'cividis'
CV_CLIP = 10       # clip CV before plotting because 3-param models produce unbounded CV
DEBUG = False
N_PARAMS = 2


def plot_sty0_n_scatter(fig, gs_cell, data, clusters, palette, show_legend, panel_label):
    # STY0 vs n scatter with marginal KDEs — called for the full dataset and the oxygen-free subset
    sub_gs = gs_cell.subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0)
    ax_joint = fig.add_subplot(sub_gs[1, 0])
    ax_x = fig.add_subplot(sub_gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(sub_gs[1, 1], sharey=ax_joint)

    sns.scatterplot(ax=ax_joint, data=data, x='STY0', y='n', hue='Cluster_title', hue_order=clusters, palette=palette, alpha=0.6)
    if show_legend:
        sns.move_legend(ax_joint, loc='center left', bbox_to_anchor=(1.25, 0.5), frameon=False, title=None)
    else:
        ax_joint.legend().remove()
    ax_joint.set(xscale='symlog', xlim=(0, None), ylim=(0, 1),
                 xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='n /')
    ax_joint.set_yscale('symlog', linthresh=1e-2)
    ax_joint.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_joint.yaxis.set_major_formatter(mticker.ScalarFormatter())

    sns.kdeplot(ax=ax_x, data=data, x='STY0', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
    sns.kdeplot(ax=ax_y, data=data, y='n', hue='Cluster_title', hue_order=clusters, palette=palette, fill=True, alpha=0.25, clip=(0, None), cut=1)
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()

    # Number plot
    ax_joint.text(-0.2/5*6, 1.1/5*6, panel_label, fontsize=10, fontweight='bold', fontfamily='arial', transform=ax_joint.transAxes)


if __name__ == "__main__":

    # Load data
    suffix = '_noSi' if DROP_Si else ''
    data = utils.load_data(ROOT / f'data/processed/data_engineered{suffix}.csv')
    data['doi'] = utils.process_doi(data['doi'])
    initial_columns = data.columns.to_list()

    # Load deactivation data
    doi_digitized, sheet_digitized, df_digitized = deactivation_modelling.extract_deact_data(ROOT / 'data/raw/data_deactivation')
    if VERBOSE:
        print('Number of deactivation files: ', len(df_digitized))

    # Merge deact data in main df, and drop NaN values
    data_deactivation = deactivation_modelling.merge_deact_data(data, doi_digitized, sheet_digitized, df_digitized
    ).dropna(subset=['STY_Acryl_mmolhg_t']).copy()
    if VERBOSE:
        print('Number of rows with deactivation data: ', len(data_deactivation))

    # Plot raw deactivation curves
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, row in data_deactivation.iterrows():
        ax.plot(row['Time_h_t'], row['STY_Acryl_mmolhg_t'], marker='o')
    ax.set(xlim=(0, 20), ylim=(0, None),
           xlabel='Time-on-stream / h', ylabel='STY$_{Acryl}$ / mmol h$^{-1}$ g$^{-1}$')
    fig.savefig(ROOT / 'figures/deactivation_modelling/sty-tos_plot.png', dpi=600)

    # ---------------------------------------------

    # Fit deactivation functions

    # Model configurations
    models = {
        # 2-parameter models (STY_inf = 0)
        'pow2': {'name': 'Power eq.',       'func': function_fitting.power_law_model_2, 'p0': [0, 0], 'bounds': ([0, 0], [100, 2]),
                 'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
        'exp2': {'name': 'Exponential eq.', 'func': function_fitting.exp_model_2,       'p0': [0, 0], 'bounds': ([0, 0], [100, 2]),
                 'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
        'lan2': {'name': 'Langmuir eq.',    'func': function_fitting.langmuir_model_2,  'p0': [0, 0], 'bounds': ([0, 0], [100, 2]),
                 'param': np.empty((0, 2)), 'perr': np.empty((0, 2)), 'cv': np.empty((0, 2)), 'pcorr': np.empty((0, 2, 2))},
    } if N_PARAMS == 2 else {
        # 3-parameter models (STY_inf > 0)
        'pow3': {'name': 'Power eq.',       'func': function_fitting.power_law_model_3, 'p0': [0,0,0], 'bounds': ([0,0,0],[100,2,50]), 
                 'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))},
        'exp3': {'name': 'Exponential eq.', 'func': function_fitting.exp_model_3,       'p0': [0,0,0], 'bounds': ([0,0,0],[100,2,50]), 
                 'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))},
        'lan3': {'name': 'Langmuir eq.',    'func': function_fitting.langmuir_model_3,  'p0': [0,0,0], 'bounds': ([0,0,0],[100,2,50]),
                 'param': np.empty((0, 3)), 'perr': np.empty((0, 3)), 'cv': np.empty((0, 3)), 'pcorr': np.empty((0, 3, 3))},
    }

    # Fit all deactivation equations
    data_deactivation, models = deactivation_modelling.fit_all_models(data_deactivation, models, debug=DEBUG)

    # ---------------------------------------------

    # Plots comparing fitting functions
    fit_palette = sns.color_palette(plt.get_cmap(FIT_CMAP)(np.linspace(0, 1, len(models))), as_cmap=False, desat=0.8)

    # NRMSE distribution across models
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    for i, name in enumerate(models.keys()):
        sns.kdeplot(ax=ax, data=data_deactivation[f'{name}_nrmse'], color=fit_palette[i],
                    label=models[name]['name'], cut=0)
        ax.axvline(x=models[name]['NRMSE_mean'], color=fit_palette[i], linestyle='--', linewidth=0.75)
    ax.legend(frameon=False)
    ax.set(xlabel='NRMSE', xlim=(0, 0.2), ylim=(0, 14), xticks=[0, 0.05, 0.1, 0.15, 0.2])
    fig.savefig(ROOT / f'figures/deactivation_modelling/NRMSE_kdeplot_{N_PARAMS}params.svg', dpi=300, format='svg', bbox_inches='tight')

    # Parameter coefficient-of-variation scatter with marginal KDEs
    # Plot structure
    fig_spec = (2.5 * N_PARAMS + 1, 2.5) if N_PARAMS < 3 else (2.5 * 2 + 1, 2.5 * 2 + 1)
    grid_spec = [1, N_PARAMS] if N_PARAMS < 3 else [2, 2]
    fig = plt.figure(figsize=fig_spec)
    top_gs = list(fig.add_gridspec(*grid_spec, wspace=0.5, hspace=0.5))
    sub_gs_list, ax_joint, ax_x, ax_y = [], [], [], []
    for i in range(N_PARAMS):
        sub_gs_list.append(top_gs[i].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
        ax_joint.append(fig.add_subplot(sub_gs_list[i][1, 0]))
        ax_x.append(fig.add_subplot(sub_gs_list[i][0, 0], sharex=ax_joint[i]))
        ax_y.append(fig.add_subplot(sub_gs_list[i][1, 1], sharey=ax_joint[i]))
        ax_x[i].set_axis_off()
        ax_y[i].set_axis_off()
    # Scatter and kde plots
    for i, name in enumerate(models.keys()):
        for j in range(models[name]['param'].shape[1]):
            cv_j = np.clip(models[name]['cv'][:, j], 0, CV_CLIP)
            sns.scatterplot(ax=ax_joint[j], x=models[name]['param'][:, j], y=cv_j,
                            color=fit_palette[i], label=models[name]['name'])
            sns.kdeplot(ax=ax_x[j], x=models[name]['param'][:, j], color=fit_palette[i])
            sns.kdeplot(ax=ax_y[j], y=cv_j, color=fit_palette[i])
    # Axes formatting
    ax_joint[0].set(xlabel='STY$_{0}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='Coefficient of variation /',
                    xscale='symlog', yscale='symlog', xlim=(0, None), ylim=(0, 2))
    ax_joint[0].set_yscale('symlog', linthresh=0.05, linscale=0.3)
    ax_joint[0].legend().remove()
    ax_joint[0].text(-0.2/5*6, 1.1/5*6, 'a', fontsize=10, fontweight='bold', fontfamily='arial', transform=ax_joint[0].transAxes)
    ax_joint[1].set(xlabel='n /', ylabel='Coefficient of variation /',
                    xscale='symlog', yscale='symlog', xlim=(0, None), ylim=(0, 10))
    ax_joint[1].set_xscale('symlog', linthresh=0.01, linscale=0.3)
    ax_joint[1].set_yscale('symlog', linthresh=0.1, linscale=0.3)
    ax_joint[1].text(-0.2/5*6, 1.1/5*6, 'b', fontsize=10, fontweight='bold', fontfamily='arial', transform=ax_joint[1].transAxes)
    if len(ax_joint) == 3:
        ax_joint[1].legend().remove()
        ax_joint[2].set(xlabel='STY$_{\inf}$ / mmol h$^{-1}$ g$^{-1}$', ylabel='Coefficient of variation /',
                        xscale='symlog', yscale='symlog', xlim=(0, None), ylim=(0, 2))
        ax_joint[2].set_yscale('symlog', linthresh=0.05, linscale=0.3)
        ax_joint[2].text(-0.2/5*6, 1.1/5*6, 'c', fontsize=10, fontweight='bold', fontfamily='arial', transform=ax_joint[2].transAxes)
    ax_joint[-1].legend(frameon=False, bbox_to_anchor=(1.25, 0.5), loc='center left')
    fig.savefig(ROOT / f'figures/deactivation_modelling/CV_scatterplot_{N_PARAMS}params.svg', dpi=300, format='svg', bbox_inches='tight')

    if N_PARAMS == 3:
        exit()

    # Save data with the activity and deactivation metrics
    data_deactivation = data_deactivation.rename(columns={BEST_MODEL + '_STY0': 'STY0', BEST_MODEL + '_n': 'n'})
    data_deactivation.loc[:, initial_columns + ['STY0', 'n']].to_csv(ROOT / f'data/processed/data_deactivation{suffix}.csv', index=False)

    # STY-n scatter plot per cluster
    clusters = np.sort(data['Cluster_title'].unique())
    fig = plt.figure(figsize=(6, 2.5))
    gs = fig.add_gridspec(1, 2, wspace=0.5)
    plot_sty0_n_scatter(fig, gs[0, 0], data_deactivation, clusters, PALETTE, show_legend=False, panel_label='a')
    plot_sty0_n_scatter(fig, gs[0, 1], data_deactivation.loc[data_deactivation['O_content'] < 0.00001], clusters, PALETTE, show_legend=True, panel_label='b')
    fig.savefig(ROOT / f'figures/deactivation_modelling/{BEST_MODEL}_sty0-n_scatterplot.svg', dpi=300, format='svg', bbox_inches='tight')

    # Parity plot: STY0 prdicted vs STY experimental
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.scatterplot(ax=ax, data=data_deactivation, x='STY0', y='STY_Acryl_mmolhg',
                    hue='Cluster_title', hue_order=clusters, palette=PALETTE)
    ax.plot([0, 50], [0, 50], '-k', linewidth=0.5)
    sns.move_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.05), frameon=False, ncols=4, title=None)
    ax.set(xlim=(0, 50), ylim=(0, 50),
           xlabel='STY$_{\\mathdefault{0, model}}$ / mmol h$^{-1}$ g$^{-1}$',
           ylabel='STY$_{\\mathdefault{0, reported}}$ / mmol h$^{-1}$ g$^{-1}$')
    fig.savefig(ROOT / f'figures/deactivation_modelling/{BEST_MODEL}_sty0-sty_scatterplot.svg', dpi=300, format='svg')
