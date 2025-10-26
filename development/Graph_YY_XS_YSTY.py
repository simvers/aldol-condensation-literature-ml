import matplotlib.pyplot as plt
import matplotlib as mplt
import numpy as np

mplt.rcParams['font.sans-serif'] = 'Arial'
mplt.rcParams['font.family'] = 'sans-serif'
mplt.rcParams['mathtext.fontset'] = 'dejavusans'  # 'dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom'


def plot_yy(df_yy, filter_type, filter_data, color, source, save):

    # Plot structure
    fig, axes = plt.subplots(1, 1, figsize=(3.5, 3.5))
    fig.tight_layout(pad=2, h_pad=4, w_pad=4)

    # Colors
    color_n = np.linspace(0, 1, len(filter_data))
    colormap = mplt.colormaps.get_cmap(color)

    # Plot y vs y
    ax = axes
    for i in range(len(filter_data)):
        ax.scatter(df_yy[(df_yy[filter_type] == filter_data[i]) & (df_yy['Best'] == 0)]['Yield MA wrt MAc'] * 100,
                   df_yy[(df_yy[filter_type] == filter_data[i]) & (df_yy['Best'] == 0)][f'Yield MA wrt {source}'] * 100,
                   edgecolor=colormap(color_n[i]), facecolor='none')
    for i in range(len(filter_data)):
        ax.scatter(df_yy[(df_yy[filter_type] == filter_data[i]) & (df_yy['Best'] == 1)]['Yield MA wrt MAc'] * 100,
                   df_yy[(df_yy[filter_type] == filter_data[i]) & (df_yy['Best'] == 1)][f'Yield MA wrt {source}'] * 100,
                   color=colormap(color_n[i]), label=filter_data[i])
    # for i in range(len(filter_data)):
    #     ax.scatter(df_yy[df_yy[filter_type] == filter_data[i]]['Yield MA wrt MAc'] * 100,
    #                df_yy[df_yy[filter_type] == filter_data[i]][f'Yield MA wrt {source}'] * 100,
    #                color=colormap(color_n[i]), label=filter_data[i])

    # Axis scale
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 100))

    # Ticks
    ax.minorticks_on()
    ax.xaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
    ax.xaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
    ax.yaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))

    # Labels
    # ax.set_title('Yield based on methyl acetate vs formaldehyde', fontsize='large')
    ax.set_xlabel(rf'$\eta_\mathrm{{MA}}$ based on MAc / %', fontsize='medium')
    ax.set_ylabel(rf'$\eta_\mathrm{{MA}}$ based on {source} / %', fontsize='medium')
    ax.legend(loc='lower right', fontsize='small', frameon=False, markerscale=0.5)

    plt.show()
    if save:
        fig.savefig(f'I:/PhD/Literature/Figures/{filter_type}/Cat_Perf_in_{source}_{filter_type}.png', dpi=300)


def plot_xs_ysty(df_xs, df_ysty, filter_type, filter_data, color, molecule, target, source, save):

    # Plot structure
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.tight_layout(pad=2, h_pad=4, w_pad=4)

    # Colors
    color_n = np.linspace(0, 1, len(filter_data))
    colormap = mplt.colormaps.get_cmap(color)

    # Yield lines
    x = np.arange(1, 95, 1)
    y_10 = 10 / x * 100
    y_25 = 25 / x * 100
    y_50 = 50 / x * 100

    # Plot X vs S
    ax = axes[0]
    for i in range(len(filter_data)):
        ax.scatter(df_xs[(df_xs[filter_type] == filter_data[i]) & (df_xs['Best'] == 0)][f'Conversion {molecule}'] * 100,
                   df_xs[(df_xs[filter_type] == filter_data[i]) & (df_xs['Best'] == 0)][f'Selectivity {target} wrt {molecule}'] * 100,
                   edgecolor=colormap(color_n[i]), facecolor='none')
    for i in range(len(filter_data)):
        ax.scatter(df_xs[(df_xs[filter_type] == filter_data[i]) & (df_xs['Best'] == 1)][f'Conversion {molecule}'] * 100,
                   df_xs[(df_xs[filter_type] == filter_data[i]) & (df_xs['Best'] == 1)][f'Selectivity {target} wrt {molecule}'] * 100,
                   color=colormap(color_n[i]), label=filter_data[i])
        # ax.scatter(df_xs[df_xs[filter_type] == filter_data[i]][f'Conversion {molecule}'] * 100,
        #            df_xs[df_xs[filter_type] == filter_data[i]][f'Selectivity {target} wrt {molecule}'] * 100,
        #            color=colormap(color_n[i]), label=filter_data[i])
    ax.plot(x, y_10, 'k-', linewidth=0.5)
    ax.plot(x, y_25, 'k-.', linewidth=0.5)
    ax.plot(x, y_50, 'k--', linewidth=0.5)

    # Axis scale
    ax.set_xlim((0, 100))
    ax.set_ylim((0, 100))

    # Ticks
    ax.minorticks_on()
    ax.xaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
    ax.xaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
    ax.yaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))
    # ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
    # ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))

    # Labels
    # ax.set_title('Conversion - Selectivity')
    ax.set_xlabel(rf'$X_\mathrm{{{molecule}}}$ / %')
    ax.set_ylabel(rf'$S_\mathrm{{{target}}}$ based on {molecule} / %')
    # ax.legend(loc='upper right')

    # Text
    ax.text(0.97, 0.1, r'$10\%$', fontsize='small', ha='right', va='center',
            backgroundcolor='w', transform=ax.transAxes)
    ax.text(0.97, 0.275, r'$25\%$', fontsize='small', ha='right', va='center',
            backgroundcolor='w', transform=ax.transAxes)
    ax.text(0.97, 0.53, r'$\eta = 50\%$', fontsize='small', ha='right', va='center',
            backgroundcolor='w', transform=ax.transAxes)

    # Plot Y vs STY
    ax = axes[1]
    for i in range(len(filter_data)):
        ax.scatter(df_ysty[(df_ysty[filter_type] == filter_data[i]) & (df_ysty['Best'] == 0)][f'Yield {target} wrt {molecule}'] * 100,
                   df_ysty[(df_ysty[filter_type] == filter_data[i]) & (df_ysty['Best'] == 0)][f'STY {target} (mmol/h/g)'],
                   edgecolor=colormap(color_n[i]), facecolor='none')
    for i in range(len(filter_data)):
        ax.scatter(df_ysty[(df_ysty[filter_type] == filter_data[i]) & (df_ysty['Best'] == 1)][f'Yield {target} wrt {molecule}'] * 100,
                   df_ysty[(df_ysty[filter_type] == filter_data[i]) & (df_ysty['Best'] == 1)][f'STY {target} (mmol/h/g)'],
                   color=colormap(color_n[i]), label=filter_data[i])
        # ax.scatter(df_ysty[df_ysty[filter_type] == filter_data[i]][f'Yield {target} wrt {molecule}'] * 100,
        #            df_ysty[df_ysty[filter_type] == filter_data[i]][f'STY {target} (mmol/h/g)'],
        #            color=colormap(color_n[i]), label=filter_data[i])

    # Axis scale
    if target == 'MA':
        max_y = 30
    elif target == 'F':
        max_y = 300
    else:
        max_y = 1000
    ax.set_xlim((0, 100))
    ax.set_ylim((0, max_y))
    if max_y < max(df_ysty[f'STY {target} (mmol/h/g)']):
        print('Problem with STY y-scale')

    # Ticks
    ax.minorticks_on()
    ax.xaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
    ax.xaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
    ax.yaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))

    # Labels
    # ax.set_title('Yield - Space time yield')
    ax.set_xlabel(rf'$\eta_\mathrm{{{target}}}$ based on {molecule} / %')
    ax.set_ylabel(rf'$STY_\mathrm{{{target}}}$ / mmol h$^{{-1}}$ g$^{{-1}}$')
    ax.legend(loc='upper right', fontsize='small', frameon=False, markerscale=0.5)

    plt.show()
    if save:
        fig.savefig(f'I:/PhD/Literature/Figures/{filter_type}/Cat_Perf_{target}_from_{molecule}_in_{source}_{filter_type}.png', dpi=300)


def plot_ssty(df_ssty_mac, df_ssty_form, filter_type, filter_data, color, molecule, target, source, save):

    # Plot structure
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    fig.tight_layout(pad=2, h_pad=4, w_pad=4)

    # Colors
    color_n = np.linspace(0, 1, len(filter_data))
    colormap = mplt.colormaps.get_cmap(color)

    # Plot S MAc vs STY
    ax = axes[0]
    for i in range(len(filter_data)):
        ax.scatter(df_ssty_mac[(df_ssty_mac[filter_type] == filter_data[i]) & (df_ssty_mac['Best'] == 0)][f'Selectivity {target} wrt {molecule[0]}'] * 100,
                   df_ssty_mac[(df_ssty_mac[filter_type] == filter_data[i]) & (df_ssty_mac['Best'] == 0)][f'STY {target} (mmol/h/g)'],
                   edgecolor=colormap(color_n[i]), facecolor='none')
    for i in range(len(filter_data)):
        ax.scatter(df_ssty_mac[(df_ssty_mac[filter_type] == filter_data[i]) & (df_ssty_mac['Best'] == 1)][f'Selectivity {target} wrt {molecule[0]}'] * 100,
                   df_ssty_mac[(df_ssty_mac[filter_type] == filter_data[i]) & (df_ssty_mac['Best'] == 1)][f'STY {target} (mmol/h/g)'],
                   color=colormap(color_n[i]), label=filter_data[i])
        # ax.scatter(df_ssty_mac[df_ssty_mac[filter_type] == filter_data[i]][f'Selectivity {target} wrt {molecule[0]}'] * 100,
        #            df_ssty_mac[df_ssty_mac[filter_type] == filter_data[i]][f'STY {target} (mmol/h/g)'],
        #            color=colormap(color_n[i]), label=filter_data[i])

    # Axis scale
    if target == 'MA':
        max_y = 30
    elif target == 'F':
        max_y = 300
    else:
        max_y = 1000
    ax.set_xlim((0, 100))
    ax.set_ylim((0, max_y))
    if max_y < max(df_ssty_mac[f'STY {target} (mmol/h/g)']):
        print('Problem with STY y-scale')

    # Ticks
    ax.minorticks_on()
    ax.xaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
    ax.xaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
    ax.yaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))

    # Labels
    # ax.set_title('Yield - Space time yield')
    ax.set_xlabel(rf'$S_\mathrm{{{target}}}$ based on {molecule[0]} / %')
    ax.set_ylabel(rf'$STY_\mathrm{{{target}}}$ / mmol h$^{{-1}}$ g$^{{-1}}$')
    # ax.legend(loc='upper right', fontsize='small', frameon=False, markerscale=0.5)

    # Plot S form vs STY
    ax = axes[1]
    for i in range(len(filter_data)):
        ax.scatter(df_ssty_form[(df_ssty_form[filter_type] == filter_data[i]) & (df_ssty_form['Best'] == 0)][f'Selectivity {target} wrt {molecule[1]}'] * 100,
                   df_ssty_form[(df_ssty_form[filter_type] == filter_data[i]) & (df_ssty_form['Best'] == 0)][f'STY {target} (mmol/h/g)'],
                   edgecolor=colormap(color_n[i]), facecolor='none')
    for i in range(len(filter_data)):
        ax.scatter(df_ssty_form[(df_ssty_form[filter_type] == filter_data[i]) & (df_ssty_form['Best'] == 1)][f'Selectivity {target} wrt {molecule[1]}'] * 100,
                   df_ssty_form[(df_ssty_form[filter_type] == filter_data[i]) & (df_ssty_form['Best'] == 1)][f'STY {target} (mmol/h/g)'],
                   color=colormap(color_n[i]), label=filter_data[i])
        # ax.scatter(df_ssty_form[df_ssty_form[filter_type] == filter_data[i]][f'Selectivity {target} wrt {molecule[1]}'] * 100,
        #            df_ssty_form[df_ssty_form[filter_type] == filter_data[i]][f'STY {target} (mmol/h/g)'],
        #            color=colormap(color_n[i]), label=filter_data[i])

    # Axis scale
    if target == 'MA':
        max_y = 30
    elif target == 'F':
        max_y = 300
    else:
        max_y = 1000
    ax.set_xlim((0, 100))
    ax.set_ylim((0, max_y))
    if max_y < max(df_ssty_form[f'STY {target} (mmol/h/g)']):
        print('Problem with STY y-scale')

    # Ticks
    ax.minorticks_on()
    ax.xaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
    ax.xaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))
    ax.yaxis.set_major_locator(mplt.ticker.LinearLocator(numticks=6))
    ax.yaxis.set_minor_locator(mplt.ticker.AutoMinorLocator(2))

    # Labels
    # ax.set_title('Yield - Space time yield')
    ax.set_xlabel(rf'$S_\mathrm{{{target}}}$ based on {molecule[1]} / %')
    ax.set_ylabel(rf'$STY_\mathrm{{{target}}}$ / mmol h$^{{-1}}$ g$^{{-1}}$')
    ax.legend(loc='upper right', fontsize='small', frameon=False, markerscale=0.5)

    plt.show()
    if save:
        fig.savefig(f'I:/PhD/Literature/Figures/{filter_type}/Cat_Perf_SSTY_{target}_in_{source}_{filter_type}.png', dpi=300)
