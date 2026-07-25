import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import string
import numpy as np
import matplotlib.pyplot as plt
import joblib

from src import ml_plotting

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8

# -----------------------------------------------------------------------------------------
# Configuration

MODEL_COMP = ['rf_reg', 'xgboost_reg', 'lgbm_reg']
FIGURE_DIR = ROOT / 'figures/ML_STY/robustness/comparison'

# Colors per model (extend if MODEL_COMP grows beyond 4)
COLORS = ['#7EC8A8', '#89ABD4', '#D4898A', '#AD8FD4']

# -----------------------------------------------------------------------------------------

if __name__ == "__main__":

    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Load saved robustness data for each model
    loaded = {}
    for mname in MODEL_COMP:
        path = ROOT / f'data/ML_models_STY/{mname}_robustness_data.pkl'
        print(f'Loading {path} ...')
        loaded[mname] = joblib.load(path)

    n = len(loaded)

    # -----------------------------------------------------------------------------------------
    # Box plot of CV R² comparison

    rng_cv = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(2 + 1.5 * n, 3))

    for ax, metric, ylabel in zip(
        axes,
        ['cv_best_scores', 'cv_best_stds'],
        ['CV mean R² score', 'CV R² std']
    ):
        for i, (mname, d) in enumerate(loaded.items()):
            values = d[metric]
            ax.boxplot(values, positions=[i], widths=0.35, patch_artist=True,
                       boxprops=dict(facecolor=COLORS[i], alpha=0.7),
                       medianprops=dict(color='k', linewidth=1.5),
                       whiskerprops=dict(linewidth=0.75),
                       capprops=dict(linewidth=0.75),
                       flierprops=dict(marker=''))
            x_jitter = i + rng_cv.uniform(-0.12, 0.12, len(values))
            ax.scatter(x_jitter, values, color='k', s=12, alpha=0.6, zorder=3)
        ax.set_xticks(range(n))
        ax.set_xticklabels([d['abb'] for d in loaded.values()])
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1)
        ax.set_box_aspect(1)
    axes[0].text(0.05, 0.95, f'Reproducibility across {len(values)} RS', ha='left', va='top', transform=axes[0].transAxes)
    for i, ax in enumerate(axes):
        ax.text(-0.2, 1.1, string.ascii_lowercase[i], fontsize=10, fontweight='bold',
                fontfamily='arial', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'cv_comparison.svg', dpi=600, bbox_inches='tight')

    # -----------------------------------------------------------------------------------------
    # Mean SHAP beeswarm

    n_feats = max(len(d['feature_order']) for d in loaded.values())
    fig, axes = plt.subplots(n, 1, figsize=(4, 3 * n))
    if n == 1:
        axes = [axes]

    for i, (ax, (mname, d)) in enumerate(zip(axes, loaded.items())):
        ml_plotting.plot_shap_beeswarm_mean(
            ax, d['mean_shap'], d['std_shap'],
            d['feature_order'], d['feature_names'],
            d['n_rs']
        )
        ax.set_box_aspect(1)
        ax.text(0.95, 0.05, d['abb'], ha='right', va='bottom', transform=ax.transAxes, fontsize=7)
        ax.text(-0.2, 1.1, string.ascii_lowercase[i], fontsize=10, fontweight='bold',
                fontfamily='arial', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'shap_beeswarm_comparison.svg', dpi=600, bbox_inches='tight')

    # -----------------------------------------------------------------------------------------
    # Feature importance stability — one subplot per model

    fig, axes = plt.subplots(n, 1, figsize=(4, 3 * n),
                              sharey=False)
    if n == 1:
        axes = [axes]

    for i, (ax, (mname, d), color) in enumerate(zip(axes, loaded.items(), COLORS)):
        ml_plotting.plot_shap_importance_stability(
            ax, d['importance_df'], d['feature_order'],
            d['n_rs'], color=color
        )
        ax.set_xlim(0, 0.5)
        ax.set_box_aspect(1)
        ax.text(0.95, 0.05, d['abb'], ha='right', va='bottom', transform=ax.transAxes, fontsize=7)
        ax.text(-0.2, 1.1, string.ascii_lowercase[i], fontsize=10, fontweight='bold',
                fontfamily='arial', transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'shap_importance_comparison.svg', dpi=600, bbox_inches='tight')
