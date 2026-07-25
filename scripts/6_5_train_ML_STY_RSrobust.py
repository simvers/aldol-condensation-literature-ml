import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config.ML_models_STY import model_config
from src import ml_training, ml_plotting, utils

ROOT = Path(__file__).resolve().parents[1]

warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8

# -----------------------------------------------------------------------------------------
# Configuration

DATA_TYPE = '_noSi'

REAC_INPUT = ['LHSV_mlhg', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'Temperature_K', 'Ac_source']
CAT_INPUT = ['SSA_m2g', 'av_cov_rad', 'av_n_val', 'var_cov_rad']
INPUT = REAC_INPUT + CAT_INPUT
OUTPUT = 'STY_Acryl_mmolhg'

# Run the models sequentially: runtime ∝ len(param_grid) × N_RS
MODEL_NAME = 'lgbm_reg'  # lgbm_reg, rf_reg, xgboost_reg
TCV_TYPE = 'stratified-grouped'
N_FOLD = 4
N_RS = 20  # number of random states

FIGURE_DIR = ROOT / 'figures/ML_STY/robustness/individual'


# -----------------------------------------------------------------------------------------

def compute_shap(fitted_pipe, x):
    """Return (mean_abs Series, shap_values ndarray N_obs x N_features, x_shap DataFrame)."""
    preprocess = fitted_pipe.named_steps['preprocessor']
    feature_names = preprocess.get_feature_names_out()
    x_shap = pd.DataFrame(preprocess.transform(x), columns=feature_names)
    explainer = shap.Explainer(fitted_pipe.named_steps['model'], x_shap)
    sv = explainer(x_shap, check_additivity=False)
    # median_abs = pd.Series(np.median(np.abs(sv.values), axis=0), index=feature_names)
    mean_abs = pd.Series(np.mean(np.abs(sv.values), axis=0), index=feature_names)
    return mean_abs, sv.values.copy(), x_shap


# -----------------------------------------------------------------------------------------

if __name__ == "__main__":

    start_time = datetime.now()
    os.makedirs(FIGURE_DIR, exist_ok=True)

    config = model_config[MODEL_NAME]

    data = utils.load_data(ROOT / f'data/processed/data_engineered{DATA_TYPE}.csv')
    x, y, _, doi = ml_training.extract_x_y(data, INPUT, OUTPUT)

    numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = x.select_dtypes(include=['object', 'category']).columns

    preprocessor = ml_training.build_preprocessor(numerical_cols, categorical_cols,
                                       drop_columns=['Ac_source_MAc', 'Ac_source_EAc'])

    # -----------------------------------------------------------------------------------------
    # GridSearchCV model optimization at N_RS different random states
    print(f"\nRunning {N_RS} GridSearchCV runs ({MODEL_NAME}, {N_FOLD}-fold) ...")

    # Initialize output
    cv_best_scores, cv_best_stds = [], []
    importance_df = pd.DataFrame()
    all_shap_arrays = []   # list of N_RS arrays, each (N_obs, N_features)

    for rs in range(N_RS):
        print(f"  RS={rs:2d} ...", end='', flush=True)

        # Build pipeline and CV folds
        estimator = config['model']
        if 'random_state' in estimator.get_params():
            estimator = estimator.set_params(random_state=rs)
        pipe = Pipeline([('preprocessor', preprocessor), ('model', estimator)])
        cv = ml_training.initialize_cv(TCV_TYPE, N_FOLD, strat=x['LHSV_mlhg'], group=doi, rs=rs)

        # Train model (GridSearchCV directly)
        gs = GridSearchCV(
            pipe, config['param_grid'], cv=cv, scoring='r2',
            n_jobs=-1, return_train_score=False, verbose=0
        )
        gs.fit(x, y)

        # Collect CV scores and SHAP values
        best_idx = gs.best_index_
        cv_best_scores.append(gs.best_score_)
        cv_best_stds.append(gs.cv_results_['std_test_score'][best_idx])

        median_imp, shap_arr, x_shap_out = compute_shap(gs.best_estimator_, x)
        importance_df[f'RS{rs}'] = median_imp
        all_shap_arrays.append(shap_arr)

        print(f"CV R²={gs.best_score_:.3f} ± {gs.cv_results_['std_test_score'][best_idx]:.3f}")

    cv_best_scores = np.array(cv_best_scores)
    cv_best_stds = np.array(cv_best_stds)

    # Derived quantities for stability plots
    shap_stack = np.stack(all_shap_arrays, axis=2)  # (N_obs, N_features, N_RS)
    mean_shap = shap_stack.mean(axis=2)              # (N_obs, N_features)
    std_shap = shap_stack.std(axis=2)                # (N_obs, N_features)
    feature_names_list = list(importance_df.index)
    # sort features by mean importance descending (most important first)
    feature_order = importance_df.mean(axis=1).sort_values(ascending=False).index.tolist()

    # -----------------------------------------------------------------------------------------
    # Save robustness data for cross-model comparison plots

    save_data = {
        'model_name': MODEL_NAME,
        'abb': config['abb'],
        'data_type': DATA_TYPE,
        'n_rs': N_RS,
        'n_fold': N_FOLD,
        'cv_best_scores': cv_best_scores,
        'cv_best_stds': cv_best_stds,
        'importance_df': importance_df,       # (features × N_RS) median |SHAP|
        'shap_stack': shap_stack,             # (N_obs, N_features, N_RS)
        'mean_shap': mean_shap,               # (N_obs, N_features)
        'std_shap': std_shap,                 # (N_obs, N_features)
        'feature_names': feature_names_list,
        'feature_order': feature_order,
    }
    save_path = ROOT / f'data/ML_models_STY/{MODEL_NAME}_robustness_data.pkl'
    joblib.dump(save_data, save_path)
    print(f'Robustness data saved to {save_path}')

    # -----------------------------------------------------------------------------------------
    # Box plot of CV R² mean and std

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5))
    for ax, values, ylabel in zip(
        axes,
        [cv_best_scores, cv_best_stds],
        ['Best CV R² (mean across folds)', 'Best CV R² (std across folds)']
    ):
        ml_plotting.plot_cv_boxstrip(ax, values, ylabel=ylabel)
    axes[0].set(ylim=(0, 1))
    axes[1].set(ylim=(0, 1))
    axes[0].text(0.95, 0.05, f'{config["abb"]} over {N_RS} RS',
                 ha='right', va='bottom', transform=axes[0].transAxes)
    axes[0].text(-0.2, 1.1, 'a', fontsize=10, fontweight='bold', fontfamily='arial', transform=axes[0].transAxes)
    axes[1].text(-0.2, 1.1, 'b', fontsize=10, fontweight='bold', fontfamily='arial', transform=axes[1].transAxes)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f'rs_robustness_boxstrip_{MODEL_NAME}.svg', dpi=600, bbox_inches='tight')

    # -----------------------------------------------------------------------------------------
    # Feature importance stability: mean ± std of mean |SHAP| across RS runs

    fig, ax = plt.subplots(figsize=(5, 3.5))  #len(feature_order) * 0.45 + 0.8))
    ml_plotting.plot_shap_importance_stability(ax, importance_df, feature_order, N_RS)
    ax.text(0.95, 0.05, f'{config["abb"]} model', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f'shap_stability_importance_{MODEL_NAME}.svg', dpi=600, bbox_inches='tight')

    # -----------------------------------------------------------------------------------------
    # SHAP stability: mean SHAP beeswarm across all RS runs

    n_feats = len(feature_order)
    fig, ax = plt.subplots(figsize=(5, 3.5))  # n_feats * 0.6 + 1.0))
    ml_plotting.plot_shap_beeswarm_mean(ax, mean_shap, std_shap, feature_order, feature_names_list, N_RS)
    ax.text(0.95, 0.05, f'{config["abb"]} model', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f'shap_beeswarm_mean_{MODEL_NAME}.svg', dpi=600, bbox_inches='tight')

    print(f'Time for completion: {datetime.now() - start_time} s')
