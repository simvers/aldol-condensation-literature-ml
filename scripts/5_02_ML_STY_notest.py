import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from data.ML_models_STY.ML_models_STY import model_config
from functions.ml_models import load_engineered_data, build_cluster_palette, build_preprocessor, build_cv
from functions.ml_plots import plot_feature_correlation, plot_target_correlation, plot_model_diagnostics, plot_observations_per_publication
from functions.ml_models import fit_model_gridsearch

warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8

# This script screens models via cross-validation only (no held-out test set), so the reported
# score reflects purely the CV strategy - not a lucky/unlucky single split, which is the point
# 5_01_ML_STY_test.py makes instead.
RS = 10
# RS = 16

DATA_TYPE = '_noSi'  # ''

REAC_INPUT = ['LHSV_mlhg', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'Temperature_K']
CAT_INPUT = ['SSA_m2g', 'av_cov_rad', 'av_n_val', 'var_cov_rad', 'var_n_val']
INPUT = REAC_INPUT + CAT_INPUT
OUTPUT = 'STY_Acryl_mmolhg'

TCV_TYPE = 'stratified-grouped'  # 'stratified', 'grouped', 'stratified-grouped' - grouped by doi
N_FOLD = 4

MODEL_TO_TRAIN = ['xgboost']  # best_model, xgboost, lgbm, rf, knn, svr, gp

SAVE_PATH = './data/tmp/ML_STY/'


def main():
    start_time = datetime.now()

    data, elements, eng_feat = load_engineered_data(DATA_TYPE)
    cluster_list, clusters, palette = build_cluster_palette(data)

    # Drop observations with missing output
    data.dropna(subset=OUTPUT, inplace=True)
    if data[INPUT].isna().any().any():
        print('Dropping obsevations with NaNs in input feature!')
        data.dropna(subset=INPUT, inplace=True)

    plot_observations_per_publication(data, OUTPUT)

    # Features and target
    x = data[INPUT]
    if OUTPUT == 'STY_Acryl_mmolhg':
        print('Transforming target space')
        y = np.log1p(data[OUTPUT])
    else:
        y = data[OUTPUT]

    # No held-out test set: "train" and "test" are the same full dataset here
    x_train, x_test, y_train, y_test = x, x, y, y
    cluster_train, cluster_test = data.loc[x_train.index, 'Cluster_title'], data.loc[x_test.index, 'Cluster_title']
    doi_train, doi_test = data.loc[x_train.index, 'doi'], data.loc[x_test.index, 'doi']

    categorical_cols = x.select_dtypes(include=['object', 'category']).columns
    numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns
    print(categorical_cols, numerical_cols)

    plot_feature_correlation(x, numerical_cols)
    plot_target_correlation(x, y, numerical_cols, cluster_list, clusters, palette)

    preprocessor = build_preprocessor(numerical_cols, categorical_cols)

    # Loop over and train models
    for model in MODEL_TO_TRAIN:

        if model not in model_config:
            print(f'Model {model} not in config json')
            continue
        config = model_config.get(model)
        print(f'Training model {model}')

        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', config.get('model'))])
        cv = build_cv(TCV_TYPE, N_FOLD, RS, doi_train)

        best_model, grid_search, train_score, test_score = fit_model_gridsearch(
            pipe, config.get('param_grid'), cv, x_train, y_train, x_test, y_test, model, SAVE_PATH, verbose=0)

        # SHAP isn't computed for KNN (no efficient explainer for it)
        compute_shap = model != 'knn'
        if not compute_shap:
            print('No feature importance or shapley analysis for KNN models')

        # Score here is really a "train" score, since x_test/y_test are the same as x_train/y_train
        plot_model_diagnostics(
            model, config.get('abb'), best_model, x_train, y_train, x_test, y_test, cv,
            cluster_train, cluster_test, clusters, palette, f"figures/ML_STY/{model}",
            parity_score_label='train', compute_shap=compute_shap)

    print(f'Time for completion: {datetime.now() - start_time} s')


if __name__ == "__main__":
    main()
