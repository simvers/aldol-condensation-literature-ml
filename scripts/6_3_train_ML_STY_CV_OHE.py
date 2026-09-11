"""
STY model variant: same cross-validated (doi-grouped) pipeline as 6_2, but using
one-hot-encoded catalyst clusters instead of engineered composition features.

Reads:  data/processed/data_engineered{DATA_TYPE}.csv
Writes: figures/ML_STY/CV_OHE/<model>/*.svg
Edit before running: DATA_TYPE, RS, REAC_INPUT, TCV_TYPE, MODEL_TO_TRAIN
"""

import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from config.ML_models_STY import model_config
from src import ml_training, ml_plotting, utils

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["font.size"] = 8

RS = 10  # random seed for CV folding and model fitting

DATA_TYPE = '_noSi'  # '' to include Si in the engineered feature set

REAC_INPUT = ['LHSV_mlhg', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'Temperature_K', 'Ac_source']
CAT_INPUT = ['Cluster_title']  # one-hot-encoded catalyst cluster, in place of engineered features
INPUT = REAC_INPUT + CAT_INPUT
OUTPUT = 'STY_Acryl_mmolhg'

# CV folding: 'stratified' balances a continuous variable (below: LHSV) across folds,
# 'grouped' keeps each doi entirely in one fold (no leakage across papers),
# 'stratified-grouped' does both at once
TCV_TYPE = 'stratified-grouped'
N_FOLD = 4

MODEL_TO_TRAIN = ['lgbm_reg', 'xgboost_reg']  # keys from config/ML_models_STY.py's model_config
# MODEL_TO_TRAIN = ['xgboost_reg', 'lgbm_reg', 'rf_reg', 'knn_reg', 'svr_reg', 'gp']

PALETTE = ["#009688", "#1565C0", "#AD1457"]


if __name__ == "__main__":

    start_time = datetime.now()

    # Import data
    data = utils.load_data(ROOT / f'data/processed/data_engineered{DATA_TYPE}.csv')
    clusters = np.sort(data['Cluster_title'].unique())

    # Extract clean x and y
    # No test
    x, y, cluster_list, doi_list = ml_training.extract_x_y(data, INPUT, OUTPUT)
    x_train, y_train = x, y
    cluster_train, cluster_test = cluster_list, cluster_list
    doi_train, doi_test = doi_list, doi_list

    # Initialize preprocessor
    categorical_cols = x.select_dtypes(include=['object', 'category']).columns
    numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ml_training.build_preprocessor(numerical_cols, categorical_cols, drop_columns=['Ac_source_MAc', 'Ac_source_EAc', 'Cluster_title_Al-Si-Cs'])

    # Loop over and train models
    for model in MODEL_TO_TRAIN:

        # Import model configuration
        if model not in model_config:
            print(f'Model {model} not in config json')
            continue
        config = model_config.get(model)
        print(f'Training model {model}')

        # Figure directory
        FIGURE_DIR = ROOT / f"figures/ML_STY/CV_OHE/{model}"
        os.makedirs(FIGURE_DIR, exist_ok=True)

        # Build pipeline and CV folds
        estimator = config.get('model')
        if 'random_state' in estimator.get_params():
            estimator = estimator.set_params(random_state=RS)
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', estimator)])
        cv = ml_training.initialize_cv(TCV_TYPE, N_FOLD, strat=x_train['LHSV_mlhg'], group=doi_train, rs=RS)

        # Train model
        best_model, grid_search, train_score, test_score = ml_training.fit_model_gridsearch(
            pipe, config.get('param_grid'), cv, x_train, y_train, x_train, y_train, model, verbose=0)

        # Plot learning analysis
        ml_plotting.plot_model_diagnostics(
            model, config.get('abb'), best_model, x_train, y_train, x_train, y_train, cv,
            FIGURE_DIR, parity_score_label='train')

        # Plot SHAP analysis
        if config.get('abb') in ['KNN', 'SVR', 'GP']:
            print('No shapley analysis for KNN, SVR, and GP models')
        else:
            ml_plotting.plot_shapley_analysis(
                model, config.get('abb'), best_model, x_train, x_train,
                cluster_train, cluster_test, clusters, PALETTE, FIGURE_DIR)

    print(f'Time for completion: {datetime.now() - start_time} s')
