"""
Illustrates why a single train/test split score is not reproducible: trains a shallow
XGBoost on STY across a few random seeds and compares train vs. held-out test scores.

Reads:  data/processed/data_engineered{DATA_TYPE}.csv
Writes: figures/ML_STY/overfitting/<model>_RS<seed>/*.svg
Edit before running: RSS, DATA_TYPE, MODEL_TO_TRAIN, TT_TYPE, TCV_TYPE
"""

import sys, os
os.environ["PYTHONWARNINGS"] = "ignore"
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

RSS = [18, 12]  # random seeds compared to show split-to-split score variance

DATA_TYPE = '_noSi'  # '' to include Si in the engineered feature set

REAC_INPUT = ['LHSV_mlhg', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'Temperature_K',
              'Ac_source', 'Fa_source', 'Stabilizer', 'O_content',
]
CAT_INPUT = ['av_cov_rad', 'av_n_val', 'var_cov_rad', 'var_n_val', 'SSA_m2g']
INPUT = REAC_INPUT + CAT_INPUT
OUTPUT = 'STY_Acryl_mmolhg'

# Train-test split: 'stratified' balances a continuous variable (below: LHSV) across the
# split, 'grouped' keeps each doi entirely on one side (no leakage across papers),
# 'stratified-grouped' does both at once
TT_TYPE = 'stratified'
TEST_SIZE = 0.2

# Train-CV folding: same three options as above, applied per fold instead of once
TCV_TYPE = 'stratified'
N_FOLD = 4

MODEL_TO_TRAIN = ['xgboost_hreg']  # keys from config/ML_models_STY.py's model_config

PALETTE = ["#009688", "#1565C0", "#AD1457"]


if __name__ == "__main__":

    start_time = datetime.now()

    # Import data
    data = utils.load_data(ROOT / f'data/processed/data_engineered{DATA_TYPE}.csv')
    clusters = np.sort(data['Cluster_title'].unique())

    # Extract clean x and y
    x, y, cluster_list, doi_list = ml_training.extract_x_y(data, INPUT, OUTPUT)

    # Initialize preprocessor
    categorical_cols = x.select_dtypes(include=['object', 'category']).columns
    numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns
    preprocessor = ml_training.build_preprocessor(numerical_cols, categorical_cols)

    # Loop over and train models
    for model in MODEL_TO_TRAIN:

        # Import model configuration
        if model not in model_config:
            print(f'Model {model} not in config json')
            continue
        config = model_config.get(model)
        print(f'Training model {model}')

        for RS in RSS:

            # Figure directory
            FIGURE_DIR = ROOT / f"figures/ML_STY/overfitting/{model}_RS{RS}"
            os.makedirs(FIGURE_DIR, exist_ok=True)

            # Train-test split
            x_train, x_test, y_train, y_test = ml_training.split_train_test(x, y, test_size=TEST_SIZE, tt_type=TT_TYPE, strat=x['LHSV_mlhg'], group=doi_list, rs=RS)
            cluster_train, cluster_test = cluster_list.loc[x_train.index], cluster_list.loc[x_test.index]
            doi_train, doi_test = doi_list.loc[x_train.index], doi_list.loc[x_test.index]

            # Build pipeline and CV folds
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', config.get('model'))])
            cv = ml_training.initialize_cv(TCV_TYPE, N_FOLD, strat=x_train['LHSV_mlhg'], rs=RS)

            # Train model
            best_model, grid_search, train_score, test_score = ml_training.fit_model_gridsearch(
                pipe, config.get('param_grid'), cv, x_train, y_train, x_test, y_test, model, verbose=2)

            # Plot learning analysis
            ml_plotting.plot_model_diagnostics(
                model, config.get('abb'), best_model, x_train, y_train, x_test, y_test, cv,
                FIGURE_DIR, parity_score_label='test')

            # Plot SHAP analysis
            if config.get('abb') in ['KNN', 'SVR', 'GP']:
                print('No shapley analysis for KNN, SVR, and GP models')
            else:
                ml_plotting.plot_shapley_analysis(
                    model, config.get('abb'), best_model, x_train, x_test,
                    cluster_train, cluster_test, clusters, PALETTE, FIGURE_DIR)


        print(f'Time for completion: {datetime.now() - start_time} s')

