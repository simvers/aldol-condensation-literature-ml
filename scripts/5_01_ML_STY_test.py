import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from data.ML_models_STY.ML_models_STY import model_config
from functions.ml_models import load_engineered_data, build_cluster_palette, build_preprocessor, build_cv
from functions.ml_models import stratified_grouped_train_test_split, fit_model_gridsearch
from functions.ml_plots import plot_feature_correlation, plot_target_correlation, plot_model_diagnostics

warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8

# This script demonstrates that a single train/test split's test score is NOT reproducible
# across random states - compare a run with RS=18 against RS=12 on the same shallow XGBoost
# model. 5_02_ML_STY_notest.py makes the opposite case (robust CV-based screening instead).
RS = 18  # 12

DATA_TYPE = '_noSi'  # ''

REAC_INPUT = ['LHSV_mlhg', 'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'Temperature_K', 'SSA_m2g']
CAT_INPUT = ['av_cov_rad', 'av_n_val', 'var_cov_rad']
INPUT = REAC_INPUT + CAT_INPUT
OUTPUT = 'STY_Acryl_mmolhg'

TT_TYPE = 'stratified'  # 'stratified', 'grouped', 'stratified-grouped', else plain random split
TEST_SIZE = 0.2

TCV_TYPE = 'stratified'  # 'stratified', 'grouped', 'stratified-grouped' - grouped by doi
N_FOLD = 4

MODEL_TO_TRAIN = ['xgboost_no_overfit']  # best_model, xgboost, rf, knn

SAVE_PATH = './data/tmp/ML_STY/'


def split_train_test(x, y, data, tt_type, test_size, rs):
    if tt_type == 'stratified':
        # Split as a function of LHSV quantiles, to keep the same range of conditions in both sets
        qcut = pd.qcut(x['LHSV_mlhg'], q=5, duplicates="drop")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=qcut, random_state=rs)
    elif tt_type == 'grouped':
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_idx, test_idx = next(gss.split(x, y, groups=data['doi']))
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]
    elif tt_type == 'stratified-grouped':
        # Train/test split - group integrity first
        lhsv_bins = pd.qcut(data['LHSV_mlhg'], q=5, labels=False)
        train_idx, test_idx = stratified_grouped_train_test_split(groups=data['doi'], strat=lhsv_bins, test_size=test_size, random_state=rs)
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rs)

    # Sanity check: train + test clusters should reconstruct the full dataset's cluster assignment
    cluster_train, cluster_test = data.loc[x_train.index, 'Cluster_title'], data.loc[x_test.index, 'Cluster_title']
    print((pd.concat([cluster_train, cluster_test]).sort_index() == data['Cluster_title']).all())

    return x_train, x_test, y_train, y_test


def main():
    start_time = datetime.now()

    data, elements, eng_feat = load_engineered_data(DATA_TYPE, na_values=[''], keep_default_na=False)
    cluster_list, clusters, palette = build_cluster_palette(data)

    # Drop observations with missing output
    data.dropna(subset=OUTPUT, inplace=True)
    if data[INPUT].isna().any().any():
        print('Dropping obsevations with NaNs in input feature!')
        data.dropna(subset=INPUT, inplace=True)
    print('Data with STY: ', len(data))

    # Features and target
    x = data[INPUT]
    if OUTPUT == 'STY_Acryl_mmolhg':
        print('Transforming target space')
        y = np.log1p(data[OUTPUT])
    else:
        y = data[OUTPUT]

    x_train, x_test, y_train, y_test = split_train_test(x, y, data, TT_TYPE, TEST_SIZE, RS)
    cluster_train, cluster_test = data.loc[x_train.index, 'Cluster_title'], data.loc[x_test.index, 'Cluster_title']
    doi_train, doi_test = data.loc[x_train.index, 'doi'], data.loc[x_test.index, 'doi']

    categorical_cols = x.select_dtypes(include=['object', 'category']).columns
    numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns

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
            pipe, config.get('param_grid'), cv, x_train, y_train, x_test, y_test, model, SAVE_PATH, verbose=2)

        # SHAP isn't computed for KNN (no efficient explainer for it)
        compute_shap = model != 'knn'
        if not compute_shap:
            print('No feature importance or shapley analysis for KNN models')

        plot_model_diagnostics(
            model, config.get('abb'), best_model, x_train, y_train, x_test, y_test, cv,
            cluster_train, cluster_test, clusters, palette, f"figures/ML_STY/{model}",
            parity_score_label='test', compute_shap=compute_shap)

    print(f'Time for completion: {datetime.now() - start_time} s')


if __name__ == "__main__":
    main()
