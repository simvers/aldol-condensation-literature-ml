# ================================================================================
# Random Forest Regression with GridSearchCV for STY Prediction
# This script uses GridSearchCV (from sklearn.model_selection) for hyperparameter
# tuning of Random Forest Regressor, replacing BayesSearchCV approach.
# Uses ContinuousStratifiedKFold for cross-validation to ensure train/test
# distribution similarity for small datasets.
#
# Inputs:
#   - data/tmp/processed_data_mols_as_num.csv (preprocessed data with molecular features)
#
# Outputs:
#   Model artifacts (data/tmp/):
#     - rf_gridsearch_model.pkl (trained pipeline with best estimator)
#     - rf_gridsearch_best_estimator.json (best hyperparameters)
#     - rf_gridsearch_train_data.pkl (X_train, y_train)
#     - rf_gridsearch_test_data.pkl (X_test, y_test)
#
#   Figures (figures/):
#     - 5_rf_gridsearch_predictions.png (predicted vs experimental STY)
#     - 5_rf_gridsearch_learning_curve.png (learning curve with CV distribution)
# ================================================================================

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from model_IO import save_output
from helpers_for_sklearn import ContinuousStratifiedKFold
from visualization_helpers import get_learning_curve, plot_learning_curve, plot_cv_distribution

# configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8


if __name__=="__main__":

    # Random state
    rs = 30  #12

    # Import data
    df = pd.read_csv("data/data_engineered.csv")
    elements = pd.read_csv('data/elements.csv', header=None).squeeze('columns').to_list()
    eng_feat = pd.read_csv('data/comp_features.csv', header=None).squeeze('columns').to_list()

    # Drop columns
    df.drop(elements + ['Y_Acryl_Ac', 'Y_Acryl_Fa', 'doi', 'Link_to_excel', 'Cluster_title', 
            # 'Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming',
            'Ac_source', 'Fa_source', 'Stabilizer', 
            'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'LHSV_mlhg',
            'Pressure_bar', # 'STY0', 'n',
            'Cluster_n'], axis=1, inplace=True)
    print(df.columns)

    # Address NaNs, and optionally high STY
    if 'Stabilizer' in df.columns:
        df['Stabilizer'].fillna('None', inplace=True)
    df.dropna(subset=['STY_Acryl_mmolhg'], inplace=True)
    # df = df[df['STY_Acryl_mmolhg'] < 10]
    print(len(df))

    # for col in df.columns:
    #     print(col, df[col].isna().sum())

    # Features and target
    X = df.drop(columns='STY_Acryl_mmolhg')
    y  = np.log1p(df['STY_Acryl_mmolhg'])

    qcut = pd.qcut(y, 5, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=qcut, random_state=rs)

    # Building the pipeline
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    print(categorical_cols)

    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ])


    param_grid = {
    'model__n_estimators': [500, 750, 1000],
    'model__max_depth': [4, 5, 6],
    # 'model__min_samples_split': [7, 8, 9],
    # 'model__min_samples_leaf': [3, 4, 5],
    # 'model__max_features': [0.35, 0.4, 0.45],
    # 'model__min_impurity_decrease': [0.0, 0.001, 0.005],
    # 'model__max_samples': [0.8, 0.9, 1.0]
}

    base_model = RandomForestRegressor(
    random_state=rs,
    n_jobs=-1
    )

    # The pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    cv = ContinuousStratifiedKFold(n_splits=10, n_iter=50, random_state=rs)
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        verbose=2,
        return_train_score=True
    )

    # Executing the pipline with train_data
    grid_search.fit(X_train, y_train)

    # Get the best model out. 
    best_pipe = grid_search.best_estimator_

    # print test and train score
    train_score = best_pipe.score(X_train, y_train)
    test_score = best_pipe.score(X_test, y_test)
    print(f"Train score:{train_score : .2f}")
    print(f"Test score:{test_score : .2f}")

    # Save model and results
    print("\nSaving model outputs...")
    path = './data/tmp/'
    model = "rf_gridsearch"
    save_output(model, path, (X_train, y_train), (X_test, y_test), grid_search)
    print(f"Model saved to {path}")

    # ----------------------------------------------

    # Learning curve
    # cv_lc = cv
    cv_lc = ContinuousStratifiedKFold(n_splits=10, n_iter=50, random_state=rs)
    # cv_lc = KFold(n_splits=10, shuffle=True, random_state=8)
    train_sizes, train_scores, val_scores = get_learning_curve(
        best_pipe, X_train, y_train, cv=cv_lc, scoring="r2"
    )

    # Dual plot
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax1, ax2 = ax

    # Plot average learning curve
    plot_learning_curve(train_sizes, train_scores, val_scores, ax=ax1, title=None)

    # Correlation plot between experimental and predicted values
    sns.regplot(x=np.expm1(y_test), y=np.expm1(best_pipe.predict(X_test)), ax=ax2)
    ax2.text(0.2, 0.8, f"test score: {test_score : .2f}", transform = ax2.transAxes)
    ax2.set(xlabel="Experimental STY / mmol h$^{-1}$ g$^{-1}$", ylabel="Predicted STY / mmol h$^{-1}$ g$^{-1}$")

    plt.tight_layout()
    plt.savefig("figures/ML_STY/rf_grid_feateng_predictions.png", dpi=600, bbox_inches='tight')
    # plt.show()

    # Dual plot
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax1, ax2 = ax

    # Plot average learning curve
    plot_learning_curve(train_sizes, train_scores, val_scores, ax=ax1, title=None)

    # Plot fold-level distribution
    plot_cv_distribution(train_sizes, val_scores, ax=ax2, title=None)

    plt.tight_layout()
    plt.savefig("figures/ML_STY/rf_grid_feateng_learning_curve.png", dpi=600, bbox_inches='tight')
    # plt.show()

    # ----------------------------------------------

    # Feature importance using permutation importance

    # Feature names after encoding
    if categorical_cols.empty:
        feature_names = numerical_cols.tolist()
    else:
        onehot_feature_names = best_pipe.named_steps['preprocessor'] \
            .named_transformers_['cat'] \
            .get_feature_names_out(categorical_cols)
        feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()
    print(feature_names)

    # Feature importance
    importances = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=10, random_state=rs, n_jobs=-1
    )
    sorted_idx = importances.importances_mean.argsort()[::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=importances.importances_mean[sorted_idx],
                y=pd.Series([feat.replace('_cordero', '') for feat in feature_names])[sorted_idx],
                orient='h', ax=ax)
    ax.set(xlabel='Feature improtance', ylabel=None, xticks=[])
    # ax.set_title("Permutation Feature Importance (Test Set)")
    plt.tight_layout()
    plt.savefig("figures/ML_STY/rf_grid_feateng_pfi.png", dpi=600, bbox_inches='tight')
    # plt.show()