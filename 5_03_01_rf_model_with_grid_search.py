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
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from model_IO import save_output
from helpers_for_sklearn import ContinuousStratifiedKFold
from visualization_helpers import get_learning_curve, plot_learning_curve, plot_cv_distribution

# configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8


if __name__=="__main__":
    # Getting the data from tmp dir (Rember to run the 5_01_data_preprocessing_for_models.py before running this.)
    # df = pd.read_csv("data/tmp/processed_data.csv")
    df = pd.read_csv("data/tmp/processed_data_mols_as_num.csv")
    # df = pd.read_csv("data/tmp/processed_data_with_reactant_score.csv")
    X = df.drop(columns="STY_MA+AA_(mmol/h/g)")
    y  = df["STY_MA+AA_(mmol/h/g)"]


    qcut = pd.qcut(y, 5, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=qcut, random_state=27)

    # Building the pipeline
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    print(categorical_cols)

    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        # ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ])


    param_grid = {
    'model__n_estimators': [250, 300, 350],
    'model__max_depth': [4, 5, 6],
    'model__min_samples_split': [7, 8, 9],
    'model__min_samples_leaf': [3, 4, 5],
    'model__max_features': [0.35, 0.4, 0.45],
    'model__min_impurity_decrease': [0.0, 0.001, 0.005],
    'model__max_samples': [0.8, 0.9, 1.0]
}

    base_model = RandomForestRegressor(
    random_state=42,
    n_jobs=-1
    )

    # The pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_model)
    ])
    cv = ContinuousStratifiedKFold(n_splits=10, n_iter=50, random_state=8)
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

    # Show a correlation plot between experimental and predicted values
    fig, ax = plt.subplots()
    sns.regplot(x=y_test, y = best_pipe.predict(X_test), ax = ax)
    ax.text(0.2, 0.8, f"test score: {test_score : .2f}", transform = ax.transAxes)
    ax.set(xlabel = "Experimental STY",ylabel = "Predicted STY")
    plt.savefig("figures/5_rf_gridsearch_predictions.png", dpi=300, bbox_inches='tight')


    cv_lc = ContinuousStratifiedKFold(n_splits=5, n_iter=50, random_state=8)
    # cv_lc = KFold(n_splits=10, shuffle=True, random_state=8)
    train_sizes, train_scores, val_scores = get_learning_curve(
        best_pipe, X_train, y_train, cv=cv_lc, scoring="r2"
    )

    fig = plt.figure(figsize=(12/2.54, 6/2.54))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # Plot average learning curve
    plot_learning_curve(train_sizes, train_scores, val_scores, title=f"rf_numeric_mols", ax=ax1)

    # Plot fold-level distribution
    plot_cv_distribution(train_sizes, val_scores, title=f"rf_numeric_mols", ax=ax2)
    plt.tight_layout()
    plt.savefig("figures/5_rf_gridsearch_learning_curve.png", dpi=600, bbox_inches='tight')
    plt.show()

