import json
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
from sklearn.inspection import permutation_importance


def save_output(model, path, train, test, opt):
    if not os.path.isdir(path):
        os.mkdir(path)

    # Save the optimized model hyperparams
    with open(f"{path}{model}_best_estimator.json", "w") as f:
        json.dump(opt.best_params_,f,indent=4)

    # Save the trained model
    joblib.dump(opt.best_estimator_, f"{path}{model}_model.pkl")

    # Save the test/train dataset to a pickle file
    joblib.dump(train, f"{path}{model}_train_data.pkl")
    joblib.dump(test, f"{path}{model}_test_data.pkl")


def load_model_from_tmp(path, model):
    best_estimator = joblib.load(f"{path}{model}_model.pkl")
    train = joblib.load(f"{path}{model}_train_data.pkl")
    test = joblib.load(f"{path}{model}_test_data.pkl")
    return best_estimator, train, test


def plot_feature_importance(opt, feature_names):

    # Extract regression step from best model and set feature names
    xgboost_model = opt.best_estimator_.named_steps['reg']
    xgboost_model.get_booster().feature_names = feature_names
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize = (10, 8))
    plot_importance(xgboost_model, grid=False, ax=ax, height=0.5)
    ax.xaxis.label.set_size('medium')
    ax.yaxis.label.set_size('medium')
    ax.tick_params(axis='both', labelsize='medium')
    plt.show()


def permutation_feature_importance(opt, feature_names, X_test, y_test):

    # Feature importance using permutation importance
    importances = permutation_importance(opt.best_estimator_, X_test, y_test, n_repeats=10, random_state=8, n_jobs=-1)
    sorted_idx = importances.importances_mean.argsort()[::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=importances.importances_mean[sorted_idx],
                y=pd.Series(feature_names)[sorted_idx],
                orient='h', ax=ax)
    ax.set_title("Permutation Feature Importance (Test Set)")
    plt.tight_layout()
    plt.show()
