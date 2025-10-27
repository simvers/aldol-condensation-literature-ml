import json
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
from sklearn.inspection import permutation_importance


def save_output(model, path, train, test, opt, tracker=None):
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
    # Save tracker data if provided
    if tracker is not None:
        # Save optimization history as CSV
        optimization_history = pd.DataFrame({
            'iteration': range(1, len(tracker.iteration_scores) + 1),
            'r2_score': tracker.iteration_scores
        })
        
        # Add parameter columns
        if tracker.iteration_params:
            param_df = pd.DataFrame(tracker.iteration_params)
            optimization_history = pd.concat([optimization_history, param_df], axis=1)
        
        optimization_history.to_csv(f"{path}{model}_optimization_history.csv", index=False)
        
        # Save tracker object itself
        joblib.dump(tracker, f"{path}{model}_tracker.pkl")
        
        # Save summary stats
        tracker_summary = {
            'total_iterations': len(tracker.iteration_scores),
            'best_r2_score': tracker.best_score,
            'final_r2_score': tracker.iteration_scores[-1] if tracker.iteration_scores else None,
            'improvement_over_time': tracker.best_score - tracker.iteration_scores[0] if tracker.iteration_scores else None
        }
        
        with open(f"{path}{model}_tracker_summary.json", "w") as f:
            json.dump(tracker_summary, f, indent=4)


def load_model_from_tmp(path, model, include_tracker=False):
    best_estimator = joblib.load(f"{path}{model}_model.pkl")
    train = joblib.load(f"{path}{model}_train_data.pkl")
    test = joblib.load(f"{path}{model}_test_data.pkl")

    if include_tracker:
        try:
            tracker = joblib.load(f"{path}{model}_tracker.pkl")
            return best_estimator, train, test, tracker
        except FileNotFoundError:
            print(f"\t[Warning....]Tracker file not found at {path}{model}_tracker.pkl")
            return best_estimator, train, test, None

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
