import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve, KFold
from sklearn.inspection import permutation_importance
from xgboost import plot_importance
from sklearn.inspection import partial_dependence
# from scipy.signal import savgol_filter
import shap
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import time, datetime


# ------------------------------------------------------------------------------------------------------------------

# Save-load models

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


# ------------------------------------------------------------------------------------------------------------------

# Model training

class ContinuousStratifiedKFold:
    """
    Continuous equivalent of StratifiedKFold for regression using KDE similarity.
    Chooses the split with the lowest average Jensen-Shannon divergence
    between train/test target distributions.
    """

    def __init__(self, n_splits=5, n_iter=50, random_state=None):
        self.n_splits = n_splits
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_seed_ = None
        self.splits_ = None

    def _kde_js_distance(self, train, test, grid_points=1000):
        kde_train = gaussian_kde(train)
        kde_test = gaussian_kde(test)
        x_grid = np.linspace(min(train.min(), test.min()), 
                             max(train.max(), test.max()), grid_points)
        p = kde_train(x_grid)
        q = kde_test(x_grid)
        p /= p.sum()
        q /= q.sum()
        return jensenshannon(p, q)

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set."""
        rng = np.random.default_rng(self.random_state)
        best_js = float('inf')
        best_seed = None
        best_splits = None

        for seed in range(self.n_iter):
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
            js_total = 0
            splits = []
            for train_idx, test_idx in kf.split(X):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                js = self._kde_js_distance(y_train, y_test)
                js_total += js
                splits.append((train_idx, test_idx))
            avg_js = js_total / self.n_splits
            if avg_js < best_js:
                best_js = avg_js
                best_seed = seed
                best_splits = splits

        self.best_seed_ = best_seed
        self.splits_ = best_splits

        for train_idx, test_idx in best_splits:
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splitting iterations."""
        return self.n_splits


# Convergence checkers to use with skopt optimizer object (not applicable to the grid search)
# Convergence criteria
class ConvergenceChecker:
    '''
    This will be passed as callback to the optimizer object.
    If this returns True, then optimization will stop.
    '''
    def __init__(self, min_improvement = 0.02, patience=15):
        """
        Initialize the ConvergenceChecker.
        Paramters:
            min_improvement (float): cutoff criteria in term of delta_r2
            patience (int): number of cycles to averagve over.
        """
        self.min_improvement = min_improvement
        self.patience = patience
        self.r2_scores = []
    
    def __call__(self, result):
        current_r2 = -result.func_vals[-1] # For some reason this is a negative number (Figure out why.)
        self.r2_scores.append(current_r2)

        # Don't check the stopping criteria in the begining, at least a few (=patience in this case)
        if len(self.r2_scores) < self.patience:
            return False

        recent_scores = self.r2_scores[-self.patience:] # This will check last few (=patience)
        best_recent = max(recent_scores)
        worst_recent = min(recent_scores)
        improvement = best_recent - worst_recent

        if improvement < self.min_improvement:
            print(f"\nConverged! Improvement of {improvement:.4f} over last {self.patience} iterations is below threshold {self.min_improvement}")
            return True

        return False # Continue to optimize
        

class OptimizationTracker:
    def __init__(self):
        self.iteration_scores = []
        self.iteration_params = []
        self.start_time = time.time()
        self.best_score = -np.inf

    def __call__(self, res):
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Get current iteration info
        iteration = len(res.func_vals)
        current_score = -res.func_vals[-1] # Again the same thing here, this is a negative number
        # current_params = res.x_iters[-1] # Get the latest paramers
        self.iteration_scores.append(current_score)

        param_dict = dict(zip(res.space.dimension_names, res.x_iters[-1]))
        self.iteration_params.append(param_dict)

        # Update best score
        if current_score > self.best_score:
            self.best_score = current_score
            improvement = "\tNew Best!"
        else:
            improvement = ""

        # Print progress
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Iteration {iteration:3d} | "
              f"R² = {current_score:.4f} | Best R² = {self.best_score:.4f} | "
              f"Elapsed: {elapsed/60:.1f}min{improvement}")
        
        # Print best parameters every 25 iterations
        if iteration % 25 == 0 and iteration > 0:
            best_idx = np.argmax(self.iteration_scores)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Best parameters so far (iteration {best_idx + 1}):")
            for param, value in self.iteration_params[best_idx].items():
                if isinstance(value, float):
                    print(f"\t{param}: {value:.4f}")
                else:
                    print(f"\t{param}: {value}")


# ------------------------------------------------------------------------------------------------------------------

# Learning curve

def get_learning_curve(estimator, X, y, cv=5, scoring="r2", train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Compute training and validation scores for learning curve.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
        return_times=False
    )

    return train_sizes, train_scores, val_scores


def plot_learning_curve(train_sizes, train_scores, val_scores,
                        title="Learning Curve", ylabel="R² Score", ax=None):
    """
    Plot mean learning curve with error bands.
    """
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    ax.set_ylim(0,1)
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(ylabel)

    # Training score curve
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")

    # Cross-validation score curve
    ax.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
    ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.2, color="g")

    ax.legend(loc="best")
    ax.grid(True)


def plot_cv_distribution(train_sizes, val_scores, title="Cross-Validation Distribution", ylabel="R² per Fold", ax = None):
    """
    Plot boxplots of cross-validation scores for each training size.
    """
    sns.boxplot(data=[val_scores[i] for i in range(len(train_sizes))], ax=ax)
    ax.set_xticks(ticks=range(len(train_sizes)), labels=train_sizes.astype(int), rotation=45)
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(0,1)


# ------------------------------------------------------------------------------------------------------------------

# Feature importance

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


# ------------------------------------------------------------------------------------------------------------------

# Shapley analysis

def plot_partial_dependence(model, x: pd.DataFrame, individual_plot=False):
        
    # Compute expected values
    # expected_output_val = explainer.expected_value
    expected_output_val = model.predict(x).mean()
    expected_feature_val = x.mean()

    # Figure
    fig, ax = plt.subplots(5, 2, figsize=(10, 25))
    # fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    ax = ax.ravel()

    # Partial dependence plot
    for i, feature in enumerate(x.columns):
        
        # Plot feature partial dependence
        results = partial_dependence(model, x, features=[i], kind='average')
        ax[i].scatter(results['grid_values'][0], results['average'][0], color='k')
        # Shap plot not working with custom ax
        # Ice for correlation with another feature
        # shap.plots.partial_dependence(feature, model.predict, x_test_shap, ax=ax[i],model_expected_value=True, feature_expected_value=True, show=False, ice=False,)

        # Trend line
        ax[i].plot(results['grid_values'][0], results['average'][0], color='DarkRed')
        # window = int(np.ceil(len(results['grid_values'][0])/15)*2+1)
        # ax[i].plot(results['grid_values'][0], savgol_filter(results['average'][0], window_length=window, polyorder=1), color='DarkRed')
        # sns.regplot(x=results['grid_values'][0], y=results['average'][0], order=0.5, color='DarkRed', ax=ax[i])

        # Labels
        ax[i].set(xlabel=feature, ylabel=f'E[f(x) | {feature}]', ylim=(0.7, 2.3))

        # Plot hist
        ax_hist = ax[i].inset_axes([0, 0, 1, 0.2], zorder=0)  # [x, y, width, height]
        ax_hist.hist(x.loc[:, feature], color='grey', bins=20)
        ax_hist.set(ylim=(0, 70))
        ax_hist.axis('off')

        # Display expected values
        ax[i].axhline(y=expected_output_val, linestyle='--', color='grey')
        ax[i].axvline(x=expected_feature_val[feature], linestyle='--', color='grey')
        y_min, y_max = ax[i].get_ylim()
        ax[i].text(x=expected_feature_val[feature], y=y_max+(y_max-y_min)*0.02, s=f'E[{feature}]', horizontalalignment='center')
        x_min, x_max = ax[i].get_xlim()
        ax[i].text(x=x_max+(x_max-x_min)*0.02, y=expected_output_val, s=f'E[f(x)]', verticalalignment='center', rotation = 270)
    
        # Partial dependence plot
        if individual_plot:
            # Plot feature partial dependence
            fig_i, ax_i = shap.plots.partial_dependence(feature, model.predict, x, model_expected_value=True, feature_expected_value=True, show=False, ice=False,)
            ax_i.scatter(results['grid_values'][0], results['average'][0], color='k')
            fig_i.savefig(f"figures/ML_STY/model_grid_feateng_{feature}.png", dpi=600, bbox_inches='tight')

    return fig, ax
    

def plot_feature_output(x: pd.DataFrame, y):

    fig, ax = plt.subplots(5, 2, figsize=(8, 20))
    # fig, ax = plt.subplots(3, 4, figsize=(16, 12))
    ax = ax.ravel()
    # Partial dependence plot
    for i, feature in enumerate(x.columns):
        
        # Plot feature partial dependence
        ax[i].scatter(x.loc[:, feature], y, color='k')
        ax[i].set(xlabel=feature, ylabel=f'log1p(STY)', ylim=(0, 4))
    
    return fig, ax
