import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold, StratifiedShuffleSplit, StratifiedGroupKFold, GroupKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

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

# Data loading

def load_engineered_data(data_type, na_values=None, keep_default_na=True):
    data = pd.read_csv(f"data/catalysts/data_engineered{data_type}.csv", na_values=na_values, keep_default_na=keep_default_na)
    elements = pd.read_csv('data/catalysts/elements.csv', header=None).squeeze('columns').to_list()
    eng_feat = pd.read_csv('data/catalysts/comp_features.csv', header=None).squeeze('columns').to_list()
    print(data.columns, len(data))
    return data, elements, eng_feat


def build_cluster_palette(data, cmap_name='cividis'):
    cluster_list = data['Cluster_title']
    clusters = np.sort(cluster_list.unique())
    cmap = plt.get_cmap(cmap_name)
    palette = cmap(np.linspace(0, 1, len(clusters)))
    return cluster_list, clusters, palette


# ------------------------------------------------------------------------------------------------------------------

# Train-test / cross-validation splitting

def stratified_grouped_train_test_split(groups, strat, test_size=0.25, random_state=42):

    # Get unique papers with their dominant LHSV bin
    paper_df = pd.DataFrame({
        'paper': groups,
        'stratum': strat
    }).groupby('paper')['stratum'].agg(lambda x: x.mode()[0]).reset_index()

    # Stratify at paper level
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)  # init generator with 1 split
    paper_train, paper_test = next(sss.split(paper_df['paper'], paper_df['stratum']))  # get 1st split from generator

    train_papers = set(paper_df.iloc[paper_train]['paper'])
    test_papers = set(paper_df.iloc[paper_test]['paper'])

    train_idx = np.where(groups.isin(train_papers))[0]
    test_idx = np.where(groups.isin(test_papers))[0]

    # Check train-test split
    print(f'Fitted vs target test size: {len(test_idx)/(len(groups))}, {test_size}')

    return train_idx, test_idx


class GroupedKFold:

    def __init__(self, n_splits=5, random_state=None, groups=None):
        self.n_splits = n_splits
        self.random_state = random_state
        self.groups = groups

    def split(self, X, y, groups=None):

        kfolder = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, cv_idx in kfolder.split(X, y, groups=self.groups):
            # print(train_idx, cv_idx)
            yield train_idx, cv_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splitting iterations."""
        return self.n_splits



class ContinuousStratifiedGroupKFold:

    def __init__(self, n_splits=5, random_state=None, strat='LHSV_mlhg', groups=None, q=3):
        self.n_splits = n_splits
        self.random_state = random_state
        self.groups = groups
        self.strat = strat
        self.q = q

    def split(self, X, y, groups=None):

        bins = pd.qcut(X[self.strat], q=self.q, labels=False)  # assign bins number here

        kfolder = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, cv_idx in kfolder.split(X, y=bins, groups=self.groups):
            # print(train_idx, cv_idx)
            yield train_idx, cv_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splitting iterations."""
        return self.n_splits


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


def build_cv(tcv_type, n_fold, rs, doi_train):
    if tcv_type == 'stratified':
        return ContinuousStratifiedKFold(n_splits=n_fold, n_iter=50, random_state=rs)
    elif tcv_type == 'grouped':
        return GroupedKFold(n_splits=n_fold, groups=doi_train, random_state=rs)
    elif tcv_type == 'stratified-grouped':
        return ContinuousStratifiedGroupKFold(n_splits=n_fold, groups=doi_train, strat='LHSV_mlhg', random_state=rs)
    return None


# ------------------------------------------------------------------------------------------------------------------

# Model fitting

def build_preprocessor(numerical_cols, categorical_cols):
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ], verbose_feature_names_out=False)


def fit_model_gridsearch(pipe, param_grid, cv, x_train, y_train, x_test, y_test, model_name, save_path, verbose=2):
    # Fit `pipe` via grid search, report train/test R2 and the chosen hyperparameters, and save
    # the fitted estimator + data via save_output. x_test/y_test are scored but not used for fitting -
    # for a "notest" caller, x_test/y_test are simply the same data as x_train/y_train.
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=cv, scoring='r2',
                                n_jobs=-1, verbose=verbose, return_train_score=True)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    train_score = best_model.score(x_train, y_train)
    test_score = best_model.score(x_test, y_test)
    print(f"{model_name} train score:{train_score : .2f}")
    print(f"{model_name} test score:{test_score : .2f}")

    # Matches the naming convention 5_01/5_02 always used for saved artifacts
    save_output(f"{model_name}_gridsearch_feateng", save_path, (x_train, y_train), (x_test, y_test), grid_search)
    print(f"Model saved to {save_path}")
    print(grid_search.best_params_)

    return best_model, grid_search, train_score, test_score
