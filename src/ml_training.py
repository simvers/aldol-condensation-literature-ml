import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV, KFold, StratifiedGroupKFold, GroupKFold, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon


# Extract clean x and y from the dataset
def extract_x_y(data, feature_cols, target, log1p_output=True, doi_col='doi', cluster_col='Cluster_title'):

    # Drop observations with missing values
    data = data.dropna(subset=target)
    if data[feature_cols].isna().any().any():
        print('Dropping observations with NaNs in input feature!')
        data = data.dropna(subset=feature_cols)

    # Features and target
    x = data[feature_cols]
    y = np.log1p(data[target]) if log1p_output else data[target]

    # Clusters and DOIs
    cluster_list = data[cluster_col]
    doi_list = data[doi_col]

    return x, y, cluster_list, doi_list


# ------------------------------------------------------------------------------------------------------------------

# Train-test splitting

def split_train_test(x, y, test_size, tt_type, strat=None, group=None, rs=0):

    # Stratified train-test split
    if tt_type == 'stratified' and strat is not None:
        qcut = pd.qcut(strat, q=5, duplicates="drop")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, stratify=qcut, random_state=rs)

    # Grouped train-test split
    elif tt_type == 'grouped' and group is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_idx, test_idx = next(gss.split(x, y, groups=group))
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]

    # Stratified-grouped train-test split
    elif tt_type == 'stratified-grouped' and strat is not None and group is not None:
        qcut = pd.qcut(strat, q=5, labels=False)
        train_idx, test_idx = stratified_grouped_train_test_split(group=group, strat=qcut, test_size=test_size, random_state=rs)
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = x.iloc[test_idx], y.iloc[test_idx]
    
    # Standard train-test split
    else:
        print('Standard train-test split')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rs)

    # Sanity check
    assert pd.concat([x_train, x_test]).sort_index().index.equals(x.sort_index().index), \
        "Train+test indices don't reconstruct x"

    return x_train, x_test, y_train, y_test


def stratified_grouped_train_test_split(group, strat, test_size, random_state=0):

    # Get unique papers with their dominant strat bin
    grouped_df = pd.DataFrame({
        'group': group,
        'stratum': strat
    }).groupby('group')['stratum'].agg(lambda x: x.mode()[0]).reset_index()

    # Stratify at paper level
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)  # init generator with 1 split
    paper_train, paper_test = next(sss.split(grouped_df['group'], grouped_df['stratum']))  # get 1st split from generator

    # Get train and test doi
    train_papers = set(grouped_df.iloc[paper_train]['group'])
    test_papers = set(grouped_df.iloc[paper_test]['group'])

    # Get train and test index
    train_idx = np.where(group.isin(train_papers))[0]
    test_idx = np.where(group.isin(test_papers))[0]

    # Check train-test split
    print(f'Fitted vs target test size: {len(test_idx)/(len(group))}, {test_size}')

    return train_idx, test_idx


# ------------------------------------------------------------------------------------------------------------------

# Cross-validation folding

def initialize_cv(tcv_type, n_fold, strat=None, group=None, rs=0):
    
    # Stratified KFold
    if tcv_type == 'stratified' and strat is not None:
        return Stratified_KFold(n_splits=n_fold, strat=strat, random_state=rs)

    # Grouped KFold
    elif tcv_type == 'grouped' and group is not None:
        return Grouped_KFold(n_splits=n_fold, groups=group, random_state=rs)

    # Stratified-grouped KFold
    elif tcv_type == 'stratified-grouped' and strat is not None and group is not None:
        return Stratified_Grouped_KFold(n_splits=n_fold, groups=group, strat=strat, random_state=rs)

    # Target-stratified KFold
    elif tcv_type == 'target-stratified':
        return TargetStratified_KFold(n_splits=n_fold, n_iter=50, random_state=rs)

    else:
        print('Standard cross-validation')
        return None


class Stratified_KFold:

    def __init__(self, n_splits=5, strat=None, q=3, random_state=None):
        self.n_splits = n_splits
        self.strat = strat
        self.q = q
        self.random_state = random_state

    def split(self, X, y, groups=None):
        bins = pd.qcut(self.strat, q=self.q, labels=False)
        kfolder = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, cv_idx in kfolder.split(X, y=bins):
            yield train_idx, cv_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class Grouped_KFold:

    def __init__(self, n_splits=5, groups=None, random_state=None):
        self.n_splits = n_splits
        self.groups = groups
        self.random_state = random_state

    def split(self, X, y, groups=None):
        kfolder = GroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, cv_idx in kfolder.split(X, y, groups=self.groups):
            yield train_idx, cv_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class Stratified_Grouped_KFold:

    def __init__(self, n_splits=5, strat=None, groups=None, q=3, random_state=None):
        self.n_splits = n_splits
        self.groups = groups
        self.strat = strat
        self.q = q
        self.random_state = random_state

    def split(self, X, y, groups=None):
        bins = pd.qcut(self.strat, q=self.q, labels=False)
        kfolder = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for train_idx, cv_idx in kfolder.split(X, y=bins, groups=self.groups):
            yield train_idx, cv_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class TargetStratified_KFold:
    """
    Continuous equivalent of StratifiedKFold for regression using KDE similarity.
    Chooses the split with the lowest average Jensen-Shannon divergence
    between train/test target distributions.
    """

    def __init__(self, n_splits=5, n_iter=50, random_state=None):
        self.n_splits = n_splits
        self.n_iter = n_iter
        self.best_seed_ = None
        self.splits_ = None
        self.random_state = random_state

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
        return self.n_splits


# ------------------------------------------------------------------------------------------------------------------

# Model fitting

class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops named columns from a pandas DataFrame input. Columns not present are silently skipped.
    Requires the preceding pipeline step to use set_output(transform='pandas') so column names flow through."""

    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        drop_set = set(self.columns_to_drop)
        self.keep_cols_ = [c for c in X.columns if c not in drop_set]
        return self

    def transform(self, X):
        return X[self.keep_cols_].to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.array(self.keep_cols_)


def build_preprocessor(numerical_cols, categorical_cols, drop_columns=None):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False) if drop_columns else OneHotEncoder(handle_unknown='ignore')
    column_transformer = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', ohe, categorical_cols)
    ], verbose_feature_names_out=False)

    if not drop_columns:
        return column_transformer

    # set_output('pandas') makes ColumnTransformer output a DataFrame so ColumnDropper can match by name;
    # sparse_output=False on OHE is required because pandas output is incompatible with sparse arrays
    column_transformer.set_output(transform='pandas')
    return Pipeline(steps=[
        ('encode', column_transformer),
        ('drop_columns', ColumnDropper(list(drop_columns)))
    ])


def fit_model_gridsearch(pipe, param_grid, cv, x_train, y_train, x_test, y_test, model_name, save_path=None, verbose=2):
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
    print(grid_search.best_params_)

    # Save output if required
    if save_path is not None:
        save_output(f"{model_name}_gridsearch_feateng", save_path, (x_train, y_train), (x_test, y_test), grid_search)
        print(f"Model saved to {save_path}")

    return best_model, grid_search, train_score, test_score


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
        # Score evolution
        optimization_history = pd.DataFrame({
            'iteration': range(1, len(tracker.iteration_scores) + 1),
            'r2_score': tracker.iteration_scores
        })

        # Add parameter evolution
        if tracker.iteration_params:
            param_df = pd.DataFrame(tracker.iteration_params)
            optimization_history = pd.concat([optimization_history, param_df], axis=1)

        # Save optimization history as CSV
        optimization_history.to_csv(f"{path}{model}_optimization_history.csv", index=False)

        # Save tracker object itself
        joblib.dump(tracker, f"{path}{model}_tracker.pkl")

        # Tracker summary
        tracker_summary = {
            'total_iterations': len(tracker.iteration_scores),
            'best_r2_score': tracker.best_score,
            'final_r2_score': tracker.iteration_scores[-1] if tracker.iteration_scores else None,
            'improvement_over_time': tracker.best_score - tracker.iteration_scores[0] if tracker.iteration_scores else None
        }

        # Save summary stats
        with open(f"{path}{model}_tracker_summary.json", "w") as f:
            json.dump(tracker_summary, f, indent=4)


def load_model(path, model, include_tracker=False):

    # Load model and train-test sets
    best_estimator = joblib.load(f"{path}{model}_model.pkl")
    train = joblib.load(f"{path}{model}_train_data.pkl")
    test = joblib.load(f"{path}{model}_test_data.pkl")

    # Load training evolution
    if include_tracker:
        try:
            tracker = joblib.load(f"{path}{model}_tracker.pkl")
            return best_estimator, train, test, tracker
        except FileNotFoundError:
            print(f"\t[Warning....]Tracker file not found at {path}{model}_tracker.pkl")
            return best_estimator, train, test, None

    return best_estimator, train, test
