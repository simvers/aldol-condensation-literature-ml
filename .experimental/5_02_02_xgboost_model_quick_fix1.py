import warnings
import time
from sklearn.model_selection import learning_curve
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold 
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
import numpy as np
from model_IO import save_output
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
# configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8


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
if __name__=="__main__":
    # Getting the data from tmp dir (Rember to run the 5_01_data_preprocessing_for_models.py before running this.)
    # df = pd.read_csv("data/tmp/processed_data.csv")
    df = pd.read_csv("data/tmp/processed_data_mols_as_num.csv")
    # df = pd.read_csv("data/tmp/processed_data_with_reactant_score.csv")
    X = df.drop(columns="STY_MA+AA_(mmol/h/g)")
    y  = df["STY_MA+AA_(mmol/h/g)"]

    # plt.figure(figsize = (20,20))
    # corr = X.corr(numeric_only=True)
    # anot = corr.round(2).astype("str")
    # sns.heatmap(corr, cmap="coolwarm", annot=anot, fmt="")
    # plt.show()
    # exit()

    # fig, (ax,ax2) = plt.subplots(1,2)
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=27)
    # sns.kdeplot(y_train,ax =ax )
    # sns.kdeplot(y_test,ax =ax)
    # ax.legend(["Train", "Test"])

    qcut = pd.qcut(y, 5, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=qcut, random_state=27)
    # sns.kdeplot(y_train,ax =ax2)
    # sns.kdeplot(y_test,ax =ax2)
    # ax.legend(["Train", "Test"])
    # plt.tight_layout()
    # plt.show()

    # Building the pipeline
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    print(categorical_cols)

    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_cols),
        # ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
    ])

    # test_model = XGBRegressor(
    # n_estimators=300,
    # eval_metric='rmse',
    # learning_rate=0.06,
    # max_depth=5,
    # min_child_weight=15,
    # subsample=0.7,
    # colsample_bytree=0.6,
    # reg_alpha=0.3,
    # reg_lambda=0.3,
    # gamma=0.05,
    # random_state=8,
    # verbosity = 0,
    # nthread = -1
    # )

    test_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=5,              
    min_samples_split=8,      
    min_samples_leaf=4,       
    max_features=0.4,         
    # bootstrap=True,
    # oob_score=True,           
    random_state=42,
    n_jobs=-1
    )

    # The pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('reg', test_model)
    ])

    # Executing the pipline with train_data
    pipe.fit(X_train, y_train)

    # # Feature names are lost during the one hot encoding, so get them back
    # onehot_feature_names = pipe.named_steps['preprocessor'] \
    #     .named_transformers_['cat'] \
    #     .get_feature_names_out(categorical_cols)

    # feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()

    # print test and train score
    train_score = pipe.score(X_train, y_train)
    test_score = pipe.score(X_test, y_test)
    print(f"Train score:{train_score : .2f}")
    print(f"Test score:{test_score : .2f}")

    # Show a correlation plot between experimental and predicted values
    fig, ax = plt.subplots()
    sns.regplot(x=y_test, y = pipe.predict(X_test), ax = ax)
    ax.text(0.2, 0.8, f"test score: {test_score : .2f}", transform = ax.transAxes)
    ax.set(xlabel = "Experimental STY",ylabel = "Predicted STY")
    plt.show()

    # # Show feature importance
    # xgboost_model = pipe.named_steps['reg']
    # xgboost_model.feature_names = feature_names
    # fig, ax = plt.subplots(figsize = (10, 8))
    # plot_importance(xgboost_model, grid=False, ax=ax, height=0.5)
    # plt.show()

    cv = ContinuousStratifiedKFold(n_splits=5, n_iter=50, random_state=8)
    # cv = KFold(n_splits=10, shuffle=True, random_state=8)
    train_sizes, train_scores, val_scores = get_learning_curve(
        pipe, X_train, y_train, cv=cv, scoring="r2"
    )

    fig = plt.figure(figsize=(12/2.54, 6/2.54))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    # Plot average learning curve
    plot_learning_curve(train_sizes, train_scores, val_scores, title=f"rf_numeric_mols", ax=ax1)

    # Plot fold-level distribution
    plot_cv_distribution(train_sizes, val_scores, title=f"rf_numeric_mols", ax=ax2)
    plt.tight_layout()
    # plt.savefig(f"figures/5_{model}_model_generalization.png", dpi = 600, bbox_inches = 'tight')
    plt.show()

