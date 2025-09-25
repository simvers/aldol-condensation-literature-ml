import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from model_IO import load_model_from_tmp   # your custom loader
from helpers_for_sklearn import ContinuousStratifiedKFold
plt.rcParams.update({
    "font.size":8
})

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


if __name__ == "__main__":
    path = "./data/tmp/"
    models = [
        "xgboost_with_own_cv_splitter_fixed_test_train_splitter",
        # "xgBoost_conservative_10fold",
        # "xgBoost_conservative",
        # "xgBoost",
        # "rf",
        # "SVR",
        # "KNN",
        # "lightGBM"
    ]


    for model in models:
        best_model, (X_train, y_train), (X_test, y_test) = load_model_from_tmp(path, model)

        # Compute learning curve
        cv = ContinuousStratifiedKFold(n_splits=5, n_iter=100, random_state=8) ###############NEW############
        train_sizes, train_scores, val_scores = get_learning_curve(
            best_model, X_train, y_train, cv=10, scoring="r2"
        )

        fig = plt.figure(figsize=(12/2.54, 6/2.54))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        # Plot average learning curve
        plot_learning_curve(train_sizes, train_scores, val_scores, title=f"{model}", ax=ax1)

        # Plot fold-level distribution
        plot_cv_distribution(train_sizes, val_scores, title=f"{model}", ax=ax2)
        plt.tight_layout()
        plt.savefig(f"figures/5_{model}_model_generalization.png", dpi = 600, bbox_inches = 'tight')
        plt.show()
