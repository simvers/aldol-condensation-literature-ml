from sklearn.model_selection import learning_curve
import numpy as np
import seaborn as sns

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
