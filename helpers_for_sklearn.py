from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import KFold
import numpy as np

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