from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import KFold
import numpy as np
import time, datetime

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
            print()