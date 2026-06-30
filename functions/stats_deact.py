import numpy as np
from numpy.linalg import lstsq
from scipy import stats


def residuals_1covar(a, b):
    reg = stats.linregress(b, a)
    residuals = a - (reg.slope * b + reg.intercept)
    print(residuals)
    return np.where(np.abs(residuals) < 1e-10, 0, residuals)  # zero out numerical noise


def partial_corr_1covar(x, y, covar):
    """Partial Spearman correlation between x and y controlling for covar."""
    
    rho, pval = stats.spearmanr(residuals_1covar(x, covar), residuals_1covar(y, covar))
    return rho, pval


def residuals_xcovar(a, covars):
    """Remove effect of multiple covariates from a using OLS."""
    X = np.column_stack(covars)  # shape (n, n_covars)
    X = np.column_stack([np.ones(len(a)), X])  # add intercept
    coefs, _, _, _ = lstsq(X, a, rcond=None)
    residuals = a - X @ coefs
    return np.where(np.abs(residuals) < 1e-10, 0, residuals)  # zero out numerical noise


def partial_corr_xcovar(x, y, covars: list):
    """Partial Spearman correlation controlling for multiple covariates."""
    res_x = residuals_xcovar(x, covars)
    res_y = residuals_xcovar(y, covars)
    return stats.spearmanr(res_x, res_y)