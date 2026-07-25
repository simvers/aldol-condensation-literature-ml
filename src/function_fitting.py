import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import linregress


def power_law_model_3(t, STY_0, n, STY_inf):
    return STY_0 / (1 + t)**n + STY_inf


def power_law_model_2(*args):
    return power_law_model_3(*args, STY_inf=0)


def exp_model_3(t, STY_0, n, STY_inf):
    return STY_0 * np.exp(- n * t) + STY_inf


def exp_model_2(*args):
    return exp_model_3(*args, STY_inf=0)


def langmuir_model_3(t, STY_0, n, STY_inf):
    return STY_0 / (1 + n*t) + STY_inf


def langmuir_model_2(*args):
    return langmuir_model_3(*args, STY_inf=0)


def analyze_fit(popt, pcov):

    # Compute metrics from popt and pcov

    # Standard deviation
    perr = np.sqrt(np.diag(pcov))

    # Coefficient of variation
    cv = np.where(popt < 0.00001, 0, np.abs(perr/popt))

    # Correlation matrix; denom is 0 wherever a parameter has perr == 0 (e.g. fixed at a bound),
    # so divide-by-zero there is expected and replaced with 0 instead of nan/inf
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.outer(perr, perr)
        pcorr = np.where(denom != 0, pcov / denom, 0)
    if np.isnan(pcorr).sum() != 0:
        raise ValueError("pcorr contains unexpected NaNs after the zero-division guard")

    return perr, cv, pcorr


def fit_model(t, sty, model, p0, bounds):

    # Return the parameters of the fitted model, and the metrics

    # If sty == 0, no fitting needed
    if sty[0] < 1e-10:
        popt  = np.zeros(len(p0))
        perr  = np.zeros(len(p0))
        cv    = np.zeros(len(p0))
        pcorr = np.eye(len(p0))
        return popt, perr, cv, pcorr, 1, 0

    # Fit linear regression
    slope, intercept, _, _, _ = linregress(t, sty)

    # Facilitate convergence with bounds and initialization
    p0[0] = intercept  # Intercept of linreg as initial STY value
    bounds[1][0] = intercept*2 + 0.1  # Max STY_0 value
    if len(p0) >= 3:
        bounds[1][-1] = intercept*2 + 0.1  # Max STY_inf value
        p0[-1] = sty.min()
    
    # Fit function
    popt, pcov = curve_fit(model, t, sty, p0=p0, bounds=bounds, maxfev=50000, ftol=1e-10, xtol=1e-10)

    # Calculate sty_pred and sty_mean
    sty_pred = model(t, *popt)
    sty_mean = sty.mean()

    # Calculate metrics
    r2 = r2_score(sty, sty_pred)
    nrmse = 0 if abs(sty_mean) < 1e-10 else root_mean_squared_error(sty, sty_pred) / sty_mean
    perr, cv, pcorr = analyze_fit(popt, pcov)

    return popt, perr, cv, pcorr, r2, nrmse

