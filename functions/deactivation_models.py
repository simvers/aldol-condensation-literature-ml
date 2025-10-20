import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.stats import linregress


def power_law_model_3(t, STY_0, n, STY_inf):
    # return (STY_0) / (1 + t)**n + STY_inf
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
    cv = np.abs(perr/popt)

    # Correlation matrix and check for nans
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.outer(perr, perr)
        pcorr = np.where(denom != 0, pcov / denom, 0)
    assert np.isnan(pcorr).sum() == 0

    return perr, cv, pcorr


def fit_model(t, sty, model, p0, bounds):

    # Return the parameters of the fitted model, and the metrics

    # If sty == 0, no fitting needed
    if sty[0] == 0:
        popt = np.array([0, 0, 0][:len(p0)])
        perr = np.array([0, 0, 0][:len(p0)])
        cv = np.array([0, 0, 0][:len(p0)])
        pcorr = np.eye(len(p0))
        r2 = 1
        nrmse = 0
        return popt, perr, cv, pcorr, r2, nrmse

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
    nrmse = 0 if sty_mean == 0 else root_mean_squared_error(sty, sty_pred)/sty_mean
    perr, cv, pcorr = analyze_fit(popt, pcov)

    return popt, perr, cv, pcorr, r2, nrmse


# ---------------------------------------------------------------------------------

if __name__ == '__main__':

    # Set variables range
    t = np.linspace(0, 50, 26)
    sty_0 = 10
    sty_inf = np.linspace(0, 5, 6)
    n_power = np.linspace(0, 1, 101)  # from stable to linear deactivation
    n_exp = np.linspace(0, 1, 11)  # from stable to linear deactivation

    # Models
    models = [power_law_model_3, exp_model_3, langmuir_model_3]

    # Colors
    colors = mplt.colormaps.get_cmap('coolwarm')(np.linspace(0, 1, len(n_power)))

    for model in models:
        fig, ax = plt.subplots(2, int(len(sty_inf)/2))
        for i, sty_inf_val in enumerate(sty_inf):

            # Select subplot
            row = int(i//(len(sty_inf)/2))
            col = int(i%(len(sty_inf)/2))

            # Format plot
            ax[row, col].set(xlim=(0, 50), ylim=(0, 12))

            # Plot curve for every deactivation rate
            for j, n_val in enumerate(n_power):
                ax[row, col].plot(t, model(t, sty_0, n_val, sty_inf_val), color=colors[j])

        plt.show()

