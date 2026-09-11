import json
import math
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import learning_curve
from sklearn.inspection import partial_dependence
import shap

_labels_path = Path(__file__).resolve().parents[1] / 'config/feature_labels.json'
with open(_labels_path) as _f:
    _labels = json.load(_f)
LONG_LABELS = _labels['long']
SHORT_LABELS = _labels['short']


def round_50(x):
    return int(math.ceil(x / 50.0)) * 50


# ------------------------------------------------------------------------------------------------------------------

# Model performance plots

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
                        ylabel="R² Score", ax=None, model='My model'):
    """
    Plot mean learning curve with error bands.
    """
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    ax.set(xlim=(0, round_50(train_sizes.max())), ylim=(0,1))
    ax.set_xlabel("Training examples")
    ax.set_ylabel(ylabel)

    # Training score curve
    ax.plot(train_sizes, train_scores_mean, 'o-', color='#D4898A', label=f"{model} training")
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color='#D4898A')

    # Cross-validation score curve
    ax.plot(train_sizes, val_scores_mean, 'o-', color='#7EC8A8', label=f"{model} CV")
    ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.2, color='#7EC8A8')

    ax.set_box_aspect(1)
    ax.legend(loc="upper left", frameon=False)
    # ax.grid(True)


def plot_parity_curve(y_test, y_pred, test_score, ax=None, max_=4, score='test', model='My model',
                      target='STY / mmol h$^{-1}$ g$^{-1}$', log1p=True):

    """
    Plot correlation between experimental and predicted values
    """
    prefix, suffix = ('ln( 1 + ', ' )') if log1p else ('', '')
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, color='#89ABD4')
    sns.lineplot(x=[0, max_], y=[0, max_], ax=ax, color='#89ABD4')
    ax.text(0.05, 0.9, f"{model} {score} score: {test_score : .2f}", transform = ax.transAxes)
    ax.set(xlabel=f"{prefix}Experimental {target}{suffix}", 
           ylabel=f"{prefix}Predicted {target}{suffix}",
           xlim=[0, max_], ylim=[0, max_], xticks=np.linspace(0, max_, 5), yticks=np.linspace(0, max_, 5))
    ax.set_aspect('equal')


def plot_cv_distribution(train_sizes, val_scores, ylabel="R² per Fold", ax=None):
    """
    Plot boxplots of cross-validation scores for each training size.
    """
    sns.boxplot(data=[val_scores[i] for i in range(len(train_sizes))], ax=ax)
    ax.set_xticks(ticks=range(len(train_sizes)), labels=train_sizes.astype(int), rotation=45)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax.set_ylim(0,1)


def plot_model_diagnostics(model_name, model_abb, best_model, x_train, y_train, x_test, y_test, cv,
                            figure_dir, parity_score_label='test', parity_kwargs=None):
    # Produce the standard diagnostic figures for a fitted model: learning curve + parity,
    # learning curve + CV-fold distribution

    train_sizes, train_scores, val_scores = get_learning_curve(best_model, x_train, y_train, cv=cv, scoring="r2")
    test_score = best_model.score(x_test, y_test)

    # Predictions figure: learning curve + parity
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    plot_learning_curve(train_sizes, train_scores, val_scores, ax=ax[0], model=model_abb)
    plot_parity_curve(y_test, best_model.predict(x_test), test_score, ax=ax[1],
                      score=parity_score_label, model=model_abb, **(parity_kwargs or {}))
    ax[0].text(-0.2, 1.1, 'a', fontsize=10, fontweight='bold', fontfamily='arial', transform=ax[0].transAxes)
    ax[1].text(-0.2, 1.1, 'b', fontsize=10, fontweight='bold', fontfamily='arial', transform=ax[1].transAxes)
    fig.tight_layout()
    fig.savefig(f"{figure_dir}/{model_name}_predictions.svg", dpi=600, bbox_inches='tight')

    # Learning curve figure: learning curve + CV-fold distribution
    # fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    # plot_learning_curve(train_sizes, train_scores, val_scores, ax=ax[0], model=model_abb)
    # plot_cv_distribution(train_sizes, val_scores, ax=ax[1])
    # fig.tight_layout()
    # fig.savefig(f"{figure_dir}/{model_name}_learning_curve.svg", dpi=600, bbox_inches='tight')


# ------------------------------------------------------------------------------------------------------------------

# Shapley analysis

def plot_partial_dependence(model, x: pd.DataFrame, shap_values=None):

    # Sort features by mean |SHAP| descending (matches beeswarm / SHAP scatter order)
    if shap_values is not None:
        mean_abs = np.mean(np.abs(shap_values.values), axis=0)
        col_order = np.argsort(mean_abs)[::-1]
        features_sorted = x.columns[col_order].tolist()
    else:
        features_sorted = x.columns.tolist()

    # Compute expected values
    expected_output_val = model.predict(x).mean()
    expected_feature_val = x.mean()

    # Sort features
    n_feats = len(features_sorted)
    n_col = 3 if n_feats >= 9 else 2  # switch to a 3rd column once the grid gets tall
    n_rows = int(np.ceil(n_feats / n_col))

    # Figure
    fig, axes = plt.subplots(n_rows, n_col, figsize=(n_col * 4, n_rows * 4),
                              gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    axes = axes.ravel()

    # Partial dependence subplots
    for plot_i, feature in enumerate(features_sorted):
        feat_col_idx = x.columns.get_loc(feature)
        ax = axes[plot_i]

        # Plot feature partial dependence
        results = partial_dependence(model, x, features=[feat_col_idx], kind='average')
        ax.scatter(results['grid_values'][0], results['average'][0], color='k')

        # Labels
        long_label = LONG_LABELS.get(feature, feature)
        ax.set(xlabel=long_label, ylabel='E[f(x) | x$_i$]', ylim=(0.7, 2.3))

        # Plot hist
        ax_hist = ax.inset_axes([0, 0, 1, 0.2], zorder=0)
        ax_hist.hist(x.loc[:, feature], color='grey', bins=20)
        ax_hist.set(ylim=(0, 70))
        ax_hist.axis('off')

        # Display expected values
        ax.axhline(y=expected_output_val, linestyle='--', color='grey')
        ax.axvline(x=expected_feature_val[feature], linestyle='--', color='grey')
        y_min, y_max = ax.get_ylim()
        ax.text(expected_feature_val[feature], y_max + (y_max - y_min) * 0.02,
                'E[x$_i$]', ha='center', fontsize=7)
        x_min, x_max = ax.get_xlim()
        ax.text(x_max + (x_max - x_min) * 0.02, expected_output_val,
                'E[f(x)]', va='center', rotation=270, fontsize=7)

        ax.text(-0.2, 1.1, string.ascii_lowercase[plot_i],
                fontsize=10, fontweight='bold', fontfamily='arial', transform=ax.transAxes)
        ax.set_box_aspect(1)

    for ax in axes[n_feats:]:
        ax.axis('off')

    return fig, axes


def plot_shap_values(model, x: pd.DataFrame, shap_values, hue, hue_order, palette,
                     ylim=(0.5, 2.5), x_display=None):
    # x_display: original (unscaled) DataFrame. When provided, numerical features use
    # original units on the x-axis; OHE columns not present in x_display fall back to x.

    from matplotlib.lines import Line2D

    # Sort features by mean |SHAP| descending (matches beeswarm order)
    mean_abs = np.mean(np.abs(shap_values.values), axis=0)
    col_order = np.argsort(mean_abs)[::-1]
    features_sorted = x.columns[col_order].tolist()

    n_feats = len(features_sorted)
    n_col = 3 if n_feats >= 9 else 2  # switch to a 3rd column once the grid gets tall
    n_rows = int(np.ceil(n_feats / n_col))

    expected_output_val = model.predict(x).mean()

    fig, axes = plt.subplots(n_rows, n_col, figsize=(n_col * 3, n_rows * 3),
                              gridspec_kw={'hspace': 0.5, 'wspace': 0.5})
    axes = axes.ravel()

    for plot_i, feature in enumerate(features_sorted):
        feat_idx = col_order[plot_i]
        ax = axes[plot_i]

        # Use original values for numerical features; fall back to scaled for OHE columns
        use_display = x_display is not None and feature in x_display.columns
        x_feat = x_display[feature] if use_display else x.loc[:, feature]
        feat_mean = x_feat.mean()

        long_label = LONG_LABELS.get(feature, feature)

        sns.scatterplot(x=x_feat,
                        y=shap_values.values[:, feat_idx] + expected_output_val,
                        ax=ax, hue=hue.to_numpy(), hue_order=hue_order, palette=palette,
                        legend=False)

        ax.set(xlabel=long_label, ylabel='E[f(x)] + φ(x$_i$)', ylim=ylim, yticks=(0.5, 1, 1.5, 2, 2.5))
        ax.text(-0.2, 1.1, string.ascii_lowercase[plot_i],
                fontsize=10, fontweight='bold', fontfamily='arial', transform=ax.transAxes)

        ax_hist = ax.inset_axes([0, 0, 1, 0.2], zorder=0)
        ax_hist.hist(x_feat, color='grey', bins=20)
        ax_hist.set(ylim=(0, 70))
        ax_hist.axis('off')

        ax.axhline(y=expected_output_val, linestyle='--', color='grey')
        ax.axvline(x=feat_mean, linestyle='--', color='grey')
        y_min, y_max = ax.get_ylim()
        ax.text(feat_mean, y_max + (y_max - y_min) * 0.02,
                f'E[x$_i$]', ha='center', fontsize=7)
        x_min, x_max = ax.get_xlim()
        ax.text(x_max + (x_max - x_min) * 0.02, expected_output_val,
                f'E[f(x)]', va='center', rotation=270, fontsize=7)
        ax.set_box_aspect(1)

    # Single legend: in the first empty trailing subplot if any, else in the top-left plot
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i],
                              label=str(label), markersize=6)
                      for i, label in enumerate(hue_order)]
    if n_feats < len(axes):
        for ax in axes[n_feats + 1:]:
            ax.axis('off')
        legend_ax = axes[n_feats]
        legend_ax.axis('off')
        legend_ax.legend(handles=legend_handles, loc='center', frameon=False)
    else:
        axes[0].legend(handles=legend_handles, loc='best', frameon=False, fontsize=7)

    return fig, axes


def plot_feature_output(x: pd.DataFrame, y):

    fig, ax = plt.subplots(5, 2, figsize=(8, 20))
    ax = ax.ravel()
    # Partial dependence plot
    for i, feature in enumerate(x.columns):

        # Plot feature partial dependence
        ax[i].scatter(x.loc[:, feature], y, color='k')
        ax[i].set(xlabel=feature, ylabel=f'log1p(STY)', ylim=(0, 4))

    return fig, ax


def plot_shapley_analysis(model_name, model_abb, best_model, x_train, x_test,
                            cluster_train, cluster_test, clusters, palette, figure_dir):

    # Shapley analysis
    # clustered scatter, and beeswarm plots for both train and test sets

    # Preprocess x and extract ml_model
    preprocess = best_model.named_steps['preprocessor']
    feature_names = preprocess.get_feature_names_out()
    ml_model = best_model.named_steps['model']
    x_train_shap = pd.DataFrame(preprocess.transform(x_train), columns=feature_names)
    x_test_shap = pd.DataFrame(preprocess.transform(x_test), columns=feature_names)

    # Initialize explainer, x_train for expected output value
    explainer = shap.Explainer(ml_model, x_train_shap)  # automatically select shap.TreeExplainer
    train_shap_values = explainer(x_train_shap)
    test_shap_values = explainer(x_test_shap)

    # PDP plots
    fig, ax = plot_partial_dependence(model=ml_model, x=x_train_shap, shap_values=train_shap_values)
    fig.savefig(f"{figure_dir}/{model_name}_train_pdp.svg", dpi=600, bbox_inches='tight')
    # fig, ax = plot_partial_dependence(model=ml_model, x=x_test_shap)
    # fig.savefig(f"{figure_dir}/{model_name}_testshap.svg", dpi=600, bbox_inches='tight')

    # SHAP plots
    fig, ax = plot_shap_values(model=ml_model, x=x_train_shap, shap_values=train_shap_values,
                               hue=cluster_train, hue_order=clusters, palette=palette,
                               x_display=x_train)
    fig.savefig(f"{figure_dir}/{model_name}_train_shap.svg", dpi=600, bbox_inches='tight')
    # fig, ax = plot_shap_values(model=ml_model, x=x_test_shap, shap_values=test_shap_values,
    #                            hue=cluster_test, hue_order=clusters, palette=palette,
    #                            x_display=x_test)
    # fig.savefig(f"{figure_dir}/{model_name}_testshap_clustered.svg", dpi=600, bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
    shap_display = test_shap_values[:, :]
    shap_display.feature_names = [SHORT_LABELS.get(f, f) for f in test_shap_values.feature_names]
    shap.plots.beeswarm(shap_display, ax=ax, show=False, plot_size=None, max_display=15)
    fs = plt.rcParams.get('font.size', 8)
    ax.set(xlabel='SHAP value')
    ax.tick_params(labelsize=fs)
    ax.xaxis.label.set_fontsize(fs)
    ax.text(-0.2, 1.1, 'a', fontsize=10, fontweight='bold', fontfamily='arial', transform=ax.transAxes)
    ytick_labels = [l.get_text() for l in ax.get_yticklabels()]
    ytick_labels = ['Other features' if l.startswith('Sum of') else l for l in ytick_labels]
    ax.set_yticklabels(ytick_labels, fontsize=fs)
    for cb_ax in fig.get_axes():
        if cb_ax is not ax:
            cb_ax.tick_params(labelsize=fs)
            cb_ax.set_ylabel('Feature value', rotation=270, va='center', fontsize=fs)
    fig.savefig(f"{figure_dir}/{model_name}_beeswarm.svg", dpi=600, bbox_inches='tight')


# ------------------------------------------------------------------------------------------------------------------

# RS-robustness plots

def plot_cv_boxstrip(ax, cv_scores, ylabel='Best CV R²', color='#7EC8A8', point_color='#2E6F5E'):
    """Box + strip overlay for one set of CV scores on a given ax."""
    rng = np.random.default_rng(42)
    ax.boxplot(cv_scores, positions=[0], widths=0.35, patch_artist=True,
               boxprops=dict(facecolor=color, alpha=0.7),
               medianprops=dict(color='k', linewidth=1.5),
               whiskerprops=dict(linewidth=0.75),
               capprops=dict(linewidth=0.75),
               flierprops=dict(marker=''))
    ax.scatter(rng.uniform(-0.12, 0.12, len(cv_scores)), cv_scores,
               color=point_color, s=15, alpha=0.8, zorder=3)
    ax.axhline(cv_scores.mean(), color='k', linestyle='--', linewidth=0.75,
               label=f'mean = {cv_scores.mean():.3f}')
    ax.set(ylabel=ylabel)
    ax.set_xticks([])
    ax.legend(loc='upper right', frameon=False, fontsize=7)


def plot_shap_importance_stability(ax, importance_df, feature_order, N_RS, color='#7EC8A8'):
    """Horizontal bar chart: mean ± std of mean |SHAP| across RS runs, ordered by importance."""
    ordered_asc = [f for f in reversed(feature_order) if f in importance_df.index]
    mean_imp = importance_df.mean(axis=1).loc[ordered_asc]
    std_imp = importance_df.std(axis=1).loc[ordered_asc]
    tick_labels = [SHORT_LABELS.get(f, f) for f in mean_imp.index]
    ax.barh(tick_labels, mean_imp.values, xerr=std_imp.values,
            color=color, ecolor='k', capsize=3, linewidth=0)
    ax.set_xlabel(f'Mean |SHAP| / mean ± std across {N_RS} RS')


def plot_shap_beeswarm_mean(ax, mean_shap, std_shap, feature_order, feature_names_list, N_RS, plot_cbar=True):
    """Mean-SHAP beeswarm colored by relative std (clipped at 1)."""
    rng = np.random.default_rng(0)
    n_feats = len(feature_order)
    mean_abs_per_feature = np.abs(mean_shap).mean(axis=0)
    rel_std = std_shap / (mean_abs_per_feature[None, :] + 1e-6)
    rel_std_clipped = np.clip(rel_std, 0, 1)

    sc = None
    for i, feat in enumerate(feature_order[::-1]):
        feat_idx = feature_names_list.index(feat)
        y_pos = i + rng.normal(0, 0.1, mean_shap.shape[0])
        sc = ax.scatter(mean_shap[:, feat_idx], y_pos,
                        c=rel_std_clipped[:, feat_idx], cmap='YlOrRd',
                        vmin=0, vmax=1, s=8, alpha=0.8)

    ax.axvline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels([SHORT_LABELS.get(f, f) for f in feature_order[::-1]])
    ax.set_xlabel(f'Mean SHAP value across {N_RS} RS')

    if plot_cbar and sc is not None:
        cbar = ax.get_figure().colorbar(sc, ax=ax)
        cbar.set_label('Relative |SHAP| std', rotation=270, labelpad=12)
