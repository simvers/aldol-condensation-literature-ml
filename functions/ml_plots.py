import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import StandardScaler
import shap


def round_50(x):
    return int(math.ceil(x / 50.0)) * 50


# ------------------------------------------------------------------------------------------------------------------

# Input data plots

def plot_feature_correlation(x, numerical_cols, save_path="figures/ML_STY/feature_correlation.svg"):
    if len(x.columns) <= 1:
        return
    norm_array = StandardScaler().fit_transform(x[numerical_cols])
    corr_matrix = np.abs(np.corrcoef(norm_array, rowvar=False))
    fig = plt.figure()
    sns.heatmap(corr_matrix, cmap='plasma_r', xticklabels=x.columns, yticklabels=x.columns, vmin=0, vmax=1, square=True)
    fig.savefig(save_path, dpi=600, bbox_inches='tight')


def plot_target_correlation(x, y, numerical_cols, cluster_list, clusters, palette, save_path="figures/ML_STY/target_correlation.png"):
    fig = plt.figure(figsize=(10, 15))
    gs = fig.add_gridspec(int(np.ceil(len(x.columns)/2)), 2)
    ax = []

    for i, feature in enumerate(numerical_cols):
        ax.append(gs[i//2, i%2].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
        ax_joint = fig.add_subplot(ax[i][1, 0])
        ax_x = fig.add_subplot(ax[i][0, 0], sharex=ax_joint)
        ax_y = fig.add_subplot(ax[i][1, 1], sharey=ax_joint)

        sns.scatterplot(x=x.loc[:, feature], y=y, ax=ax_joint, hue=cluster_list, hue_order=clusters, palette=palette)
        sns.regplot(x=x.loc[:, feature], y=y, ax=ax_joint, order=1, scatter=False, line_kws={"color": 'k'})
        ax_joint.set(xlabel=feature, ylabel='STY', ylim=(0, 4), yticks=[0, 1, 2, 3, 4])
        ax_joint.get_legend().remove()
        sns.kdeplot(ax=ax_x, x=x.loc[:, feature], hue=cluster_list, hue_order=clusters, palette=palette, cut=0)
        sns.kdeplot(ax=ax_y, y=y, hue=cluster_list, hue_order=clusters, palette=palette, cut=0)
        for item in [ax_x, ax_y]:
            item.set_axis_off()
            item.get_legend().remove()
    fig.savefig(save_path, dpi=600, bbox_inches='tight')


def plot_observations_per_publication(data, output):
    # How many data points come from the same paper - motivates grouping CV folds by doi below,
    # since a paper with many points could otherwise leak into both train and validation folds
    observations = data.groupby('doi').count()[output].values
    print('Data with STY: ', len(data), 'over ', len(data['doi'].unique()), 'max ', observations.max(), 'mean ', observations.mean())

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.histplot(ax=ax, x=observations, discrete=True, binwidth=1, color="#AD1457", alpha=0.8)
    ax.axvline(x=observations.mean(), color="#AD1457", linestyle='--', linewidth=0.75)
    ax.set(xlabel='Observations per publication /', ylabel='Count /')
    fig.savefig('figures/ML_STY/observations_count.png', dpi=600)


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
                        title="Learning Curve", ylabel="R² Score", ax=None, model='My model'):
    """
    Plot mean learning curve with error bands.
    """
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    ax.set(xlim=(0, round_50(train_sizes.max())), ylim=(0,1))
    ax.set_title(title)
    ax.set_xlabel("Training examples")
    ax.set_ylabel(ylabel)

    # Training score curve
    ax.plot(train_sizes, train_scores_mean, 'o-', color='#D4898A', label=f"{model} training")
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color='#D4898A')

    # Cross-validation score curve
    ax.plot(train_sizes, val_scores_mean, 'o-', color='#7EC8A8', label=f"{model} cross-validation")
    ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                     val_scores_mean + val_scores_std, alpha=0.2, color='#7EC8A8')

    ax.legend(loc="upper left", frameon=False)
    # ax.grid(True)


def plot_parity_curve(y_test, y_pred, test_score, ax=None, max_=4, score='test', model='My model'):

    """
    Plot correlation between experimental and predicted values
    """
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, color='#89ABD4')
    sns.lineplot(x=[0, max_], y=[0, max_], ax=ax, color='#89ABD4')
    ax.text(0.05, 0.9, f"{model} {score} score: {test_score : .2f}", transform = ax.transAxes)
    ax.set(xlabel="ln( 1 + Experimental STY / mmol h$^{-1}$ g$^{-1}$ )", ylabel="ln( 1 + Predicted STY / mmol h$^{-1}$ g$^{-1}$ )",
           xlim=[0, max_], ylim=[0, max_], xticks=np.linspace(0, max_, 5), yticks=np.linspace(0, max_, 5))


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


def plot_model_diagnostics(model_name, model_abb, best_model, x_train, y_train, x_test, y_test, cv,
                            cluster_train, cluster_test, clusters, palette, figure_dir,
                            parity_score_label='test', compute_shap=True):
    # Produce the standard diagnostic figures for a fitted model: learning curve + parity,
    # learning curve + CV-fold distribution, and (if compute_shap) SHAP partial dependence,
    # clustered scatter, and beeswarm plots for both train and test sets. compute_shap=False
    # for models (e.g. KNN) that SHAP can't explain efficiently.

    train_sizes, train_scores, val_scores = get_learning_curve(best_model, x_train, y_train, cv=cv, scoring="r2")
    test_score = best_model.score(x_test, y_test)

    # Predictions figure: learning curve + parity
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    plot_learning_curve(train_sizes, train_scores, val_scores, ax=ax[0], title=None, model=model_abb)
    plot_parity_curve(y_test, best_model.predict(x_test), test_score, ax=ax[1], score=parity_score_label, model=model_abb)
    fig.tight_layout()
    fig.savefig(f"{figure_dir}/{model_name}_grid_feateng_predictions.png", dpi=600, bbox_inches='tight')

    # Learning curve figure: learning curve + CV-fold distribution
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    plot_learning_curve(train_sizes, train_scores, val_scores, ax=ax[0], title=None, model=model_abb)
    plot_cv_distribution(train_sizes, val_scores, ax=ax[1], title=None)
    fig.tight_layout()
    fig.savefig(f"{figure_dir}/{model_name}_grid_feateng_learning_curve.png", dpi=600, bbox_inches='tight')

    if not compute_shap:
        return

    # Shapley analysis

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
    fig, ax = plot_partial_dependence(model=ml_model, x=x_train_shap)
    fig.savefig(f"{figure_dir}/{model_name}_grid_feateng_trainshap.png", dpi=600, bbox_inches='tight')
    fig, ax = plot_partial_dependence(model=ml_model, x=x_test_shap)
    fig.savefig(f"{figure_dir}/{model_name}_grid_feateng_testshap.png", dpi=600, bbox_inches='tight')

    # SHAP plots
    fig, ax = plot_shap_values(model=ml_model, x=x_train_shap, shap_values=train_shap_values, hue=cluster_train, hue_order=clusters, palette=palette)
    fig.savefig(f"{figure_dir}/{model_name}_grid_feateng_trainshap_clustered.png", dpi=600, bbox_inches='tight')
    fig, ax = plot_shap_values(model=ml_model, x=x_test_shap, shap_values=test_shap_values, hue=cluster_test, hue_order=clusters, palette=palette)
    fig.savefig(f"{figure_dir}/{model_name}_grid_feateng_testshap_clustered.png", dpi=600, bbox_inches='tight')

    fig, ax = plt.subplots()
    shap.plots.beeswarm(test_shap_values, ax=ax, show=False, plot_size=None, max_display=15)
    ax.set(xlabel='SHAP value')
    fig.savefig(f"{figure_dir}/{model_name}_grid_feateng_beeswarm.png", dpi=600, bbox_inches='tight')


# ------------------------------------------------------------------------------------------------------------------

# Shapley analysis

def plot_partial_dependence(model, x: pd.DataFrame, individual_plot=False):

    # Compute expected values
    expected_output_val = model.predict(x).mean()
    expected_feature_val = x.mean()

    # Figure
    fig, ax = plt.subplots(int(np.ceil(len(x.columns)/2)), 2, figsize=(10, int(np.ceil(len(x.columns)/2))*5))
    ax = ax.ravel()

    # Partial dependence plot
    for i, feature in enumerate(x.columns):

        # Plot feature partial dependence
        results = partial_dependence(model, x, features=[i], kind='average')
        ax[i].scatter(results['grid_values'][0], results['average'][0], color='k')

        # Labels
        ax[i].set(xlabel=feature, ylabel=f'E[f(x) | {feature}]', ylim=(0.7, 2.3))

        # Plot hist
        ax_hist = ax[i].inset_axes([0, 0, 1, 0.2], zorder=0)  # [x, y, width, height]
        ax_hist.hist(x.loc[:, feature], color='grey', bins=20)
        ax_hist.set(ylim=(0, 70))
        ax_hist.axis('off')

        # Display expected values
        ax[i].axhline(y=expected_output_val, linestyle='--', color='grey')
        ax[i].axvline(x=expected_feature_val[feature], linestyle='--', color='grey')
        y_min, y_max = ax[i].get_ylim()
        ax[i].text(x=expected_feature_val[feature], y=y_max+(y_max-y_min)*0.02, s=f'E[{feature}]', horizontalalignment='center')
        x_min, x_max = ax[i].get_xlim()
        ax[i].text(x=x_max+(x_max-x_min)*0.02, y=expected_output_val, s=f'E[f(x)]', verticalalignment='center', rotation = 270)

        # Partial dependence plot
        if individual_plot:
            # Plot feature partial dependence
            fig_i, ax_i = shap.plots.partial_dependence(feature, model.predict, x, model_expected_value=True, feature_expected_value=True, show=False, ice=False,)
            ax_i.scatter(results['grid_values'][0], results['average'][0], color='k')
            fig_i.savefig(f"figures/ML_STY/model_grid_feateng_{feature}.png", dpi=600, bbox_inches='tight')

    return fig, ax


def plot_shap_values(model, x: pd.DataFrame, shap_values, hue, hue_order, palette, ylim=(0.5, 2.5)):

    # Compute expected values
    expected_output_val = model.predict(x).mean()
    expected_feature_val = x.mean()

    # Figure
    fig, ax = plt.subplots(int(np.ceil(len(x.columns)/2)), 2, figsize=(10, int(np.ceil(len(x.columns)/2))*5))
    ax = ax.ravel()

    # Partial dependence plot
    for i, feature in enumerate(x.columns):

        # Plot feature partial dependence
        sns.scatterplot(x=x.loc[:, feature], y=shap_values.values[:, i] + expected_output_val, ax=ax[i], hue=hue.to_numpy(), hue_order=hue_order, palette=palette)

        # Labels
        ax[i].set(xlabel=feature, ylabel=f'E[f(x) | {feature}]', ylim=ylim)

        # Plot hist
        ax_hist = ax[i].inset_axes([0, 0, 1, 0.2], zorder=0)  # [x, y, width, height]
        ax_hist.hist(x.loc[:, feature], color='grey', bins=20)
        ax_hist.set(ylim=(0, 70))
        ax_hist.axis('off')

        # Display expected values
        ax[i].axhline(y=expected_output_val, linestyle='--', color='grey')
        ax[i].axvline(x=expected_feature_val[feature], linestyle='--', color='grey')
        y_min, y_max = ax[i].get_ylim()
        ax[i].text(x=expected_feature_val[feature], y=y_max+(y_max-y_min)*0.02, s=f'E[{feature}]', horizontalalignment='center')
        x_min, x_max = ax[i].get_xlim()
        ax[i].text(x=x_max+(x_max-x_min)*0.02, y=expected_output_val, s=f'E[f(x)]', verticalalignment='center', rotation = 270)

    return fig, ax


def plot_feature_output(x: pd.DataFrame, y):

    fig, ax = plt.subplots(5, 2, figsize=(8, 20))
    ax = ax.ravel()
    # Partial dependence plot
    for i, feature in enumerate(x.columns):

        # Plot feature partial dependence
        ax[i].scatter(x.loc[:, feature], y, color='k')
        ax[i].set(xlabel=feature, ylabel=f'log1p(STY)', ylim=(0, 4))

    return fig, ax
