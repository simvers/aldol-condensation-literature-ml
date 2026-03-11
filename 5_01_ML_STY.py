import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
import shap
from data.ML_models_STY.ML_models_STY import model_config
from functions.functions_MLmodels import save_output, load_model_from_tmp, get_learning_curve, plot_learning_curve, plot_cv_distribution, plot_feature_output, plot_partial_dependence, ContinuousStratifiedKFold


# Configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8
mycolor = 'cividis'

# Random state
rs = 8

# ------------------------------------------------------------------------------------------------------------------

# Import data
data = pd.read_csv("data/catalysts/data_engineered.csv")
# data = pd.read_csv("data/catalysts/data_engineered_noSi.csv")
elements = pd.read_csv('data/catalysts/elements.csv', header=None).squeeze('columns').to_list()
eng_feat = pd.read_csv('data/catalysts/comp_features.csv', header=None).squeeze('columns').to_list()
print(data.columns, len(data))

# Cluster colors
cluster_list = data['Cluster_title']
clusters = np.sort(cluster_list.unique())
cmap = plt.get_cmap(mycolor)
palette = cmap(np.linspace(0, 1, len(clusters)))

# Input features
input = [
    'Ac_mmolming', 'Fa_mmolming', 'MeOH_mmolming', 'Water_mmolming',
    # 'Ac_source', 'Fa_source', 'Stabilizer', 
    'Temperature_K', 'O_content',
    'av_cov_rad', 'av_n_val', 'SSA_m2g',
    # 'var_cov_rad', 'var_n_val',
    'var_pca',
] 
output = 'STY_Acryl_mmolhg'

# ------------------------------------------------------------------------------------------------------------------

# Drop observations with missing output
data.dropna(subset=output, inplace=True)
assert ~data[input].isna().sum().all()
print('Data with STY: ', len(data))

# Features and target
# x = data.drop(columns=output)
x = data[input]
y = np.log1p(data[output])

# Train-test split
# Split as function of quantiles
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=rs)
qcut = pd.qcut(y, q=5, duplicates="drop")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=qcut, random_state=rs)

# ------------------------------------------------------------------------------------------------------------------

# Correlation of features
norm_array = StandardScaler().fit_transform(x)
corr_matrix = np.abs(np.corrcoef(norm_array, rowvar=False))
fig = plt.figure()
sns.heatmap(corr_matrix, cmap='plasma_r', xticklabels=x.columns, yticklabels=x.columns, vmin=0, vmax=1, square=True)
fig.savefig(f"figures/ML_STY/feature_correlation.png", dpi=600, bbox_inches='tight')

# Correlation with STY

# Figure
fig = plt.figure(figsize=(10, 25))
gs = fig.add_gridspec(5, 2)
# gs = np.ravel(gs)
ax = []

# Partial dependence plot
for i, feature in enumerate(x.columns):

    # Create subplot
    ax.append(gs[i//2, i%2].subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0, hspace=0))
    ax_joint = fig.add_subplot(ax[i][1, 0])
    ax_x = fig.add_subplot(ax[i][0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(ax[i][1, 1], sharey=ax_joint)
    
    # Scatter and regression plot
    sns.scatterplot(x=x.loc[:, feature], y=y, ax=ax_joint, hue=cluster_list, hue_order=clusters, palette=palette)
    sns.regplot(x=x.loc[:, feature], y=y, ax=ax_joint, order=1, scatter=False, line_kws={"color": 'k'})
    # sns.regplot(x=x.loc[:, feature], y=y, ax=ax[i], order=1, scatter=False, hue=cluster_list, hue_order=clusters, palette=palette)

    # Labels
    ax_joint.set(xlabel=feature, ylabel='STY')

    ax_joint.get_legend().remove()
    sns.kdeplot(ax=ax_x, x=x.loc[:, feature], hue=cluster_list, hue_order=clusters, palette=palette, cut=0)
    sns.kdeplot(ax=ax_y, y=y, hue=cluster_list, hue_order=clusters, palette=palette, cut=0)
    for item in [ax_x, ax_y]:
        item.set_axis_off()
        item.get_legend().remove()
fig.savefig(f"figures/ML_STY/target_correlation.png", dpi=600, bbox_inches='tight')


# ------------------------------------------------------------------------------------------------------------------

# Preprocessing

# Identify numerical and categorical features
categorical_cols = x.select_dtypes(include=['object', 'category']).columns
numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
], verbose_feature_names_out=False)

# ------------------------------------------------------------------------------------------------------------------

# Loop over and optimize models
model_to_train = ['best_model']  #  best_model, xgboost, rf, knn

# Train models
# for model, config in model_config.items():
for model in model_to_train:

    # Get model configuration
    if model not in model_config.keys():
        print(f'Model {model} not in config json')
        break
    config = model_config.get(model)
    print(f'Training model {model}')

    # --------------------------

    # Train model

    # ML pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', config.get('model'))
    ])

    # Bayesian hyperparameter optimization
    # bayes_search = BayesSearchCV(pipe, config.get('param_grid'), cv=10, n_iter=100, scoring='r2', random_state=8)
    # bayes_search.fit(x_train, y_train)

    # Callback instances to check bayes search convergence
    # tracker = OptimizationTracker()
    # convergence_checker = ConvergenceChecker()
    # Executing the pipline with train_data with callback
    # bayes_search.fit(X_train, y_train, callback=[tracker, convergence_checker])

    # Gridsearch hyperparameter optimization
    # cv = KFold(n_splits=10, shuffle=True)
    cv = ContinuousStratifiedKFold(n_splits=5, n_iter=50, random_state=rs)
    grid_search = GridSearchCV(estimator=pipe, param_grid=config.get('param_grid'), cv=cv, scoring='r2',
                                n_jobs=-1, verbose=2, return_train_score=True
    )
    grid_search.fit(x_train, y_train)

    # Extract best model
    best_model = grid_search.best_estimator_

    # Train and test score
    train_score = best_model.score(x_train, y_train)
    test_score = best_model.score(x_test, y_test)
    print(f"{model} train score:{train_score : .2f}")
    print(f"{model} test score:{test_score : .2f}")

    # Save model and results
    path = './data/tmp/'
    save_output(model + '_gridsearch_feateng', path, (x_train, y_train), (x_test, y_test), grid_search)
    print(f"Model saved to {path}")

    # --------------------------

    # Learning curve
    train_sizes, train_scores, val_scores = get_learning_curve(best_model, x_train, y_train, cv=cv, scoring="r2")

    # Dual plot
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    # Plot average learning curve
    plot_learning_curve(train_sizes, train_scores, val_scores, ax=ax[0], title=None)

    # Plot correlation between experimental and predicted values
    # sns.regplot(x=np.expm1(y_test), y=np.expm1(best_pipe.predict(X_test)), ax=ax[1])
    # sns.regplot(x=y_test, y=best_model.predict(x_test), ax=ax[1])
    sns.scatterplot(x=y_test, y=best_model.predict(x_test), ax=ax[1])
    max_ = 4  # max(ax[1].get_xlim()[-1], ax[1].get_ylim()[-1])
    sns.lineplot(x=[0, max_], y=[0, max_], ax=ax[1])
    ax[1].text(0.2, 0.8, f"test score: {test_score : .2f}", transform = ax[1].transAxes)
    ax[1].set(xlabel="ln( 1 + Experimental STY / mmol h$^{-1}$ g$^{-1}$ )", ylabel="ln( 1 + Predicted STY / mmol h$^{-1}$ g$^{-1}$ )",
              xlim=[0, max_], ylim=[0, max_], )

    fig.tight_layout()
    fig.savefig(f"figures/ML_STY/{model}/{model}_grid_feateng_predictions.png", dpi=600, bbox_inches='tight')

    # Dual plot
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    # Plot average learning curve
    plot_learning_curve(train_sizes, train_scores, val_scores, ax=ax[0], title=None)

    # Plot fold-level distribution
    plot_cv_distribution(train_sizes, val_scores, ax=ax[1], title=None)

    fig.tight_layout()
    fig.savefig(f"figures/ML_STY/{model}/{model}_grid_feateng_learning_curve.png", dpi=600, bbox_inches='tight')

    # --------------------------

    # Feature importance using permutation importance

    # Only if model is not knn
    if model == 'knn':
        print('No feature importance or shapley analysis for KNN models')
        break

    # Feature names after encoding
    if categorical_cols.empty:
        feature_names = numerical_cols.tolist()
    else:
        onehot_feature_names = best_model.named_steps['preprocessor'] \
            .named_transformers_['cat'] \
            .get_feature_names_out(categorical_cols)
        feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()

    # if config.get('func_feature_importance'):
    #     for func in config.get('func_feature_importance'):
    #         func(opt, feature_names, x_test, y_test)

    # Feature importance
    importances = permutation_importance(best_model, x_test, y_test, n_repeats=10, random_state=rs, n_jobs=-1)
    sorted_idx = importances.importances_mean.argsort()[::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=importances.importances_mean[sorted_idx],
                y=pd.Series(feature_names)[sorted_idx],
                orient='h', ax=ax)
    ax.set(xlabel='Permutation feature importance - test set', ylabel=None, xticks=[])
    fig.tight_layout()
    fig.savefig(f"figures/ML_STY/{model}/{model}_grid_feateng_pfi.png", dpi=600, bbox_inches='tight')

    # --------------------------

    # Shapley analysis

    # Preprocess x and extract ml_model
    preprocess = best_model.named_steps['preprocessor'] 
    feature_names = preprocess.get_feature_names_out()
    ml_model = best_model.named_steps['model']
    x_train_shap = pd.DataFrame(preprocess.transform(x_train), columns=feature_names)
    x_test_shap = pd.DataFrame(preprocess.transform(x_test), columns=feature_names)

    # Initialize explainer, x_train for expected output value
    explainer = shap.Explainer(ml_model, x_train_shap)  # automatically select shap.TreeExplainer
    shap_values = explainer(x_test_shap)
    
    fig, ax = plot_partial_dependence(model=ml_model, x=x_train_shap)
    fig.savefig(f"figures/ML_STY/{model}/{model}_grid_feateng_trainshap.png", dpi=600, bbox_inches='tight')
    fig, ax = plot_partial_dependence(model=ml_model, x=x_test_shap)
    fig.savefig(f"figures/ML_STY/{model}/{model}_grid_feateng_testshap.png", dpi=600, bbox_inches='tight')

    fig, ax = plt.subplots()
    shap.plots.beeswarm(shap_values, ax=ax, show=False, plot_size=None)  #, color=plt.get_cmap("red_blue")
    ax.set(xlabel='SHAP value')
    fig.savefig(f"figures/ML_STY/{model}/{model}_grid_feateng_beeswarm.png", dpi=600, bbox_inches='tight')
        
