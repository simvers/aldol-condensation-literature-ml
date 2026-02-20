import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from data.ML_models_n.ML_models_n import model_config_grid, model_config_bayes
from functions.functions_MLmodels import save_output, load_model_from_tmp
from helpers_for_sklearn import ContinuousStratifiedKFold
from visualization_helpers import get_learning_curve, plot_learning_curve, plot_cv_distribution


# Configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8


# ------------------------------------------------------------------------------------------------------------------

rs = 30

# Import data
df = pd.read_csv("data/data_deactivation.csv")
elements = pd.read_csv('data/elements.csv', header=None).squeeze('columns').to_list()

# Drop columns
df.drop(elements + ['Y_Acryl_Ac', 'Y_Acryl_Fa', 'doi', 'Link_to_excel', 'Cluster_title', 
        'Ac_mmolming', 'Fa_mmolming', 
        # 'MeOH_mmolming', 'Water_mmolming',
        'Ac_source', 'Fa_source', 'Stabilizer', 
        'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 'LHSV_mlhg',
        'STY_Acryl_mmolhg', 'Pressure_bar', # 'STY0', 'n',
        'Cluster_n'], axis=1, inplace=True)
print(df.columns)

df.drop(df.loc[df['n'] > 0.5, :].index, axis=0, inplace=True)

# Features and target
X = df.drop(columns="n")
y  = df["n"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

# ------------------------------------------------------------------------------------------------------------------

# Preprocessing

# Identify numerical and categorical features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)  # Drop one column, to avoid linear combination
])

# ------------------------------------------------------------------------------------------------------------------

# Loop over and optimize models
model_to_run = ['lgbm']  # xgboost

for model, config in model_config_grid.items():
    if model in model_to_run:

        # ML pipeline
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('reg', config.get('model'))
        ])

        # pipe = Pipeline(steps=[
        #     ('preprocessor', preprocessor),
        #     ('reg', RandomForestRegressor(max_depth= 12,max_features=0.38602124182347686,min_samples_leaf=8,min_samples_split=17, n_estimators=890))])

        # Setup hyperparameter optimization
        # opt = BayesSearchCV(pipe, config.get('search_space'), cv=10, n_iter=20, scoring='r2', random_state=rs)
        cv = ContinuousStratifiedKFold(n_splits=5, n_iter=50, random_state=rs)
        opt = GridSearchCV(estimator=pipe, param_grid=config.get('search_space'), cv=cv, scoring='r2', n_jobs=-1, verbose=2, return_train_score=True)

        # Fit model with train_data
        # opt = pipe
        opt.fit(X_train, y_train)

        # Extract best model
        best_model = opt.best_estimator_

        # Retrieve feature names from one hot encoding
        if categorical_cols.empty:
            feature_names = numerical_cols.tolist()
        else:
            onehot_feature_names = best_model.named_steps['preprocessor'] \
                .named_transformers_['cat'] \
                .get_feature_names_out(categorical_cols)
            feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()
        print(feature_names)

        # Train and test score
        train_score = best_model.score(X_train, y_train)
        test_score = best_model.score(X_test, y_test)
        print(f"{model} train score:{train_score : .2f}")
        print(f"{model} test score:{test_score : .2f}")

        # Save optimized model
        path = './data/ML_models_n/'
        save_output(model, path, (X_train, y_train), (X_test, y_test), opt)

        # ----------------------------------------------

        # Learning curve
        cv_lc = ContinuousStratifiedKFold(n_splits=5, n_iter=50, random_state=rs)
        train_sizes, train_scores, val_scores = get_learning_curve(best_model, X_train, y_train, cv=cv, scoring="r2")

        # Dual plot
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax1, ax2 = ax

        # Plot average learning curve
        plot_learning_curve(train_sizes, train_scores, val_scores, ax=ax1, title=None)

        # Correlation plot between experimental and predicted values
        sns.regplot(x=y_test, y=best_model.predict(X_test), ax=ax2)
        ax2.text(0.2, 0.8, f"test score: {test_score : .2f}", transform = ax2.transAxes)
        ax2.set(xlabel="Experimental n", ylabel="Predicted n")

        plt.tight_layout()
        plt.savefig(f"figures/ML_models_n/{model}_grid_feateng_predictions.png", dpi=600, bbox_inches='tight')
        # plt.show()

        # Dual plot
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax1, ax2 = ax

        # Plot average learning curve
        plot_learning_curve(train_sizes, train_scores, val_scores, ax=ax1, title=None)

        # Plot fold-level distribution
        plot_cv_distribution(train_sizes, val_scores, ax=ax2, title=None)

        plt.tight_layout()
        plt.savefig(f"figures/ML_models_n/{model}_grid_feateng_learning_curve.png", dpi=600, bbox_inches='tight')
        # plt.show()

        # ----------------------------------------------

        # Feature importance using permutation importance

        # Feature importance
        # if config.get('func_feature_importance'):
        #     for func in config.get('func_feature_importance'):
        #         func(opt, feature_names, X_test, y_test)

        # Feature names after encoding
        if categorical_cols.empty:
            feature_names = numerical_cols.tolist()
        else:
            onehot_feature_names = best_model.named_steps['preprocessor'] \
                .named_transformers_['cat'] \
                .get_feature_names_out(categorical_cols)
            feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()
        print(feature_names)

        # Feature importance
        importances = permutation_importance(
            best_model, X_test, y_test, n_repeats=10, random_state=rs, n_jobs=-1
        )
        sorted_idx = importances.importances_mean.argsort()[::-1]

        # Plot
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(x=importances.importances_mean[sorted_idx],
                    y=pd.Series([feat.replace('_cordero', '') for feat in feature_names])[sorted_idx],
                    orient='h', ax=ax)
        ax.set(xlabel='Feature improtance', ylabel=None, xticks=[])
        # ax.set_title("Permutation Feature Importance (Test Set)")
        plt.tight_layout()
        plt.savefig(f"figures/ML_models_n/{model}_grid_feateng_pfi.png", dpi=600, bbox_inches='tight')
        # plt.show()
# ------------------------------------------------------------------------------------------------------------------

exit()

# Comparison of models' fitting
models_to_compare = ["svr", "xgboost", "lightGBM", "knn", "rf"]

# Plot
fig = plt.figure(figsize=(18/2.54, 18/2.54))
for n, model in enumerate(models_to_compare):

    # Load saved model
    opt, (X_train,y_train), (X_test,y_test) = load_model_from_tmp("data/tmp/", model)

    # Add subplot
    ax = fig.add_subplot(3, 2, n+1)
    sns.regplot(x=y_test, y=opt.predict(X_test), ax=ax)
    score = opt.score(X_test, y_test)
    ax.set(title = f"model : {model}, test_score : {score : .2f}",
           xlabel = r"STY_Experimental",
           ylabel = r"STY_predicted",
           )

plt.tight_layout()
# plt.savefig("figures/5_11_model_comparison.png", dpi = 600, bbox_inches='tight')
plt.show()

