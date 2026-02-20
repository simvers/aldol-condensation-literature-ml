import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions.functions_MLmodels import load_model_from_tmp
from sklearn.inspection import permutation_importance
import warnings


plt.rcParams["font.size"] = 8

models = ["svr", "xgboost", "lightGBM", "knn", "rf"]

# #correlation plot
# fig = plt.figure(figsize=(18/2.54, 18/2.54))
# for n,model in enumerate(models):
#     opt, (X_train,y_train), (X_test,y_test) = load_model_from_tmp("data/tmp/", model)
#     ax = fig.add_subplot(3, 2, n+1)
#     sns.regplot(x = y_test, y = opt.predict(X_test), ax = ax)
#     score = opt.score(X_test, y_test)
#     ax.set(title = f"model : {model}, test_score : {score : .2f}",
#            xlabel = r"STY_Experimental",
#            ylabel = r"STY_predicted",
#            )
    

# plt.tight_layout()
# plt.savefig("figures/5_11_model_comparison.png", dpi = 600, bbox_inches='tight')
# plt.show()


#Residual plot
fig = plt.figure(figsize=(18/2.54, 18/2.54))
for n,model in enumerate(models):
    opt, (X_train,y_train), (X_test,y_test) = load_model_from_tmp("data/tmp/", model)
    ax = fig.add_subplot(3, 2, n+1)
    sns.scatterplot(x = y_test, y = y_test - opt.predict(X_test), ax = ax)
    ax.plot(y_test, np.zeros(len(y_test)))
    ax.set(title = f"model : {model}",
           xlabel = r"STY_Experimental",
           ylabel = "residual",
           )
    

plt.tight_layout()
plt.savefig("figures/5_12_model_comparison_residual.png", dpi = 600, bbox_inches='tight')
plt.show()

#Residual distribution
fig = plt.figure(figsize=(8/2.54, 8/2.54))
ax = fig.add_subplot()
for n,model in enumerate(models):
    opt, (X_train,y_train), (X_test,y_test) = load_model_from_tmp("data/tmp/", model)
    sns.kdeplot(y_test - opt.predict(X_test), ax = ax, label = model)
    ax.axvline(0, c = 'k', alpha = 0.5, linestyle = '--')
    ax.set(
           xlabel = "Residuals",
           )

ax.legend(frameon = False) 

plt.tight_layout()
plt.savefig("figures/5_12_model_comparison_residual_distribution.png", dpi = 600, bbox_inches='tight')
plt.show()


#Configurations
plt.rcParams["font.size"] = 10 
warnings.filterwarnings("ignore")



models = ['XGBoost', 'RF', 'LightGBM', 'SVR', 'KNN']

for model in models:
    opt, (X_train, y_train), (X_test, y_test) = load_model_from_tmp("data/tmp/", model)

    feature_names = [fn.split("__", 1)[1] for fn in opt[0].get_feature_names_out()]

    # Feature importance using permutation importance
    importances = permutation_importance(
        opt, X_test, y_test, n_repeats=10, random_state=8, n_jobs=-1
    )
    sorted_idx = importances.importances_mean.argsort()[::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=importances.importances_mean[sorted_idx],
                y=pd.Series(feature_names)[sorted_idx],
                orient='h', ax=ax)
    ax.set_title(f"Permutation Feature Importance (Test Set), {model}")
    ax.set(xlabel = "Feature importance", ylabel = "")
    plt.tight_layout()

    plt.savefig(f"figures/{5}_{model}_pfi.png",dpi = 600, bbox_inches = 'tight')
    plt.show()

# # Comparison of models' fitting
# models_to_compare = ["svr", "xgboost", "lightGBM", "knn", "rf"]

# # Plot
# fig = plt.figure(figsize=(18/2.54, 18/2.54))
# for n, model in enumerate(models_to_compare):

#     # Load saved model
#     opt, (x_train,y_train), (x_test,y_test) = load_model_from_tmp("data/tmp/", model)

#     # Add subplot
#     ax = fig.add_subplot(3, 2, n+1)
#     sns.regplot(x=y_test, y=opt.predict(x_test), ax=ax)
#     score = opt.score(x_test, y_test)
#     ax.set(title = f"model : {model}, test_score : {score : .2f}",
#            xlabel = r"STY_Experimental",
#            ylabel = r"STY_predicted",
#            )

# plt.tight_layout()
# # plt.savefig("figures/5_11_model_comparison.png", dpi = 600, bbox_inches='tight')
# plt.show()

# Load models config from json file
# Map json directory to ML model
# with open('data/ML_models_STY/ML_models_STY.json') as f:
#     model_config = json.load(f)
# model_mapping = {
#     "xgboost": XGBRegressor,
# }