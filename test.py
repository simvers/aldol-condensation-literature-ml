# import os
# import numpy as np
# import shap
# import matplotlib.pyplot as plt
# import joblib
# from sklearn.inspection import permutation_importance
# import warnings
# warnings.warn("Ignore")

# # Setup
# shap.initjs()
# model_dir = 'data/tmp/'
# models_to_plot = ['XGBoost', 'RF', 'LightGBM', 'SVR', 'KNN']

# # Helper function
# def load_model_from_tmp(path, modelname):
#     pipe = joblib.load(f"{path}{modelname}_model.pkl")
#     X_train, y_train = joblib.load(f"{path}{modelname}_train_data.pkl")
#     X_test, y_test = joblib.load(f"{path}{modelname}_test_data.pkl")
#     return pipe, (X_train, y_train), (X_test, y_test)

# # SHAP plots
# for model_name in models_to_plot:
#     print(f"\n🔍 Plotting SHAP for: {model_name}")
    
#     pipe, (X_train, y_train), (X_test, y_test) = load_model_from_tmp(model_dir, model_name)
#     reg = pipe.named_steps['reg']
#     preprocessor = pipe.named_steps['preprocessor']
    
#     X_test_transformed = preprocessor.transform(X_test)
    
#     try:
#         # Try SHAP with TreeExplainer (works for tree models)
#         explainer = shap.Explainer(reg, X_test_transformed, feature_names=[f"f{i}" for i in range(X_test_transformed.shape[1])], index = feature_names)
#         shap_values = explainer(X_test_transformed)

#         # Plot SHAP summary
#         shap.summary_plot(shap_values, feature_names=shap_values.feature_names, show=True)
    
#     except Exception as e:
#         print(f"⚠️ SHAP not supported for {model_name}, using permutation importance instead.")

#         # Fallback: permutation importance
#         result = permutation_importance(reg, X_test_transformed, y_test, n_repeats=10, random_state=42)
#         importances = result.importances_mean
#         feature_names = [f"f{i}" for i in range(len(importances))]

#         top_idx = np.argsort(importances)[-10:][::-1]
#         top_features = [feature_names[j] for j in top_idx]
#         top_values = importances[top_idx]

#         # Bar plot
#         import seaborn as sns
#         fig, ax = plt.subplots(figsize=(6, 4))
#         sns.barplot(x=top_values, y=top_features, ax=ax, palette='viridis')
#         ax.set_title(f"{model_name} (Permutation Importance)")
#         ax.set_xlabel("Importance")
#         ax.set_ylabel("Feature")
#         plt.tight_layout()
#         plt.show()

from model_IO import load_model_from_tmp
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

#Configurations
plt.rcParams["font.size"] = 10 
plt.rcParams["font.family"] = 'verdana' 
warnings.filterwarnings("ignore")



models = ['XGBoost', 'RF', 'LightGBM', 'SVR', 'KNN']

for model in models:
    opt, (X_train, y_train), (X_test, y_test) = load_model_from_tmp("data/tmp/", models[0] )

    num_features = opt[0]['num'].get_feature_names_out()
    cat_features = opt[0]['cat'].get_feature_names_out()
    feature_names = np.concatenate((num_features, cat_features))

    # Feature importance using permutation importance
    importances = permutation_importance(
        opt, X_test, y_test, n_repeats=10, random_state=8, n_jobs=-1
    )
    sorted_idx = importances.importances_mean.argsort()[::-1]

    fig, ax = plt.subplots(figsize=(10, 8), dpi = 600)
    sns.barplot(x=importances.importances_mean[sorted_idx],
                y=pd.Series(feature_names)[sorted_idx],
                orient='h', ax=ax)
    ax.set_title(f"Permutation Feature Importance (Test Set), {model}")
    ax.set(xlabel = "Feature importance", ylabel = "")
    plt.tight_layout()

    plt.savefig(f"figures/{5}_{model}_pfi.png", bbox_inches = 'tight')