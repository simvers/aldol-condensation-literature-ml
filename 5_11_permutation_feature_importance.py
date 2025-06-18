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

    feature_names = [fn.split("__", 1)[1] for fn in opt[0].get_feature_names_out()]

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