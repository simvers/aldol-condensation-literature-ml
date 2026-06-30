import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions.ml_models import load_model_from_tmp
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 8


# Variables

# Excluded silicium
data_type = '_noSi'  # ''

# Input features
reac_input = [
    'LHSV_mlhg',
    'Ratio_Ac_Fa', 'Ratio_Stab_Fa', 
    'Temperature_K', 
    'Ac_source', 'Fa_source', 
    'Stabilizer', 
    'O_content', 
]
cat_input = [
    'SSA_m2g',
    'av_cov_rad', 
    'av_n_val',
    'var_cov_rad',  
    'var_n_val'
    # 'var_pca', # 'var_lat_cst', 
] 
input = reac_input + cat_input
output = 'STY_Acryl_mmolhg'
# output = 'Y_Acryl_Ac'

# ------------------------------------------------------------------------------------------------------------------

# Import data
data = pd.read_csv(f"data/catalysts/data_engineered{data_type}.csv", na_values=[''], keep_default_na=False)
elements = pd.read_csv('data/catalysts/elements.csv', header=None).squeeze('columns').to_list()
eng_feat = pd.read_csv('data/catalysts/comp_features.csv', header=None).squeeze('columns').to_list()
print(data.columns, len(data))

# Cluster colors
cluster_list = data['Cluster_title']
clusters = np.sort(cluster_list.unique())

# ------------------------------------------------------------------------------------------------------------------

# Drop observations with missing output
# data.dropna(subset=output, inplace=True)
# assert not data[input].isna().any().any()
data.dropna(subset=output, inplace=True)
print(len(data))
if data[input].isna().any().any():
    print('Dropping obsevations with NaNs in input feature!')
    data.dropna(subset=input, inplace=True)
print('Data with STY: ', len(data))

# Features and target
# x = data.drop(columns=output)
x = data[input]

# ------------------------------------------------------------------------------------------------------------------

# Identify numerical and categorical features
categorical_cols = x.select_dtypes(include=['object', 'category']).columns
numerical_cols = x.select_dtypes(include=['int64', 'float64']).columns

# Condensation
mask = (x['Stabilizer'] == 'MeOH') | (x['Stabilizer'] == 'MeOH \n+ H$_2$O') | (x['Stabilizer'] == 'EtOH')
x.loc[mask, 'Stabilizer'] = 'Stabilizer'

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
], verbose_feature_names_out=False)
norm_array = preprocessor.fit_transform(x)
onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()
df = pd.DataFrame(norm_array, columns=feature_names)

# Columns
remove_col = categorical_cols.tolist() + ['Ac_source_MAc', 'Ac_source_EAc', 'Fa_source_TRX', 'Fa_source_DMM', 'Fa_source_MeOH', 'Stabilizer_Stabilizer']
selected_col = [col for col in (reac_input + onehot_feature_names.tolist() + cat_input) if col not in remove_col]
feature_mapper = {'Ac_source_HAc': 'HAc or MAc', 'Fa_source_FORM': 'Form or Trx', 'Stabilizer_None': 'Add. or None', 'LHSV_mlhg': 'LHSV', 'SSA_m2g': 'SSA', 'Temperature_K': 'Temperature',
                  'O_content': 'O$_2$ content', 'Ratio_Ac_Fa': 'Ac/Fa', 'Ratio_Stab_Fa': 'Add/Fa', 
                  'av_cov_rad': 'Cov. rad.', 'av_n_val': 'n val.', 'var_cov_rad': 'Cov. rad.', 'var_n_val': 'n val.',}

# Rename and select columns
corr_df = df.corr(method='pearson').abs()
# corr_df = pd.DataFrame(np.abs(np.corrcoef(norm_array, rowvar=False)), index=feature_names, columns=feature_names)
corr_selected = corr_df.loc[selected_col, selected_col]
corr_selected.rename(index=feature_mapper, columns=feature_mapper, inplace=True)

# Correlation of features
fig, ax = plt.subplots(1, 1)
mask = np.triu(np.ones_like(corr_selected, dtype=bool))
sns.heatmap(corr_selected, cmap='plasma_r', vmin=0, vmax=1, square=True, mask=mask, ax=ax)
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Correlation coefficient', rotation=270, labelpad=15)
# Horizontal text
title, subtitle = -0.3, -0.2
line, subline = -0.25, -0.16
ax.text(4/13, title, 'Reaction conditions', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
# ax.text(6.5/13, title, 'Reactants', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.text(10.5/13, title, 'Catalyst', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.text(10/13, subtitle, 'Mean', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.text(12/13, subtitle, 'Variance', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.annotate('', xytext=(0+0.005, line), xy=(8/13-0.005, line), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
# ax.annotate('', xytext=(5/13+0.005, line), xy=(8/13-0.005, line), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
ax.annotate('', xytext=(8/13+0.005, line), xy=(13/13-0.005, line), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
ax.annotate('', xytext=(9/13+0.005, subline), xy=(11/13-0.005, subline), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
ax.annotate('', xytext=(11/13+0.005, subline), xy=(13/13-0.005, subline), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
# Vertical text
ax.text(subtitle, 1/13, 'Variance', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
ax.text(subtitle, 3/13, 'Mean', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
ax.text(title, 2.5/13, 'Catalyst', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
# ax.text(title, 6.5/13, 'Reactants', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
ax.text(title, 9/13, 'Reaction conditions', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
ax.annotate('', xytext=(line, 0+0.005), xy=(line, 5/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
# ax.annotate('', xytext=(line, 5/13+0.005), xy=(line, 8/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
ax.annotate('', xytext=(line, 5/13+0.005), xy=(line, 13/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
ax.annotate('', xytext=(subline, 0+0.005), xy=(subline, 2/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
ax.annotate('', xytext=(subline, 2/13+0.005), xy=(subline, 4/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
fig.savefig(f"figures/ML_STY/feature_correlation_pearson.svg", dpi=600, bbox_inches='tight')

# Rename and select columns
corr_df = df.corr(method='spearman').abs()
# corr_df = pd.DataFrame(np.abs(np.corrcoef(norm_array, rowvar=False)), index=feature_names, columns=feature_names)
corr_selected = corr_df.loc[selected_col, selected_col]
corr_selected.rename(index=feature_mapper, columns=feature_mapper, inplace=True)

# Correlation of features
fig, ax = plt.subplots(1, 1)
mask = np.triu(np.ones_like(corr_selected, dtype=bool))
sns.heatmap(corr_selected, cmap='plasma_r', vmin=0, vmax=1, square=True, mask=mask, ax=ax)
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Correlation coefficient', rotation=270, labelpad=15)
# Horizontal text
title, subtitle = -0.3, -0.2
line, subline = -0.25, -0.16
ax.text(2.5/13, title, 'Reaction conditions', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.text(6.5/13, title, 'Reactants', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.text(10.5/13, title, 'Catalyst', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.text(10/13, subtitle, 'Mean', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.text(12/13, subtitle, 'Variance', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='horizontal')
ax.annotate('', xytext=(0+0.005, line), xy=(5/13-0.005, line), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
ax.annotate('', xytext=(5/13+0.005, line), xy=(8/13-0.005, line), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
ax.annotate('', xytext=(8/13+0.005, line), xy=(13/13-0.005, line), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
ax.annotate('', xytext=(9/13+0.005, subline), xy=(11/13-0.005, subline), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
ax.annotate('', xytext=(11/13+0.005, subline), xy=(13/13-0.005, subline), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
# Vertical text
ax.text(subtitle, 1/13, 'Variance', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
ax.text(subtitle, 3/13, 'Mean', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
ax.text(title, 2.5/13, 'Catalyst', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
ax.text(title, 6.5/13, 'Reactants', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
ax.text(title, 10.5/13, 'Reaction conditions', ha='center', va='center', transform=ax.transAxes, weight='bold', rotation='vertical')
ax.annotate('', xytext=(line, 0+0.005), xy=(line, 5/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
ax.annotate('', xytext=(line, 5/13+0.005), xy=(line, 8/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
ax.annotate('', xytext=(line, 8/13+0.005), xy=(line, 13/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=3), xycoords=ax.transAxes)
ax.annotate('', xytext=(subline, 0+0.005), xy=(subline, 2/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
ax.annotate('', xytext=(subline, 2/13+0.005), xy=(subline, 4/13-0.005), arrowprops=dict(arrowstyle="|-|", lw=0.75, mutation_scale=2), xycoords=ax.transAxes)
fig.savefig(f"figures/ML_STY/feature_correlation_spearman.svg", dpi=600, bbox_inches='tight')

exit()

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