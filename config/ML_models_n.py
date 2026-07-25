from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor


model_config = {
    "xgboost_reg": {
        "abb": "XGB",
        "model": XGBRegressor(verbosity=0, nthread=-1),
        "param_grid": {
            'model__n_estimators': [5, 10],           # This is important, which was not there before 
            'model__max_depth': [5, 6],  
            'model__learning_rate': [0.4,],  # [0.1]
            'model__subsample': [0.5],
            'model__colsample_bytree': [0.5],          
            'model__min_child_weight': [4, 5],           # this prevents overfitting  [10, 20]
            'model__reg_alpha': [0.1],  # L1 regularization for sparse model [1]
            'model__reg_lambda': [8],  # L2 regularization for spread weights [1, 2]
            'model__gamma': [0]
        },
    },

    "xgboost": {
        "abb": "XGB",
        "model": XGBRegressor(verbosity=0, nthread=-1),
        "param_grid": {
            'model__n_estimators': [50],           # This is important, which was not there before 
            'model__max_depth': [2],
            'model__learning_rate': [0.02, 0.1],
            'model__subsample': [0.5, 1],
            'model__colsample_bytree': [0.5, 0.7],
            'model__min_child_weight': [3, 5],           # this prevents overfitting  [10, 20]
            'model__reg_alpha': [0.1],
            'model__reg_lambda': [5, 8],
            'model__gamma': [0.0]
        },
    },

    "rf": {
        "abb": "RF",
        "model": RandomForestRegressor(n_jobs=-1, verbose=0),
        "param_grid": {
            "model__n_estimators": [50, 100],
            "model__max_depth": [3, 6],
            "model__min_samples_split": [2, 20],
            "model__min_samples_leaf": [1, 5, 10],
            "model__max_features": [0.3, 1.0],
            "model__min_impurity_decrease": [0.0, 0.005],
            "model__max_samples": [0.8, 1.0]
        },
    },

    "lgbm_reg": {
        "abb": "LGB",
        "model": LGBMRegressor(verbosity=-1, n_jobs=1),
        "param_grid": {
            'model__num_leaves': [5, 10, 20],
            'model__max_depth': [3, 6],
            'model__learning_rate': [0.1, 0.3],
            'model__n_estimators': [10, 50],
            'model__subsample': [0.5, 0.8],
            'model__colsample_bytree': [0.5, 0.8],
            'model__reg_alpha': [0, 0.1, 1],
            'model__reg_lambda': [1, 5, 10]
        },
    }
}
