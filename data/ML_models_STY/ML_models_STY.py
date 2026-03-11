from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from functions import functions_MLmodels


model_config = {
    "best_model": {
        "model": XGBRegressor(verbosity=1, nthread=-1, random_state=8),
        "param_grid": {
            'model__n_estimators': [500, 1000],           # This is important, which was not there before 
            'model__max_depth': [5],
            'model__learning_rate': [0.1],
            'model__subsample': [0.5],
            'model__colsample_bytree': [1.0],          
            'model__min_child_weight': [10, 20],           # this prevents overfitting
            'model__reg_alpha': [1],
            'model__reg_lambda': [1, 2],
            'model__gamma': [0]
        },
        "func_feature_importance": [functions_MLmodels.plot_feature_importance]
    },

    "xgboost": {
        "model": XGBRegressor(verbosity=1, nthread=-1, random_state=8),
        "param_grid": {
            'model__n_estimators': [500, 1000],
            'model__max_depth': [2, 4],  # Decrease to prevent overfitting
            'model__learning_rate': [0.01, 0.1],
            'model__subsample': [0.5, 0.8],
            'model__colsample_bytree': [0.5, 0.8],          
            'model__min_child_weight': [10],  # splitting barrier preventing overfitting
            'model__reg_alpha': [3],  # reg term 1
            'model__reg_lambda': [2],  # reg term 2
            'model__gamma': [0, 0.5]
        },
        "func_feature_importance": [functions_MLmodels.plot_feature_importance]
    },

    "rf": {
        "model": RandomForestRegressor(n_jobs=-1, verbose=0, random_state=8),
        "param_grid": {
            'model__n_estimators': [500, 750, 1000],
            'model__max_depth': [4, 6],
            'model__min_samples_split': [4, 8],  # [7, 8, 9],
            'model__min_samples_leaf': [2, 4],  # [3, 4, 5],
            'model__max_features': [1],  # [0.35, 0.4, 0.45],
            'model__min_impurity_decrease': [0, 0.001],  # [0.0, 0.001, 0.005],
            'model__max_samples': [0.8, 1.0]  #[0.8, 0.9, 1.0]
        },
        "func_feature_importance": [functions_MLmodels.permutation_feature_importance]
    },

    "svr": {
        "model": SVR(),
        "param_grid": {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__epsilon': [0.001, 0.01, 0.1, 1.0],
            'model__kernel': ['rbf', 'poly', 'sigmoid'],
            'model__gamma': ['scale', 'auto']
        },
        "func_feature_importance": None
    },

    "knn": {
        "model": KNeighborsRegressor(),
        "param_grid": {
            'model__n_neighbors': [1, 5, 10, 20, 30],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
        },
        "func_feature_importance": None
    },

    # Very heavy computation
    "lgbm": {
        "model": LGBMRegressor(),
        "param_grid": {
            'model__n_estimators': [500, 1000],
            'model__max_depth': [3, 5],
            'model__learning_rate': [0.01, 0.1],
            'model__num_leaves': [10, 50],
            'model__subsample': [0.5, 1.0],
            'model__colsample_bytree': [0.5, 1.0],
            'model__min_child_weight': [10],           # this prevents overfitting
            'model__reg_alpha': [1, 3],
            'model__reg_lambda': [1, 2]
        },
        "func_feature_importance": [functions_MLmodels.plot_feature_importance]
    }
}
