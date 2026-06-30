from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor


model_config = {
    "best_model": {
        "model": XGBRegressor(verbosity=0, nthread=-1, random_state=8),
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
        "model": XGBRegressor(verbosity = 0, nthread = 8),
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
        "model": RandomForestRegressor(random_state=8, n_jobs=-1, verbose=0),
        "search_space": {
            "reg__n_estimators": [500, 1000],
            "reg__max_depth": [3, 6],
            "reg__min_samples_split": [2, 20],
            "reg__min_samples_leaf": [1, 5, 10],
            "reg__max_features": [0.3, 1.0],
            "reg__min_impurity_decrease": [0.0, 0.005],
            "reg__max_samples": [0.8, 1.0]
            # "reg__n_estimators": [100, 500, 100],
            # "reg__max_depth": [2, 10, 20],
            # "reg__min_samples_split": [2, 10, 20],
            # "reg__min_samples_leaf": [1, 5, 10],
            # "reg__max_features": [0.3, 0.6, 1.0],
            # "reg__min_impurity_decrease": [0.0, 0.001, 0.005],
            # "reg__max_samples": [0.8, 0.9, 1.0]
        },
    },

    "svr": {
        "model": SVR(),
        "search_space": {
            'reg__C': [0.01, 1, 100],
            'reg__epsilon': [0.001, 0.1, 1],
            'reg__kernel': ['rbf', 'poly', 'sigmoid'],
            'reg__gamma': ['scale', 'auto']
        },
    },

    "knn": {
        "model": KNeighborsRegressor(),
        "search_space": {
            'reg__n_neighbors': [1, 10, 20, 30],
            'reg__weights': ['uniform', 'distance'],
            'reg__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
        },
    },

    "lgbm": {
        "model": LGBMRegressor(),
        "search_space": {
            'reg__num_leaves': [1, 5],
            'reg__max_depth': [3, 6],
            'reg__learning_rate': [0.001, 0.1, 0.5],
            'reg__n_estimators': [500, 1000],
            # 'reg__subsample': [0.1, 1],
            # 'reg__colsample_bytree': [0.5, 1],
            'reg__reg_alpha': [1, 10],
            'reg__reg_lambda': [1, 10]
        },
    }
}
