from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor


model_config = {
    "best_model": {
        "name": "Extreme Gradient Boosting",
        "abb": "XGBoost",
        "model": XGBRegressor(verbosity=0, nthread=-1, random_state=8),
        "param_grid": {
            'model__n_estimators': [500, 1000],           # This is important, which was not there before [500, 1000]
            'model__max_depth': [2],  # 5
            'model__learning_rate': [0.05],  # [0.1]
            'model__subsample': [0.5],
            'model__colsample_bytree': [1.0],          
            'model__min_child_weight': [10],           # this prevents overfitting  [10, 20]
            'model__reg_alpha': [1],  # L1 regularization for sparse model [1]
            'model__reg_lambda': [1, 2],  # L2 regularization for spread weights [1, 2]
            'model__gamma': [0.5]  # 0
        },
    },

    "xgboost_no_overfit": {
        "name": "Extreme Gradient Boosting",
        "abb": "XGBoost",
        "model": XGBRegressor(verbosity=0, nthread=-1, random_state=8),
        "param_grid": {
            'model__n_estimators': [300],
            'model__max_depth': [2],  # Decrease to prevent overfitting
            'model__learning_rate': [0.01],
            'model__subsample': [0.8],
            'model__colsample_bytree': [0.8],          
            'model__min_child_weight': [10],  # splitting barrier preventing overfitting
            'model__reg_alpha': [3],  # reg term 1
            'model__reg_lambda': [3],  # reg term 2
            'model__gamma': [0.5]
        },
    },

    "xgboost": {
        "name": "Extreme Gradient Boosting",
        "abb": "XGBoost",
        "model": XGBRegressor(verbosity=0, nthread=-1, random_state=12, device='cuda'),
        "param_grid": {
            'model__n_estimators': [300, 500],
            'model__max_depth': [2, 3],  # Decrease to prevent overfitting
            'model__learning_rate': [0.005, 0.01],
            'model__subsample': [0.4, 0.6],
            'model__colsample_bytree': [0.4, 0.6],          
            'model__min_child_weight': [10, 15],  # splitting barrier preventing overfitting
            'model__reg_alpha': [3, 5],  # reg term 1
            'model__reg_lambda': [3, 5],  # reg term 2
            'model__gamma': [0, 0.5]
        },
    },

    "xgboost_yield": {
        "name": "Extreme Gradient Boosting",
        "abb": "XGBoost",
        "model": XGBRegressor(verbosity=0, nthread=-1, random_state=8, device='cuda'),
        "param_grid": {
            'model__n_estimators': [300, 500],
            'model__max_depth': [2, 3, 5],  # Decrease to prevent overfitting
            'model__learning_rate': [0.1, 0.2],
            'model__subsample': [0.5, 0.8],
            'model__colsample_bytree': [0.5, 0.8],          
            'model__min_child_weight': [5, 10],  # splitting barrier preventing overfitting
            'model__reg_alpha': [1, 2],  # reg term 1
            'model__reg_lambda': [1, 2],  # reg term 2
            'model__gamma': [0, 0.5]
        },
    },

    "rf_noreg": {
        "name": "Random Forest",
        "abb": "RF",
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
    },

    "rf": {
        "name": "Random Forest",
        "abb": "RF",
        "model": RandomForestRegressor(n_jobs=-1, verbose=0, random_state=8),
        "param_grid": {
            'model__n_estimators': [200, 300, 400],
            'model__max_depth': [3, 4],
            # 'model__min_samples_split': [4, 8],  # [7, 8, 9],
            'model__min_samples_leaf': [10, 20, 30],  # [3, 4, 5],
            'model__max_features': [0.6],  # [0.35, 0.4, 0.45],
            # 'model__min_impurity_decrease': [0, 0.001],  # [0.0, 0.001, 0.005],
            'model__max_samples': [0.6]  #[0.8, 0.9, 1.0]
        },
    },
    
    "svr_noreg": {
        "name": "Support Vector Machine",
        "abb": "SVR",
        "model": SVR(),
        "param_grid": {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__epsilon': [0.001, 0.01, 0.1, 1.0],
            'model__kernel': ['rbf', 'poly', 'sigmoid'],
            'model__gamma': ['scale', 'auto']
        },
    },

    "svr": {
        "name": "Support Vector Machine",
        "abb": "SVR",
        "model": SVR(),
        "param_grid": {
            'model__C': [0.01, 0.1],
            'model__epsilon': [0.001, 0.01, 0.1, 1],
            'model__kernel': ['poly', 'sigmoid', 'rbf'],
            'model__gamma': ['scale', 'auto']
        },
    },

    "knn": {
        "name": "K-Nearest Neighbors",
        "abb": "KNN",
        "model": KNeighborsRegressor(),
        "param_grid": {
            'model__n_neighbors': [20, 30],
            'model__weights': ['uniform'],
            'model__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
        },
    },

    "knn_noreg": {
        "abb": "KNN",
        "name": "K-Nearest Neighbors",
        "model": KNeighborsRegressor(),
        "param_grid": {
            'model__n_neighbors': [20, 30],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
        },
    },

    # Very heavy computation
    "lgbm_noreg": {
        "name": "Light Gradient Boosting Machine",
        "abb": "LGBM",
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
    },

    "lgbm": {
        "name": "Light Gradient Boosting Machine",
        "abb": "LGBM",
        "model": LGBMRegressor(verbosity=0),
        "param_grid": {
            'model__n_estimators': [300, 500],
            'model__max_depth': [2, 3, 4],
            'model__learning_rate': [0.01, 0.1],
            'model__num_leaves': [10, 20],
            'model__subsample': [0.8],
            'model__colsample_bytree': [0.8],
            'model__min_child_weight': [10, 20],           # this prevents overfitting
            'model__reg_alpha': [3],
            'model__reg_lambda': [3]
        },
    },

    "gp": {
        "name": "Gaussian Process",
        "abb": "GP",
        "model": GaussianProcessRegressor(n_restarts_optimizer=10, normalize_y=True, 
                                          kernel=ConstantKernel(1.0) * RBF(length_scale=5.0) + WhiteKernel(noise_level=1.0)),
        "param_grid": {
            'model__alpha': [5, 10],  # High for regularization
            'model__kernel__k1__k2__length_scale': [5.0, 10.0],  # RBF length scale, high for regularization
            'model__kernel__k2__noise_level': [5, 10]
        },
    }
}
