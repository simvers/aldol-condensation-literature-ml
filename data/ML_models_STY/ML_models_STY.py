from skopt.space import Real, Integer, Categorical
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from functions import functions_MLmodels


model_config = {
    "xgboost": {
        "model": XGBRegressor(verbosity = 2, nthread = 8),
        "search_space": {
            'reg__max_depth': Integer(2,8),
            'reg__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
            'reg__subsample': Real(0.5, 1.0),
            'reg__colsample_bytree': Real(0.5, 1.0),
            'reg__colsample_bylevel': Real(0.5, 1.0),
            'reg__colsample_bynode' : Real(0.5, 1.0),
            'reg__reg_alpha': Real(0.0, 10.0),
            'reg__reg_lambda': Real(0.0, 10.0),
            'reg__gamma': Real(0.0, 10.0)
        },
        "func_feature_importance": [functions_MLmodels.gboost_feature_importance]
    },

    "rf": {
        "model": RandomForestRegressor(random_state=8, n_jobs=-1, verbose=0),
        "search_space": {
            'reg__n_estimators': Integer(100, 1000),
            'reg__max_depth': Integer(2, 20),
            'reg__min_samples_split': Integer(2, 20),
            'reg__min_samples_leaf': Integer(1, 10),
            'reg__max_features': Real(0.3, 1.0)
        },
        "func_feature_importance": [functions_MLmodels.permutation_feature_importance]
    },

    "svr": {
        "model": SVR(),
        "search_space": {
            'reg__C': Real(0.01, 100, prior='log-uniform'),
            'reg__epsilon': Real(0.001, 1.0, prior='log-uniform'),
            'reg__kernel': Categorical(['rbf', 'poly', 'sigmoid']),
            'reg__gamma': Categorical(['scale', 'auto'])
        },
        "func_feature_importance": None
    },

    "knn": {
        "model": KNeighborsRegressor(),
        "search_space": {
            'reg__n_neighbors': Integer(1, 30),
            'reg__weights': Categorical(['uniform', 'distance']),
            'reg__p': Integer(1, 2)  # 1 = Manhattan, 2 = Euclidean
        },
        "func_feature_importance": None
    },

    "lgbm": {
        "model": LGBMRegressor,
        "search_space": {
            'reg__num_leaves': Integer(15, 150),
            'reg__max_depth': Integer(3, 12),
            'reg__learning_rate': Real(0.001, 0.5, prior='log-uniform'),
            'reg__n_estimators': Integer(50, 500),
            'reg__subsample': Real(0.5, 1.0),
            'reg__colsample_bytree': Real(0.5, 1.0),
            'reg__reg_alpha': Real(0.0, 10.0),
            'reg__reg_lambda': Real(0.0, 10.0)
        },
        "func_feature_importance": [functions_MLmodels.gboost_feature_importance]
    }
}
