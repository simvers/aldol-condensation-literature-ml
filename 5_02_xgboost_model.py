import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance

# configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8

# Getting the data from tmp dir (Rember to run the 5_01_data_preprocessing_for_models.py before running this.)
df = pd.read_csv("data/tmp/processed_data.csv")
X = df.drop(columns="STY_MA+AA_(mmol/h/g)")
y  = df["STY_MA+AA_(mmol/h/g)"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=8)


# Building the pipeline
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
])

# The pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('reg', XGBRegressor(verbosity = 2, nthread = 8))
])

# Building hyperparams tuning space with BayerSearchCV
search_space = {
    'reg__max_depth': Integer(2,8),
    'reg__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'reg__subsample': Real(0.5, 1.0),
    'reg__colsample_bytree': Real(0.5, 1.0),
    'reg__colsample_bylevel': Real(0.5, 1.0),
    'reg__colsample_bynode' : Real(0.5, 1.0),
    'reg__reg_alpha': Real(0.0, 10.0),
    'reg__reg_lambda': Real(0.0, 10.0),
    'reg__gamma': Real(0.0, 10.0)
}

opt = BayesSearchCV(pipe, search_space, cv=10, n_iter=100, scoring='r2', random_state=8)

# Executing the pipline with train_data
opt.fit(X_train, y_train)

# Feature names are lost during the one hot encoding, so get them back
onehot_feature_names = opt.best_estimator_.named_steps['preprocessor'] \
    .named_transformers_['cat'] \
    .get_feature_names_out(categorical_cols)

feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()

# print test and train score
train_score = opt.score(X_train, y_train)
test_score = opt.score(X_test, y_test)
print(f"Train score:{train_score : .2f}")
print(f"Test score:{test_score : .2f}")

# Show a correlation plot between experimental and predicted values
fig, ax = plt.subplots()
sns.regplot(x=y_test, y = opt.predict(X_test), ax = ax)
ax.text(0.2, 0.8, f"test score: {test_score : .2f}", transform = ax.transAxes)
ax.set(xlabel = "Experimental STY",ylabel = "Predicted STY")
plt.show()

# Show feature importance
xgboost_model = opt.best_estimator_.named_steps['reg']
xgboost_model.get_booster().feature_names = feature_names
fig, ax = plt.subplots(figsize = (10, 8))
plot_importance(xgboost_model, grid=False, ax=ax, height=0.5)
plt.show()

from model_IO import save_output

path = './data/tmp/'
model = "xgboost"

save_output(model, path, (X_train, y_train), (X_test, y_test), opt)