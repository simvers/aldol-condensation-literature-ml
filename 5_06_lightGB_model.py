import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import matplotlib.pyplot as plt
import seaborn as sns

# configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8

# Getting the data
df = pd.read_csv("data/tmp/processed_data.csv")
X = df.drop(columns="STY_MA+AA_(mmol/h/g)")
y = df["STY_MA+AA_(mmol/h/g)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Column types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
])

# Pipeline
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('reg', LGBMRegressor(verbose=-1))
])

# Hyperparameter search space for LightGBM
search_space = {
    'reg__num_leaves': Integer(15, 150),
    'reg__max_depth': Integer(3, 12),
    'reg__learning_rate': Real(0.001, 0.5, prior='log-uniform'),
    'reg__n_estimators': Integer(50, 500),
    'reg__subsample': Real(0.5, 1.0),
    'reg__colsample_bytree': Real(0.5, 1.0),
    'reg__reg_alpha': Real(0.0, 10.0),
    'reg__reg_lambda': Real(0.0, 10.0)
}

# Bayesian Optimization
opt = BayesSearchCV(pipe, search_space, cv=10, n_iter=75, scoring='r2', random_state=8, verbose=True)
opt.fit(X_train, y_train)

# Get back feature names
onehot_feature_names = opt.best_estimator_.named_steps['preprocessor'] \
    .named_transformers_['cat'] \
    .get_feature_names_out(categorical_cols)
feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()

# Print scores
train_score = opt.score(X_train, y_train)
test_score = opt.score(X_test, y_test)
print(f"Train score:{train_score : .2f}")
print(f"Test score:{test_score : .2f}")

# Experimental vs Predicted Plot
fig, ax = plt.subplots()
sns.regplot(x=y_test, y=opt.predict(X_test), ax=ax)
ax.text(0.2, 0.8, f"test score: {test_score : .2f}", transform=ax.transAxes)
ax.set(xlabel="Experimental STY", ylabel="Predicted STY")
plt.show()

# Feature importance plot
lgb_model = opt.best_estimator_.named_steps['reg']
importances = pd.Series(lgb_model.feature_importances_, index=feature_names)
importances = importances.sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
importances.head(20).plot(kind='barh', ax=ax)
ax.invert_yaxis()
ax.set_title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

from functions.functions_MLmodels import save_output

path = './data/tmp/'
model = "lightGBM"

save_output(model, path, (X_train, y_train), (X_test, y_test), opt)