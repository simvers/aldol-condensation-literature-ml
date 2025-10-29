import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
import seaborn as sns

# configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8

# Getting the data from tmp dir (Remember to run the 5_01_data_preprocessing_for_models.py before running this.)
df = pd.read_csv("data/tmp/processed_data.csv")
X = df.drop(columns="STY_MA+AA_(mmol/h/g)")
y = df["STY_MA+AA_(mmol/h/g)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

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
    ('reg', RandomForestRegressor(random_state=8, n_jobs=-1, verbose=1))
])

# Hyperparameter tuning space with BayesSearchCV
search_space = {
    'reg__n_estimators': Integer(100, 1000),
    'reg__max_depth': Integer(2, 20),
    'reg__min_samples_split': Integer(2, 20),
    'reg__min_samples_leaf': Integer(1, 10),
    'reg__max_features': Real(0.3, 1.0)
}

opt = BayesSearchCV(pipe, search_space, cv=10, n_iter=50, scoring='r2', random_state=8)

# Executing the pipeline with train_data
opt.fit(X_train, y_train)

# Feature names after encoding
onehot_feature_names = opt.best_estimator_.named_steps['preprocessor'] \
    .named_transformers_['cat'] \
    .get_feature_names_out(categorical_cols)
feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()

# Print test and train score
train_score = opt.score(X_train, y_train)
test_score = opt.score(X_test, y_test)
print(f"Train score:{train_score : .2f}")
print(f"Test score:{test_score : .2f}")

# Correlation plot
fig, ax = plt.subplots()
sns.regplot(x=y_test, y=opt.predict(X_test), ax=ax)
ax.text(0.2, 0.8, f"test score: {test_score : .2f}", transform=ax.transAxes)
ax.set(xlabel="Experimental STY", ylabel="Predicted STY")
plt.show()

# Feature importance using permutation importance
importances = permutation_importance(
    opt.best_estimator_, X_test, y_test, n_repeats=10, random_state=8, n_jobs=-1
)
sorted_idx = importances.importances_mean.argsort()[::-1]

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x=importances.importances_mean[sorted_idx],
            y=pd.Series(feature_names)[sorted_idx],
            orient='h', ax=ax)
ax.set_title("Permutation Feature Importance (Test Set)")
plt.tight_layout()
plt.show()

from model_IO import save_output

path = './data/tmp/'
model = "rf"

save_output(model, path, (X_train, y_train), (X_test, y_test), opt)