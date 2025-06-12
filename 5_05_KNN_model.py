import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import matplotlib.pyplot as plt
import seaborn as sns

# configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8

# Load dataset
df = pd.read_csv("data/tmp/processed_data.csv")
X = df.drop(columns="STY_MA+AA_(mmol/h/g)")
y = df["STY_MA+AA_(mmol/h/g)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# Feature types
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
])

# Pipeline with KNN
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('reg', KNeighborsRegressor())
])

# Search space for KNN hyperparameters
search_space = {
    'reg__n_neighbors': Integer(1, 30),
    'reg__weights': Categorical(['uniform', 'distance']),
    'reg__p': Integer(1, 2)  # 1 = Manhattan, 2 = Euclidean
}

# Bayesian optimization
opt = BayesSearchCV(pipe, search_space, cv=10, n_iter=50, scoring='r2', random_state=8, verbose=True)
opt.fit(X_train, y_train)

# Get feature names after one-hot encoding
onehot_feature_names = opt.best_estimator_.named_steps['preprocessor'] \
    .named_transformers_['cat'] \
    .get_feature_names_out(categorical_cols)
feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()

# Train/test scores
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

from model_IO import save_output

path = './data/tmp/'
model = "knn"

save_output(model, path, (X_train, y_train), (X_test, y_test), opt)