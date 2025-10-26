import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from skopt import BayesSearchCV
from data.ML_models_STY.ML_models_STY import model_config
from functions.functions_MLmodels import save_output, load_model_from_tmp


# Configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8


# ------------------------------------------------------------------------------------------------------------------

# Import data

# Getting the data from tmp dir (Rember to run the 5_01_data_preprocessing_for_models.py before running this.)
df = pd.read_csv("data/tmp/processed_data.csv")
X = df.drop(columns="STY_MA+AA_(mmol/h/g)")
y  = df["STY_MA+AA_(mmol/h/g)"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

# ------------------------------------------------------------------------------------------------------------------

# Preprocessing

# Identify numerical and categorical features
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_cols)
])

# ------------------------------------------------------------------------------------------------------------------

# Loop over and optimize models
model_to_run = ['lgbm']

# Correlation plot of df features

for model, config in model_config.items():
    if model in model_to_run:

        # ML pipeline
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('reg', config.get('model'))
        ])

        # Setup hyperparameter optimization
        opt = BayesSearchCV(pipe, config.get('search_space'), cv=10, n_iter=100, scoring='r2', random_state=8)

        # Fit model with train_data
        opt.fit(X_train, y_train)

        # Retrieve feature names from one hot encoding
        onehot_feature_names = opt.best_estimator_.named_steps['preprocessor'] \
            .named_transformers_['cat'] \
            .get_feature_names_out(categorical_cols)
        feature_names = numerical_cols.tolist() + onehot_feature_names.tolist()

        # Train and test score
        train_score = opt.score(X_train, y_train)
        test_score = opt.score(X_test, y_test)
        print(f"{model} train score:{train_score : .2f}")
        print(f"{model} test score:{test_score : .2f}")

        # Plot correlation between experimental and predicted values
        fig, ax = plt.subplots()
        sns.regplot(x=y_test, y=opt.predict(X_test), ax = ax)
        ax.text(0.2, 0.8, f"test score: {test_score : .2f}", transform=ax.transAxes)
        ax.set(xlabel="Experimental STY", ylabel="Predicted STY", title=f"{model} model")
        plt.show()

        # Feature importance
        if config.get('func_feature_importance'):
            for func in config.get('func_feature_importance'):
                func(opt, feature_names, X_test, y_test)

        # Save optimized model
        path = './data/tmp/'
        # save_output(model, path, (X_train, y_train), (X_test, y_test), opt)

# ------------------------------------------------------------------------------------------------------------------

# Comparison of models' fitting
models_to_compare = ["svr", "xgboost", "lightGBM", "knn", "rf"]

# Plot
fig = plt.figure(figsize=(18/2.54, 18/2.54))
for n, model in enumerate(models_to_compare):

    # Load saved model
    opt, (X_train,y_train), (X_test,y_test) = load_model_from_tmp("data/tmp/", model)

    # Add subplot
    ax = fig.add_subplot(3, 2, n+1)
    sns.regplot(x=y_test, y=opt.predict(X_test), ax=ax)
    score = opt.score(X_test, y_test)
    ax.set(title = f"model : {model}, test_score : {score : .2f}",
           xlabel = r"STY_Experimental",
           ylabel = r"STY_predicted",
           )

plt.tight_layout()
# plt.savefig("figures/5_11_model_comparison.png", dpi = 600, bbox_inches='tight')
plt.show()

# Load models config from json file
# Map json directory to ML model
# with open('data/ML_models_STY/ML_models_STY.json') as f:
#     model_config = json.load(f)
# model_mapping = {
#     "xgboost": XGBRegressor,
# }
