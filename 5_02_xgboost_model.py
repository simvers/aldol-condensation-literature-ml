import warnings
import time
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import plot_importance
import numpy as np
from model_IO import save_output

# configuration
warnings.filterwarnings("ignore")
plt.rcParams["font.size"] = 8
np.random.seed(8) ##############NEW############

# Convergence criteria
class ConvergenceChecker:
    '''
    This will be passed as callback to the optimizer object.
    If this returns True, then optimization will stop.
    '''
    def __init__(self, min_improvement = 0.02, patience=15):
        """
        Initialize the ConvergenceChecker.
        Paramters:
            min_improvement (float): cutoff criteria in term of delta_r2
            patience (int): number of cycles to averagve over.
        """
        self.min_improvement = min_improvement
        self.patience = patience
        self.r2_scores = []
    
    def __call__(self, result):
        current_r2 = -result.func_vals[-1] # For some reason this is a negative number (Figure out why.)
        self.r2_scores.append(current_r2)

        # Don't check the stopping criteria in the begining, at least a few (=patience in this case)
        if len(self.r2_scores) < self.patience:
            return False

        recent_scores = self.r2_scores[-self.patience:] # This will check last few (=patience)
        best_recent = max(recent_scores)
        worst_recent = min(recent_scores)
        improvement = best_recent - worst_recent

        if improvement < self.min_improvement:
            print(f"\nConverged! Improvement of {improvement:.4f} over last {self.patience} iterations is below threshold {self.min_improvement}")
            return True

        return False # Continue to optimize
        
class OptimizationTracker:
    def __init__(self):
        self.iteration_scores = []
        self.iteration_params = []
        self.start_time = time.time()
        self.best_score = -np.inf

    def __call__(self, res):
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Get current iteration info
        iteration = len(res.func_vals)
        current_score = -res.func_vals[-1] # Again the same thing here, this is a negative number
        # current_params = res.x_iters[-1] # Get the latest paramers
        self.iteration_scores.append(current_score)

        param_dict = dict(zip(res.space.dimension_names, res.x_iters[-1]))
        self.iteration_params.append(param_dict)

        # Update best score
        if current_score > self.best_score:
            self.best_score = current_score
            improvement = "New Best!"
        else:
            improvement = ""

        # Print progress
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Iteration {iteration:3d} | "
              f"R² = {current_score:.4f} | Best R² = {self.best_score:.4f} | "
              f"Elapsed: {elapsed/60:.1f}min{improvement}")
        
        # Print best parameters every 25 iterations
        if iteration % 25 == 0 and iteration > 0:
            best_idx = np.argmax(self.iteration_scores)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Best parameters so far (iteration {best_idx + 1}):")
            for param, value in self.iteration_params[best_idx].items():
                if isinstance(value, float):
                    print(f"\t{param}: {value:.4f}")
                else:
                    print(f"\t{param}: {value}")
            print()
if __name__=="__main__":
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
        ('reg', XGBRegressor(verbosity = 0, nthread = -1, random_state = 8))
    ])

    # Building hyperparams tuning space with BayerSearchCV
    search_space = {
        'reg__n_estimators': Integer(100, 1000),           # This is important, which was not there before 
        'reg__max_depth': Integer(2, 8),
        'reg__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
        'reg__subsample': Real(0.5, 1.0),
        'reg__colsample_bytree': Real(0.5, 1.0),          
        'reg__min_child_weight': Integer(1, 10),           # this prevents overfitting
        'reg__reg_alpha': Real(0.0, 10.0),
        'reg__reg_lambda': Real(0.0, 10.0),
        'reg__gamma': Real(0.0, 10.0)
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=8) ###############NEW############

    opt = BayesSearchCV(pipe, search_space, cv=cv, n_iter=1000, scoring='r2', random_state=8)
    # the callback instances
    tracker = OptimizationTracker()
    convergence_checker = ConvergenceChecker()

    # Executing the pipline with train_data
    opt.fit(X_train, y_train, callback=[tracker, convergence_checker])

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


    path = './data/tmp/'
    model = "xgboost"

    # This will now save tracker object with all its stuff
    save_output(model, path, (X_train, y_train), (X_test, y_test), opt, tracker)

