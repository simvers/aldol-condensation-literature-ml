import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import pandas as pd

def kde_similarity(train, test, grid_points=1000):
    kde_train = gaussian_kde(train)
    kde_test = gaussian_kde(test)
    
    x_grid = np.linspace(min(train.min(), test.min()), max(train.max(), test.max()), grid_points)
    p = kde_train(x_grid)
    q = kde_test(x_grid)
    
    # Normalize just in case
    p /= p.sum()
    q /= q.sum()
    
    # JS distance (0 = identical, 1 = very different)
    js_dist = jensenshannon(p, q)
    return js_dist

# Example
df = pd.read_csv("data/tmp/processed_data.csv")
X = df.drop(columns="STY_MA+AA_(mmol/h/g)")
y  = df["STY_MA+AA_(mmol/h/g)"]

best_seed = None
best_js = float('inf')

for seed in range(100):
    qcut = pd.qcut(y, 5, duplicates="drop")
    train_X, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=qcut)
    js = kde_similarity(train_y, test_y)
    if js < best_js:
        best_js = js
        best_seed = seed

print("Best seed:", best_seed, "with JS distance:", best_js)
