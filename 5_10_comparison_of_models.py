from model_IO import load_model_from_tmp
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["font.size"] = 8

models = ["svr", "xgboost", "lightGBM", "knn", "rf"]
fig = plt.figure(figsize=(18/2.54, 18/2.54))
for n,model in enumerate(models):
    opt, (X_train,y_train), (X_test,y_test) = load_model_from_tmp("data/tmp/", model)
    ax = fig.add_subplot(3, 2, n+1)
    sns.regplot(x = y_test, y = opt.predict(X_test), ax = ax)
    score = opt.score(X_test, y_test)
    ax.set(title = f"model : {model}, test_score : {score : .2f}",
           xlabel = r"STY_Experimental",
           ylabel = r"STY_predicted",
           )
    

plt.tight_layout()
plt.savefig("figures/5_11_model_comparison.png", dpi = 600, bbox_inches='tight')
plt.show()
