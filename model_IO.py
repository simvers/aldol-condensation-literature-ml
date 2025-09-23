import json
import os
import joblib
import pandas as pd


def save_output(model, path, train, test, opt, tracker=None):
    if not os.path.isdir(path):
        os.mkdir(path)

    # Save the optimized model hyperparams
    with open(f"{path}{model}_best_estimator.json", "w") as f:
        json.dump(opt.best_params_,f,indent=4)

    # Save the trained model
    joblib.dump(opt.best_estimator_, f"{path}{model}_model.pkl")

    # Save the test/train dataset to a pickle file
    joblib.dump(train, f"{path}{model}_train_data.pkl")
    joblib.dump(test, f"{path}{model}_test_data.pkl")
    # Save tracker data if provided
    if tracker is not None:
        # Save optimization history as CSV
        optimization_history = pd.DataFrame({
            'iteration': range(1, len(tracker.iteration_scores) + 1),
            'r2_score': tracker.iteration_scores
        })
        
        # Add parameter columns
        if tracker.iteration_params:
            param_df = pd.DataFrame(tracker.iteration_params)
            optimization_history = pd.concat([optimization_history, param_df], axis=1)
        
        optimization_history.to_csv(f"{path}{model}_optimization_history.csv", index=False)
        
        # Save tracker object itself
        joblib.dump(tracker, f"{path}{model}_tracker.pkl")
        
        # Save summary stats
        tracker_summary = {
            'total_iterations': len(tracker.iteration_scores),
            'best_r2_score': tracker.best_score,
            'final_r2_score': tracker.iteration_scores[-1] if tracker.iteration_scores else None,
            'improvement_over_time': tracker.best_score - tracker.iteration_scores[0] if tracker.iteration_scores else None
        }
        
        with open(f"{path}{model}_tracker_summary.json", "w") as f:
            json.dump(tracker_summary, f, indent=4)

def load_model_from_tmp(path, model, include_tracker=False):
    best_estimator = joblib.load(f"{path}{model}_model.pkl")
    train = joblib.load(f"{path}{model}_train_data.pkl")
    test = joblib.load(f"{path}{model}_test_data.pkl")

    if include_tracker:
        try:
            tracker = joblib.load(f"{path}{model}_tracker.pkl")
            return best_estimator, train, test, tracker
        except FileNotFoundError:
            print(f"\t[Warning....]Tracker file not found at {path}{model}_tracker.pkl")
            return best_estimator, train, test, None

    return best_estimator, train, test
