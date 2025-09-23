'''
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
        current_score = res.func_vals[-1]  # Latest score (negative because skopt minimizes)
        current_params = res.x_iters[-1]   # Latest parameters
        
        # Convert to positive R²
        current_r2 = -current_score
        
        # Track scores and parameters
        self.iteration_scores.append(current_r2)
        
        # Convert parameter values to dictionary
        param_dict = {}
        param_names = list(search_space.keys())
        for i, param_name in enumerate(param_names):
            param_dict[param_name] = current_params[i]
        self.iteration_params.append(param_dict)
        
        # Update best score
        if current_r2 > self.best_score:
            self.best_score = current_r2
            improvement = " ⭐ NEW BEST!"
        else:
            improvement = ""
        
        # Print progress
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Iteration {iteration:3d} | "
              f"R² = {current_r2:.4f} | Best R² = {self.best_score:.4f} | "
              f"Elapsed: {elapsed/60:.1f}min{improvement}")
        
        # Print best parameters every 25 iterations
        if iteration % 25 == 0 and iteration > 0:
            best_idx = np.argmax(self.iteration_scores)
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Best parameters so far (iteration {best_idx + 1}):")
            for param, value in self.iteration_params[best_idx].items():
                if isinstance(value, float):
                    print(f"    {param}: {value:.4f}")
                else:
                    print(f"    {param}: {value}")
            print()

'''
