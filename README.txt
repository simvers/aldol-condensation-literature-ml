This project analyzes literature data for the aldol condensation of acetate with formaldehyde.
It focuses on reaction performed in the vapor phase on heterogeneous catalysts.

# -------------------

Preprocess encodes the composition of the catalyst as 0 and 1 if the element is absent or present, respectively.

# -------------------

Clustering attempts to group catalysts that are similar
The k-mean algorithm identifies 3 different catalyst families
- Al-Cs-Ti-P
- Si-Al-Cs-P 
- P-V-Ti-Si

# -------------------

Description summarizes in figures what has been done so fare in the literature: what catalysts, what conditions?

# -------------------

Deactivation quantifies the deactivation of catalysts from the literature over time 
Then, ML is used to identify the most important features that affect deactivation

# -------------------

Runfiles are used to run the different scripts

# -------------------

Utils contains useful functions that are used in the different scripts
