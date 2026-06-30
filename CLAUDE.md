# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

This repo analyzes literature data on the vapor-phase aldol condensation of acetic acid (acetate) and formaldehyde over heterogeneous catalysts. It ingests digitized literature data (catalyst composition, reaction conditions, performance, deactivation behavior), engineers features, and trains ML models to find which catalyst/condition features drive performance (STY, yield) and deactivation.

## Commands

There is no build system, linter, or test harness in this repo — it is a sequence of analysis scripts run directly with Python.

Install dependencies (the checked-in `requirements.txt` is UTF-16 encoded with some Windows-only conda wheel URLs; on Linux/WSL install core packages manually instead of using it directly):
```bash
pip install numpy pandas scikit-learn xgboost lightgbm torch joblib matplotlib seaborn shap scikit-optimize openpyxl scipy
```

Run the early pipeline stages in sequence via the tiny runner (only covers stages 0-2):
```bash
python run_files.py
```

Run a single stage directly (preferred when iterating). **Always run from the repo root** — scripts use paths relative to the repo root (e.g. `data/catalysts/...`, `figures/...`) and a `sys.path` shim that assumes `parents[1]` of the script is the repo root:
```bash
python scripts/0_preprocess.py
python scripts/3_features_engineering.py
python scripts/5_01_ML_STY_test.py
```

Scripts are not parameterized via CLI args — edit variables/paths near the top of the script (e.g. `data_type`, `rs` random seed, input file paths) and re-run.

## Architecture: numbered pipeline

Pipeline scripts live in `scripts/` and are numbered to indicate pipeline order — preserve the prefixes when adding or reordering stages. Each stage reads CSVs written by the previous stage from `data/catalysts/` and writes its own output CSV plus figures. Since the scripts live in `scripts/` rather than the repo root, each one that imports from `functions/` or `data/` starts with a small shim to put the repo root on `sys.path`:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
```

- `0_preprocess.py` — loads the raw literature dataset from an external `Catalysts.xlsx` (path hardcoded near the top, OneDrive location — must be edited per machine), encodes catalyst composition (element present/absent), computes molar flow rates, imputes SSA. Writes `data/catalysts/data_processed.csv` and `elements.csv`.
- `1_clustering.py` — k-means clustering (binarized composition) on `data_processed.csv` to group catalysts into families (e.g. Al-Cs-Ti-P, Si-Al-Cs-P, P-V-Ti-Si). Writes `data_clustered.csv` and `centroids.csv`.
- `2_description.py` — descriptive figures of the literature dataset (what catalysts/conditions have been studied) from `data_clustered.csv`.
- `3_features_engineering.py` — builds ML-ready features from composition + element physicochemical properties (weighted-average atomic properties, composition-level descriptors — see `ml_featurization.md` for the exact descriptor list and rationale). Writes `data_engineered.csv` / `data_engineered_noSi.csv`.
- `4_deactivation_modelling.py` — extracts digitized deactivation curves from an external directory of per-paper Excel files (`DigitizedData/`, path hardcoded, OneDrive location), fits deactivation curve models (`functions/deactivation_models.py`: power-law, exponential, Langmuir forms) per dataset, merges with reaction conditions. Writes `data_deactivation.csv`.
- `5_00_models_comparison.py`, `5_01_ML_STY_test.py`, `5_02_ML_STY_notest.py`, `5_03_ML_Yield_notest.py` — ML regression on `data_engineered*.csv` to predict STY / yield. Model configs (estimator + hyperparameter search space per algorithm) live in `data/ML_models_STY/ML_models_STY.py` as the `model_config` dict, keyed by model name (e.g. `"best_model"` = tuned XGBoost). `_test` variants hold out a test split; `_notest` variants fit on all data for final model artifacts.
- `6_01_ML_n.py` (and `6_ML_Deact_old.py`, superseded) — ML on deactivation rate `n` using `data/ML_models_n/ML_models_n.py` model configs.
- `7_Stats_Deact.py` — statistical analysis (partial correlations: `functions/stats_deact.py`) of deactivation drivers, with custom figure layouts (gridspec, colorbars).

`.development/` holds earlier/experimental versions of these scripts (gitignored, not part of the maintained pipeline) — useful only as historical reference, not for reuse.

## Shared code (`functions/`)

Always prefer reusing these over re-implementing:
- `ml_models.py` — the core ML toolkit shared by all `5_*`/`6_*` scripts: custom CV splitters for grouped/continuous-stratified regression data (`ContinuousStratifiedKFold`, `ContinuousStratifiedGroupKFold`, `GroupedKFold`), `stratified_grouped_train_test_split`, `OptimizationTracker`/`ConvergenceChecker` for BayesSearchCV progress, `save_output`/`load_model_from_tmp` for model + data persistence, and the plotting functions used for learning curves, parity plots, partial dependence, SHAP, and (permutation) feature importance.
- `preprocessing.py` — composition/molar flow rate calculation and SSA imputation used by `0_preprocess.py`.
- `features_engineering.py` — composition scaling and weighted-average element property featurization used by `3_features_engineering.py`.
- `deactivation_modelling.py` — Excel extraction and merging for deactivation data, used by `4_deactivation_modelling.py`.
- `deactivation_models.py` — the deactivation curve fit functions (power-law/exponential/Langmuir, 2- and 3-parameter variants) and `fit_model`/`analyze_fit`.
- `stats_deact.py` — partial correlation helpers used by `7_Stats_Deact.py`.
- `utils.py` — small generic helpers (`sorted_alphanumeric`, `filter_type`, `process_doi`, `clean_df_deactivation`).

## Data layout (`data/`)

- `data/catalysts/` — the canonical CSVs flowing through the pipeline (`data_processed.csv` → `data_clustered.csv` → `data_engineered*.csv` → `data_deactivation*.csv`), plus `elements.csv` (element list) and `centroids.csv` (cluster centroids).
- `data/mol_properties/` — JSON property tables for acetic acid/formaldehyde/stabilizer used in featurization.
- `data/features.json` — display labels (with LaTeX-style units) for feature names, used in plots.
- `data/ML_models_STY/ML_models_STY.py`, `data/ML_models_n/ML_models_n.py` — per-target `model_config` dicts (estimator + `skopt` search space + which feature-importance function to use) consumed by the `5_*`/`6_*` scripts. Add new models/hyperparameter ranges here rather than inline in the training scripts.
- Trained model artifacts are saved alongside these configs as `<model>_model.pkl`, `<model>_best_estimator.json`, `<model>_train_data.pkl`, `<model>_test_data.pkl` (joblib + JSON), via `ml_models.save_output`/`load_model_from_tmp` — follow this naming convention for any new model writer.
- `data/tmp/` — scratch/intermediate artifacts (gitignored).

Figures are written to `figures/<stage>/...` (e.g. `figures/clustering/`, `figures/ML_STY/`, `figures/Stats_n/`) mirroring the script that produced them; avoid changing these output paths without checking for downstream consumers.

## Conventions and gotchas

- Numeric prefixes on scripts in `scripts/` encode pipeline order — preserve them when adding or reordering stages.
- Several scripts hardcode absolute Windows/OneDrive input paths near the top (commented-out WSL/Windows alternates are usually left in place) — these need updating per machine/environment rather than being made configurable.
- Model training scripts build an sklearn `Pipeline` (`ColumnTransformer` with `StandardScaler`/`OneHotEncoder`) wrapped in `BayesSearchCV` (skopt) or `GridSearchCV`; when adding a model, follow the existing pattern in `5_01_ML_STY_test.py` and add its config to the relevant `ML_models_*.py` file.
- `data_type = '_noSi'` (or `''`) toggles between the full feature set and the silicon-excluded variant across the `5_*` scripts — check which is active when comparing results.
