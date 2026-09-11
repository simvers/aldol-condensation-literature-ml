# ML Literature — Aldol Condensation Catalyst Analysis

Machine learning analysis of literature data for the vapor-phase aldol condensation of acetic acid and formaldehyde over heterogeneous catalysts.

The repository ingests a digitized literature dataset (catalyst composition, reaction conditions, performance, deactivation behavior), engineers features from catalyst composition and elemental physicochemical properties, and trains ML models (XGBoost, LightGBM, random forest) to identify which catalyst and condition features drive space-time yield (STY), acrylate yield, and deactivation rate.

## Setup

```bash
conda env create -f environment.yml
conda activate simvers_ml_py311
```

There is no build system or test harness. Scripts are not parameterized via CLI — each script has a short docstring at the top listing what it reads/writes and which variables to edit before running. **Always run from the repository root**, since all paths are relative to it:

```bash
python scripts/0_preprocess_dataset.py
python scripts/6_2_train_ML_STY_CV_feateng.py
```

## Pipeline

Scripts in `scripts/` are numbered to indicate pipeline order. Each stage reads CSVs from `data/processed/` written by the previous stage and writes its own outputs (CSV + figures).

| Script | Purpose |
|---|---|
| `0_preprocess_dataset.py` | Loads `data/raw/data_raw.xlsx`, encodes catalyst composition, computes molar flow rates, imputes SSA. Writes `data_processed.csv`, `elements.csv`. |
| `1_cluster_catalysts.py` | K-means clustering on binarized composition to group catalysts into families. Writes `data_clustered.csv`, `centroids.csv`. |
| `2_describe_dataset.py` | Descriptive figures of the literature dataset (catalysts, conditions, performance coverage). |
| `3_engineer_features.py` | Builds ML-ready features from composition and elemental properties (composition-weighted mean/variance of atomic radius, valence, …). Writes `data_engineered.csv` / `data_engineered_noSi.csv`. |
| `4_fit_deactivation.py` | Extracts digitized deactivation curves from `data/raw/data_deactivation/`, fits power-law / exponential / Langmuir models per dataset, merges with conditions. Writes `data_deactivation.csv`. |
| `5_check_feature_correlation.py` | Feature–feature and feature–target (STY, n) correlation plots. |
| `6_1_train_ML_STY_overfit.py` | Single train/test split on a shallow XGBoost — illustrates why a single split score is not reproducible. |
| `6_2_train_ML_STY_CV_feateng.py` | Main STY model: cross-validated, doi-grouped, with engineered features. |
| `6_3_train_ML_STY_CV_OHE.py` | STY model variant using one-hot-encoded catalyst clusters instead of engineered features. |
| `6_4_train_ML_STY_CV_elem.py` | STY model variant using element composition. |
| `6_5_train_ML_STY_RSrobust.py` | Robustness check: trains each model across multiple random seeds, saves SHAP values per seed. |
| `6_6_plot_ML_STY_RSrobust.py` | Plots SHAP stability (beeswarm + feature importance boxplot) from `6_5` artifacts. |
| `6_7_train_ML_Y_CV_feateng.py` | Same CV pipeline for acrylate yield (Y) target. |
| `7_1_train_ML_n_CV_feateng.py` | CV pipeline for deactivation rate `n`. |
| `7_2_plot_stat_n.py` | Partial-correlation (Spearman, covariance-controlled) analysis of deactivation drivers. |

## Repository layout

```
data/
  raw/                        digitized source data
    data_raw.xlsx             catalyst / condition / performance
    data_deactivation/        one Excel per literature source (time-on-stream curves)
    atomic_features.json      elemental property table
    mol_properties/           physicochemical properties for reactants / stabilizers
  processed/                  generated data, pipeline intermediates
  ML_models_STY/              per-model robustness data (*.pkl) saved by 6_5_train_ML_STY_RSrobust.py

config/
  ML_models_STY.py            model configs for STY/yield targets
  ML_models_n.py              model configs for deactivation rate n
  feature_labels.json         display labels (short / long) for feature names used in plots

src/
  preprocessing.py            composition encoding, molar flow, SSA imputation
  features_engineering.py     weighted-average featurization
  deactivation_modelling.py   Excel extraction and deactivation curve fitting
  function_fitting.py         power-law / exponential / Langmuir fit functions
  ml_training.py              CV splitters, train/test split, GridSearchCV wrapper, preprocessor builder
  ml_plotting.py              learning curves, parity plots, SHAP, feature importance
  stats_analysis.py           partial correlation helpers
  utils.py                    small generic helpers

figures/                      all output figures, organised by script
```

## Citing

If you use this code or dataset, please cite the associated publication:

> Verstraeten, S., Palai, Y. N., Makshina, E., Sels, B. TBD.

## License

MIT — see `LICENSE`.
