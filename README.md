# SHAP-Guided RBF-KAN Reference Code

This repository provides a compact reference implementation of the main modelling workflow described in **SHAP-Guided Variable Selection with Radial Basis Function Kolmogorov-Arnold Networks for Interpretable Infrared Spectral Quantification**.

The retained scientific components are:

- Gaussian RBF-KAN and B-spline KAN regression layers
- spectral baseline correction, SNV, Savitzky-Golay smoothing, and first or second derivatives
- Gradient-SHAP attribution and deterministic Top-K selection with stable tie handling
- calibration-only Optuna-TPE hyperparameter optimization
- runnable SVR, XGBoost, MLP, 1D-CNN, and B-spline KAN comparisons
- internal neural-network epoch selection followed by complete-calibration refitting

## Repository layout

```text
src/shap_rbf_kan/
  models.py           RBF-KAN, B-spline KAN, and 1D-CNN definitions
  model_search.py     Model fitting used by cross-validation and final refitting
  preprocessing.py    Spectral preprocessing operations
  shap_selection.py   SHAP aggregation and Top-K selection
  tuning.py           Optuna spaces, trial budgets, and five-fold objective
  workflow.py         Full-spectrum and SHAP-selected RBF-KAN workflow
scripts/
  run_bayesian_search.py
  run_reference_workflow.py
tests/
```

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Bayesian optimization protocol

Each model is optimized with Optuna's seeded `TPESampler`. The first 10 completed trials use the sampler's startup phase; subsequent configurations are proposed adaptively from the accumulated study results. Every trial uses the same shuffled calibration-only five-fold partitions and minimizes mean validation RMSE. Fold-level RMSE values and their standard deviation are retained with the trial.

Pruning is disabled. Search stops after the declared number of completed trials. A resumed SQLite study retains failed trials, converts a stale running trial from an interrupted process to failed, and runs only the additional trials needed to reach the completed-trial total. Run one process per study.

| Model | Trials per dataset | Hyperparameter space |
|---|---:|---|
| SVR | 48 | `C`: 0.1–1000 log; `gamma`: 1e-5–0.1 log; `epsilon`: 0.001–0.2 log |
| XGBoost | 36 | depth: 2–8; learning rate: 0.005–0.2 log; estimators: 200–1200; child weight: 1–10; row/column sampling: 0.6–1.0; L1: 1e-8–10 log; L2: 1e-3–100 log |
| MLP | 24 | six one-to-three-layer architectures; ReLU/Tanh; alpha: 1e-6–0.1 log; learning rate: 1e-5–0.01 log; batch: 32/64/128 |
| 1D-CNN | 24 | channels: 8–48; kernel: 3–15 odd; dense width: 16–128; dropout: 0–0.4; learning rate: 1e-5–0.005 log; weight decay: 1e-6–0.01 log; batch: 32/64/128 |
| B-spline KAN | 40 | dataset-specific hidden architectures; bases: 3/5/7/9/11; grid half-width: 2.0–5.0; dropout: 0–0.2; learning rate: 1e-5–0.005 log; weight decay: 1e-6–0.1 log; batch: 32/64/128 |

The sampler seed and fold seed default to `20260726`. Random seed is controlled rather than tuned.

## Run a model search

The calibration CSV must contain numeric spectral columns and the named target. A prediction CSV can be supplied with the same columns.

```bash
python scripts/run_bayesian_search.py \
  --calibration-csv calibration.csv \
  --prediction-csv prediction.csv \
  --target reference_value \
  --model xgboost \
  --dataset fruit \
  --preprocessing snv d2 sg \
  --output-dir outputs/fruit_xgboost
```

Use `--resume` to continue the SQLite study. `--trials` specifies the total completed-trial target; when omitted, the model budget in the table is used.

The command writes:

- `optuna_studies.sqlite3`: resumable Optuna study
- `<dataset>_<model>_trials.csv`: parameters and fold-level results for every trial
- `<dataset>_<model>_summary.json`: protocol, best parameters, validation results, selected epoch, and optional prediction metrics

Prediction labels are read only after hyperparameter optimization and final complete-calibration refitting. They are used solely to calculate the final held-out metrics.

## Neural-network stopping criteria

MLP, 1D-CNN, and B-spline KAN use an internal 15% split of each fold-training partition for epoch selection. The selected model is then reinitialized and trained on the entire fold-training partition for the selected number of epochs. The final selected configuration follows the same procedure on the complete calibration set.

| Model | Maximum epochs | Earliest stopping epoch | Patience | Minimum improvement |
|---|---:|---:|---:|---:|
| MLP | 2000 | 80 | 80 | 1e-5 |
| 1D-CNN | 900 | 100 | 100 | 1e-7 |
| B-spline KAN | 700 | 80 | 80 | 1e-7 |

All three neural models use AdamW, mean-squared error, cosine learning-rate decay, and gradient clipping. Command-line epoch options can be used for short smoke tests.

## SHAP-guided RBF-KAN workflow

```bash
python scripts/run_reference_workflow.py \
  --calibration-csv calibration.csv \
  --prediction-csv prediction.csv \
  --target reference_value \
  --top-k 30 \
  --preprocessing snv d2 sg \
  --output reference_summary.json
```

Use either command with `--help` for the complete argument list.

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

The test suite checks model gradients, preprocessing invariants, SHAP ranking, Bayesian spaces and budgets, exact trial stopping, resumable studies, fold-local training, and the separation of prediction labels from tuning and checkpoint selection.
