# reproducible-ml-pipeline

A reproducible property-price regression workflow managed with DVC. The pipeline prepares raw CSV data, engineers features, cross-validates a model, saves a raw-input model artifact, and evaluates it on a held-out test set.

## Run the Pipeline

Install the locked environment and run all stages:

```bash
uv sync
MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run dvc repro
```

Run an individual stage and its dependencies with, for example, `uv run dvc repro train` or `uv run dvc repro evaluate`.

To inspect runs in MLflow, start the UI against the same tracking database:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Configuration

- `configs/params.yaml` is tracked directly in Git for readable diffs. DVC uses the parameter declarations in `dvc.yaml` to track the values used by each stage and rerun only affected stages and their downstream dependents.
- `configs/preprocess.yaml` configures raw-column transformations.
- `configs/featurizer.yaml` configures property-specific feature engineering.
- `configs/model.yaml` defines the scikit-learn estimator pipeline.

Change these files before running DVC so stage parameters and dependencies are recorded in the lockfile.

## Experiment Runs

`experiment-name` in `configs/params.yaml` is the MLflow experiment and run-name base. Each training invocation creates a parent run named `<base>-training-<timestamp>-<id>`. Cross-validation runs are nested children named with the same parent identity and a `-cv-N` suffix. The training stage saves the parent run ID and naming metadata to `models/run_info.json`; evaluation resumes that run and adds holdout metrics and report artifacts to the parent.

DVCLive does not use MLflow-style nested runs here. With the configured five folds, training writes five independent metric/plot sets under `models/dvclive/cv-1/` through `models/dvclive/cv-5/`. Evaluation writes the overall holdout metrics under `reports/evaluation/dvclive/`, giving DVC six independently tracked metric files: five fold runs and one overall run.

## Data Modules

- `pipeline/data/utils.py` provides shared CSV-directory loading and labeled-data splitting.
- `pipeline/data/prepare.py` creates the reproducible train/holdout split, fits preprocessing on training features only, and saves the untouched holdout rows.
- `pipeline/data/featurize.py` fits property feature engineering on the prepared training set and optionally adds configured noise to eligible features. The target and encoded categorical fields are excluded from noise.

## Artifacts

| Location | Contents |
| --- | --- |
| `data/processed/` | Prepared training and holdout CSVs plus the fitted preprocessor. |
| `data/features/` | Featurized training data and fitted feature engineer. |
| `models/` | Composite `model.joblib`, parent `run_info.json`, cross-validation metrics and predictions, plots, and per-fold DVCLive logs. |
| `reports/evaluation/` | Holdout metrics, per-row predictions, plots, DVCLive logs, and `evaluation_report.html`. |

The saved composite model accepts raw property rows and applies the fitted preprocessor and feature engineer before prediction. Training uses the featurized training CSV; evaluation uses the untouched raw holdout CSV.

## Evaluation Report

The HTML report is self-contained and includes regression metrics, the ten largest absolute errors, predicted-versus-actual and residual plots, absolute-error distribution, feature importance, partial dependence for up to five important features, and a SHAP summary. Diagnostic plots are logged to both MLflow and DVCLive. If an estimator or plotting dependency cannot provide a diagnostic, the report records the reason.
