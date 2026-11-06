"""Cross-validate, fit, and persist the raw-input regression model."""

from argparse import ArgumentParser, Namespace
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator, KFold, RepeatedKFold, ShuffleSplit
from sklearn.pipeline import Pipeline
import mlflow
from dvclive import Live
from omegaconf import OmegaConf
from hydra.utils import instantiate
from pipeline.data.utils import load_labeled_data
from pipeline.model.utils import (
    create_run_identity,
    cv_run_name,
    log_metrics,
    log_pipeline_configuration,
    regression_metrics,
    save_figure,
    save_run_identity,
)


logger = logging.getLogger(__name__)


def build_cv(args: Namespace) -> BaseCrossValidator:
    """Construct the configured cross-validation splitter."""
    if args.cv_type == "kfold":
        return KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.cv_random_state)
    if args.cv_type == "shuffle":
        return ShuffleSplit(n_splits=args.cv_splits, random_state=args.cv_random_state)
    return RepeatedKFold(
        n_splits=args.cv_splits,
        n_repeats=args.cv_repeats,
        random_state=args.cv_random_state,
    )


def load_model_template(config_path: Path) -> Pipeline:
    """Instantiate the configured scikit-learn model pipeline."""
    model = instantiate(OmegaConf.load(config_path), _convert_="all")
    if not isinstance(model, Pipeline):
        raise TypeError("The model configuration must instantiate a scikit-learn Pipeline.")
    return model


def get_feature_names(model: Pipeline, fallback: list[str]) -> list[str]:
    """Get transformed feature names, falling back to training columns."""
    try:
        return list(model[:-1].get_feature_names_out())
    except (AttributeError, ValueError):
        return fallback


def cross_validate(
    model_template: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    run_identity: dict[str, str],
    dvclive_dir: Path,
    params_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit isolated fold models and collect metrics and out-of-fold predictions."""
    fold_metrics: list[dict[str, float]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold, (train_indices, valid_indices) in enumerate(cv.split(X), start=1):
        fold_model = clone(model_template)
        X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
        y_train, y_valid = y.iloc[train_indices], y.iloc[valid_indices]
        fold_model.fit(X_train, y_train)
        train_predictions = fold_model.predict(X_train)
        valid_predictions = fold_model.predict(X_valid)
        metrics = {
            **regression_metrics(y_train, train_predictions, "train"),
            **regression_metrics(y_valid, valid_predictions, "valid"),
            "fold": float(fold),
        }
        fold_metrics.append(metrics)
        fold_predictions = pd.DataFrame({
            "fold": fold,
            "actual": y_valid.to_numpy(),
            "predicted": valid_predictions,
            "residual": y_valid.to_numpy() - valid_predictions,
        })
        prediction_frames.append(fold_predictions)
        child_run_name = cv_run_name(run_identity, fold)
        with mlflow.start_run(nested=True, run_name=child_run_name):
            log_pipeline_configuration(params_path)
            with Live(
                dir=str(dvclive_dir / f"cv-{fold}"),
                exp_name=child_run_name,
                save_dvc_exp=False,
                dvcyaml=False,
            ) as live:
                log_metrics({name: value for name, value in metrics.items() if name != "fold"}, live)
                live.log_plot(
                    "valid_predictions", fold_predictions, x="actual", y="predicted",
                    template="scatter", title=f"CV {fold}: predicted vs actual",
                    x_label="Actual", y_label="Predicted",
                )
                live.log_plot(
                    "valid_residuals", fold_predictions, x="predicted", y="residual",
                    template="scatter", title=f"CV {fold}: residuals vs predicted",
                    x_label="Predicted", y_label="Residual",
                )

    return pd.DataFrame(fold_metrics), pd.concat(prediction_frames, ignore_index=True)


def summarize_cv_metrics(cv_metrics: pd.DataFrame, cv_type: str) -> dict:
    """Summarize the mean and standard deviation of each fold metric."""
    return {
        "cv_type": cv_type,
        "folds": len(cv_metrics),
        "metrics": {
            metric: {
                "mean": float(cv_metrics[metric].mean()),
                "std": float(cv_metrics[metric].std(ddof=0)),
            }
            for metric in cv_metrics.columns
            if metric.startswith(("train_", "valid_"))
        },
    }


def save_cv_results(
    output_dir: Path,
    cv_metrics: pd.DataFrame,
    cv_predictions: pd.DataFrame,
    summary: dict,
) -> None:
    """Write fold-level tables and the DVC metrics summary."""
    cv_metrics.to_csv(output_dir / "cv_metrics.csv", index=False)
    cv_predictions.to_csv(output_dir / "cv_predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(summary, metrics_file, indent=2)


def log_cv_plots(
    cv_metrics: pd.DataFrame,
    cv_predictions: pd.DataFrame,
    output_dir: Path,
    live: Live | None = None,
) -> None:
    """Log fold metrics, prediction scatter, and residual diagnostics."""
    if live is not None:
        live.log_plot(
            "cv_metrics", cv_metrics, x="fold", y=["valid_rmse", "valid_mae", "valid_r2"],
            template="linear", title="Cross-validation metrics by fold",
            x_label="Fold", y_label="Score",
        )
        live.log_plot(
            "cv_predictions", cv_predictions, x="actual", y="predicted", template="scatter",
            title="Cross-validation predictions", x_label="Actual", y_label="Predicted",
        )
        live.log_plot(
            "cv_residuals", cv_predictions, x="predicted", y="residual", template="scatter",
            title="Cross-validation residuals", x_label="Predicted", y_label="Residual",
        )

    figure, axis = plt.subplots(figsize=(8, 5))
    cv_metrics[["train_rmse", "valid_rmse"]].plot(kind="box", ax=axis)
    axis.set(title="Train and validation RMSE across folds", ylabel="RMSE")
    save_figure(figure, "cv_rmse", output_dir, live)

    figure, axis = plt.subplots(figsize=(7, 7))
    axis.scatter(cv_predictions["actual"], cv_predictions["predicted"], alpha=0.55)
    limits = [
        min(cv_predictions["actual"].min(), cv_predictions["predicted"].min()),
        max(cv_predictions["actual"].max(), cv_predictions["predicted"].max()),
    ]
    axis.plot(limits, limits, linestyle="--", color="black")
    axis.set(title="Cross-validation: predicted vs actual", xlabel="Actual", ylabel="Predicted")
    save_figure(figure, "cv_predictions", output_dir, live)


def fit_composite_model(
    model_template: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor_path: Path,
    featurizer_path: Path,
) -> Pipeline:
    """Fit the estimator and prepend the saved raw-data transforms."""
    final_model = clone(model_template).fit(X, y)
    preprocessor = joblib.load(preprocessor_path)
    preprocessor.set_output(transform="pandas")
    featurizer = joblib.load(featurizer_path)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("featurizer", featurizer),
        ("model", final_model),
    ])


def log_feature_importance(
    model: Pipeline,
    X: pd.DataFrame,
    output_dir: Path,
    live: Live | None = None,
) -> None:
    """Log a feature-importance plot when the estimator exposes importances."""
    estimator = model.steps[-1][1]
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return
    feature_names = get_feature_names(model, list(X.columns))
    importance = pd.Series(importances, index=feature_names).sort_values().tail(20)
    figure, axis = plt.subplots(figsize=(9, max(5, len(importance) * 0.28)))
    importance.plot.barh(ax=axis)
    axis.set(title="Top model feature importances", xlabel="Importance", ylabel="Feature")
    save_figure(figure, "feature_importance", output_dir, live)


def build_parser() -> ArgumentParser:
    """Define the command-line options for training and cross-validation."""
    parser = ArgumentParser(description="Train a model.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Featurized training CSV.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for model and metrics artifacts.")
    parser.add_argument("--config", type=Path, default="configs/model.yaml", help="Model configuration YAML.")
    parser.add_argument("--preprocessor-path", type=Path, required=True, help="Fitted preprocessing artifact.")
    parser.add_argument("--featurizer-path", type=Path, required=True, help="Fitted feature engineering artifact.")
    parser.add_argument("--target-column", default="price", help="Regression target column.")
    parser.add_argument("--experiment-name", default="property-price-models", help="MLflow experiment name.")
    parser.add_argument("--params-file", type=Path, default="configs/params.yaml", help="Pipeline parameter YAML.")
    parser.add_argument("--cv-type", choices=["kfold", "shuffle", "repeated"], default="kfold")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--cv-random-state", type=int, default=37)
    return parser


def main() -> None:
    """Run cross-validation and save the complete raw-input model artifact."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()

    X, y = load_labeled_data(args.input_csv, args.target_column)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_template = load_model_template(args.config)
    mlflow.set_experiment(args.experiment_name)
    logger.info("Evaluating the model with %s cross-validation", args.cv_type)

    run_identity = create_run_identity(args.experiment_name)
    run_info_path = args.output_dir / "run_info.json"
    with mlflow.start_run(run_name=run_identity["run_name"]) as parent_run:
        run_identity["mlflow_run_id"] = parent_run.info.run_id
        mlflow.log_params({
            "experiment_base": args.experiment_name,
            "run_timestamp": run_identity["timestamp"],
            "run_identifier": run_identity["unique_id"],
            "cv_type": args.cv_type,
            "cv_splits": args.cv_splits,
            "cv_repeats": args.cv_repeats,
            "cv_random_state": args.cv_random_state,
            "model_config": str(args.config),
        })
        log_pipeline_configuration(args.params_file)
        cv_metrics, cv_predictions = cross_validate(
            model_template,
            X,
            y,
            build_cv(args),
            run_identity,
            args.output_dir / "dvclive",
            args.params_file,
        )
        summary = summarize_cv_metrics(cv_metrics, args.cv_type)
        save_cv_results(args.output_dir, cv_metrics, cv_predictions, summary)
        for metric, values in summary["metrics"].items():
            log_metrics({f"cv_{metric}_mean": values["mean"], f"cv_{metric}_std": values["std"]})
        log_cv_plots(cv_metrics, cv_predictions, args.output_dir)

        composite_model = fit_composite_model(
            model_template,
            X,
            y,
            args.preprocessor_path,
            args.featurizer_path,
        )
        model_path = args.output_dir / "model.joblib"
        joblib.dump(composite_model, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        log_feature_importance(composite_model.steps[-1][1], X, args.output_dir)
        save_run_identity(run_info_path, run_identity, parent_run.info.run_id)
        mlflow.log_artifact(str(run_info_path), artifact_path="metadata")

    logger.info("Saved composite model and CV metrics to %s", args.output_dir)


if __name__ == "__main__":
    main()