"""Shared helpers for model training and evaluation stages."""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mlflow
import numpy as np
import pandas as pd
import yaml
from dvclive import Live


def regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    prefix: str = "",
) -> dict[str, float]:
    """Calculate consistently named regression metrics for a prediction set."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    label = f"{prefix}_" if prefix else ""
    mse = mean_squared_error(actual, predicted)
    return {
        f"{label}mse": float(mse),
        f"{label}rmse": float(np.sqrt(mse)),
        f"{label}mae": float(mean_absolute_error(actual, predicted)),
        f"{label}r2": float(r2_score(actual, predicted)),
    }


def log_metrics(metrics: dict[str, float | int], live: Live | None = None) -> None:
    """Send numeric metrics to the active MLflow run and optional DVCLive run."""
    normalized = {name: float(value) for name, value in metrics.items()}
    mlflow.log_metrics(normalized)
    if live is not None:
        for name, value in normalized.items():
            live.log_metric(name, value)


def create_run_identity(experiment_name: str) -> dict[str, str]:
    """Build a unique, readable parent run name from the configured base name."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    unique_id = uuid4().hex[:8]
    return {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "unique_id": unique_id,
        "run_name": f"{experiment_name}-training-{timestamp}-{unique_id}",
    }


def cv_run_name(run_identity: dict[str, str], fold: int) -> str:
    """Append a fold suffix while retaining the parent timestamp and ID."""
    return f"{run_identity['run_name']}-cv-{fold}"


def save_run_identity(path: Path, run_identity: dict[str, str], run_id: str) -> None:
    """Persist the MLflow parent ID and naming metadata for evaluation to resume."""
    metadata = {**run_identity, "mlflow_run_id": run_id}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_run_identity(path: Path) -> dict[str, str]:
    """Load the parent run metadata created by the training stage."""
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_params(params: dict, prefix: str = "") -> dict[str, str]:
    """Flatten params.yaml for MLflow, omitting configured output directories."""
    flattened: dict[str, str] = {}
    for key, value in params.items():
        if key == "output":
            continue
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_params(value, name))
        else:
            flattened[name] = str(value)
    return flattened


def log_pipeline_configuration(params_path: Path) -> dict[str, str]:
    """Log shared pipeline params, config files, and their DVC-compatible MD5s."""
    params_path = params_path.resolve()
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    mlflow.log_params(flatten_params(params))
    mlflow.log_artifact(str(params_path), artifact_path="configuration")

    hashes: dict[str, str] = {}
    for stage in ("prepare", "featurize", "train"):
        config_value = params.get(stage, {}).get("config")
        if not config_value:
            continue
        config_path = (params_path.parent.parent / config_value).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file for {stage} not found: {config_path}")
        digest = hashlib.md5(config_path.read_bytes()).hexdigest()
        hashes[f"dvc_md5.{stage}_config"] = digest
        mlflow.log_artifact(str(config_path), artifact_path=f"configuration/{stage}")

    mlflow.log_params(hashes)
    return hashes


def save_figure(
    figure: Figure,
    name: str,
    output_dir: Path,
    live: Live | None = None,
) -> Path:
    """Save a plot locally and register the same image with MLflow and DVCLive."""
    path = output_dir / "plots" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    mlflow.log_artifact(str(path), artifact_path="plots")
    if live is not None:
        live.log_image(name, str(path))
    plt.close(figure)
    return path