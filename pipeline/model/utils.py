"""Shared helpers for model training and evaluation stages."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import mlflow
import numpy as np
import pandas as pd
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


def save_figure(figure: Figure, name: str, output_dir: Path, live: Live) -> Path:
    """Save a plot locally and register the same image with MLflow and DVCLive."""
    path = output_dir / "plots" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    mlflow.log_artifact(str(path), artifact_path="plots")
    live.log_image(name, str(path))
    plt.close(figure)
    return path