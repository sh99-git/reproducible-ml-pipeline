"""Evaluate a saved raw-input model and write holdout diagnostics."""

from argparse import ArgumentParser
import base64
import html
import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from dvclive import Live
from sklearn.pipeline import Pipeline
from sklearn.inspection import PartialDependenceDisplay
from pipeline.data.utils import load_labeled_data
from pipeline.model.utils import log_metrics, regression_metrics, save_figure


logger = logging.getLogger(__name__)


def unwrap_estimator(model):
	"""Return the final estimator nested inside scikit-learn pipelines."""
	while isinstance(model, Pipeline):
		model = model.steps[-1][1]
	return model


def get_model_features(
	model: Pipeline,
	X: pd.DataFrame,
) -> tuple[Pipeline, object, pd.DataFrame, pd.DataFrame, list[str]]:
	"""Transform input rows for the estimator and recover model feature names."""
	engineered = model[:-1].transform(X)
	model_pipeline = model.named_steps["model"]
	transformer = model_pipeline[:-1]
	estimator = unwrap_estimator(model_pipeline)
	try:
		transformed = transformer.transform(engineered)
		feature_names = list(transformer.get_feature_names_out())
	except (AttributeError, ValueError):
		transformed = engineered
		feature_names = list(engineered.columns)

	if isinstance(transformed, pd.DataFrame):
		transformed_frame = transformed.copy()
		transformed_frame.columns = feature_names
	else:
		transformed_frame = pd.DataFrame(transformed, columns=feature_names, index=engineered.index)
	return model_pipeline, estimator, engineered, transformed_frame, feature_names


def plot_partial_dependence(
	model_pipeline: Pipeline,
	engineered: pd.DataFrame,
	features: list[str],
	output_dir: Path,
	live: Live,
) -> Path:
	"""Plot partial dependence for up to five selected input features."""
	pdp_data = engineered.copy()
	numeric_columns = pdp_data.select_dtypes(include=np.number).columns
	pdp_data[numeric_columns] = pdp_data[numeric_columns].astype(float)
	figure, axis = plt.subplots(figsize=(10, 7))
	PartialDependenceDisplay.from_estimator(
		model_pipeline,
		pdp_data,
		features=features,
		ax=axis,
		n_cols=2,
		line_kw={"color": "#166534"},
	)
	figure.suptitle("Partial dependence of the five most important features")
	return save_figure(figure, "partial_dependence", output_dir, live)


def plot_shap_summary(estimator, transformed: pd.DataFrame, output_dir: Path, live: Live) -> Path:
	"""Create a SHAP summary for a reproducible sample of holdout rows."""
	import shap

	sample = transformed.sample(min(200, len(transformed)), random_state=0)
	background = transformed.sample(min(100, len(transformed)), random_state=1)
	explanation = shap.Explainer(estimator, background)(sample)
	figure = plt.figure(figsize=(10, 7))
	shap.summary_plot(explanation, sample, show=False, max_display=20)
	return save_figure(figure, "shap_summary", output_dir, live)


def create_diagnostic_plots(
	model: Pipeline,
	X: pd.DataFrame,
	output_dir: Path,
	live: Live,
) -> tuple[list[Path], str]:
	"""Create feature-importance, partial-dependence, and SHAP plots."""
	plot_paths: list[Path] = []
	diagnostics: list[str] = []
	model_pipeline, estimator, engineered, transformed, feature_names = get_model_features(model, X)
	importances = getattr(estimator, "feature_importances_", None)
	if importances is None and hasattr(estimator, "coef_"):
		importances = np.abs(np.asarray(estimator.coef_)).reshape(-1)

	if importances is not None and len(importances) == len(feature_names):
		importance = pd.Series(importances, index=feature_names).sort_values().tail(20)
		figure, axis = plt.subplots(figsize=(9, max(5, len(importance) * 0.28)))
		importance.plot.barh(ax=axis)
		axis.set(title="Top model feature importances", xlabel="Importance", ylabel="Feature")
		plot_paths.append(save_figure(figure, "feature_importance", output_dir, live))
		top_features = [name.split("__", 1)[-1] for name in importance.sort_values(ascending=False).index]
		top_features = [name for name in top_features if name in engineered.columns][:5]
		if top_features:
			try:
				plot_paths.append(plot_partial_dependence(model_pipeline, engineered, top_features, output_dir, live))
			except Exception as error:
				plt.close("all")
				diagnostics.append(f"Partial dependence plots were unavailable: {error}")
	else:
		diagnostics.append("Feature importance is unavailable for this estimator.")

	try:
		plot_paths.append(plot_shap_summary(estimator, transformed, output_dir, live))
	except Exception as error:
		plt.close("all")
		diagnostics.append(f"SHAP summary was unavailable: {error}")
	return plot_paths, " ".join(diagnostics)


def render_metric_rows(metrics: dict[str, float | int | str]) -> str:
	"""Render scalar evaluation metrics as escaped HTML table rows."""
	return "".join(
		f"<tr><th>{html.escape(str(name))}</th><td>{html.escape(str(value))}</td></tr>"
		for name, value in metrics.items()
	)


def render_worst_errors(predictions: pd.DataFrame) -> str:
	"""Render the ten largest absolute prediction errors as table rows."""
	worst_errors = predictions.nlargest(10, "absolute_error")
	return "".join(
		"<tr>" + "".join(f"<td>{value:.4f}</td>" for value in row) + "</tr>"
		for row in worst_errors[["actual", "predicted", "residual", "absolute_error"]]
		.itertuples(index=False, name=None)
	)


def render_embedded_plots(plot_paths: list[Path]) -> str:
	"""Embed generated PNG plots in the HTML report."""
	sections = []
	for path in plot_paths:
		encoded = base64.b64encode(path.read_bytes()).decode("ascii")
		title = html.escape(path.stem.replace("_", " ").title())
		sections.append(
			f'<section><h2>{title}</h2><img src="data:image/png;base64,{encoded}" alt="{title}"></section>'
		)
	return "".join(sections)


def write_report(
	output_path: Path,
	metrics: dict[str, float | int | str],
	predictions: pd.DataFrame,
	plot_paths: list[Path],
	diagnostics: str,
) -> None:
	"""Write a self-contained HTML report with metrics and holdout diagnostics."""
	diagnostic_section = f"<p class=\"notice\">{html.escape(diagnostics)}</p>" if diagnostics else ""
	output_path.write_text(
		"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
		"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
		"<title>Model evaluation report</title><style>"
		"body{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#17221c}"
		"h1,h2{color:#14532d}table{border-collapse:collapse;width:100%;margin:16px 0 32px}"
		"th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #d5ded8}"
		"img{display:block;max-width:100%;height:auto;margin:12px auto 28px}"
		"section{border-top:1px solid #d5ded8;padding-top:12px}.notice{color:#7c2d12}"
		"</style></head><body><h1>Model evaluation report</h1>"
		f"<p>Holdout rows: {metrics['rows']} | Target: {html.escape(str(metrics['target_column']))}</p>"
		f"<h2>Metrics</h2><table><tbody>{render_metric_rows(metrics)}</tbody></table>"
		f"<h2>Largest absolute errors</h2><table><thead><tr><th>Actual</th><th>Predicted</th>"
		f"<th>Residual</th><th>Absolute error</th></tr></thead><tbody>{render_worst_errors(predictions)}</tbody></table>"
		f"{diagnostic_section}{render_embedded_plots(plot_paths)}</body></html>",
		encoding="utf-8",
	)


def evaluate_holdout(
	model: Pipeline,
	X: pd.DataFrame,
	y: pd.Series,
	target_column: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
	"""Predict a holdout set and calculate its regression metrics and residuals."""
	predictions = model.predict(X)
	residuals = y.to_numpy() - predictions
	metrics: dict[str, float | int | str] = {
		"target_column": target_column,
		"rows": int(len(y)),
		**regression_metrics(y, predictions),
		"median_absolute_error": float(np.median(np.abs(residuals))),
		"mean_residual": float(np.mean(residuals)),
	}
	prediction_data = pd.DataFrame({
		"actual": y.to_numpy(),
		"predicted": predictions,
		"residual": residuals,
		"absolute_error": np.abs(residuals),
	})
	return metrics, prediction_data


def save_evaluation_results(
	output_dir: Path,
	metrics: dict[str, float | int | str],
	predictions: pd.DataFrame,
) -> tuple[Path, Path]:
	"""Write holdout metrics and per-row predictions to DVC-tracked files."""
	predictions_path = output_dir / "predictions.csv"
	predictions.to_csv(predictions_path, index=False)
	metrics_path = output_dir / "evaluation_metrics.json"
	metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
	return metrics_path, predictions_path


def plot_holdout_diagnostics(
	y: pd.Series,
	predictions: pd.DataFrame,
	output_dir: Path,
	live: Live,
) -> list[Path]:
	"""Create and log prediction, residual, and absolute-error plots."""
	plot_paths = []
	live.log_plot(
		"holdout_predictions", predictions, x="actual", y="predicted", template="scatter",
		title="Holdout: predicted vs actual", x_label="Actual", y_label="Predicted",
	)
	live.log_plot(
		"holdout_residuals", predictions, x="predicted", y="residual", template="scatter",
		title="Holdout residuals vs predicted", x_label="Predicted", y_label="Residual",
	)

	figure, axis = plt.subplots(figsize=(7, 7))
	axis.scatter(y, predictions["predicted"], alpha=0.6)
	limits = [min(y.min(), predictions["predicted"].min()), max(y.max(), predictions["predicted"].max())]
	axis.plot(limits, limits, linestyle="--", color="black")
	axis.set(title="Holdout: predicted vs actual", xlabel="Actual", ylabel="Predicted")
	plot_paths.append(save_figure(figure, "actual_vs_predicted", output_dir, live))

	figure, axis = plt.subplots(figsize=(8, 5))
	axis.scatter(predictions["predicted"], predictions["residual"], alpha=0.6)
	axis.axhline(0, color="black", linestyle="--")
	axis.set(title="Holdout residuals vs predicted", xlabel="Predicted", ylabel="Residual")
	plot_paths.append(save_figure(figure, "residuals_vs_predicted", output_dir, live))

	figure, axis = plt.subplots(figsize=(8, 5))
	axis.hist(predictions["residual"], bins=30, color="#0f766e", alpha=0.85)
	axis.axvline(0, color="black", linestyle="--")
	axis.set(title="Holdout residual distribution", xlabel="Residual", ylabel="Count")
	plot_paths.append(save_figure(figure, "residual_distribution", output_dir, live))

	figure, axis = plt.subplots(figsize=(8, 5))
	axis.scatter(y, predictions["absolute_error"], alpha=0.6)
	axis.set(title="Absolute error by actual target", xlabel="Actual", ylabel="Absolute error")
	plot_paths.append(save_figure(figure, "absolute_error_by_actual", output_dir, live))
	return plot_paths


def build_parser() -> ArgumentParser:
	"""Define command-line options for holdout model evaluation."""
	parser = ArgumentParser(description="Evaluate a model and create an HTML report.")
	parser.add_argument("--input-csv", type=Path, required=True, help="Raw holdout CSV file.")
	parser.add_argument("--model-path", type=Path, required=True, help="Composite model joblib file.")
	parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics, plots, and report.")
	parser.add_argument("--target-column", default="price", help="Regression target column.")
	parser.add_argument("--experiment-name", default="property-price-models", help="MLflow experiment name.")
	return parser


def main() -> None:
	"""Evaluate the holdout set and persist metrics, plots, and the HTML report."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
	args = build_parser().parse_args()
	args.output_dir.mkdir(parents=True, exist_ok=True)

	X, y = load_labeled_data(args.input_csv, args.target_column)
	model = joblib.load(args.model_path)
	metrics, prediction_data = evaluate_holdout(model, X, y, args.target_column)
	metrics_path, predictions_path = save_evaluation_results(args.output_dir, metrics, prediction_data)

	mlflow.set_experiment(args.experiment_name)
	plot_paths: list[Path] = []
	with mlflow.start_run(run_name="holdout-evaluation"):
		mlflow.log_params({"target_column": args.target_column, "holdout_rows": len(y)})
		with Live(
			dir=str(args.output_dir / "dvclive"),
			save_dvc_exp=False,
			dvcyaml=False,
		) as live:
			log_metrics({name: value for name, value in metrics.items() if isinstance(value, (float, int))}, live)
			plot_paths.extend(plot_holdout_diagnostics(y, prediction_data, args.output_dir, live))
			diagnostic_paths, diagnostics = create_diagnostic_plots(model, X, args.output_dir, live)
			plot_paths.extend(diagnostic_paths)
		report_path = args.output_dir / "evaluation_report.html"
		write_report(report_path, metrics, prediction_data, plot_paths, diagnostics)
		mlflow.log_artifact(str(predictions_path), artifact_path="evaluation")
		mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")
		mlflow.log_artifact(str(report_path), artifact_path="evaluation")

	logger.info("Wrote holdout metrics and evaluation report to %s", args.output_dir)


if __name__ == "__main__":
	main()