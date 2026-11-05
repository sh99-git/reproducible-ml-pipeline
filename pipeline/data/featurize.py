"""Engineer property features, optionally add training noise, and save artifacts."""

from argparse import ArgumentParser
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from omegaconf import OmegaConf
from hydra.utils import instantiate
import joblib
from pipeline.data.utils import load_labeled_data

logger = logging.getLogger(__name__)
NOISE_EXCLUDED_COLUMNS = {"furnishing"}
NOISE_EXCLUDED_PREFIXES = ("amenities_", "property_type_", "location_")

def add_noise_to_dataframe(
	df: pd.DataFrame,
	columns: list[str],
	noise_level: float = 0.1,
	noise_type: str = "gaussian",
	random_state: int = 57,
	fraction: float = 0.25
	) -> pd.DataFrame:
	"""Add noise to specified columns of a DataFrame."""
	if not 0 <= fraction <= 1:
		raise ValueError("fraction must be between 0 and 1.")
	if not 0 <= noise_level <= 1:
		raise ValueError("noise_level must be between 0 and 1.")
	if noise_type not in {"gaussian", "uniform"}:
		raise ValueError("noise_type must be 'gaussian' or 'uniform'.")

	np.random.seed(random_state)
	noisy_df = df.copy()
	num_rows = len(df)
	num_noisy_rows = int(num_rows * fraction)
	noisy_indices = np.random.choice(num_rows, num_noisy_rows, replace=False)

	for col in columns:
		if col not in df.columns:
			logger.warning("Column %s not found in DataFrame. Skipping.", col)
			continue
		if noise_type == "gaussian":
			std_dev = df[col].std() * noise_level
			noise = np.random.normal(0, std_dev, size=num_noisy_rows)
		else:  # uniform
			range_ = df[col].max() - df[col].min()
			noise = np.random.uniform(-range_ * noise_level, range_ * noise_level, size=num_noisy_rows)

		noisy_df[col] = noisy_df[col].astype(float)
		noisy_df.loc[noisy_indices, col] += noise

	return noisy_df


def select_noise_columns(data: pd.DataFrame, target_column: str) -> list[str]:
	"""Return numeric candidates except the target and encoded categorical fields."""
	excluded_columns = NOISE_EXCLUDED_COLUMNS | {target_column}
	return [
		column
		for column in data.columns
		if column not in excluded_columns and not column.startswith(NOISE_EXCLUDED_PREFIXES)
	]


class PropertyFeatureEngineer(TransformerMixin, BaseEstimator):
	"""Create property features from a pandas feature matrix."""

	def __init__(
		self,
		size_column: str = "size_sqft",
		bedrooms_column: str = "num_bedrooms",
		bathrooms_column: str = "num_bathrooms",
		floor_column: str = "floor_number",
		total_floors_column: str = "total_floors",
		amenity_prefix: str = "amenities_",
	):
		self.size_column = size_column
		self.bedrooms_column = bedrooms_column
		self.bathrooms_column = bathrooms_column
		self.floor_column = floor_column
		self.total_floors_column = total_floors_column
		self.amenity_prefix = amenity_prefix

	def _validate_input(self, X: pd.DataFrame) -> None:
		if not isinstance(X, pd.DataFrame):
			raise TypeError("PropertyFeatureEngineer requires a pandas DataFrame.")
		missing_columns = self._required_columns().difference(X.columns)
		if missing_columns:
			raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

	def _required_columns(self) -> set[str]:
		return {
			self.size_column,
			self.bedrooms_column,
			self.bathrooms_column,
			self.floor_column,
			self.total_floors_column,
		}

	def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PropertyFeatureEngineer":
		"""Validate the schema and remember input and amenity columns."""
		self._validate_input(X)

		self.amenity_columns_ = [
			column for column in X.columns if column.startswith(self.amenity_prefix)
		]
		if not self.amenity_columns_:
			raise ValueError(
				f"Expected at least one amenity column with the {self.amenity_prefix!r} prefix."
			)

		self.feature_names_in_ = np.asarray(X.columns, dtype=object)
		self.n_features_in_ = len(self.feature_names_in_)
		return self

	def transform(self, X: pd.DataFrame) -> pd.DataFrame:
		"""Add normalized property ratios and aggregate amenity count."""
		check_is_fitted(self, ["amenity_columns_", "feature_names_in_"])
		self._validate_input(X)

		featured = X.reindex(columns=self.feature_names_in_, fill_value=0).copy()
		featured["size_sqft_per_bedroom"] = featured[self.size_column].div(
			featured[self.bedrooms_column]
		).where(featured[self.bedrooms_column].ne(0), 0)
		featured["bathrooms_per_bedroom"] = featured[self.bathrooms_column].div(
			featured[self.bedrooms_column]
		).where(featured[self.bedrooms_column].ne(0), 0)
		featured["floor_ratio"] = featured[self.floor_column].div(
			featured[self.total_floors_column]
		).where(featured[self.total_floors_column].ne(0), 0)
		featured["amenity_count"] = featured[self.amenity_columns_].sum(axis=1)
		return featured

	def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
		"""Return original and engineered output feature names."""
		check_is_fitted(self, ["amenity_columns_", "feature_names_in_"])
		if input_features is not None and list(input_features) != list(self.feature_names_in_):
			raise ValueError("input_features must match the columns seen during fit.")
		generated = [
			"size_sqft_per_bedroom",
			"bathrooms_per_bedroom",
			"floor_ratio",
			"amenity_count",
		]
		names = [name for name in self.feature_names_in_ if name not in generated]
		return np.asarray([*names, *generated], dtype=object)


def build_parser() -> ArgumentParser:
	"""Define command-line options for feature engineering and noise."""
	parser = ArgumentParser(description="Engineer features from the prepared training CSV.")
	parser.add_argument("--input-csv", type=Path, required=True, help="Prepared training CSV.")
	parser.add_argument("--output-dir", type=Path, required=True, help="Directory for features and transformer.")
	parser.add_argument("--target-column", default="price", help="Target column retained outside feature engineering.")
	parser.add_argument("--config", type=Path, default="configs/featurizer.yaml", help="Featurizer configuration YAML.")
	parser.add_argument("--noise-level", type=float, default=0.1, help="Noise magnitude relative to each feature range.")
	parser.add_argument("--noise-type", choices=["gaussian", "uniform"], default="gaussian")
	parser.add_argument("--noise-state", type=int, default=57, help="Random seed for noise selection and values.")
	parser.add_argument("--noise-fraction", type=float, default=0.25, help="Fraction of rows receiving noise.")
	return parser


def load_featurizer(config_path: Path):
	"""Instantiate the feature engineer from its Hydra configuration."""
	return instantiate(OmegaConf.load(config_path))


def transform_features(
	input_path: Path,
	target_column: str,
	featurizer,
) -> pd.DataFrame:
	"""Fit feature engineering on training features and retain the target column."""
	features, target = load_labeled_data(input_path, target_column)
	logger.info("Fitting featurizer on %d rows and %d input columns", len(features), len(features.columns))
	transformed = featurizer.fit_transform(features)
	transformed[target_column] = target
	return transformed


def add_configured_noise(data: pd.DataFrame, args) -> pd.DataFrame:
	"""Apply configured noise only to eligible engineered numeric features."""
	return add_noise_to_dataframe(
		data,
		columns=select_noise_columns(data, args.target_column),
		noise_level=args.noise_level,
		noise_type=args.noise_type,
		random_state=args.noise_state,
		fraction=args.noise_fraction,
	)


def save_features(output_dir: Path, data: pd.DataFrame, featurizer) -> None:
	"""Write the feature CSV and fitted feature-engineering artifact."""
	output_dir.mkdir(parents=True, exist_ok=True)
	data.to_csv(output_dir / "features.csv", index=False)
	joblib.dump(featurizer, output_dir / "featurizer.joblib")


def main() -> None:
	"""Run feature engineering and save the configured training artifacts."""
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
	args = build_parser().parse_args()
	logger.info("Loading featurizer configuration from %s", args.config)
	featurizer = load_featurizer(args.config)
	logger.info("Reading input data from %s", args.input_csv)
	features = transform_features(args.input_csv, args.target_column, featurizer)
	features = add_configured_noise(features, args)
	save_features(args.output_dir, features, featurizer)
	logger.info("Wrote %d rows and %d columns to %s", len(features), len(features.columns), args.output_dir)


if __name__ == "__main__":
	main()
