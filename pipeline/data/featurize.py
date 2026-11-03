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

logger = logging.getLogger(__name__)


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

	def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PropertyFeatureEngineer":
		if not isinstance(X, pd.DataFrame):
			raise TypeError("PropertyFeatureEngineer requires a pandas DataFrame.")

		required_columns = {
			self.size_column,
			self.bedrooms_column,
			self.bathrooms_column,
			self.floor_column,
			self.total_floors_column,
		}
		missing_columns = required_columns.difference(X.columns)
		if missing_columns:
			raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")

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
		check_is_fitted(self, ["amenity_columns_", "feature_names_in_"])
		if not isinstance(X, pd.DataFrame):
			raise TypeError("PropertyFeatureEngineer requires a pandas DataFrame.")

		required_columns = {
			self.size_column,
			self.bedrooms_column,
			self.bathrooms_column,
			self.floor_column,
			self.total_floors_column,
		}
		missing_required = required_columns.difference(X.columns)
		if missing_required:
			raise ValueError(f"Missing required columns: {', '.join(sorted(missing_required))}")

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


def main():
	parser = ArgumentParser(description="Test the PropertyFeatureEngineer.")
	parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV file.")
	parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
	parser.add_argument("--target-column", default="price", help="Target column retained outside preprocessing.")
	parser.add_argument("--config", type=Path, default="configs/featurizer.yaml", help="Path to the configuration YAML file.")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
	logger.info("Loading featurizer configuration from %s", args.config)
	cfg = OmegaConf.load(args.config)
	fe = instantiate(cfg)
	logger.info("Reading input data from %s", args.input_csv)
	df = pd.read_csv(args.input_csv)
	if args.target_column not in df.columns:
		raise ValueError(f"Target column {args.target_column!r} is missing from {args.input_csv}.")
	y = df.pop(args.target_column)
	logger.info("Fitting featurizer on %d rows and %d input columns", len(df), len(df.columns))
	df = fe.fit_transform(df)
	df[args.target_column] = y
	args.output_dir.mkdir(parents=True, exist_ok=True)
	df.to_csv(args.output_dir / "features.csv", index=False)
	joblib.dump(fe, args.output_dir / "featurizer.joblib")
	logger.info("Wrote %d rows and %d columns to %s", len(df), len(df.columns), args.output_dir)


if __name__ == "__main__":
	main()
