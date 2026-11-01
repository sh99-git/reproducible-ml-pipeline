import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PropertyFeatureEngineer(TransformerMixin, BaseEstimator):
	"""Create property features from a pandas feature matrix."""

	def __init__(
		self,
		target_column: str = "price",
		size_column: str = "size_sqft",
		bedrooms_column: str = "num_bedrooms",
		bathrooms_column: str = "num_bathrooms",
		floor_column: str = "floor_number",
		total_floors_column: str = "total_floors",
		amenity_prefix: str = "amenities_",
	):
		self.target_column = target_column
		self.size_column = size_column
		self.bedrooms_column = bedrooms_column
		self.bathrooms_column = bathrooms_column
		self.floor_column = floor_column
		self.total_floors_column = total_floors_column
		self.amenity_prefix = amenity_prefix

	def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PropertyFeatureEngineer":
		if not isinstance(X, pd.DataFrame):
			raise TypeError("PropertyFeatureEngineer requires a pandas DataFrame.")
		if self.target_column in X.columns:
			raise ValueError(f"'{self.target_column}' is the target and must not be included in X.")

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
		if self.target_column in X.columns:
			raise ValueError(f"'{self.target_column}' is the target and must not be included in X.")

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