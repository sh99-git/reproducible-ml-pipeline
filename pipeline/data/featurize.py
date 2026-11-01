from argparse import ArgumentParser
from pathlib import Path

import joblib
import pandas as pd
from pipeline.data.features import PropertyFeatureEngineer


def featurize_frame(frame: pd.DataFrame, target_column: str = "price") -> pd.DataFrame:
    """Add property features while keeping the target separate from transformer inputs."""
    if target_column not in frame.columns:
        raise ValueError(f"Missing target column: {target_column}")

    X = frame.drop(columns=[target_column])
    transformer = PropertyFeatureEngineer(target_column=target_column)
    featured = transformer.fit_transform(X)
    featured[target_column] = frame[target_column]
    return featured


def featurize_directory(
    input_dir: Path,
    output_dir: Path,
    target_column: str = "price",
    size_column: str = "size_sqft",
    bedrooms_column: str = "num_bedrooms",
    bathrooms_column: str = "num_bathrooms",
    floor_column: str = "floor_number",
    total_floors_column: str = "total_floors",
    amenity_prefix: str = "amenities_",
) -> None:
    """Transform train.csv and test.csv from a prepared-data directory."""

    input_path = Path(input_dir) / "train.csv"

    if not input_path.is_file():
        raise FileNotFoundError(f"Expected prepared split at {input_path}")

    train_frame = pd.read_csv(input_path)
    if target_column not in train_frame:
        raise ValueError(f"Target column '{target_column}' not found in train data.")

    transformer = PropertyFeatureEngineer(
        target_column=target_column,
        size_column=size_column,
        bedrooms_column=bedrooms_column,
        bathrooms_column=bathrooms_column,
        floor_column=floor_column,
        total_floors_column=total_floors_column,
        amenity_prefix=amenity_prefix,
    )
    featured = transformer.fit_transform(train_frame.drop(columns=[target_column]))
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(transformer, output_dir / "featurizer.joblib")
    featured[target_column] = train_frame[target_column]
    featured.to_csv(output_dir / "features.csv", index=False)


def main() -> None:
    parser = ArgumentParser(description="Add property features to prepared train/test CSVs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-column", default="price")
    parser.add_argument("--size-column", default="size_sqft")
    parser.add_argument("--bedrooms-column", default="num_bedrooms")
    parser.add_argument("--bathrooms-column", default="num_bathrooms")
    parser.add_argument("--floor-column", default="floor_number")
    parser.add_argument("--total-floors-column", default="total_floors")
    parser.add_argument("--amenity-prefix", default="amenities_")
    args = parser.parse_args()
    featurize_directory(
        args.input_dir,
        args.output_dir,
        args.target_column,
        args.size_column,
        args.bedrooms_column,
        args.bathrooms_column,
        args.floor_column,
        args.total_floors_column,
        args.amenity_prefix,
    )


if __name__ == "__main__":
    main()
