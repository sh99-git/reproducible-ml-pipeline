"""Split raw CSV data and fit the configured preprocessor."""

import logging
from argparse import ArgumentParser
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
from hydra.utils import instantiate
from pipeline.data.utils import load_csv_directory, split_target

logger = logging.getLogger(__name__)


def build_parser() -> ArgumentParser:
    """Define command-line options for data preparation."""
    parser = ArgumentParser(description="Prepare data for modeling.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for prepared data and transformer.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of rows reserved for testing.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the train/test split.")
    parser.add_argument("--target-column", default="price", help="Target column retained outside preprocessing.")
    parser.add_argument("--config", type=Path, default="configs/preprocess.yaml", help="Preprocessor configuration YAML.")
    return parser


def split_data(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible training and untouched holdout frames."""
    return train_test_split(data, test_size=test_size, random_state=random_state)


def load_preprocessor(config_path: Path):
    """Instantiate the preprocessing transformer from its Hydra config."""
    preprocessor = instantiate(OmegaConf.load(config_path), _convert_="all")
    preprocessor.set_output(transform="pandas")
    return preprocessor


def transform_training_data(
    train_data: pd.DataFrame,
    target_column: str,
    preprocessor,
) -> pd.DataFrame:
    """Fit preprocessing on training features and reattach the target."""
    features, target = split_target(train_data, target_column)
    transformed = preprocessor.fit_transform(features)
    transformed[target_column] = target
    return transformed


def save_prepared_data(
    output_dir: Path,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    preprocessor,
) -> None:
    """Write prepared CSVs and the fitted preprocessing artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(output_dir / "train.csv", index=False)
    test_data.to_csv(output_dir / "test.csv", index=False)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")


def main():
    """Run the preparation stage and persist its data and fitted transformer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(f"Configuration file {args.config} not found.")

    df = load_csv_directory(args.input_dir)
    logger.info("Loaded %d rows and %d columns", *df.shape)
    logger.info("Splitting data with test_size=%s and random_state=%s", args.test_size, args.random_state)
    train_df, test_df = split_data(df, args.test_size, args.random_state)
    logger.info("Loading preprocessing configuration from %s", args.config)
    preprocessor = load_preprocessor(args.config)
    logger.info("Fitting preprocessor and transforming training data")
    train_df = transform_training_data(train_df, args.target_column, preprocessor)
    save_prepared_data(args.output_dir, train_df, test_df, preprocessor)
    logger.info("Saved train.csv, test.csv, and preprocessor.joblib to %s", args.output_dir)


if __name__ == "__main__":
    main()
