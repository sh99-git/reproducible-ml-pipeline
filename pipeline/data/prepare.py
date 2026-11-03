import re
import logging
from argparse import ArgumentParser
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
from hydra.utils import instantiate

logger = logging.getLogger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    logger.info("Loading CSV files from %s", path)
    return pd.concat(
        [
            pd.read_csv(f) for f in path.glob('*.csv')
        ], 
        ignore_index=True
    )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = ArgumentParser(description="Prepare data for modeling.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory containing CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory to save prepared data.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--target-column", default="price", help="Target column retained outside preprocessing.")
    parser.add_argument("--config", type=Path, default="configs/preprocess.yaml", help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load and prepare data
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory.")

    if not args.config.is_file():
        raise FileNotFoundError(f"Configuration file {args.config} not found.")

    df = load_data(input_dir)
    logger.info("Loaded %d rows and %d columns", *df.shape)

    if args.target_column not in df.columns:
        raise ValueError(f"Target column {args.target_column!r} not found in input data.")

    logger.info("Splitting data with test_size=%s and random_state=%s", args.test_size, args.random_state)
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.random_state)
    y = train_df.pop(args.target_column)

    logger.info("Loading preprocessing configuration from %s", args.config)
    cfg = OmegaConf.load(args.config)
    pre = instantiate(cfg, _convert_="all")
    pre.set_output(transform='pandas')
    logger.info("Fitting preprocessor and transforming training data")
    train_df = pre.fit_transform(train_df)
    train_df[args.target_column] = y
    
    # Save prepared data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving prepared data to %s", output_dir)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    joblib.dump(pre, output_dir / "preprocessor.joblib")
    logger.info("Saved train.csv, test.csv, and preprocessor.joblib")


if __name__ == "__main__":
    main()
