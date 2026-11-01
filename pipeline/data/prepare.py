import re
from argparse import ArgumentParser
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
from hydra.utils import instantiate


def normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        raise ValueError(f"Invalid empty column name derived from {value!r}")
    return normalized


def load_data(path: Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.concat(
        [
            pd.read_csv(f) for f in path.glob('*.csv')
        ], 
        ignore_index=True
    )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare the data for modeling."""
    # Normalize column names
    trf = instantiate(OmegaConf.load("configs/preprocess.yaml"), _convert_="all")
    trf.set_output(transform='pandas')
    df = trf.fit_transform(df)
    df.columns = [normalize_name(c) for c in df.columns]
    return df, trf


def main():
    parser = ArgumentParser(description="Prepare data for modeling.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input directory containing CSV files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory to save prepared data.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--target-column", default="price", help="Target column retained outside preprocessing.")
    args = parser.parse_args()

    # Load and prepare data
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory.")

    df = load_data(input_dir)
    if args.target_column not in df.columns:
        raise ValueError(f"Target column {args.target_column!r} not found in input data.")

    df_prepared, tf = prepare_data(df)
    # Split into train and test sets
    train_df, test_df = train_test_split(df_prepared, test_size=args.test_size, random_state=args.random_state)

    # Save prepared data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    joblib.dump(tf, output_dir / "preprocessor.joblib")


if __name__ == "__main__":
    main()
