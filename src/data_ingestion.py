"""Stage 1 - load the raw CSV, drop exact duplicates and split into train/test."""
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("data_ingestion")


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    params = load_params()
    cfg = params["data_ingestion"]
    raw_path = ROOT / cfg["raw_path"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw_path}. Copy your CSV there (see README)."
        )

    df = pd.read_csv(raw_path)
    log.info("Loaded %s rows x %s columns from %s", *df.shape, raw_path.name)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("Removed %s duplicate rows", before - len(df))

    train, test = train_test_split(
        df, test_size=cfg["test_size"], random_state=params["base"]["random_state"]
    )

    out_dir = ROOT / "data" / "ingested"
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    log.info("Saved train (%s rows) and test (%s rows) to %s", len(train), len(test), out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Data ingestion failed")
        sys.exit(1)
