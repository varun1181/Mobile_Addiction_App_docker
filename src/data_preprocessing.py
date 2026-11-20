"""Stage 2 - clean the data: drop identifier columns, fix the column typo, tidy text."""
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("data_preprocessing")


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def preprocess(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    cfg = params["data_preprocessing"]
    target = params["base"]["target"]

    df = df.drop(columns=cfg["drop_columns"], errors="ignore")
    df = df.rename(columns=cfg["rename_columns"])

    # remove stray spaces in text columns (e.g. "Male " vs "Male")
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].str.strip()

    df = df.dropna(subset=[target]).reset_index(drop=True)
    return df


def main() -> None:
    params = load_params()
    in_dir = ROOT / "data" / "ingested"
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        df = pd.read_csv(in_dir / f"{split}.csv")
        clean = preprocess(df, params)
        clean.to_csv(out_dir / f"{split}_preprocessed.csv", index=False)
        log.info("%s: %s -> %s columns, %s rows", split, df.shape[1], clean.shape[1], len(clean))
        missing = int(clean.isnull().sum().sum())
        if missing:
            log.info("%s: %s missing values left (imputed later inside the model pipeline)", split, missing)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Data preprocessing failed")
        sys.exit(1)
