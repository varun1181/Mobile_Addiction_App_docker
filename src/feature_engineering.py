"""Stage 3 - create the ratio features (share of daily use spent on each activity)."""
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("feature_engineering")


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def add_ratio_features(df: pd.DataFrame, ratio_features: dict, eps: float) -> pd.DataFrame:
    df = df.copy()
    for new_col, (num, den) in ratio_features.items():
        df[new_col] = df[num] / (df[den] + eps)
    return df


def main() -> None:
    cfg = load_params()["feature_engineering"]
    data_dir = ROOT / "data" / "processed"

    for split in ("train", "test"):
        df = pd.read_csv(data_dir / f"{split}_preprocessed.csv")
        out = add_ratio_features(df, cfg["ratio_features"], float(cfg["epsilon"]))
        out.to_csv(data_dir / f"{split}_features.csv", index=False)
        log.info("%s: added %s features -> %s columns", split, len(cfg["ratio_features"]), out.shape[1])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("Feature engineering failed")
        sys.exit(1)
