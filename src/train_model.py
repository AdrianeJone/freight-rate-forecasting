"""Fit the final model on all labeled data and save it for the API.

Usage (from the repo root):
    python src/train_model.py --data-dir data --model-path models/freight_model.joblib
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from pipeline import build_final_model, make_features


def find(data_dir, name):
    """Accept both hyphen and underscore file names."""
    for n in (name, name.replace("_", "-"), name.replace("-", "_")):
        if (data_dir / n).is_file():
            return data_dir / n
    raise FileNotFoundError(f"{name} not found in {data_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-path", default="models/freight_model.joblib")
    args = parser.parse_args()

    labeled = pd.read_csv(find(Path(args.data_dir), "train-test.csv"), parse_dates=["date"])
    labeled = labeled.sort_values("date").reset_index(drop=True)  # same order as the notebook

    model = build_final_model().fit(make_features(labeled), labeled["posted_rate"])

    path = Path(args.model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Fit on {len(labeled):,} loads. Saved model to {path}")


if __name__ == "__main__":
    main()