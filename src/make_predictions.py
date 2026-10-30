"""Rebuild validation_predictions.csv and the December chart predictions from the raw data.

Usage (from the repo root):
    python src/make_predictions.py --data-dir data --out-dir .
"""
import argparse
from pathlib import Path

import numpy as np
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
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load ----
    labeled = pd.read_csv(find(data_dir, "train-test.csv"), parse_dates=["date"])
    labeled = labeled.sort_values("date").reset_index(drop=True)
    val = pd.read_csv(find(data_dir, "validation.csv"), parse_dates=["date"])
    template = pd.read_csv(find(data_dir, "validation-predictions-template.csv"))
    dec = pd.read_csv(find(data_dir, "december-chart-inputs.csv"))

    # ---- final fit on all labeled data ----
    model = build_final_model().fit(make_features(labeled), labeled["posted_rate"])
    print(f"Final model fit on {len(labeled):,} labeled loads")

    # ---- validation predictions ----
    pred_val = model.predict(make_features(val))
    submission = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": pred_val.round(2)})
    submission = template[["load_id"]].merge(submission, on="load_id", how="left")
    assert len(submission) == 12_000
    assert submission["predicted_rate"].notna().all() and (submission["predicted_rate"] > 0).all()
    submission.to_csv(out_dir / "validation_predictions.csv", index=False)
    print("Saved validation_predictions.csv")

    # ---- December chart predictions ----
    # Inputs lack market_index and quote_signal, so use the per-day mean from validation.csv.
    dec_feat = dec.drop(columns="predicted_rate").copy()
    dec_feat["date"] = pd.to_datetime(dec_feat["date"])
    pick = labeled.drop_duplicates("pickup").set_index("pickup")[["pickup_lat", "pickup_lon"]]
    deli = labeled.drop_duplicates("delivery").set_index("delivery")[["delivery_lat", "delivery_lon"]]
    signals = val.groupby("date")[["market_index", "quote_signal"]].mean()
    dec_feat = dec_feat.join(pick, on="pickup").join(deli, on="delivery").join(signals, on="date")
    assert dec_feat.isna().sum().sum() == 0

    dec["predicted_rate"] = model.predict(make_features(dec_feat)).round(2)
    dec.to_csv(out_dir / "december-chart-predictions.csv", index=False)
    print("Saved december-chart-predictions.csv")


if __name__ == "__main__":
    main()