import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_loads():
    """Small synthetic dataset with the same schema as the real data."""
    rng = np.random.default_rng(0)
    n = 600
    equipment = rng.choice(["Dry Van", "Flatbed", "Reefer"], size=n)
    rpm = pd.Series(equipment).map({"Dry Van": 2.05, "Flatbed": 2.2, "Reefer": 2.3}).to_numpy()
    distance = rng.uniform(70, 3000, n).round(1)
    df = pd.DataFrame({
        "pickup_lat": rng.uniform(28, 44, n),
        "pickup_lon": rng.uniform(-120, -70, n),
        "delivery_lat": rng.uniform(28, 44, n),
        "delivery_lon": rng.uniform(-120, -70, n),
        "distance": distance,
        "equipment": equipment,
        "weight": rng.uniform(10_000, 47_500, n).round(0),
        "date": pd.date_range("2025-01-01", periods=n, freq="6h").strftime("%Y-%m-%d"),
        "market_index": rng.uniform(0.8, 1.4, n),
        "quote_signal": rng.normal(2.0, 0.25, n),
    })
    df["posted_rate"] = (distance * rpm * rng.normal(1.0, 0.03, n)).round(2)

    # inject the same data-quality problems as the real data
    df.loc[:9, "weight"] = -df.loc[:9, "weight"]   # 10 negative weights
    df.loc[10:19, "weight"] = np.nan               # 10 missing weights
    df.loc[20:29, "market_index"] = np.nan         # 10 missing market_index
    return df