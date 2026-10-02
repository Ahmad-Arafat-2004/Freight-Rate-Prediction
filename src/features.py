import numpy as np
import pandas as pd

# Reference date for the linear time trend
T0 = pd.Timestamp("2025-01-01")
QUARTER_END_MONTHS = [3, 6, 9, 12]

# Linear part: components that must extrapolate into Nov-Dec
TIME_FEATURES = ["market_index", "t", "qe_day"]

# Tree part: static load characteristics
STATIC_FEATURES = [
    "distance", "log_distance", "weight", "equipment",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]
CITY_FEATURES = ["pickup", "delivery"]
CATEGORICAL = ["equipment", "pickup", "delivery"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Linear time trend (days since start of training data)
    df["t"] = (df["date"] - T0).dt.days

    # Quarter-end ramp: day of month in Mar/Jun/Sep/Dec, 0 otherwise
    is_qe = df["date"].dt.month.isin(QUARTER_END_MONTHS)
    df["qe_day"] = np.where(is_qe, df["date"].dt.day, 0)

    # Rate per mile falls with distance (fixed costs) -> log distance helps
    df["log_distance"] = np.log(df["distance"])

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Log target: equipment / market effects are multiplicative."""
    df = df.copy()
    df["y"] = np.log(df["posted_rate"])
    return df