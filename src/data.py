import numpy as np
import pandas as pd

from src.config import RPM_HIGH, RPM_LOW


def load(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def flag_label_outliers(df: pd.DataFrame) -> pd.Series:
    """True for rows whose rate per mile falls outside the normal cluster (corrupted labels)."""
    rpm = df["posted_rate"] / df["distance"]
    return (rpm < RPM_LOW) | (rpm > RPM_HIGH)


def fit_cleaning_stats(train: pd.DataFrame) -> dict:
    """Statistics learned from training data only, reused for every dataset."""
    return {"weight_median": train["weight"].abs().median()}


def clean(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply the same cleaning to train, validation and December inputs."""
    df = df.copy()

    # Negative weights are sign errors -> take absolute value; fill missing with train median
    df["weight"] = df["weight"].abs().fillna(stats["weight_median"])

    # market_index is ~98% day-level -> fill missing with the same-date mean
    if "market_index" in df.columns:
        daily_mean = df.groupby("date")["market_index"].transform("mean")
        df["market_index"] = df["market_index"].fillna(daily_mean)

    # quote_signal flips sign between months and is uninformative in Nov-Dec -> drop
    return df.drop(columns=["quote_signal"], errors="ignore")