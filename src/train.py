"""Validate candidate models on time-based holdouts, then train the final model on all data."""
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, SEED, TRAIN_PATH
from src.data import clean, fit_cleaning_stats, flag_label_outliers, load
from src.features import (CATEGORICAL, CITY_FEATURES, STATIC_FEATURES, TIME_FEATURES,
                          add_features, add_target)
from src.model import LGB_PARAMS, CityAwareModel, HybridModel

HOLDOUT_START = "2025-09-01"  # train Jan-Aug, evaluate Sep-Oct (two months ahead, includes a quarter end)
N_HELDOUT_CITIES = 8          # mirrors the 8 unseen cities in the validation file


def metrics(y_true, y_pred) -> dict:
    err = y_pred - y_true
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE_%": np.mean(np.abs(err) / y_true) * 100,
    }


def prepare_training_data():
    raw = load(TRAIN_PATH)
    outliers = flag_label_outliers(raw)
    print(f"Dropping {outliers.sum()} corrupted labels ({outliers.mean():.1%})")
    raw = raw[~outliers]
    stats = fit_cleaning_stats(raw)
    return add_target(add_features(clean(raw, stats))), stats


# ---------- candidate models: each takes (train, test) and returns predicted rates ----------

def baseline_rpm_table(tr, te):
    """Median rate per mile by equipment and distance band, times distance."""
    bins = [0, 150, 300, 500, 700, 1000, 1500, 2000, 2500, 3500]
    band = lambda d: pd.cut(d["distance"], bins)
    rpm = (tr["posted_rate"] / tr["distance"]).groupby([tr["equipment"], band(tr)], observed=True).median()
    keys = pd.MultiIndex.from_arrays([te["equipment"], band(te)])
    return rpm.reindex(keys).to_numpy() * te["distance"].to_numpy()


def plain_lightgbm(tr, te):
    """Single LightGBM with every feature, including the time features, as tree inputs."""
    cols = STATIC_FEATURES + TIME_FEATURES
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    for c in [c for c in CATEGORICAL if c in cols]:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)
    return np.exp(lgb.LGBMRegressor(**LGB_PARAMS).fit(Xtr, tr["y"]).predict(Xte))


def fit_predict(model):
    return lambda tr, te: model().fit(tr, tr["y"]).predict(te)


CANDIDATES = {
    "Baseline: rpm table": baseline_rpm_table,
    "Plain LightGBM": plain_lightgbm,
    "Hybrid": fit_predict(lambda: HybridModel(STATIC_FEATURES)),
    "Hybrid + city IDs": fit_predict(lambda: HybridModel(STATIC_FEATURES + CITY_FEATURES)),
    "City-aware hybrid": fit_predict(lambda: CityAwareModel(STATIC_FEATURES, CITY_FEATURES)),
}


def evaluate(tr, te, groups=None) -> pd.DataFrame:
    """Score every candidate on the test set, optionally broken down by row groups."""
    y = te["posted_rate"].to_numpy()
    rows = {}
    for name, fn in CANDIDATES.items():
        pred = fn(tr, te)
        rows[(name, "all")] = metrics(y, pred)
        for label, mask in (groups or {}).items():
            rows[(name, label)] = metrics(y[mask], pred[mask])
    return pd.DataFrame(rows).T


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df, stats = prepare_training_data()
    is_test = df["date"] >= HOLDOUT_START

    # 1) Time holdout: does the model extrapolate two months ahead?
    print(f"\n=== Time holdout: train Jan-Aug ({(~is_test).sum():,}) -> test Sep-Oct ({is_test.sum():,}) ===")
    time_results = evaluate(df[~is_test], df[is_test])
    print(time_results.round(3))

    # 2) Time + unseen-city holdout: also remove 8 cities from training, like the validation file
    cities = sorted(set(df["pickup"]) | set(df["delivery"]))
    held = np.random.default_rng(SEED).choice(cities, N_HELDOUT_CITIES, replace=False)
    touches_held = df["pickup"].isin(held) | df["delivery"].isin(held)
    tr, te = df[~is_test & ~touches_held], df[is_test]
    unseen = touches_held[is_test].to_numpy()
    print(f"\n=== Time + unseen-city holdout: {N_HELDOUT_CITIES} cities removed from training "
          f"({unseen.mean():.0%} of test loads touch them) ===")
    city_results = evaluate(tr, te, groups={"seen cities": ~unseen, "unseen cities": unseen})
    print(city_results.round(3))

    pd.concat({"time_holdout": time_results, "unseen_city_holdout": city_results}) \
      .round(4).to_csv(OUTPUT_DIR / "validation_metrics.csv")

    # 3) Final model on all clean data
    print("\n=== Training final City-aware hybrid on all clean data (Jan-Oct) ===")
    final = CityAwareModel(STATIC_FEATURES, CITY_FEATURES).fit(df, df["y"])
    print("Time coefficients:", {k: round(float(v), 5) for k, v in final.time_coefficients().items()})
    joblib.dump({"model": final, "cleaning_stats": stats}, OUTPUT_DIR / "model.joblib")
    print(f"Saved model to {OUTPUT_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()