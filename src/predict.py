"""Generate validation predictions and fill the December chart inputs using the trained model."""
import joblib
import pandas as pd

from src.config import DECEMBER_PATH, OUTPUT_DIR, PREDICTIONS_PATH, TEMPLATE_PATH, TRAIN_PATH, VAL_PATH
from src.data import clean, load
from src.features import add_features

COORD_COLS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]


def city_coordinates(*frames) -> pd.DataFrame:
    """One (lat, lon) per city, collected from every file that has coordinates."""
    parts = []
    for df in frames:
        for side in ["pickup", "delivery"]:
            parts.append(df[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def predict_validation(model, stats) -> pd.DataFrame:
    val = add_features(clean(load(VAL_PATH), stats))
    val["predicted_rate"] = model.predict(val).round(2)

    template = pd.read_csv(TEMPLATE_PATH)
    out = template[["load_id"]].merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert len(out) == len(template) and out["predicted_rate"].notna().all(), "missing predictions"

    out.to_csv(PREDICTIONS_PATH, index=False)
    known = model.uses_city_model(val).mean()
    print(f"Saved {len(out):,} predictions to {PREDICTIONS_PATH.name} "
          f"({known:.1%} via city model, {1 - known:.1%} via fallback)")
    return val


def predict_december(model, stats, val: pd.DataFrame):
    """December inputs lack coordinates and market_index: fill from city lookup and validation same-date mean."""
    dec_raw = pd.read_csv(DECEMBER_PATH)
    original_columns = list(dec_raw.columns)
    dec = dec_raw.copy()
    dec["date"] = pd.to_datetime(dec["date"])

    coords = city_coordinates(load(TRAIN_PATH), load(VAL_PATH))
    for side in ["pickup", "delivery"]:
        dec[f"{side}_lat"] = dec[side].map(coords["lat"])
        dec[f"{side}_lon"] = dec[side].map(coords["lon"])

    daily_mi = val.groupby("date")["market_index"].mean()
    dec["market_index"] = dec["date"].map(daily_mi)
    assert dec[COORD_COLS + ["market_index"]].notna().all().all(), "could not fill December inputs"

    dec = add_features(clean(dec, stats))
    dec_raw["predicted_rate"] = model.predict(dec).round(2)
    dec_raw[original_columns].to_csv(DECEMBER_PATH, index=False)

    print(f"Filled {len(dec_raw)} December rows in {DECEMBER_PATH.name}: "
          f"min ${dec_raw['predicted_rate'].min():,.2f}, max ${dec_raw['predicted_rate'].max():,.2f}")


def main():
    bundle = joblib.load(OUTPUT_DIR / "model.joblib")
    model, stats = bundle["model"], bundle["cleaning_stats"]
    val = predict_validation(model, stats)
    predict_december(model, stats, val)


if __name__ == "__main__":
    main()