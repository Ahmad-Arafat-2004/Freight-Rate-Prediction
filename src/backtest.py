"""Rolling backtest: repeat the 2-months-ahead test from several start points to check the model ranking is stable."""
import pandas as pd

from src.config import OUTPUT_DIR
from src.train import CANDIDATES, metrics, prepare_training_data

BACKTEST_STARTS = ["2025-05-01", "2025-07-01", "2025-09-01"]
MODELS = ["Plain LightGBM", "Hybrid", "City-aware hybrid"]


def rolling_backtest(df: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for start in BACKTEST_STARTS:
        end = pd.Timestamp(start) + pd.DateOffset(months=2)
        tr = df[df["date"] < start]
        te = df[(df["date"] >= start) & (df["date"] < end)]
        label = f"{pd.Timestamp(start):%b}-{(end - pd.Timedelta(days=1)):%b}"
        print(f"Testing {label}: train {len(tr):,} loads -> test {len(te):,} loads")
        for name in MODELS:
            rows[(label, name)] = metrics(te["posted_rate"].to_numpy(), CANDIDATES[name](tr, te))
    return pd.DataFrame(rows).T


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df, _ = prepare_training_data()
    results = rolling_backtest(df)
    print("\n=== Rolling backtest (MAPE %) ===")
    table = results["MAPE_%"].unstack()
    print(table.loc[results.index.get_level_values(0).unique(), MODELS].round(3))
    results.round(4).to_csv(OUTPUT_DIR / "rolling_backtest.csv")
    print(f"\nSaved to {OUTPUT_DIR / 'rolling_backtest.csv'}")


if __name__ == "__main__":
    main()