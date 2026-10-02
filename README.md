# Freight Rate Prediction — Spotter ML Assessment

Predicts `posted_rate` for 12,000 freight loads in Nov–Dec 2025, using labelled loads from Jan–Oct 2025.

## Results

**Main holdout** (train Jan–Aug, test Sep–Oct):

| Model | MAPE | MAE ($) |
|---|---|---|
| Baseline (median rate/mile by equipment × distance) | 3.60% | 79.0 |
| Plain LightGBM (all features as tree inputs) | 1.88% | 43.5 |
| Hybrid (linear time component + LightGBM) | 1.67% | 37.6 |
| **City-aware hybrid (final)** | **1.60%** | **35.9** |

**Unseen cities:** with 8 cities removed from training (mirroring the 8 new cities in the validation file), the final model scores 1.69% MAPE overall and 1.88% on loads touching unseen cities.

**Rolling backtest** (train on everything before each start, test the next 2 months, MAPE):

| Test period | Plain LightGBM | Hybrid | City-aware hybrid |
|---|---|---|---|
| May–Jun | 2.20% | 1.64% | **1.61%** |
| Jul–Aug | 2.12% | 1.93% | **1.90%** |
| Sep–Oct | 1.88% | 1.67% | **1.60%** |

The final model wins in every period. Jul–Aug is harder because of an abrupt market drop in late July, so a realistic expectation for Nov–Dec is roughly 1.6–1.9% MAPE.

## Approach in brief

- **Cleaning:** 677 corrupted labels (1.4%, rate per mile at ≈0.3x or ≈3.5x normal) dropped from training; negative weights treated as sign errors (`abs`); missing weight filled with the training median; missing `market_index` filled with the same-date mean; `quote_signal` dropped (its relationship with price flips between +1 and −1 across months and is ≈0 in Nov–Dec).
- **Target:** `log(posted_rate)`, since equipment and market effects are multiplicative.
- **Model:** `log(rate) = linear(market_index, time trend, quarter-end ramp) + LightGBM(distance, weight, equipment, coordinates)`, fitted by backfitting. The linear part extrapolates the time trend and quarter-end ramp into Nov–Dec, which tree models cannot do.
- **Unseen cities:** a model with city IDs is used when both cities were seen in training; otherwise a fallback model without city IDs is used.

Full analysis: `notebooks/01_eda.ipynb`. Full write-up: `report.pdf`.

## Setup

Requires Python 3.10+.

```bash
python -m venv venv
# Windows: venv\Scripts\activate   |   macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

Place the assessment data files in `data/`:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

```bash
python -m src.train      # holdout experiments + final model -> outputs/
python -m src.predict    # validation_predictions.csv + fills data/december_chart_inputs.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

python -m src.backtest   # optional: rolling backtest -> outputs/rolling_backtest.csv
```

`src.train` and `src.backtest` take ~2–3 minutes each. Results are deterministic (fixed seed).

## Repository structure

```
data/
src/
  config.py      paths and constants
  data.py        loading, label-outlier detection, cleaning
  features.py    time trend, quarter-end ramp, log distance, target
  model.py       HybridModel (backfitting) and CityAwareModel
  train.py       holdout experiments and final training
  predict.py     validation predictions and December chart inputs
  backtest.py    rolling backtest across three periods
notebooks/
  01_eda.ipynb   exploratory analysis and decisions
outputs/
  validation_metrics.csv
  rolling_backtest.csv
scorer_results/
  candidate_december.png
validation_predictions.csv
score.py
```

## Key assumptions

- The quarter-end price ramp seen in March, June and September repeats in December.
- The linear time trend continues for two months beyond the training data.
- No holiday adjustment is made for Christmas / New Year, as no holiday effects were visible in the training data. Real December markets typically soften around Christmas, so the Dec 25 prediction is likely too high.
