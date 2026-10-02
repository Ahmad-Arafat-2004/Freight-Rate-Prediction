from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "train_test.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

PREDICTIONS_PATH = ROOT / "validation_predictions.csv"

# Label-outlier thresholds (rate per mile), chosen from the gaps in the EDA histogram
RPM_LOW = 1.4
RPM_HIGH = 3.8

SEED = 42