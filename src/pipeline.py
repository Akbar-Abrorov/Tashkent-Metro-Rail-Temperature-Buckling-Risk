"""
End-to-end pipeline: real Tashkent weather -> physical baseline -> RF/ANN
(trained on physical-baseline-as-proxy target) -> plots + results CSV.

Run:
    python3 src/pipeline.py

Requires data/raw/tashkent_weather_<start>_<end>.csv to already exist
(produced by src/fetch_weather.py --start ... --end ...). Run that first
if the file is missing.

See README.md for what is real data vs. proxy/placeholder in this pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from physical_model import add_physical_baseline
from plots import plot_model_comparison, plot_weather_overview
from train_models import FEATURE_COLUMNS, PROXY_TARGET_COLUMN, train_and_evaluate_proxy_target

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"


def find_weather_csv() -> Path:
    candidates = sorted(RAW_DATA_DIR.glob("tashkent_weather_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No weather CSV found in {RAW_DATA_DIR}. Run "
            "`python3 src/fetch_weather.py` first."
        )
    return candidates[-1]


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    weather_csv = find_weather_csv()
    print(f"Loading real weather data from: {weather_csv}")
    df = pd.read_csv(weather_csv, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} real hourly rows, {df.index.min()} to {df.index.max()}")

    print("\nComputing physical baseline rail temperature (heat-balance model)...")
    df = add_physical_baseline(df)
    print(df["rail_temp_physical_c"].describe().round(2).to_string())

    processed_path = PROCESSED_DATA_DIR / "tashkent_weather_with_physical_baseline.csv"
    df.to_csv(processed_path)
    print(f"Saved processed dataset to: {processed_path}")

    print("\n" + "=" * 70)
    print("Training RF + ANN against PROXY target (physical baseline output).")
    print("These models are NOT validated against real measured rail")
    print("temperature -- see README.md.")
    print("=" * 70)
    trained = train_and_evaluate_proxy_target(df, feature_columns=FEATURE_COLUMNS)

    print(f"\nTrain rows: {trained.test_metrics['n_train']}, "
          f"Test rows: {trained.test_metrics['n_test']} (chronological split)")
    print("\nTest-set metrics (model vs. physical-baseline PROXY target, "
          "NOT real-world accuracy):")
    for model_name, m in trained.test_metrics.items():
        if isinstance(m, dict):
            print(f"  {model_name:15s} RMSE={m['rmse']:.3f} C  MAE={m['mae']:.3f} C  R2={m['r2']:.4f}")

    results_path = OUTPUTS_DIR / "results.csv"
    trained.predictions.to_csv(results_path)
    print(f"\nSaved predictions to: {results_path}")

    metrics_rows = []
    for model_name, m in trained.test_metrics.items():
        if isinstance(m, dict):
            metrics_rows.append({"model": model_name, "label": trained.label, **m})
    metrics_path = OUTPUTS_DIR / "test_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"Saved test metrics to: {metrics_path}")

    print("\nGenerating plots...")
    plot_weather_overview(df, PLOTS_DIR / "tashkent_summer_2023_weather.png")
    plot_model_comparison(
        trained.predictions, target_column=PROXY_TARGET_COLUMN,
        label="physical baseline (PROXY, not real measurements)",
        out_path=PLOTS_DIR / "model_comparison_proxy_target.png",
    )
    print(f"Saved plots to: {PLOTS_DIR}")

    print("\nDone. Reminder: no step in this pipeline used real measured rail")
    print("temperature. All rail-temperature numbers above are either the")
    print("physical baseline's own output or ML models trained to imitate it.")


if __name__ == "__main__":
    main()
