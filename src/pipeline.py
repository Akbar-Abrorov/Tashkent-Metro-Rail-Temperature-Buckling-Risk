"""
End-to-end pipeline: real Tashkent weather -> physical models -> RF/ANN
(trained on the real RailTemp/CNU model's output as a proxy target) -> plots
+ results CSV.

Run:
    python3 src/pipeline.py              # loads cached models if available
    python3 src/pipeline.py --retrain    # forces retraining, overwrites cache

Requires:
  - data/raw/tashkent_weather_<start>_<end>.csv (produced by
    src/fetch_weather.py --start ... --end ...).
  - data/processed/railtemp_cnu_baseline_<start>_<end>.csv (produced by
    tools/railtemp_cnu/run_cnu.py, run with the separate `.venv-railtemp`
    interpreter -- see README.md "Physical models" section). This is Model A,
    the real published RailTemp/CNU transient model, and is what RF/ANN are
    actually trained against.

See README.md for what is real data vs. proxy/placeholder in this pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baselines import compute_baselines
from diagnostics import run_all_diagnostics
from physical_model import add_physical_baseline
from plots import plot_model_comparison, plot_physical_model_agreement, plot_weather_overview
from train_models import (
    FEATURE_COLUMNS,
    PROXY_TARGET_COLUMN,
    STEADYSTATE_BASELINE_COLUMN,
    evaluate,
    evaluate_cached_models,
    load_cached_models,
    save_models,
    train_and_evaluate_proxy_target,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"

STATION_TZ = "Asia/Tashkent"


def find_weather_csv() -> Path:
    candidates = sorted(RAW_DATA_DIR.glob("tashkent_weather_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No weather CSV found in {RAW_DATA_DIR}. Run "
            "`python3 src/fetch_weather.py` first."
        )
    return candidates[-1]


def find_cnu_csv(weather_csv: Path) -> Path:
    suffix = weather_csv.stem.removeprefix("tashkent_weather_")
    cnu_csv = PROCESSED_DATA_DIR / f"railtemp_cnu_baseline_{suffix}.csv"
    if not cnu_csv.exists():
        raise FileNotFoundError(
            f"Missing {cnu_csv}. Generate it first with the real RailTemp/CNU "
            f"model (requires the separate `.venv-railtemp` interpreter -- see "
            f"README.md):\n"
            f"  .venv-railtemp/bin/python tools/railtemp_cnu/run_cnu.py {weather_csv}"
        )
    return cnu_csv


def load_cnu_baseline(cnu_csv: Path) -> pd.Series:
    cnu_df = pd.read_csv(cnu_csv, index_col=0, parse_dates=True)
    # run_cnu.py saves naive local timestamps (see its tz_localize comment);
    # localize back to Asia/Tashkent so the index matches the main dataframe.
    cnu_df.index = cnu_df.index.tz_localize(STATION_TZ)
    return cnu_df[PROXY_TARGET_COLUMN]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true",
                         help="Force retraining RF/ANN even if a matching cached model exists.")
    args = parser.parse_args()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    weather_csv = find_weather_csv()
    print(f"Loading real weather data from: {weather_csv}")
    df = pd.read_csv(weather_csv, index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} real hourly rows, {df.index.min()} to {df.index.max()}")

    print("\nComputing Model B: steady-state heat-balance baseline (secondary, fast)...")
    df = add_physical_baseline(df)
    print(df[STEADYSTATE_BASELINE_COLUMN].describe().round(2).to_string())

    print("\nLoading Model A: real RailTemp/CNU transient model output (primary, ML target)...")
    cnu_csv = find_cnu_csv(weather_csv)
    df[PROXY_TARGET_COLUMN] = load_cnu_baseline(cnu_csv)
    print(f"Loaded from: {cnu_csv}")
    print(df[PROXY_TARGET_COLUMN].describe().round(2).to_string())

    processed_path = PROCESSED_DATA_DIR / "tashkent_weather_with_physical_baseline.csv"
    df.to_csv(processed_path)
    print(f"Saved processed dataset to: {processed_path}")

    print("\n" + "=" * 70)
    print("Physical model agreement: Model A (RailTemp/CNU) vs. Model B (steady-state)")
    print("=" * 70)
    overlap = df.dropna(subset=[PROXY_TARGET_COLUMN, STEADYSTATE_BASELINE_COLUMN])
    agreement = evaluate(overlap[PROXY_TARGET_COLUMN].to_numpy(), overlap[STEADYSTATE_BASELINE_COLUMN].to_numpy())
    agreement["bias_c"] = float((overlap[STEADYSTATE_BASELINE_COLUMN] - overlap[PROXY_TARGET_COLUMN]).mean())
    agreement["n_overlap_rows"] = len(overlap)
    print(f"  RMSE={agreement['rmse']:.3f} C  MAE={agreement['mae']:.3f} C  "
          f"R2={agreement['r2']:.4f}  bias(B-A)={agreement['bias_c']:.3f} C  "
          f"n={agreement['n_overlap_rows']}")
    pd.DataFrame([agreement]).to_csv(OUTPUTS_DIR / "physical_model_agreement.csv", index=False)
    plot_physical_model_agreement(
        df, cnu_column=PROXY_TARGET_COLUMN, steadystate_column=STEADYSTATE_BASELINE_COLUMN,
        out_path=PLOTS_DIR / "physical_model_agreement.png",
    )

    print("\n" + "=" * 70)
    print("Training RF + ANN against PROXY target (real RailTemp/CNU model output).")
    print("These models are NOT validated against real measured rail")
    print("temperature -- see README.md.")
    print("=" * 70)

    cached = None if args.retrain else load_cached_models(MODELS_DIR, FEATURE_COLUMNS, PROXY_TARGET_COLUMN)
    if cached is not None:
        rf_model, ann_model, ann_x_scaler, ann_y_scaler, label = cached
        print(f"Loaded cached models from {MODELS_DIR} (label: {label}). "
              f"Pass --retrain to force retraining.")
        trained = evaluate_cached_models(
            df, FEATURE_COLUMNS, PROXY_TARGET_COLUMN,
            rf_model, ann_model, ann_x_scaler, ann_y_scaler, label,
        )
    else:
        print("No matching cached models found (or --retrain passed): training now...")
        trained = train_and_evaluate_proxy_target(df, feature_columns=FEATURE_COLUMNS)
        save_models(trained, MODELS_DIR)
        print(f"Saved trained models to: {MODELS_DIR}")

    print(f"\nTrain rows: {trained.test_metrics['n_train']}, "
          f"Val rows: {trained.test_metrics['n_val']} (ANN early stopping only), "
          f"Test rows: {trained.test_metrics['n_test']} (chronological, held out)")
    print("\nTest-set metrics -- the ONLY numbers reported as accuracy anywhere in "
          "this project (model vs. RailTemp/CNU PROXY target, NOT real-world accuracy):")
    for model_name, m in trained.test_metrics.items():
        if isinstance(m, dict):
            print(f"  {model_name:15s} RMSE={m['rmse']:.3f} C  MAE={m['mae']:.3f} C  R2={m['r2']:.4f}")

    results_path = OUTPUTS_DIR / "results.csv"
    trained.predictions.to_csv(results_path)
    print(f"\nSaved predictions to: {results_path}")

    print("\nNaive baselines (same held-out test split, for comparison):")
    baseline_metrics = compute_baselines(df, FEATURE_COLUMNS, PROXY_TARGET_COLUMN)
    for name, m in baseline_metrics.items():
        print(f"  {name:15s} RMSE={m['rmse']:.3f} C  MAE={m['mae']:.3f} C  R2={m['r2']:.4f}")

    metrics_rows = []
    for model_name, m in trained.test_metrics.items():
        if isinstance(m, dict):
            metrics_rows.append({"model": model_name, "label": trained.label, "rmse": m["rmse"],
                                  "mae": m["mae"], "r2": m["r2"]})
    for name, m in baseline_metrics.items():
        metrics_rows.append({"model": name, "label": "naive baseline (same test split)",
                              "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"]})
    metrics_path = OUTPUTS_DIR / "test_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    print(f"Saved held-out test metrics (RF/ANN + naive baselines) to: {metrics_path}")

    print("\nRunning diagnostics (feature importance, parity/residual plots, "
          "error-by-temperature-regime)...")
    run_all_diagnostics(
        trained.predictions, PROXY_TARGET_COLUMN, trained.rf_model, FEATURE_COLUMNS,
        OUTPUTS_DIR, PLOTS_DIR,
    )

    print("\nGenerating plots...")
    plot_weather_overview(df, PLOTS_DIR / "tashkent_weather_overview.png")
    plot_model_comparison(
        trained.predictions, target_column=PROXY_TARGET_COLUMN,
        label="RailTemp/CNU transient model (PROXY, not real measurements)",
        out_path=PLOTS_DIR / "model_comparison_proxy_target.png",
    )
    print(f"Saved plots to: {PLOTS_DIR}")

    print("\nDone. Reminder: no step in this pipeline used real measured rail")
    print("temperature. All rail-temperature numbers above are either a")
    print("physical model's own output or ML models trained to imitate it.")


if __name__ == "__main__":
    main()
