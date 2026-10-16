"""
Rolling-origin (expanding-window) chronological cross-validation for RF/ANN.

The main pipeline reports one held-out test split (the final chronological
15%). That's an honest, leak-free number, but a single split can't show
whether performance is stable across different seasons/periods, which is
exactly what a reviewer of a time-series model will ask about. This script
re-trains RF/ANN (and the naive baselines) across multiple chronological
folds and reports mean +/- std, writing outputs/cv_metrics.csv and
outputs/plots/cv_metrics.png.

This is DIAGNOSTIC, not the project's headline result -- the main
pipeline's single train/val/test split (see README's "Evaluation
methodology") remains the reported number; this quantifies how much that
number could have varied under a different split.

Run:
    python3 src/cross_validate.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from baselines import linear_airtemp_baseline, persistence_baseline
from physical_model import add_physical_baseline
from train_models import (
    FEATURE_COLUMNS,
    PROXY_TARGET_COLUMN,
    _predict_ann,
    evaluate,
    train_ann,
    train_random_forest,
)
from pipeline import find_cnu_csv, find_weather_csv, load_cnu_baseline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

N_FOLDS = 5
MIN_TRAIN_FRAC = 0.4  # first fold's training window size, as a fraction of the full dataset
VAL_FRAC_WITHIN_TRAIN = 0.2  # held out of each fold's train window for ANN early stopping


def make_folds(work: pd.DataFrame, n_folds: int = N_FOLDS, min_train_frac: float = MIN_TRAIN_FRAC):
    """Expanding-window folds: fold i's train window is everything before its
    test window (always growing), test windows are consecutive, non-overlapping,
    equal-sized blocks covering the remaining (1 - min_train_frac) of the data."""
    n = len(work)
    n_min_train = int(n * min_train_frac)
    n_remaining = n - n_min_train
    n_test = n_remaining // n_folds

    folds = []
    for i in range(n_folds):
        test_start = n_min_train + i * n_test
        test_end = n if i == n_folds - 1 else test_start + n_test
        train_df = work.iloc[:test_start]
        test_df = work.iloc[test_start:test_end]
        folds.append((train_df, test_df))
    return folds


def fit_and_score_fold(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str) -> list[dict]:
    n_val = int(len(train_df) * VAL_FRAC_WITHIN_TRAIN)
    fit_train_df = train_df.iloc[:-n_val]
    val_df = train_df.iloc[-n_val:]

    X_fit_df = cast(pd.DataFrame, fit_train_df[FEATURE_COLUMNS])
    X_val_df = cast(pd.DataFrame, val_df[FEATURE_COLUMNS])
    X_test_df = cast(pd.DataFrame, test_df[FEATURE_COLUMNS])
    y_fit = np.asarray(fit_train_df[target_column].to_numpy(), dtype=float)
    y_val = np.asarray(val_df[target_column].to_numpy(), dtype=float)
    y_test = np.asarray(test_df[target_column].to_numpy(), dtype=float)

    rf_model = train_random_forest(X_fit_df, cast(pd.Series, fit_train_df[target_column]))
    ann_model, ann_x_scaler, ann_y_scaler = train_ann(
        np.asarray(X_fit_df.to_numpy(), dtype=float), y_fit,
        np.asarray(X_val_df.to_numpy(), dtype=float), y_val,
    )

    rf_pred = rf_model.predict(X_test_df)
    ann_pred = _predict_ann(ann_model, ann_x_scaler, ann_y_scaler, np.asarray(X_test_df.to_numpy(), dtype=float))

    persistence_metrics = persistence_baseline(
        pd.concat([train_df, test_df]).sort_index(), target_column, test_df,
    )
    linear_metrics = linear_airtemp_baseline(train_df, test_df, target_column)

    return [
        {"model": "random_forest", **evaluate(y_test, np.asarray(rf_pred, dtype=float))},
        {"model": "ann", **evaluate(y_test, ann_pred)},
        {"model": "persistence", **persistence_metrics},
        {"model": "linear_airtemp", **{k: v for k, v in linear_metrics.items() if k in ("rmse", "mae", "r2")}},
    ]


def main() -> None:
    weather_csv = find_weather_csv()
    df = pd.read_csv(weather_csv, index_col=0, parse_dates=True)
    df = add_physical_baseline(df)
    cnu_csv = find_cnu_csv(weather_csv)
    df[PROXY_TARGET_COLUMN] = load_cnu_baseline(cnu_csv)

    work = df.dropna(subset=FEATURE_COLUMNS + [PROXY_TARGET_COLUMN]).copy().sort_index()
    folds = make_folds(work)

    all_rows = []
    for fold_idx, (train_df, test_df) in enumerate(folds):
        print(f"\nFold {fold_idx + 1}/{len(folds)}: "
              f"train {train_df.index.min()} to {train_df.index.max()} ({len(train_df)} rows), "
              f"test {test_df.index.min()} to {test_df.index.max()} ({len(test_df)} rows)")
        start = time.time()
        fold_metrics = fit_and_score_fold(train_df, test_df, PROXY_TARGET_COLUMN)
        for row in fold_metrics:
            row["fold"] = fold_idx + 1
            row["test_start"] = test_df.index.min()
            row["test_end"] = test_df.index.max()
            row["n_test"] = len(test_df)
            all_rows.append(row)
            print(f"  {row['model']:15s} RMSE={row['rmse']:.3f} C  MAE={row['mae']:.3f} C  R2={row['r2']:.4f}")
        print(f"  (fold took {time.time() - start:.1f}s)")

    cv_df = pd.DataFrame(all_rows)
    cv_df.to_csv(OUTPUTS_DIR / "cv_metrics.csv", index=False)

    summary = cv_df.groupby("model")[["rmse", "mae", "r2"]].agg(["mean", "std"])
    print("\n" + "=" * 70)
    print(f"Cross-validation summary across {len(folds)} chronological folds "
          f"(mean +/- std):")
    print("=" * 70)
    print(summary.round(3).to_string())
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUTPUTS_DIR / "cv_summary.csv", index=False)

    plot_cv_results(cv_df, PLOTS_DIR / "cv_metrics.png")
    print(f"\nSaved per-fold metrics to: {OUTPUTS_DIR / 'cv_metrics.csv'}")
    print(f"Saved summary to: {OUTPUTS_DIR / 'cv_summary.csv'}")
    print(f"Saved plot to: {PLOTS_DIR / 'cv_metrics.png'}")


def plot_cv_results(cv_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    models = ["persistence", "linear_airtemp", "random_forest", "ann"]
    colors = {"persistence": "#95a5a6", "linear_airtemp": "#e67e22",
              "random_forest": "#2980b9", "ann": "#27ae60"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        sub = cv_df[cv_df["model"] == model].sort_values("fold")
        ax.plot(sub["fold"], sub["rmse"], marker="o", label=model, color=colors[model])
    ax.set_xlabel("Fold (chronologically later, expanding training window)")
    ax.set_ylabel("Test RMSE, deg C")
    ax.set_title("Rolling-origin cross-validation: RMSE per fold")
    ax.set_xticks(sorted(cv_df["fold"].unique()))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
