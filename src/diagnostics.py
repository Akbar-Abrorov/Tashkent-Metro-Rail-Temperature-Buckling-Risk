"""
Research-paper-style diagnostics for the RF/ANN models: feature importance,
parity (predicted-vs-actual) and residual plots, and an error breakdown by
temperature regime -- overall average error is a weak summary for a
buckling-RISK application, where what actually matters is accuracy on the
hottest hours, not on a mild spring afternoon.

All diagnostics are computed on the TEST split only (predictions["split"]
== "test"), never train/val -- consistent with README's "Evaluation
methodology" (held-out results only).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from train_models import evaluate


def feature_importance_table(rf_model: RandomForestRegressor, feature_columns: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame({"feature": feature_columns, "importance": rf_model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance(importance_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ordered = importance_df.sort_values("importance")
    ax.barh(ordered["feature"], ordered["importance"], color="#2980b9")
    ax.set_xlabel("Random Forest feature importance (Gini-based)")
    ax.set_title("Which weather features drive the RF model's predictions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_parity(test_predictions: pd.DataFrame, target_column: str, out_path: Path) -> None:
    """Predicted-vs-actual scatter for RF and ANN, on the test split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    actual = test_predictions[target_column]
    lims = (min(actual.min(), test_predictions["rf_pred_c"].min(), test_predictions["ann_pred_c"].min()) - 2,
            max(actual.max(), test_predictions["rf_pred_c"].max(), test_predictions["ann_pred_c"].max()) + 2)

    for ax, col, name, color in zip(
        axes, ["rf_pred_c", "ann_pred_c"], ["Random Forest", "ANN"], ["#2980b9", "#27ae60"]
    ):
        ax.scatter(actual, test_predictions[col], s=4, alpha=0.25, color=color)
        ax.plot(lims, lims, color="black", linewidth=1, linestyle="--", label="y = x")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Target (Model A: RailTemp/CNU), deg C")
        ax.set_ylabel("Predicted, deg C")
        ax.set_title(name)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Parity plots -- TEST split only (held out from training and ANN early stopping)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_residuals(test_predictions: pd.DataFrame, target_column: str, out_path: Path) -> None:
    """Residual-vs-predicted and residual histogram, for RF and ANN, test split only."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    actual = test_predictions[target_column]

    for col_idx, (col, name, color) in enumerate(
        zip(["rf_pred_c", "ann_pred_c"], ["Random Forest", "ANN"], ["#2980b9", "#27ae60"])
    ):
        residual = test_predictions[col] - actual
        axes[0, col_idx].scatter(test_predictions[col], residual, s=4, alpha=0.25, color=color)
        axes[0, col_idx].axhline(0, color="black", linewidth=1, linestyle="--")
        axes[0, col_idx].set_xlabel("Predicted, deg C")
        axes[0, col_idx].set_ylabel("Residual (pred - target), deg C")
        axes[0, col_idx].set_title(f"{name}: residual vs. predicted")
        axes[0, col_idx].grid(alpha=0.3)

        axes[1, col_idx].hist(residual, bins=60, color=color, alpha=0.8)
        axes[1, col_idx].axvline(0, color="black", linewidth=1, linestyle="--")
        axes[1, col_idx].set_xlabel("Residual (pred - target), deg C")
        axes[1, col_idx].set_ylabel("Count")
        axes[1, col_idx].set_title(f"{name}: residual distribution "
                                    f"(mean={residual.mean():.3f}, std={residual.std():.3f})")
        axes[1, col_idx].grid(alpha=0.3)

    fig.suptitle("Residual diagnostics -- TEST split only")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def error_by_regime(
    test_predictions: pd.DataFrame, target_column: str, extreme_threshold_c: float = 40.0,
) -> pd.DataFrame:
    """RMSE/MAE/R2 for RF and ANN on: all test rows, the hottest decile of
    test rows (data-driven), and rows at/above a fixed buckling-risk-relevant
    threshold. Overall average error can look good while the model is worse
    exactly where it matters (the hot tail) -- this makes that visible
    instead of hiding it in one aggregate number.
    """
    rows = []
    actual = test_predictions[target_column]
    p90 = float(actual.quantile(0.90))

    regimes = {
        "overall": test_predictions,
        f"hottest_decile (target >= {p90:.1f} C, data-driven p90)": test_predictions[actual >= p90],
        f"target >= {extreme_threshold_c:.0f} C (fixed threshold)": test_predictions[actual >= extreme_threshold_c],
    }
    for regime_name, subset in regimes.items():
        if len(subset) == 0:
            continue
        sub_actual = np.asarray(subset[target_column].to_numpy(), dtype=float)
        for model_name, col in [("random_forest", "rf_pred_c"), ("ann", "ann_pred_c")]:
            m = evaluate(sub_actual, np.asarray(subset[col].to_numpy(), dtype=float))
            rows.append({"regime": regime_name, "model": model_name, "n": len(subset), **m})
    return pd.DataFrame(rows)


def run_all_diagnostics(
    predictions: pd.DataFrame, target_column: str, rf_model: RandomForestRegressor,
    feature_columns: list[str], outputs_dir: Path, plots_dir: Path,
) -> None:
    test_predictions = predictions[predictions["split"] == "test"]

    importance_df = feature_importance_table(rf_model, feature_columns)
    importance_df.to_csv(outputs_dir / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df, plots_dir / "feature_importance.png")

    plot_parity(test_predictions, target_column, plots_dir / "parity_plots.png")
    plot_residuals(test_predictions, target_column, plots_dir / "residuals.png")

    regime_df = error_by_regime(test_predictions, target_column)
    regime_df.to_csv(outputs_dir / "test_metrics_by_regime.csv", index=False)
    print("\nError by temperature regime (test split):")
    print(regime_df.to_string(index=False))
