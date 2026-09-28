"""Plotting utilities for the Tashkent rail temperature pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_weather_overview(df: pd.DataFrame, out_path: Path) -> None:
    """Plot real observed Tashkent summer 2023 air temperature + clear-sky GHI."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    axes[0].plot(df.index, df["air_temp_c"], color="#c0392b", linewidth=0.8)
    axes[0].set_ylabel("Air temperature (deg C)")
    axes[0].set_title(
        "Tashkent, summer 2023 -- REAL observed weather (Meteostat station 38457)"
    )
    axes[0].grid(alpha=0.3)

    axes[1].plot(df.index, df["clearsky_ghi_wm2"], color="#e67e22", linewidth=0.8)
    axes[1].set_ylabel("Clear-sky GHI (W/m^2)")
    axes[1].set_xlabel("Local time (Asia/Tashkent)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(predictions: pd.DataFrame, target_column: str, label: str,
                           out_path: Path, zoom_days: int | None = 10) -> None:
    """Plot physical baseline / target vs RF vs ANN predictions on real weather input.

    If zoom_days is set, an inset-style second panel zooms into the first
    `zoom_days` days for readability (hourly data over 3 months is dense).
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    for ax, sub in zip(axes, [predictions, predictions.iloc[: zoom_days * 24] if zoom_days else predictions]):
        ax.plot(sub.index, sub[target_column], label=f"Target ({label})", color="black", linewidth=1.2)
        ax.plot(sub.index, sub["rf_pred_c"], label="Random Forest", color="#2980b9", linewidth=0.9, alpha=0.85)
        ax.plot(sub.index, sub["ann_pred_c"], label="ANN", color="#27ae60", linewidth=0.9, alpha=0.85)
        ax.set_ylabel("Rail surface temp (deg C)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title(
        f"Model comparison on REAL Tashkent weather input -- target = {label}\n"
        "NOT validated against real measured rail temperature (see README)"
    )
    axes[1].set_title(f"Zoomed: first {zoom_days} days" if zoom_days else "")
    axes[1].set_xlabel("Local time (Asia/Tashkent)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
