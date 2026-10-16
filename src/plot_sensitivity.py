"""
Plot Model A's (RailTemp/CNU) sensitivity to its two assumed corridor-specific
parameters (track azimuth, ambient emissivity), from the CSV produced by
tools/railtemp_cnu/sensitivity.py (run via the separate `.venv-railtemp`
interpreter -- see README.md).

Run:
    python3 src/plot_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SENSITIVITY_CSV = PROJECT_ROOT / "outputs" / "model_a_sensitivity.csv"
OUT_PATH = PROJECT_ROOT / "outputs" / "plots" / "model_a_sensitivity.png"


def main() -> None:
    if not SENSITIVITY_CSV.exists():
        raise FileNotFoundError(
            f"Missing {SENSITIVITY_CSV}. Generate it first via the railtemp venv:\n"
            f"  .venv-railtemp/bin/python tools/railtemp_cnu/sensitivity.py"
        )
    df = pd.read_csv(SENSITIVITY_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (param, label, baseline_val) in zip(
        axes,
        [("azimuth_deg", "Track azimuth, deg", 90.0), ("ambient_emissivity", "Ambient emissivity", 0.5)],
    ):
        sub = df[df["parameter"] == param].sort_values("value")
        colors = ["#c0392b" if not baseline else "#7f8c8d" for baseline in sub["is_baseline"]]
        ax.bar(sub["value"].astype(str), sub["mean_abs_dev_c"], color=colors)
        ax.set_xlabel(label)
        ax.set_ylabel("Mean abs. deviation from baseline, deg C")
        ax.set_title(f"Model A sensitivity to {label.lower()}\n(baseline = {baseline_val}, gray bar)")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "How much do Model A's (RailTemp/CNU) UNVALIDATED assumptions matter?\n"
        "One-at-a-time local sensitivity, 3-month window"
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved sensitivity plot to: {OUT_PATH}")


if __name__ == "__main__":
    main()
