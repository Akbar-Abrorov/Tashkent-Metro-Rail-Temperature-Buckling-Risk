"""
Sensitivity analysis for Model A (RailTemp/CNU): how much does its output
change under different values of the two corridor-specific parameters that
were simply assumed (not measured) for Tashkent Metro -- track azimuth and
ambient emissivity? See run_cnu.py's module docstring for why they're
assumptions.

One-at-a-time local sensitivity around the baseline (azimuth=90,
ambient_emissivity=0.5): sweep each parameter while holding the other at
its baseline value, on the 3-month file (fast: ~2.7s/run). Reports mean/max
absolute deviation from the baseline run.

Run (via the railtemp venv):
    .venv-railtemp/bin/python tools/railtemp_cnu/sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from run_cnu import (
    AMBIENT_EMISSIVITY,
    DEFAULT_RAW_WEATHER_CSV,
    TRACK_AZIMUTH_DEG,
    load_weather,
    simulate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_CSV = PROJECT_ROOT / "outputs" / "model_a_sensitivity.csv"

AZIMUTH_SWEEP_DEG = [0.0, 45.0, 90.0, 135.0, 180.0]
AMBIENT_EMISSIVITY_SWEEP = [0.3, 0.5, 0.7, 0.9]


def deviation_stats(baseline: pd.Series, variant: pd.Series) -> dict:
    diff = variant - baseline
    return {
        "mean_bias_c": float(diff.mean()),
        "mean_abs_dev_c": float(diff.abs().mean()),
        "max_abs_dev_c": float(diff.abs().max()),
    }


def main() -> None:
    df = load_weather(DEFAULT_RAW_WEATHER_CSV)
    print(f"Loaded {len(df)} hourly rows from {DEFAULT_RAW_WEATHER_CSV.name} for sensitivity analysis")

    baseline = simulate(df, azimuth_deg=TRACK_AZIMUTH_DEG, ambient_emissivity=AMBIENT_EMISSIVITY, quiet=True)
    print(f"Baseline (azimuth={TRACK_AZIMUTH_DEG}, ambient_emissivity={AMBIENT_EMISSIVITY}): "
          f"mean={baseline.mean():.2f} C")

    rows = []
    print("\nAzimuth sweep (ambient_emissivity held at baseline):")
    for azimuth in AZIMUTH_SWEEP_DEG:
        if azimuth == TRACK_AZIMUTH_DEG:
            stats = {"mean_bias_c": 0.0, "mean_abs_dev_c": 0.0, "max_abs_dev_c": 0.0}
        else:
            variant = simulate(df, azimuth_deg=azimuth, ambient_emissivity=AMBIENT_EMISSIVITY, quiet=True)
            stats = deviation_stats(baseline, variant)
        rows.append({"parameter": "azimuth_deg", "value": azimuth, "is_baseline": azimuth == TRACK_AZIMUTH_DEG, **stats})
        print(f"  azimuth={azimuth:>5.0f} deg: mean_abs_dev={stats['mean_abs_dev_c']:.3f} C, "
              f"max_abs_dev={stats['max_abs_dev_c']:.3f} C, bias={stats['mean_bias_c']:+.3f} C")

    print("\nAmbient emissivity sweep (azimuth held at baseline):")
    for emissivity in AMBIENT_EMISSIVITY_SWEEP:
        if emissivity == AMBIENT_EMISSIVITY:
            stats = {"mean_bias_c": 0.0, "mean_abs_dev_c": 0.0, "max_abs_dev_c": 0.0}
        else:
            variant = simulate(df, azimuth_deg=TRACK_AZIMUTH_DEG, ambient_emissivity=emissivity, quiet=True)
            stats = deviation_stats(baseline, variant)
        rows.append({"parameter": "ambient_emissivity", "value": emissivity,
                      "is_baseline": emissivity == AMBIENT_EMISSIVITY, **stats})
        print(f"  ambient_emissivity={emissivity:.1f}: mean_abs_dev={stats['mean_abs_dev_c']:.3f} C, "
              f"max_abs_dev={stats['max_abs_dev_c']:.3f} C, bias={stats['mean_bias_c']:+.3f} C")

    result_df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved sensitivity results to: {OUT_CSV}")
    print("(Plot it from the main venv: python3 src/plot_sensitivity.py)")


if __name__ == "__main__":
    main()
