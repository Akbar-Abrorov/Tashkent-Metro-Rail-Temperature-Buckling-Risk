"""
Audit the real Meteostat weather dataset used throughout this project.

Checks timestamp continuity/gaps/duplicates, per-column missingness,
physical plausibility bounds, and a solar-geometry cross-check (clear-sky
GHI must be 0 whenever the sun is below the horizon). Writes findings to
outputs/weather_audit.md.

Run:
    python3 src/audit_weather.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Physically plausible ranges for Tashkent (41.27N, 489 m elevation).
PLAUSIBLE_RANGES = {
    "air_temp_c": (-30.0, 50.0),
    "rel_humidity_pct": (0.0, 100.0),
    "wind_speed_ms": (0.0, 40.0),
    # Sea-level-reduced pressure. Upper bound set to 1055 hPa (not the
    # textbook ~1040) because Central Asian winters see real Siberian-High
    # anticyclones push sea-level-equivalent pressure into the 1040s.
    "pressure_hpa": (930.0, 1055.0),
    "cloud_cover_okta": (0.0, 8.0),
}


def find_weather_csv() -> Path:
    candidates = sorted(RAW_DATA_DIR.glob("tashkent_weather_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No weather CSV found in {RAW_DATA_DIR}.")
    return candidates[-1]


def audit_timestamp_continuity(df: pd.DataFrame) -> dict:
    duplicates = int(df.index.duplicated().sum())
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="h")
    missing = full_range.difference(df.index)
    gaps = []
    if len(missing) > 0:
        missing_sorted = missing.sort_values()
        gap_start = missing_sorted[0]
        prev = missing_sorted[0]
        for ts in missing_sorted[1:]:
            if ts - prev > pd.Timedelta(hours=1):
                gaps.append((gap_start, prev))
                gap_start = ts
            prev = ts
        gaps.append((gap_start, prev))
    return {
        "expected_hours": len(full_range),
        "actual_rows": len(df),
        "duplicate_timestamps": duplicates,
        "missing_hours": len(missing),
        "n_gaps": len(gaps),
        "largest_gaps": sorted(
            gaps, key=lambda g: g[1] - g[0], reverse=True
        )[:5],
    }


def audit_missingness(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean().round(4).sort_values(ascending=False)


def audit_plausibility(df: pd.DataFrame) -> dict:
    violations = {}
    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < low) | (df[col] > high)][col]
        if len(out_of_range) > 0:
            violations[col] = {
                "n_violations": int(len(out_of_range)),
                "min": float(out_of_range.min()),
                "max": float(out_of_range.max()),
                "expected_range": (low, high),
            }
    return violations


def audit_night_ghi(df: pd.DataFrame) -> dict:
    if "clearsky_ghi_wm2" not in df.columns or "solar_elevation_deg" not in df.columns:
        return {"checked": False}
    night = df[df["solar_elevation_deg"] <= 0]
    nonzero_night_ghi = night[night["clearsky_ghi_wm2"] > 1e-6]
    return {
        "checked": True,
        "n_night_rows": len(night),
        "n_violations": int(len(nonzero_night_ghi)),
        "max_violation_wm2": float(nonzero_night_ghi["clearsky_ghi_wm2"].max()) if len(nonzero_night_ghi) else 0.0,
    }


def audit_unit_consistency(df: pd.DataFrame) -> dict:
    if "wind_speed_ms" not in df.columns or "wind_speed_kmh" not in df.columns:
        return {"checked": False}
    expected = df["wind_speed_kmh"] / 3.6
    max_abs_diff = float((df["wind_speed_ms"] - expected).abs().max())
    return {"checked": True, "max_abs_diff_ms": max_abs_diff}


def format_report(weather_csv: Path, df: pd.DataFrame, continuity: dict, missingness: pd.Series,
                   plausibility: dict, night_ghi: dict, units: dict) -> str:
    lines = []
    lines.append("# Weather dataset audit\n")
    lines.append(f"Source: `{weather_csv.relative_to(PROJECT_ROOT)}`\n")
    lines.append(f"Range: {df.index.min()} to {df.index.max()} ({len(df)} rows)\n")

    lines.append("\n## Station representativeness\n")
    lines.append(
        "Single Meteostat station (38457, Tashkent/UTTT, 41.2667N 69.2667E, "
        "489 m elevation) -- the only station within 50 km of Tashkent (next "
        "nearest is ~71 km away), so no multi-station interpolation is used or "
        "possible. This means the dataset reflects conditions at one point "
        "near the city, not necessarily at the rail corridor itself.\n"
    )

    lines.append("\n## Timestamp continuity\n")
    lines.append(f"- Expected hourly rows over the range: {continuity['expected_hours']}\n")
    lines.append(f"- Actual rows: {continuity['actual_rows']}\n")
    lines.append(f"- Duplicate timestamps: {continuity['duplicate_timestamps']}\n")
    lines.append(f"- Missing hours: {continuity['missing_hours']} across {continuity['n_gaps']} gap(s)\n")
    if continuity["largest_gaps"]:
        lines.append("- Largest gaps:\n")
        for start, end in continuity["largest_gaps"]:
            lines.append(f"  - {start} to {end}\n")

    lines.append("\n## Per-column missingness (NaN fraction)\n")
    lines.append("| column | NaN fraction |\n|---|---|\n")
    for col, frac in missingness.items():
        if frac > 0:
            lines.append(f"| {col} | {frac:.4f} |\n")
    if (missingness == 0).all():
        lines.append("| (none) | all columns fully populated |\n")

    lines.append("\n## Physical plausibility violations\n")
    if plausibility:
        for col, info in plausibility.items():
            lines.append(
                f"- `{col}`: {info['n_violations']} rows outside "
                f"{info['expected_range']} (observed min={info['min']:.2f}, max={info['max']:.2f})\n"
            )
    else:
        lines.append("None found within the checked ranges.\n")

    lines.append("\n## Solar-geometry cross-check: clear-sky GHI must be 0 when sun is below horizon\n")
    if night_ghi["checked"]:
        lines.append(f"- Night rows (solar_elevation_deg <= 0): {night_ghi['n_night_rows']}\n")
        lines.append(f"- Violations (nonzero GHI at night): {night_ghi['n_violations']}\n")
        if night_ghi["n_violations"] > 0:
            lines.append(f"  - Max violating value: {night_ghi['max_violation_wm2']:.4f} W/m^2\n")
    else:
        lines.append("Not checked -- required columns missing.\n")

    lines.append("\n## Unit consistency: wind_speed_ms == wind_speed_kmh / 3.6\n")
    if units["checked"]:
        lines.append(f"- Max absolute discrepancy: {units['max_abs_diff_ms']:.6f} m/s\n")
    else:
        lines.append("Not checked -- required columns missing.\n")

    return "".join(lines)


def main() -> None:
    weather_csv = find_weather_csv()
    df = pd.read_csv(weather_csv, index_col=0, parse_dates=True)

    continuity = audit_timestamp_continuity(df)
    missingness = audit_missingness(df)
    plausibility = audit_plausibility(df)
    night_ghi = audit_night_ghi(df)
    units = audit_unit_consistency(df)

    report = format_report(weather_csv, df, continuity, missingness, plausibility, night_ghi, units)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "weather_audit.md"
    out_path.write_text(report)

    print(report)
    print(f"\nSaved full report to: {out_path}")


if __name__ == "__main__":
    main()
