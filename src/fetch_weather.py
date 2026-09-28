"""
Fetch real hourly weather data for Tashkent (Meteostat station 38457) and
augment it with solar position + clear-sky irradiance features (pvlib).

DATA SOURCE: 100% real observed data from Meteostat station 38457
(WMO 38457 / ICAO UTTT / IATA TAS, "Tashkent", 41.2667N 69.2667E, 489m elev).
This is the only Meteostat station within 50 km of Tashkent; the next
nearest station is ~71 km away, too far for reliable interpolation, so we
query this single station directly (no ms.interpolate()).

Solar position (zenith/elevation/azimuth) and clear-sky GHI/DNI/DHI are
computed, not measured -- they come from pvlib's SPA algorithm and the
Ineichen clear-sky model, which are physically-derived, deterministic
functions of time + location (not statistical/fitted placeholders).

Output: data/raw/tashkent_weather_<start>_<end>.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import meteostat
import pandas as pd
import pvlib

STATION_ID = "38457"
STATION_LAT = 41.2667
STATION_LON = 69.2667
STATION_ELEVATION_M = 489
STATION_TZ = "Asia/Tashkent"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Columns Meteostat returns for this station but which are entirely empty
# (confirmed by inspection: snow depth, peak wind gust, sunshine duration
# are not measured/reported by station 38457). Dropped rather than kept
# as all-NaN placeholder columns.
ALWAYS_EMPTY_COLUMNS = ["snwd", "wpgt", "tsun"]


def fetch_station_metadata() -> meteostat.typing.Station: # type: ignore
    meta = meteostat.stations.meta(STATION_ID)
    if meta is None:
        raise RuntimeError(f"Meteostat station {STATION_ID} not found")
    return meta


def fetch_raw_weather(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch real hourly observations for STATION_ID, in local (Asia/Tashkent) time."""
    ts = meteostat.hourly(STATION_ID, start, end, timezone=STATION_TZ)
    df = ts.fetch()
    if df is None or df.empty:
        raise RuntimeError(
            f"Meteostat returned no data for station {STATION_ID} "
            f"between {start} and {end}. Check the local cache at "
            "~/.meteostat/ and your internet connection."
        )
    df = df.drop(columns=[c for c in ALWAYS_EMPTY_COLUMNS if c in df.columns])
    df.index.name = "time"
    return df


def add_solar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add pvlib-computed solar position + Ineichen clear-sky irradiance.

    These are physically computed from (timestamp, lat, lon, elevation),
    not observed -- clearly separate from the Meteostat-observed columns.
    """
    location = pvlib.location.Location(
        STATION_LAT, STATION_LON, tz=STATION_TZ, altitude=STATION_ELEVATION_M,
        name="Tashkent",
    )
    times = df.index
    solpos = location.get_solarposition(times)
    clearsky = location.get_clearsky(times, model="ineichen")

    out = df.copy()
    out["solar_zenith_deg"] = solpos["apparent_zenith"]
    out["solar_elevation_deg"] = solpos["apparent_elevation"]
    out["solar_azimuth_deg"] = solpos["azimuth"]
    out["clearsky_ghi_wm2"] = clearsky["ghi"]
    out["clearsky_dni_wm2"] = clearsky["dni"]
    out["clearsky_dhi_wm2"] = clearsky["dhi"]
    return out


def rename_meteostat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Give the raw Meteostat columns explicit, self-documenting names."""
    return df.rename(columns={
        "temp": "air_temp_c",
        "rhum": "rel_humidity_pct",
        "prcp": "precip_mm",
        "wdir": "wind_dir_deg",
        "wspd": "wind_speed_kmh",
        "pres": "pressure_hpa",
        "cldc": "cloud_cover_okta",
        "coco": "weather_condition_code",
    })


def build_dataset(start: datetime, end: datetime) -> pd.DataFrame:
    raw = fetch_raw_weather(start, end)
    raw = rename_meteostat_columns(raw)
    full = add_solar_features(raw)
    full["wind_speed_ms"] = full["wind_speed_kmh"] / 3.6
    return full


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-06-01", help="YYYY-MM-DD (local time)")
    parser.add_argument("--end", default="2023-09-01", help="YYYY-MM-DD (local time, exclusive-ish)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    meta = fetch_station_metadata()
    print(f"Station: {meta.name} ({meta.id}), {meta.latitude}N {meta.longitude}E, "
          f"elevation {meta.elevation} m, tz {meta.timezone}")

    df = build_dataset(start, end)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DATA_DIR / f"tashkent_weather_{args.start}_{args.end}.csv"
    df.to_csv(out_path)

    print(f"\nFetched {len(df)} real hourly rows from Meteostat station {STATION_ID}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Saved to: {out_path}")
    print("\nColumn NaN fraction (0.0 = fully populated):")
    print(df.isna().mean().round(3).to_string())
    print("\nSample rows:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.head(5))
        print("...")
        print(df.sample(5, random_state=0).sort_index())


if __name__ == "__main__":
    main()
