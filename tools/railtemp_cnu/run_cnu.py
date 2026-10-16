"""
Run the REAL published RailTemp/CNU transient rail-temperature model
(github.com/aryvini/railtemp, GPL-3.0) against the real Tashkent weather data,
and cache its output as a CSV for the main pipeline to consume.

WHY A SEPARATE SCRIPT + VENV: the `railtemp` package requires Python <3.14;
this project's main venv is 3.14. Run this script ONLY with the interpreter in
`.venv-railtemp/` (created once via:
    /opt/homebrew/bin/python3.13 -m venv .venv-railtemp
    .venv-railtemp/bin/pip install "railtemp @ git+https://github.com/aryvini/railtemp.git" pytz
), e.g.:
    .venv-railtemp/bin/python tools/railtemp_cnu/run_cnu.py

USAGE:
    .venv-railtemp/bin/python tools/railtemp_cnu/run_cnu.py [weather_csv]

Defaults to the 3-month summer 2023 file; pass the 2.5-year file explicitly
to run over the full range (in practice this takes only ~30s per ~22k rows
on this machine -- the per-row Python loop turned out not to be a scaling
problem for an hourly dataset of this size).

MODELING CHOICES (all documented, none hidden):
- Rail profile: UIC60 (bundled with the package). The exact profile used on
  Tashkent Metro track is not known to this project; UIC60 is the closest
  widely-used standard mainline/metro rail profile. `cross_area` (steel
  cross-sectional area) and `convection_area`/`radiation_area` (exposed
  surface area per 1 m of rail) are computed directly from the UIC60 profile
  coordinates shipped with the package (shoelace polygon area / polygon
  perimeter), not guessed constants.
- Track azimuth (90 deg, i.e. east-west) and ambient_emissivity (0.5, the
  value used in the package's own worked example) are both assumptions --
  the real track orientation and sky-view factor for this corridor are
  unknown. `RailMaterial` absorptivity (0.8) and emissivity (0.7) match the
  literature defaults already used in src/physical_model.py, for
  comparability between the two physical models.
- Solar input is `clearsky_dni_wm2` (direct normal irradiance), NOT
  `clearsky_ghi_wm2`. The CNU model's sun-exposed area (`As`) is already the
  beam-projected area (profile shadow x sin(solar elevation)), so multiplying
  it by GHI would double-count the elevation-angle geometry. This is an easy
  mistake to make silently; call it out explicitly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from railtemp.railtemp import CNU, Rail, RailMaterial, WeatherData
from railtemp.utils import load_section_coordinates

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_WEATHER_CSV = PROJECT_ROOT / "data" / "raw" / "tashkent_weather_2023-06-01_2023-09-01.csv"

STATION_LAT = 41.2667
STATION_LON = 69.2667
STATION_ELEVATION_M = 489
STATION_TZ = "Asia/Tashkent"

RAIL_PROFILE_NAME = "UIC60"
TRACK_AZIMUTH_DEG = 90.0  # assumption: east-west orientation, unknown for this corridor
AMBIENT_EMISSIVITY = 0.5  # assumption: same value used in the package's own worked example
SOLAR_ABSORPTIVITY = 0.8  # matches src/physical_model.py DEFAULT_ABSORPTIVITY
RAIL_EMISSIVITY = 0.7  # matches the railtemp package's own example material


def polygon_area_and_perimeter(x: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Shoelace area and perimeter of a closed 2D polygon given in order."""
    x_next = np.roll(x, -1)
    z_next = np.roll(z, -1)
    area = 0.5 * abs(np.sum(x * z_next - x_next * z))
    perimeter = float(np.sum(np.hypot(x_next - x, z_next - z)))
    return area, perimeter


def rail_geometry_from_profile(name: str) -> tuple[float, float, float]:
    """Compute (cross_area, convection_area, radiation_area) for a 1 m rail
    segment directly from the package's bundled cross-section coordinates,
    instead of guessing or copying another profile's numbers.
    """
    profile = load_section_coordinates(name)
    one_side = profile[profile["Y"] == profile["Y"].iloc[0]]
    area, perimeter = polygon_area_and_perimeter(
        one_side["X"].to_numpy(dtype=float), one_side["Z"].to_numpy(dtype=float)
    )
    # Perimeter x 1 m length = exposed surface area per 1 m of rail, used for
    # both convective and radiative exchange.
    return area, perimeter, perimeter


def main() -> None:
    raw_weather_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RAW_WEATHER_CSV
    if not raw_weather_csv.exists():
        sys.exit(f"Missing weather CSV: {raw_weather_csv}. Run src/fetch_weather.py first.")

    out_csv = (
        PROJECT_ROOT / "data" / "processed" / f"railtemp_cnu_baseline_{raw_weather_csv.stem.removeprefix('tashkent_weather_')}.csv"
    )

    print(f"Loading weather data from: {raw_weather_csv}")
    df = pd.read_csv(raw_weather_csv, index_col=0, parse_dates=True)
    # The CSV index is already tz-aware local time (+05:00); railtemp's own
    # __fetch_solar_data() calls tz_localize(tz) itself and errors on an
    # already-tz-aware index, so hand it naive local timestamps instead.
    df.index = df.index.tz_localize(None)
    print(f"Loaded {len(df)} hourly rows, {df.index.min()} to {df.index.max()}")

    cross_area, convection_area, radiation_area = rail_geometry_from_profile(RAIL_PROFILE_NAME)
    print(
        f"{RAIL_PROFILE_NAME} geometry (from bundled profile coordinates): "
        f"cross_area={cross_area:.6f} m^2, convection_area=radiation_area="
        f"{convection_area:.6f} m^2 per 1 m rail length"
    )

    material = RailMaterial(density=7850.0, solar_absort=SOLAR_ABSORPTIVITY, emissivity=RAIL_EMISSIVITY)
    rail = Rail(
        name=RAIL_PROFILE_NAME,
        azimuth=TRACK_AZIMUTH_DEG,
        lat=STATION_LAT,
        long=STATION_LON,
        elev=STATION_ELEVATION_M,
        cross_area=cross_area,
        convection_area=convection_area,
        radiation_area=radiation_area,
        ambient_emissivity=AMBIENT_EMISSIVITY,
        material=material,
    )

    weather = WeatherData(
        solar_radiation=df["clearsky_dni_wm2"].rename("SR"),
        ambient_temperature=df["air_temp_c"].rename("Tamb"),
        wind_velocity=df["wind_speed_ms"].rename("Wv"),
        timezone=pytz.timezone(STATION_TZ),
    )

    initial_temp_c = float(df["air_temp_c"].iloc[0])
    print(f"Running CNU transient simulation (Trail_initial={initial_temp_c:.2f} C)...")
    start = time.time()
    sim = CNU(rail=rail, weather=weather)
    sim.run(Trail_initial=initial_temp_c)
    print(f"Simulation finished in {time.time() - start:.1f}s")

    result = sim.result[["Tr_simu"]].rename(columns={"Tr_simu": "rail_temp_cnu_c"})
    result.index.name = "time"
    print(result["rail_temp_cnu_c"].describe().round(2).to_string())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv)
    print(f"Saved RailTemp/CNU baseline to: {out_csv}")


if __name__ == "__main__":
    main()
