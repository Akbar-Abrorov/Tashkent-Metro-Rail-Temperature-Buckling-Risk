"""
Physical baseline rail-temperature model.

IMPORTANT HONESTY NOTE:
This is a standard steady-state surface energy-balance model of the kind
widely used in pavement/rail surface temperature literature (solar
absorption balanced against convective + longwave-radiative cooling). It
is NOT a reproduction of the exact proprietary formula or fitted
coefficients from Hong, Park & Cho (2021, Sensors) or Hong et al. (2019,
IJPEM-GT) -- those papers' exact equations/coefficients were not available
to derive this from. Treat this as a physically-reasonable stand-in
baseline, not a validated replication of those papers' physical model.
Coefficients (absorptivity, emissivity, convection correlation) are typical
literature values for weathered/oxidized steel and should be recalibrated
against real rail temperature measurements once available.

Model
-----
At steady state, absorbed solar energy = convective loss + net longwave
radiative loss:

    alpha * GHI = h_c * (T_rail - T_air) + eps * sigma * (T_rail^4 - T_sky^4)

  alpha    solar absorptivity of the rail surface (weathered steel: ~0.80)
  eps      longwave emissivity of the rail surface (~0.85)
  sigma    Stefan-Boltzmann constant
  h_c      convective heat transfer coefficient, wind-dependent:
           McAdams correlation h_c = 5.7 + 3.8 * wind_speed_ms  [W/m^2/K]
  T_sky    effective sky temperature, Swinbank (1963) clear-sky formula:
           T_sky[K] = 0.0552 * T_air[K]^1.5

Solved for T_rail via vectorized Newton-Raphson (converges in ~10
iterations to well under 1e-6 K).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant, W/m^2/K^4

DEFAULT_ABSORPTIVITY = 0.80   # weathered/oxidized steel rail head
DEFAULT_EMISSIVITY = 0.85     # weathered/oxidized steel
DEFAULT_CONV_A = 5.7          # McAdams still-air term, W/m^2/K
DEFAULT_CONV_B = 3.8          # McAdams wind term, W/m^2/K per m/s


def sky_temperature_k(air_temp_k: np.ndarray) -> np.ndarray:
    """Swinbank (1963) clear-sky effective sky temperature."""
    return 0.0552 * air_temp_k ** 1.5


def convective_coefficient(wind_speed_ms: np.ndarray, a: float = DEFAULT_CONV_A,
                            b: float = DEFAULT_CONV_B) -> np.ndarray:
    """McAdams forced-convection correlation, W/m^2/K."""
    return a + b * np.asarray(wind_speed_ms)


def predict_rail_temperature(
    air_temp_c: pd.Series | np.ndarray,
    ghi_wm2: pd.Series | np.ndarray,
    wind_speed_ms: pd.Series | np.ndarray,
    alpha: float = DEFAULT_ABSORPTIVITY,
    eps: float = DEFAULT_EMISSIVITY,
    n_iter: int = 25,
) -> np.ndarray:
    """Solve the steady-state heat balance for rail surface temperature.

    All inputs same length. Returns rail temperature in degrees C.
    """
    air_temp_c = np.asarray(air_temp_c, dtype=float)
    ghi_wm2 = np.asarray(ghi_wm2, dtype=float)
    wind_speed_ms = np.asarray(wind_speed_ms, dtype=float)

    t_air_k = air_temp_c + 273.15
    t_sky_k = sky_temperature_k(t_air_k)
    h_c = convective_coefficient(wind_speed_ms)

    # Newton-Raphson, vectorized, initialized at air temperature.
    t_rail_k = t_air_k.copy()
    for _ in range(n_iter):
        f = alpha * ghi_wm2 - h_c * (t_rail_k - t_air_k) - eps * SIGMA * (
            t_rail_k ** 4 - t_sky_k ** 4
        )
        f_prime = -h_c - 4 * eps * SIGMA * t_rail_k ** 3
        t_rail_k = t_rail_k - f / f_prime

    return t_rail_k - 273.15


def add_physical_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Append a `rail_temp_physical_c` column computed by the physical model.

    Expects columns: air_temp_c, clearsky_ghi_wm2, wind_speed_ms
    (produced by src/fetch_weather.py).
    """
    out = df.copy()
    out["rail_temp_physical_c"] = predict_rail_temperature(
        air_temp_c=out["air_temp_c"],
        ghi_wm2=out["clearsky_ghi_wm2"],
        wind_speed_ms=out["wind_speed_ms"],
    )
    return out
