"""
Cloud-cover adjustment for clear-sky solar irradiance.

Both physical models (Model A: RailTemp/CNU, Model B: steady-state) were
computing absorbed solar energy from pvlib's CLEAR-SKY irradiance only,
ignoring the real observed cloud cover (`cloud_cover_okta`) that Meteostat
actually provides -- a documented limitation, now fixed here.

FORMULA: Kasten & Czeplak (1980), "Solar and terrestrial radiation
dependent on the amount and type of cloud," Solar Energy 24(2):177-189 --
a standard, widely-used empirical clearness-index reduction as a function
of total cloud cover in oktas (N, 0-8):

    Kc(N) = 1 - 0.75 * (N / 8)^3.4

Kc=1 at N=0 (no cloud, no reduction), Kc=0.25 at N=8 (fully overcast,
irradiance cut to 25% of clear-sky). This was derived/validated for GHI.

APPROXIMATION, STATED PLAINLY: there is no equally standard, simple okta-
based formula for DNI specifically (direct-beam radiation is physically
attenuated by clouds differently than global/diffuse). This module applies
the SAME Kc(N) factor to GHI, DNI and DHI alike -- a simplification, not a
validated cloud-opacity partitioning model. In reality DNI would typically
be cut MORE sharply than GHI/DHI under partial cloud (clouds block the
direct beam selectively), so this likely still somewhat overestimates
direct-beam irradiance under partial cloud. Flagged here and in README.md
"Known limitations" rather than presented as exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kasten_czeplak_transmittance(cloud_okta: pd.Series | np.ndarray) -> np.ndarray:
    """Kc(N) = 1 - 0.75 * (N/8)^3.4, N in oktas (0-8). Returns a multiplier
    in [0.25, 1.0] to apply to clear-sky irradiance."""
    n = np.clip(np.asarray(cloud_okta, dtype=float), 0.0, 8.0)
    return 1.0 - 0.75 * (n / 8.0) ** 3.4


def add_cloud_adjusted_irradiance(df: pd.DataFrame) -> pd.DataFrame:
    """Append cloud-adjusted irradiance columns (`cloudy_ghi_wm2`,
    `cloudy_dni_wm2`, `cloudy_dhi_wm2`) alongside the existing clear-sky
    ones, using real observed `cloud_cover_okta`. Clear-sky columns are
    left untouched so the before/after comparison stays inspectable.
    """
    out = df.copy()
    kc = kasten_czeplak_transmittance(out["cloud_cover_okta"])
    out["cloud_transmittance"] = kc
    out["cloudy_ghi_wm2"] = out["clearsky_ghi_wm2"] * kc
    out["cloudy_dni_wm2"] = out["clearsky_dni_wm2"] * kc
    out["cloudy_dhi_wm2"] = out["clearsky_dhi_wm2"] * kc
    return out
