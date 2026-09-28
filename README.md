# Tashkent Metro Rail Temperature / Buckling Risk — Research Pipeline

Predicting rail surface temperature for the Tashkent Metro (Uzbekistan) from
weather + solar position features, following the general methodology of
Hong, Park & Cho (2021, *Sensors*) and Hong et al. (2019, *IJPEM-GT*):
a physical baseline, a Random Forest, and a small neural network, compared
on the same weather input.

## What is real, what is proxy/placeholder — read this first

| Component | Status |
|---|---|
| Weather data (`data/raw/tashkent_weather_*.csv`) | **REAL.** Observed hourly data from Meteostat station 38457 (WMO 38457 / ICAO UTTT / IATA TAS, "Tashkent", 41.2667N 69.2667E, 489 m elevation), June 1 – Sep 1 2023. This is the only Meteostat station within 50 km of Tashkent (next nearest is ~71 km away), so it is queried directly — no multi-station interpolation. |
| Solar position + clear-sky irradiance | **REAL, but computed, not measured.** pvlib's SPA solar-position algorithm and the Ineichen clear-sky model, deterministic functions of timestamp + station location. Not a statistical/fitted placeholder, but also not a pyranometer measurement — actual cloud cover (from Meteostat) will reduce real irradiance below the "clear-sky" value on cloudy hours. |
| Physical baseline rail temperature (`rail_temp_physical_c`) | **A standard, generic steady-state heat-balance model** (solar absorption vs. convective + longwave radiative cooling), of a type widely used in pavement/rail surface temperature literature. It is **not a reproduction of the exact formula or fitted coefficients from Hong et al. (2019/2021)** — those papers' precise equations were not available when building this. Treat it as a physically-reasonable stand-in, not a validated replication. See `src/physical_model.py` for the equations and coefficients (solar absorptivity, emissivity, McAdams convection correlation, Swinbank sky temperature) and recalibrate them once real measurements exist. |
| Random Forest & ANN predictions | **Trained on the physical baseline's own output as a proxy target**, because no real measured rail temperature exists yet (see below). Their RMSE/MAE/R² measure only how well each model reproduces the physical formula from weather features — **not real-world prediction accuracy.** Every place these numbers appear (console output, `outputs/test_metrics.csv`, plot titles) is labeled PROXY. |
| Real rail temperature measurements | **Do not exist in this project.** Obtaining them requires contacting Uzbek Railways / Tashkent Metro directly — a separate, unresolved step, not something this codebase can produce. |

**Do not present any RMSE/MAE/R² number from this pipeline as real-world
accuracy in a presentation.** They currently describe agreement with an
unvalidated physical formula, nothing more.

## Project layout

```
src/
  fetch_weather.py    Pulls real hourly Meteostat data for station 38457,
                       adds pvlib solar position + clear-sky irradiance.
  physical_model.py   Heat-balance physical baseline (see caveats above).
  train_models.py     RF + ANN. train_and_evaluate_proxy_target() trains on
                       the physical baseline's own output. THE function to
                       use once real data exists is
                       train_and_evaluate_real_target() — see below.
  plots.py            Plotting helpers.
  pipeline.py          Orchestrates fetch(load) -> physical model -> proxy
                       training -> plots -> results CSV.
data/
  raw/                Fetched real weather CSV(s).
  processed/          Weather + physical baseline merged CSV.
outputs/
  results.csv          Per-timestamp target vs. RF vs. ANN predictions.
  test_metrics.csv     RMSE/MAE/R2 on the held-out (chronological) test set.
  plots/                PNG plots.
```

## How to run

Requires the `python.org` Python 3.14 install with meteostat, pvlib, pandas,
numpy, scikit-learn, xgboost, torch, matplotlib (already set up in this
environment).

```bash
# 1. Fetch real weather data (only needed once, or to change the date range)
python3 src/fetch_weather.py --start 2023-06-01 --end 2023-09-01

# 2. Run the full pipeline: physical baseline -> proxy-trained RF/ANN -> plots
python3 src/pipeline.py
```

Outputs land in `outputs/`: `results.csv`, `test_metrics.csv`, and two PNGs
in `outputs/plots/` (real weather overview, model-vs-target comparison).

## What's needed before this becomes a validated model

1. **Real measured rail temperature data from Tashkent Metro**, ideally
   paired hourly with the weather features already being computed here
   (same time range/resolution as `data/raw/tashkent_weather_*.csv`).
   Requires contacting Uzbek Railways / Tashkent Metro directly.
2. Merge that data into the processed dataframe as a column named
   `rail_temp_measured_c` (indexed by the same hourly timestamps).
3. Call `train_and_evaluate_real_target()` in `src/train_models.py` instead
   of `train_and_evaluate_proxy_target()` — same features, same model
   architectures, same train/test split logic, just a real target. That
   function is the single, clearly-marked entry point for this; nothing
   else in the codebase needs to change.
4. Recalibrate `src/physical_model.py`'s coefficients (absorptivity,
   emissivity, convection correlation) against the real measurements —
   the current values are generic literature defaults, not fitted to
   Tashkent rail steel.
5. Re-run `src/pipeline.py` (or a small variant calling the real-target
   function) and only then treat the reported RMSE/MAE/R² as real-world
   accuracy.

## Known limitations of the physical baseline as currently implemented

- Uses **clear-sky** GHI, not the real cloud-attenuated irradiance — actual
  observed cloud cover from Meteostat (`cloud_cover_okta`) is fetched but
  not yet applied to reduce GHI on cloudy hours. This will make the
  baseline (and hence the proxy targets) run hot on cloudy days.
- Steady-state assumption: no thermal mass/lag in the rail is modeled, so
  predicted rail temperature responds instantly to changes in air temp and
  irradiance, which real rail (with thermal inertia) does not.
- Coefficients (0.80 absorptivity, 0.85 emissivity, McAdams convection) are
  generic literature defaults for weathered steel, not calibrated to
  Tashkent Metro's actual rail material/condition.
