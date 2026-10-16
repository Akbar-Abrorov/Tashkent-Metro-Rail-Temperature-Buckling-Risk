# Tashkent Metro Rail Temperature / Buckling Risk — Research Pipeline

Predicting rail surface temperature for the Tashkent Metro (Uzbekistan) from
weather + solar position features: a real published physical model, a
secondary hand-rolled physical model used to cross-check it, and RF/ANN
models trained to reproduce the physical model's output from weather alone.

## What is real, what is proxy/placeholder — read this first

| Component | Status |
|---|---|
| Weather data (`data/raw/tashkent_weather_*.csv`) | **REAL.** Observed hourly data from Meteostat station 38457 (WMO 38457 / ICAO UTTT / IATA TAS, "Tashkent", 41.2667N 69.2667E, 489 m elevation), June 2023 – Jan 2026. This is the only Meteostat station within 50 km of Tashkent (next nearest is ~71 km away), so it is queried directly — no multi-station interpolation. See `outputs/weather_audit.md` for a full data-quality audit. |
| Solar position + clear-sky irradiance | **REAL, but computed, not measured.** pvlib's SPA solar-position algorithm and the Ineichen clear-sky model, deterministic functions of timestamp + station location. Not a statistical/fitted placeholder, but also not a pyranometer measurement — actual cloud cover will reduce real irradiance below the "clear-sky" value on cloudy hours. |
| **Model A — RailTemp/CNU** (`rail_temp_cnu_c`) | **A real, published, third-party model** — [github.com/aryvini/railtemp](https://github.com/aryvini/railtemp), GPL-3.0, a transient (thermal-mass) rail heat-balance model built on Hong et al. (2019, *IJPEM-GT*, DOI 10.1007/s12541-019-00015-1), the same paper this project already cited. Run via a separate script/venv (see below) because the package requires Python <3.14. This is the model RF/ANN are actually trained against. It still rests on several documented assumptions about this specific corridor (see "Physical models" below) — it is a real model run with assumed, not measured, local parameters. |
| **Model B — steady-state baseline** (`rail_temp_steadystate_c`) | A simpler, hand-rolled, fast steady-state heat-balance model (`src/physical_model.py`), **not** a reproduction of any specific paper's exact formula. Used only to cross-check Model A (see `outputs/physical_model_agreement.csv`) — it is not the ML training target. |
| Random Forest & ANN predictions | **Trained on Model A's (RailTemp/CNU) own output as a proxy target**, because no real measured rail temperature exists yet (see below). Their RMSE/MAE/R² measure only how well each model reproduces Model A's output from weather features on a **held-out chronological test split** — not real-world prediction accuracy. See "Evaluation methodology" below for how the split works and why it matters. |
| Real rail temperature measurements | **Do not exist in this project.** See `docs/measurement_feasibility.md` for a concrete assessment of how they could realistically be obtained. |

**Do not present any RMSE/MAE/R² number from this pipeline as real-world
accuracy in a presentation.** They currently describe agreement with an
unvalidated physical model (Model A), run with assumed local parameters,
nothing more.

## Physical models

Two independent physical models are computed; they are not interchangeable
and serve different roles:

- **Model A — RailTemp/CNU** (primary; the one RF/ANN are trained against).
  Run via `tools/railtemp_cnu/run_cnu.py`, the actual published package, in
  a **separate venv** (`.venv-railtemp`, Python 3.13 — the package requires
  <3.14, this project's main venv is 3.14). Its output is cached as
  `data/processed/railtemp_cnu_baseline_<start>_<end>.csv` and consumed by
  the main pipeline; the main 3.14 stack never imports `railtemp` directly.
  Assumptions baked into this run, all documented in the script itself:
  - **Rail profile: UIC60** (bundled with the package). The actual profile
    used on Tashkent Metro track is unknown to this project; UIC60 is the
    closest widely-used standard. `cross_area`/`convection_area`/
    `radiation_area` are computed directly from the UIC60 cross-section
    coordinates (polygon area / perimeter), not guessed constants.
  - **Track azimuth (90°, east-west) and ambient emissivity (0.5)** are
    assumptions — the real track orientation and sky-view factor for this
    corridor are unknown.
  - **Solar input is `clearsky_dni_wm2` (direct normal irradiance), not
    GHI** — the model's sun-exposed area is already the beam-projected
    area, so using GHI would double-count the elevation-angle geometry.
    This is an easy mistake to make silently; it's called out here on
    purpose.
  - Regenerate it (after `python3 src/fetch_weather.py`):
    ```bash
    python3.13 -m venv .venv-railtemp
    .venv-railtemp/bin/pip install "railtemp @ git+https://github.com/aryvini/railtemp.git" pytz
    .venv-railtemp/bin/python tools/railtemp_cnu/run_cnu.py data/raw/tashkent_weather_<start>_<end>.csv
    ```
    (In practice this runs in ~1s per 2,200 rows / ~25s for the full 2.5-year,
    22,681-row file — it scales fine despite the per-timestep Python loop.)
- **Model B — steady-state baseline** (`src/physical_model.py`). A generic
  solar-absorption-vs-convective/radiative-cooling steady-state model, no
  thermal mass/lag. Computed by `src/pipeline.py` directly (fast, vectorized,
  no extra venv needed). Used only as a cross-check against Model A over
  their full overlap — see `outputs/physical_model_agreement.csv` /
  `outputs/plots/physical_model_agreement.png`. Current agreement on the full
  2023-06-01–2026-01-01 range: RMSE≈9.5°C, bias≈+1.4°C (B runs hotter than
  A), R²≈0.40 — the two models broadly track each other but disagree
  substantially in magnitude, consistent with B's missing thermal mass/lag
  and B's different (simpler) convection/radiation treatment. This is
  reported, not hidden, as evidence of how much the fast surrogate diverges
  from the real published model.

## Evaluation methodology (ANN leakage fix)

An earlier version of this pipeline used the **test set itself** as the ANN's
early-stopping validation set — i.e. model selection saw the data the
"held-out" metrics were later computed on, inflating those metrics. This has
been fixed:

- `chronological_split_3way()` (`src/train_models.py`) splits the data
  **chronologically** (never shuffled — hourly weather is autocorrelated)
  into train (70%) / validation (15%) / test (15%), in that time order, so
  test is always the most recent, truly-unseen block.
- The validation split is used **only** for ANN early stopping.
- `outputs/test_metrics.csv` and every number printed by `src/pipeline.py` as
  "accuracy" come **only** from the test split. Validation-set numbers are
  never written to that file and never called accuracy.
- Current held-out test metrics (RF: n_estimators=500; ANN:
  10→128→64→32→1 with dropout — see "Model size" below), against Model A:

  | model | RMSE (°C) | MAE (°C) | R² |
  |---|---|---|---|
  | random_forest | 1.09 | 0.75 | 0.990 |
  | ann | 1.22 | 0.94 | 0.988 |

  (For comparison, the pre-fix leaked numbers were RMSE≈0.09–0.66 — i.e.
  noticeably better than these honest ones, which is exactly the signature
  of test-set leakage.)

## Model size and caching

- `RailTempANN` (`src/train_models.py`) is now `10 → 128 → 64 → 32 → 1`
  (ReLU + dropout 0.1 on the two widest layers), roughly 10x the parameters
  of the original `10→32→16→1` network. Random Forest is `n_estimators=500`
  (up from 300).
- Trained models are cached in `outputs/models/` (`rf_model.joblib`,
  `ann_model.pt`, scalers, `manifest.json`). **`python3 src/pipeline.py`
  loads the cache by default** and only recomputes held-out metrics — it
  does not retrain. Pass `--retrain` to force retraining (e.g. after
  changing `FEATURE_COLUMNS` or the target column; the manifest check also
  catches this automatically and retrains if the cache doesn't match).

## Project layout

```
tools/
  railtemp_cnu/run_cnu.py   Runs the REAL RailTemp/CNU package (separate
                            venv, see above) -> data/processed/railtemp_cnu_baseline_*.csv
src/
  fetch_weather.py    Pulls real hourly Meteostat data for station 38457,
                      adds pvlib solar position + clear-sky irradiance.
  audit_weather.py    Data-quality audit -> outputs/weather_audit.md.
  physical_model.py   Model B: steady-state heat-balance baseline (see caveats above).
  train_models.py     RF + ANN, chronological train/val/test split, model
                      caching. train_and_evaluate_proxy_target() trains on
                      Model A's own output. train_and_evaluate_real_target()
                      is the function to use once real data exists.
  plots.py            Plotting helpers.
  pipeline.py          Orchestrates: load weather -> Model B -> load cached
                        Model A -> physical-model agreement check -> RF/ANN
                        (cached or retrained) -> plots -> results CSV.
data/
  raw/                Fetched real weather CSV(s).
  processed/          Weather + Model B merged CSV; cached Model A output.
outputs/
  models/              Cached RF/ANN + scalers + manifest.
  results.csv          Per-timestamp target vs. RF vs. ANN predictions.
  test_metrics.csv     RMSE/MAE/R2 on the held-out (chronological) TEST split only.
  physical_model_agreement.csv   Model A vs. Model B agreement metrics.
  weather_audit.md     Weather data-quality audit report.
  plots/                PNG plots.
docs/
  measurement_feasibility.md   Assessment of how real measurements could be obtained.
```

## How to run

Main pipeline (Python 3.14 venv, as already set up):

```bash
# 1. Fetch real weather data (only needed once, or to change the date range)
python3 src/fetch_weather.py --start 2023-06-01 --end 2023-09-01

# 2. Generate Model A (RailTemp/CNU) -- separate venv, see "Physical models" above
.venv-railtemp/bin/python tools/railtemp_cnu/run_cnu.py data/raw/tashkent_weather_<start>_<end>.csv

# 3. Run the full pipeline (loads cached RF/ANN models by default)
python3 src/pipeline.py          # or: python3 src/pipeline.py --retrain

# 4. Audit the weather dataset
python3 src/audit_weather.py
```

Outputs land in `outputs/`: `results.csv`, `test_metrics.csv`,
`physical_model_agreement.csv`, `weather_audit.md`, cached models in
`outputs/models/`, and plots in `outputs/plots/`.

## What's needed before this becomes a validated model

1. **Real measured rail temperature data from Tashkent Metro** — see
   `docs/measurement_feasibility.md` for the concrete options assessed
   (contacting the operator, a DIY sensor pilot, why satellite data isn't
   good enough) and the recommended path. Still unresolved.
2. Merge that data into the processed dataframe as a column named
   `rail_temp_measured_c` (indexed by the same hourly timestamps).
3. Call `train_and_evaluate_real_target()` in `src/train_models.py` instead
   of `train_and_evaluate_proxy_target()` — same features, same chronological
   split, same model architectures, just a real target.
4. Recalibrate Model A's assumed parameters (rail profile, azimuth, ambient
   emissivity — see "Physical models" above) against the real measurements.
5. Re-run `src/pipeline.py --retrain` and only then treat the reported
   RMSE/MAE/R² as real-world accuracy.

## Known limitations

- Model A's local parameters (rail profile, track azimuth, ambient
  emissivity) are documented assumptions, not measured for this corridor
  (see "Physical models" above).
- Model A only models direct-beam solar absorption (via the profile's
  sun-facing area); diffuse sky irradiance is not included in its energy
  balance.
- Model B (the secondary baseline) has no thermal mass/lag — predicted rail
  temperature responds instantly to weather changes, which real rail does
  not. This is part of why Model A and Model B disagree (see "Physical
  models" above); Model B is not used as the ML target specifically because
  of this.
- The weather dataset itself is high quality (0 timestamp gaps/duplicates
  over the full 2.5-year range, only `precip_mm` has missing values at 2%) —
  see `outputs/weather_audit.md` — but it is still a single station, so it
  reflects conditions near the city generally, not necessarily at the rail
  corridor specifically.
