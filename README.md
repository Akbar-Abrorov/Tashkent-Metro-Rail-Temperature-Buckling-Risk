# Tashkent Metro Rail Temperature / Buckling Risk — Research Pipeline

Predicting rail surface temperature for the Tashkent Metro (Uzbekistan) from
weather + solar position features: a real published physical model, a
secondary hand-rolled physical model used to cross-check it, RF/ANN models
trained to reproduce the physical model's output from weather alone, and a
set of research-style diagnostics (naive-baseline comparison, cross-
validation, sensitivity analysis) meant to make this defensible as more than
a one-off number.

## Where this model's physics is even applicable — read this first

Solar rail-surface heating, the entire premise of this project, is only a
real phenomenon on **exposed** (at-grade or elevated) track. Tashkent
Metro's three original lines — Chilonzor (red), O'zbekiston (blue),
Yunusobod (green) — run entirely in underground tunnels (8–25 m deep); rail
temperature there is governed by the tunnel thermal environment, not solar
heat balance, and **this model does not apply to them at all.** The one
Tashkent Metro line this model's physics is actually relevant to is the
**Circle Line (gold)**, under construction since 2020, designed as a ~52 km
ring with ~50.5 km on elevated overpasses. Because it's a ring, its track
azimuth varies continuously over the full 0–360° range as it curves around
the city — there is no single "real" azimuth to look up for "the Tashkent
Metro track." This project was never told which specific line/segment its
weather station's readings should be paired with, so everything below
should be read as *"what would an exposed Tashkent-area rail segment's
temperature look like,"* not as a validated statement about any specific
currently-operating line. See `tools/railtemp_cnu/run_cnu.py`'s module
docstring for the same point in context, and "Model A sensitivity analysis"
below for how much the unavoidable azimuth guess actually matters.

## What is real, what is proxy/placeholder

| Component | Status |
|---|---|
| Weather data (`data/raw/tashkent_weather_*.csv`) | **REAL.** Observed hourly data from Meteostat station 38457 (WMO 38457 / ICAO UTTT / IATA TAS, "Tashkent", 41.2667N 69.2667E, 489 m elevation), June 2023 – Jan 2026. This is the only Meteostat station within 50 km of Tashkent (next nearest is ~71 km away), so it is queried directly — no multi-station interpolation. See `outputs/weather_audit.md` for a full data-quality audit. |
| Solar position + clear-sky irradiance | **REAL, but computed, not measured.** pvlib's SPA solar-position algorithm and the Ineichen clear-sky model, deterministic functions of timestamp + station location. Not a statistical/fitted placeholder, but also not a pyranometer measurement — actual cloud cover will reduce real irradiance below the "clear-sky" value on cloudy hours. Both physical baselines (Model A and B) use **clear-sky**, not cloud-adjusted, irradiance — see "Known limitations." |
| **Model A — RailTemp/CNU** (`rail_temp_cnu_c`) | **A real, published, third-party model** — [github.com/aryvini/railtemp](https://github.com/aryvini/railtemp), GPL-3.0, a transient (thermal-mass) rail heat-balance model built on Hong et al. (2019, *IJPEM-GT*, DOI 10.1007/s12541-019-00015-1), the same paper this project already cited. Run via a separate script/venv (see below) because the package requires Python <3.14. This is the model RF/ANN are actually trained against. It still rests on several documented assumptions about geometry/orientation (see "Physical models" below and the scope note above) — it is a real model run with assumed, not measured, local parameters. |
| **Model B — steady-state baseline** (`rail_temp_steadystate_c`) | A simpler, hand-rolled, fast steady-state heat-balance model (`src/physical_model.py`), **not** a reproduction of any specific paper's exact formula. Used only to cross-check Model A (see `outputs/physical_model_agreement.csv`) — it is not the ML training target. |
| Random Forest & ANN predictions | **Trained on Model A's (RailTemp/CNU) own output as a proxy target**, because no real measured rail temperature exists yet (see below). Their RMSE/MAE/R² measure only how well each model reproduces Model A's output from weather features on a **held-out chronological test split** — not real-world prediction accuracy. See "Evaluation methodology" below. |
| Real rail temperature measurements | **Do not exist in this project.** See `docs/measurement_feasibility.md` for a concrete assessment of how they could realistically be obtained. |

**Do not present any RMSE/MAE/R² number from this pipeline as real-world
accuracy in a presentation.** They currently describe agreement with an
unvalidated physical model (Model A), run with assumed local parameters,
on a line whose exposed sections don't exist in final form yet.

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
  - **Track azimuth (90°) and ambient emissivity (0.5)** are assumptions.
    Azimuth in particular can't be "looked up" even in principle for this
    corridor — see the scope note above (the only physically-relevant line
    is a ring). 90° is used as an arbitrary representative midpoint; see
    "Model A sensitivity analysis" below for how much this choice matters.
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

## Model A sensitivity analysis

How much do Model A's two unavoidable assumptions (track azimuth, ambient
emissivity) actually matter? `tools/railtemp_cnu/sensitivity.py` (run via
`.venv-railtemp`) sweeps each one-at-a-time around the baseline (azimuth=90°,
ambient_emissivity=0.5) on the 3-month window; `src/plot_sensitivity.py`
renders `outputs/plots/model_a_sensitivity.png` from the resulting
`outputs/model_a_sensitivity.csv`:

| parameter | sweep range | mean abs. deviation | max abs. deviation |
|---|---|---|---|
| track azimuth | 0°–180° | up to 1.23°C | up to 6.01°C |
| ambient emissivity | 0.3–0.9 | up to 0.28°C | up to 1.43°C |

**Azimuth is the dominant source of uncertainty in Model A** — roughly 4x
the effect of ambient emissivity over its plausible range, and consistent
with the scope note above: since the only physically-relevant line is a
ring (azimuth genuinely varies 0–360° along its length), this isn't a
parameter this project could ever pin down to one "correct" value, and the
honest response is to report the resulting range rather than claim a
specific number.

## Is the added complexity worth it? Naive-baseline comparison

The standard question for any ML result: how much better is this than
something trivial? `src/baselines.py` scores two naive baselines on the
**exact same held-out test split** as RF/ANN:

| model | RMSE (°C) | MAE (°C) | R² |
|---|---|---|---|
| persistence (target[t] = target[t-1]) | 2.35 | 1.78 | 0.961 |
| linear (rail_temp = a·air_temp + b) | 3.73 | 3.11 | 0.901 |
| random_forest | 0.97 | 0.67 | 0.993 |
| ann | 1.11 | 0.82 | 0.991 |

Both RF and ANN meaningfully beat both naive baselines, and the gap holds
up across all 5 cross-validation folds below (not just this one split) —
so the added model complexity isn't just curve-fitting noise, it's buying
real accuracy against Model A, for whatever that's worth given everything
above about Model A itself being unvalidated.

## Robustness: rolling-origin cross-validation

The single train/val/test split above is one chronological cut; a
different cut could look better or worse by chance. `src/cross_validate.py`
re-trains RF/ANN/baselines across 5 expanding-window chronological folds
(each with its own internal validation split for ANN early stopping) and
reports mean ± std across folds, written to `outputs/cv_metrics.csv` /
`outputs/cv_summary.csv` / `outputs/plots/cv_metrics.png`:

| model | RMSE mean ± std (°C) | R² mean ± std |
|---|---|---|
| persistence | 2.40 ± 0.28 | 0.933 ± 0.011 |
| linear_airtemp | 3.89 ± 0.95 | 0.828 ± 0.042 |
| random_forest | 1.15 ± 0.42 | 0.985 ± 0.007 |
| ann | 1.34 ± 0.30 | 0.980 ± 0.004 |

RF/ANN beat both naive baselines in every one of the 5 folds (different
seasons, different amounts of training history) — the single-split result
above isn't a lucky cut.

## Diagnostics: feature importance, parity/residuals, error by regime

`src/diagnostics.py` (run automatically as part of `src/pipeline.py`,
always on the test split only):

- **Feature importance** (`outputs/feature_importance.csv`,
  `outputs/plots/feature_importance.png`): `air_temp_c` dominates RF's
  splits (~88%), followed by `clearsky_dni_wm2`/`clearsky_ghi_wm2` (~9%
  combined); `cloud_cover_okta` is ~0%. That near-zero cloud importance is
  **not evidence clouds don't matter physically** — it's because Model A's
  own input is clear-sky irradiance (see "Known limitations"), so an RF
  trained to imitate Model A correctly learns that cloud cover, which Model
  A itself ignores, doesn't predict its output.
- **Parity and residual plots** (`outputs/plots/parity_plots.png`,
  `outputs/plots/residuals.png`): both models track the target closely with
  roughly symmetric, near-unbiased residuals (RF residual mean 0.04°C,
  ANN -0.28°C) across most of the range.
- **Error by temperature regime** (`outputs/test_metrics_by_regime.csv`) —
  overall average error is a weak summary for a buckling-*risk* application;
  what matters is accuracy in the hot tail specifically:

  | regime | model | n | RMSE (°C) | R² |
  |---|---|---|---|---|
  | overall | random_forest | 4536 | 0.97 | 0.993 |
  | overall | ann | 4536 | 1.11 | 0.991 |
  | hottest decile (≥39.1°C) | random_forest | 454 | 0.78 | 0.886 |
  | hottest decile (≥39.1°C) | ann | 454 | 1.26 | 0.706 |
  | ≥40°C | random_forest | 370 | 0.78 | 0.868 |
  | ≥40°C | ann | 370 | 1.25 | 0.662 |

  **Both models degrade substantially in the hot tail** — RF's R² drops from
  0.993 overall to 0.868 above 40°C, and the ANN's drops further, from 0.991
  to 0.662. This is a real, honest finding, not an error: ML models trained
  on mostly-moderate weather data generalize worse at the sparse, extreme
  end of the distribution, which is exactly the regime a buckling-risk
  application cares about most. (An earlier run of this table, on a smaller
  15%-test-split test window with only 101 rows ≥40°C, showed the ANN's R²
  going outright negative in that regime — a real result for that window,
  but also a reminder that a small test subset can be noisy; the 20%-split
  test window here has 370 rows ≥40°C and gives a more stable estimate.
  Either way: don't quote the overall R²≈0.99 as if it describes performance
  where it matters.)

## Evaluation methodology (ANN leakage fix)

An earlier version of this pipeline used the **test set itself** as the ANN's
early-stopping validation set — i.e. model selection saw the data the
"held-out" metrics were later computed on, inflating those metrics. This has
been fixed:

- `chronological_split_3way()` (`src/train_models.py`) splits the data
  **chronologically** (never shuffled — hourly weather is autocorrelated)
  into train (60%) / validation (20%) / test (20%), in that time order, so
  test is always the most recent, truly-unseen block.
- The validation split is a **separate, independent set from test** —
  touched only during training (ANN early stopping / model selection),
  never used to report accuracy. Test is the one and only split used to
  "exam" the finished model.
- `outputs/test_metrics.csv` and every number printed by `src/pipeline.py` as
  "accuracy" come **only** from the test split (now including the two naive
  baselines, for direct comparison — see above). Validation-set numbers are
  never written to that file and never called accuracy.
- (For context: the pre-fix leaked numbers were RMSE≈0.09–0.66 — noticeably
  "better" than today's honest ones, which is exactly the signature of
  test-set leakage.)

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
tools/railtemp_cnu/
  run_cnu.py            Runs the REAL RailTemp/CNU package (separate venv)
                         -> data/processed/railtemp_cnu_baseline_*.csv.
                         Exposes simulate() for reuse by sensitivity.py.
  sensitivity.py         Azimuth/emissivity sweep -> outputs/model_a_sensitivity.csv.
src/
  fetch_weather.py       Pulls real hourly Meteostat data for station 38457,
                          adds pvlib solar position + clear-sky irradiance.
  audit_weather.py       Data-quality audit -> outputs/weather_audit.md.
  physical_model.py      Model B: steady-state heat-balance baseline.
  train_models.py        RF + ANN, chronological train/val/test split, model caching.
  baselines.py            Naive persistence/linear baselines, same test split.
  diagnostics.py           Feature importance, parity/residual plots, error-by-regime.
  cross_validate.py        5-fold rolling-origin CV -> outputs/cv_metrics.csv.
  plot_sensitivity.py      Plots outputs/model_a_sensitivity.csv.
  plots.py                Plotting helpers.
  pipeline.py              Orchestrates: load weather -> Model B -> load cached
                            Model A -> agreement check -> RF/ANN (cached or
                            retrained) -> baselines -> diagnostics -> plots.
data/
  raw/                Fetched real weather CSV(s).
  processed/          Weather + Model B merged CSV; cached Model A output.
outputs/
  models/                          Cached RF/ANN + scalers + manifest.
  results.csv                      Per-timestamp target vs. RF vs. ANN predictions.
  test_metrics.csv                 RF/ANN + naive baselines, held-out TEST split only.
  test_metrics_by_regime.csv       Error broken down by temperature regime.
  physical_model_agreement.csv     Model A vs. Model B agreement metrics.
  feature_importance.csv           RF feature importances.
  cv_metrics.csv / cv_summary.csv  Rolling-origin cross-validation results.
  model_a_sensitivity.csv          Model A azimuth/emissivity sensitivity sweep.
  weather_audit.md                 Weather data-quality audit report.
  plots/                           PNG plots.
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

# 3. Run the full pipeline (RF/ANN + baselines + diagnostics; loads cached models by default)
python3 src/pipeline.py          # or: python3 src/pipeline.py --retrain

# 4. Audit the weather dataset
python3 src/audit_weather.py

# 5. (Optional, slower) Robustness + sensitivity checks
python3 src/cross_validate.py                                    # ~25s, 5 folds
.venv-railtemp/bin/python tools/railtemp_cnu/sensitivity.py       # ~25s, 9 CNU runs
python3 src/plot_sensitivity.py
```

Outputs land in `outputs/` (see "Project layout" above for the full list).

## What's needed before this becomes a validated model

1. **Real measured rail temperature data from Tashkent Metro** — see
   `docs/measurement_feasibility.md` for the concrete options assessed
   (contacting the operator, a DIY sensor pilot, why satellite data isn't
   good enough) and the recommended path. Still unresolved. Note this is
   doubly true now given the scope note above: measurements would need to
   come from the Circle Line's exposed sections specifically, not from any
   currently-operating (underground) line.
2. Merge that data into the processed dataframe as a column named
   `rail_temp_measured_c` (indexed by the same hourly timestamps).
3. Call `train_and_evaluate_real_target()` in `src/train_models.py` instead
   of `train_and_evaluate_proxy_target()` — same features, same chronological
   split, same model architectures, just a real target.
4. Recalibrate Model A's assumed parameters (rail profile, azimuth, ambient
   emissivity — see "Physical models" and "Model A sensitivity analysis"
   above) against the real measurements and the real segment's actual
   orientation.
5. Re-run `src/pipeline.py --retrain` and only then treat the reported
   RMSE/MAE/R² as real-world accuracy — and still look at the by-regime
   breakdown, not just the overall number (see "Diagnostics" above).

## Known limitations

- **Scope**: Model A/B physics only applies to exposed track; Tashkent
  Metro's three currently-operating lines are entirely underground, and the
  one line it's relevant to (the Circle Line) isn't fully built and is a
  ring with no single azimuth — see the top of this README.
- Model A's local parameters (rail profile, track azimuth, ambient
  emissivity) are documented assumptions, not measured — see "Physical
  models" and "Model A sensitivity analysis" (azimuth in particular can
  shift Model A's output by up to ~6°C at the extremes tested).
- Both Model A and Model B use **clear-sky**, not cloud-adjusted, solar
  irradiance — real cloud cover (which the weather data does include,
  `cloud_cover_okta`) isn't applied to reduce irradiance on cloudy hours.
  This is directly visible in the RF feature-importance result (cloud cover
  ≈0% importance) — see "Diagnostics" above.
- Model B (the secondary baseline) has no thermal mass/lag — predicted rail
  temperature responds instantly to weather changes, which real rail does
  not. Part of why Model A and Model B disagree; Model B is not used as the
  ML target specifically because of this.
- **RF/ANN accuracy degrades substantially in the hot tail** (ANN R² drops
  to 0.66 above 40°C, RF's to 0.87, vs. ~0.99 overall) — see "Diagnostics"
  above. Don't use the overall R²≈0.99 to characterize performance in the
  regime that actually matters for buckling risk.
- The weather dataset itself is high quality (0 timestamp gaps/duplicates
  over the full 2.5-year range, only `precip_mm` has missing values at 2%) —
  see `outputs/weather_audit.md` — but it is still a single station, so it
  reflects conditions near the city generally, not necessarily at any
  specific rail corridor.
