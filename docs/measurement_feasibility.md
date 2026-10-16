# Measurement feasibility: how could real Tashkent Metro rail temperature actually be obtained?

Every number this project currently produces (`outputs/results.csv`,
`outputs/test_metrics.csv`, both physical-model plots) is derived from weather
data run through a physical formula -- never from a real rail-temperature
sensor. This document assesses the realistic options for closing that gap,
their feasibility, and what each would and wouldn't actually validate.

## Option A: Request existing instrumentation/logs from the operator

Ask Uzbekiston Temir Yo'llari (national rail operator) and/or Tashkent Metro
directly whether rail temperature is already instrumented anywhere on their
network (continuously welded rail buckling-risk monitoring is common enough
internationally that some agencies already log it).

- **Feasibility: Medium, slow.** Requires identifying the right department
  (track maintenance / infrastructure engineering, not public affairs),
  likely a written request, and an unknown approval/response timeline --
  realistically weeks to a few months, if a response comes at all. No
  existing relationship with the operator exists today.
- **Data-sharing risk.** Even if measurements exist, infrastructure operators
  often treat track condition data as sensitive (safety/security
  classification), so sharing is not guaranteed even with a response.
- **If it works:** this is the only option that gives real, in-situ,
  continuous measurements directly comparable to this project's hourly
  weather features -- strictly better than anything else below.

## Option B: DIY sensor pilot on an accessible rail segment

Attach a contact thermocouple/RTD + a small logger (or take periodic
handheld IR-thermometer spot readings) on a short, publicly accessible
above-ground rail segment (e.g. near a station, depot, or level crossing).

- **Feasibility: Medium.** Equipment is cheap (a logging thermocouple kit is
  on the order of tens of USD; an IR thermometer is similarly cheap), and
  the method itself is simple. The real constraint is **physical/legal
  access** to track infrastructure -- rail corridors are normally restricted
  and attaching anything to the rail itself without the operator's
  permission is both a safety hazard and almost certainly prohibited. This
  realistically requires at least informal cooperation from track
  maintenance staff, even for a short pilot.
- **What it would validate:** a point measurement at one location, useful as
  a sanity check against Model A's output at that location/time, but not a
  substitute for continuous network-wide data.
- **Recommended pilot scope:** a few days of handheld IR spot-checks at a
  couple of times a day (peak solar heating + early morning minimum) is the
  lowest-effort, lowest-risk way to get *some* real data point to compare
  against Model A's predictions for the same hours.

## Option C: Satellite-derived land surface temperature as a proxy

Use thermal-infrared satellite products -- Landsat 8/9 TIRS (~100 m thermal
resolution, resampled to 30 m) or NASA ECOSTRESS (~70 m) -- to estimate
surface temperature along the rail corridor from space.

- **Feasibility: High to obtain, low to be useful here.** Both are free and
  programmatically accessible (USGS EarthExplorer / NASA Earthdata). But:
  - **Spatial resolution** (30-100 m) is far coarser than a rail (standard
    gauge is 1.5 m); a pixel over the corridor is a mix of rail, ballast,
    surrounding ground, and possibly vegetation/structures -- it measures
    "the area," not the rail itself.
  - **Temporal resolution**: 1-2 overpasses per day at fixed times, no
    continuous hourly coverage, and optical/thermal sensors give no
    nighttime coverage from the same instrument family used here.
  - Rail steel heats and cools much faster than the ballast/ground around
    it, so even a perfectly-georeferenced pixel would systematically
    under- or over-estimate peak rail temperature.
- **Verdict:** usable as a very rough independent sanity check on relative
  spatial/seasonal patterns, but not as a validation source for an
  hourly point model like the one in this project.

## Option D: Proxy from adjacent municipal infrastructure sensors

Check whether any nearby paved road, bridge, or municipal weather-station
network already logs surface (not just air) temperature that could serve as
a loose proxy.

- **Feasibility: Low/speculative.** No such network is known to exist for
  this corridor; this would require local inquiry with the city/road
  authority with no guarantee anything useful exists. Even if it did, road
  pavement and rail steel have different absorptivity/emissivity/thermal
  mass, so it would only be a weak qualitative proxy at best.

## Recommended practical path

1. **Start Option A in parallel with everything else** -- it's the only path
   to real, validating data, so the earlier the outreach starts, the sooner
   (if ever) a real answer comes back. Treat it as a long-running background
   task, not a blocker for anything else in this project.
2. **Run Option B's cheap pilot** (a few days of handheld IR spot-checks) as
   soon as any track-side access can be arranged, purely to get *a* real data
   point against which Model A's output can be sanity-checked, even
   informally.
3. **Do not use Option C or D as validation** -- satellite data is useful
   context (e.g., confirming regional heat-wave timing) but should not be
   presented as validating or calibrating the rail-temperature models in this
   project; the spatial/temporal mismatch is too large.
4. Until Option A or B produces real data, every accuracy number in this
   repository must keep carrying the PROXY label it already has (see
   README.md) -- this document does not change that, it only describes the
   path toward eventually removing it.
