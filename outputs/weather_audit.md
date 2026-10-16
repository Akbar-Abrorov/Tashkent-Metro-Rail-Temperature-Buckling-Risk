# Weather dataset audit
Source: `data/raw/tashkent_weather_2023-06-01_2026-01-01.csv`
Range: 2023-06-01 00:00:00+05:00 to 2026-01-01 00:00:00+05:00 (22681 rows)

## Station representativeness
Single Meteostat station (38457, Tashkent/UTTT, 41.2667N 69.2667E, 489 m elevation) -- the only station within 50 km of Tashkent (next nearest is ~71 km away), so no multi-station interpolation is used or possible. This means the dataset reflects conditions at one point near the city, not necessarily at the rail corridor itself.

## Timestamp continuity
- Expected hourly rows over the range: 22681
- Actual rows: 22681
- Duplicate timestamps: 0
- Missing hours: 0 across 0 gap(s)

## Per-column missingness (NaN fraction)
| column | NaN fraction |
|---|---|
| precip_mm | 0.0203 |

## Physical plausibility violations
None found within the checked ranges.

## Solar-geometry cross-check: clear-sky GHI must be 0 when sun is below horizon
- Night rows (solar_elevation_deg <= 0): 11210
- Violations (nonzero GHI at night): 0

## Unit consistency: wind_speed_ms == wind_speed_kmh / 3.6
- Max absolute discrepancy: 0.000000 m/s
