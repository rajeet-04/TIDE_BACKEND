# UTide Tide-Event Model Results

## Model

The production model is `src/utide_event_model.py`.

- UTide harmonic analysis with `68` constituents per port
- Astronomical nodal corrections for lunar and solar tidal cycles
- Historical-only Extra Trees residual calibration
- Explicit synodic lunar-phase features for full-moon and new-moon effects
- Event-context features: neighbouring tide ranges, intervals, annual phase,
  time of day, and raw UTide height

The model was trained from:

- `data/haldia.csv`: hourly values from `2000-01-01` to `2024-12-31`
- `data/diamond_harbour.csv`: hourly values from `2000-01-01` to `2023-04-15`

The April, May, and June 2026 PDF rows are reserved for validation only. They
are tide-table reference predictions, not live water-level gauge observations.

## Accuracy Definition

An event is correct when:

- high/low tide timing error is at most `30 minutes`
- tide amplitude error is at most `0.30 metres`

Joint accuracy requires both conditions to hold for the same event.

## Reserved 2026 Validation

| Port | Events | Time MAE | Height MAE | Timing accuracy | Height accuracy | Joint accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Haldia | 352 | 14.98 min | 0.130 m | 97.44% | 99.72% | 97.16% |
| Diamond Harbour | 352 | 10.51 min | 0.106 m | 98.01% | 100.00% | 98.01% |

Both ports pass the `>=95%` joint accuracy gate against all extracted
April-June 2026 PDF tide events.

## Reproduce

```powershell
uv run python src/extract_reference_events.py `
  --data-dir data `
  --out data/reference_tides_2026.csv

uv run python src/utide_event_model.py train

uv run python src/utide_event_model.py validate
```

Generate a 24-hour Haldia forecast:

```powershell
uv run python src/utide_event_model.py forecast `
  --port haldia `
  --start 2026-06-01 `
  --hours 24 `
  --out-dir output/utide_forecast_haldia_2026-06-01
```

## Haldia Smoke Test: 2026-06-01

| State | Predicted time (IST) | Predicted height |
| --- | --- | ---: |
| Low | 05:41 | 1.353 m |
| High | 10:54 | 5.703 m |
| Low | 18:03 | 1.556 m |
| High | 23:12 | 5.177 m |

The June 2026 Haldia PDF reference values are:

| State | PDF time (IST) | PDF height |
| --- | --- | ---: |
| Low | 06:01 | 1.25 m |
| High | 11:04 | 5.47 m |
| Low | 18:16 | 1.51 m |
| High | 23:18 | 4.95 m |

## Generated Files

- `data/reference_tides_2026.csv`
- `output/utide_models/haldia.pkl`
- `output/utide_models/diamond_harbour.pkl`
- `output/utide_validation_2026/metrics.json`
- `output/utide_validation_2026/event_errors.csv`
- `output/utide_validation_2026/haldia_event_errors.png`
- `output/utide_validation_2026/diamond_harbour_event_errors.png`
- `output/utide_forecast_haldia_2026-06-01/hourly_water_levels.csv`
- `output/utide_forecast_haldia_2026-06-01/predicted_tide_events.csv`
- `output/utide_forecast_haldia_2026-06-01/tide_frequency.csv`
- `output/utide_forecast_haldia_2026-06-01/water_level_forecast.png`

## Environmental Limitation

The provided historical CSVs and 2026 PDF tide tables do not contain measured
weather, pressure, river discharge, wind, or observed surge data. The model can
learn astronomical and recurring seasonal effects, but it cannot claim to
forecast storm surge or discharge anomalies without those external inputs.

Forecast CSVs include both IST and UTC timestamps. IST is the primary local
time convention; UTC is exactly `5 hours 30 minutes` earlier.
