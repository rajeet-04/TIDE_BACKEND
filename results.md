# Tide Forecast Results: Haldia, 2026-06-01

## Completed outputs

- [x] Hour-wise water-level height calculation for the next 24 hours
- [x] High- and low-tide time prediction
- [x] High- and low-tide height calculation
- [x] Frequency and timing summary for high and low tides
- [x] Graphical 24-hour water-level forecast
- [x] Graphical high- and low-tide timing errors
- [x] Graphical water-level height errors

## Reproduce the report

```powershell
uv run python src/tide_report.py `
  --forecast output/forecast.csv `
  --start 2026-06-01 `
  --observed-events data/reference_tides_haldia_2026-06-01.csv `
  --out-dir output/tide_report_haldia_2026-06-01
```

## Predicted high and low tides

| State | Predicted time (IST) | Predicted height |
| --- | --- | --- |
| High | 00:00 | 5.032 m |
| Low | 06:11 | 1.663 m |
| High | 11:31 | 5.597 m |
| Low | 18:34 | 1.625 m |
| High | 23:41 | 5.150 m |

The first `00:00` high tide is a boundary estimate because the input forecast
starts at `00:00`. Generate at least one forecast hour before the report start
when an interpolated first turning point is required.

## Frequency in the next 24 hours

| State | Count | Average interval |
| --- | --- | --- |
| High | 3 | 11.85 hours |
| Low | 2 | 12.38 hours |

## Error summary

Errors were calculated against the four supplied Haldia reference tide events:

| Metric | Value |
| --- | --- |
| Matched events | 4 |
| Mean absolute time error | 328.36 minutes |
| Time RMSE | 328.74 minutes |
| Mean absolute height error | 0.504 m |
| Height RMSE | 0.564 m |
| Height bias | +0.455 m |

## Generated files

- `output/tide_report_haldia_2026-06-01/hourly_water_levels.csv`
- `output/tide_report_haldia_2026-06-01/predicted_tide_events.csv`
- `output/tide_report_haldia_2026-06-01/tide_frequency.csv`
- `output/tide_report_haldia_2026-06-01/tide_event_errors.csv`
- `output/tide_report_haldia_2026-06-01/error_metrics.json`
- `output/tide_report_haldia_2026-06-01/water_level_forecast.png`
- `output/tide_report_haldia_2026-06-01/tide_event_errors.png`
