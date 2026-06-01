# Generate Tide Forecast Data

Run commands from the project root:

```powershell
cd "R:\Code\TIDE BACKEND"
```

## 1. Install Dependencies

Use the project `uv` environment:

```powershell
uv sync
```

## 2. Generate Tide Data

The trained model files are already stored in:

```text
output/utide_models/haldia.pkl
output/utide_models/diamond_harbour.pkl
```

Generate a Haldia forecast:

```powershell
uv run python src/utide_event_model.py forecast `
  --port haldia `
  --start 2026-06-01 `
  --hours 72
```

Generate a Diamond Harbour forecast:

```powershell
uv run python src/utide_event_model.py forecast `
  --port diamond_harbour `
  --start 2026-06-01 `
  --hours 72
```

Change:

- `--port` to `haldia` or `diamond_harbour`
- `--start` to the required date or date-time in ISO format
- `--hours` to the required forecast duration

Examples of valid start values:

```text
2026-06-01
2026-06-01T06:00:00
```

## 3. Forecast Output Files

By default, the command creates:

```text
output/utide_forecast_<port>_<date>/
```

The directory contains:

| File | Purpose |
| --- | --- |
| `hourly_water_levels.csv` | Predicted water-level height for every hour in IST and UTC |
| `predicted_tide_events.csv` | Predicted high/low tide times in IST and UTC, plus amplitudes |
| `tide_frequency.csv` | Number of high/low tides, average intervals, and event times in IST and UTC |
| `water_level_forecast.png` | Graphical 24-hour or multi-hour tide forecast |

## Time Zones

The model is trained from Indian tide tables, and `--start` is interpreted as
Indian Standard Time (IST, UTC+05:30).

Forecast CSVs include both:

- `datetime_ist`: local Indian time
- `datetime_utc`: UTC time, exactly `5 hours 30 minutes` earlier

Use `datetime_ist` when comparing against Indian local tide tables. Some
websites or browser settings display UTC, so check the time-zone label before
comparing tide events.

Use a custom output directory when needed:

```powershell
uv run python src/utide_event_model.py forecast `
  --port haldia `
  --start 2026-06-01 `
  --hours 24 `
  --out-dir output/my_haldia_forecast
```

## 4. Retrain the Model

Retrain after updating either historical CSV:

```text
data/haldia.csv
data/diamond_harbour.csv
```

Train both ports:

```powershell
uv run python src/utide_event_model.py train
```

Train one port only:

```powershell
uv run python src/utide_event_model.py train --port haldia
```

```powershell
uv run python src/utide_event_model.py train --port diamond_harbour
```

Training overwrites the corresponding model file under:

```text
output/utide_models/
```

## 5. Validate Against April-June 2026 PDFs

Extract reference high/low tide events from the PDFs:

```powershell
uv run python src/extract_reference_events.py `
  --data-dir data `
  --out data/reference_tides_2026.csv
```

Run validation:

```powershell
uv run python src/utide_event_model.py validate
```

Validation output is written to:

```text
output/utide_validation_2026/
```

Important files:

| File | Purpose |
| --- | --- |
| `metrics.json` | Accuracy metrics for both ports |
| `event_errors.csv` | Event-by-event timing and amplitude errors |
| `haldia_event_errors.png` | Haldia graphical errors |
| `diamond_harbour_event_errors.png` | Diamond Harbour graphical errors |

## Model Summary

This is a trained hybrid model, not hardcoded future tide data.

It uses:

- UTide harmonic analysis with astronomical nodal corrections
- Historical hourly tide data for both ports
- Lunar-phase features for full-moon and new-moon effects
- Extra Trees calibration trained from historical prediction errors

The April-June 2026 PDFs are used only for validation, not as forecast inputs.
They are reference tide-table predictions, not measurements from a live water
level gauge. Real-world operational accuracy requires gauge observations and,
for environmental anomalies, pressure, wind, river discharge, and surge data.
