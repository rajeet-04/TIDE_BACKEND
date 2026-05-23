# TIDE BACKEND — Quick Start Guide

**Status**: ✅ Production-ready  
**Last Updated**: May 23, 2026

---

## Installation (One-time)

```bash
cd r:\Code\TIDE BACKEND
uv sync
```

This installs all dependencies including PyTorch with CUDA 12.8 support (auto-falls-back to CPU if GPU unavailable).

---

## Run Complete Pipeline

### Option 1: Run All Phases (Fresh Start)
```bash
# Phase 1: Extract PDFs
uv run python src/extract_pdfs.py

# Phase 2-5: LSM Training
uv run python src/train_predict.py

# Phase 6: LSTM Training (GPU)
uv run python src/residual_lstm.py

# Phase 7: Generate Plots
uv run python src/plot_results.py
uv run python src/plot_lstm.py
```

### Option 2: Skip to Forecasting (Models Already Trained)
```bash
# Generate forecasts directly
uv run python src/forecast_combined.py --port haldia --start 2025-01-01 --hours 72 --mode forecast
```

---

## Generate Forecasts

### LSM-Only (Deterministic, Far-Future Capable)
```bash
# 30-day forecast for Haldia
uv run python src/forecast.py --port haldia --start 2026-01-01 --days 30 \
    --out output/jan_2026_haldia.csv

# 30-day forecast for Diamond Harbour
uv run python src/forecast.py --port diamond_harbour --start 2026-01-01 --days 30 \
    --out output/jan_2026_diamond_harbour.csv
```

### LSM+LSTM Hindcast (Most Accurate, Uses Observed History)
```bash
# 7-day hindcast for Haldia (Dec 2024)
uv run python src/forecast_combined.py --port haldia \
    --start 2024-12-01 --days 7 --mode hindcast \
    --out output/hindcast_haldia_dec2024.csv

# 7-day hindcast for Diamond Harbour (Mar 2023)
uv run python src/forecast_combined.py --port diamond_harbour \
    --start 2023-03-01 --days 7 --mode hindcast \
    --out output/hindcast_dh_mar2023.csv
```

### LSM+LSTM Forecast (Autoregressive, 6-72 Hour Horizons)
```bash
# 72-hour forecast for Haldia (next 3 days)
uv run python src/forecast_combined.py --port haldia \
    --start 2025-01-01 --hours 72 --mode forecast \
    --out output/forecast_haldia_72h.csv

# 48-hour forecast for Diamond Harbour
uv run python src/forecast_combined.py --port diamond_harbour \
    --start 2025-01-01 --hours 48 --mode forecast \
    --out output/forecast_dh_48h.csv
```

---

## Validate Model Performance

### Test on Historical Data
```bash
# Haldia: December 2024 (post-monsoon)
uv run python src/validate_forecast.py --port haldia --start 2024-12-01 --days 7

# Haldia: September 2024 (monsoon season)
uv run python src/validate_forecast.py --port haldia --start 2024-09-01 --days 14

# Diamond Harbour: February 2023 (dry season)
uv run python src/validate_forecast.py --port diamond_harbour --start 2023-02-01 --days 14
```

### Generate Validation Plots
```bash
uv run python src/plot_validation.py
```

This generates:
- `output/validation_comparison.png` — Forecast vs observations across 5 periods
- `output/horizon_degradation.png` — Accuracy by forecast horizon
- `output/error_distribution.png` — Error distribution boxplots

---

## View Results

### Check Test-Set Metrics
```bash
# View LSTM test-set performance
type output\lstm_summary.json

# View LSM metrics and tidal datums
type output\summary.json
```

### View Harmonic Constants
```bash
# LSM coefficients for Haldia
type output\harmonic_constants_haldia.csv

# LSM coefficients for Diamond Harbour
type output\harmonic_constants_diamond_harbour.csv
```

### View Validation Results
```bash
# List all validation files
dir output\validation_forecast_*.csv

# View specific validation
type output\validation_forecast_haldia_2024-12-01_to_2024-12-08.csv
```

---

## Common Use Cases

### Use Case 1: Generate Next 72-Hour Forecast
```bash
# Forecast for Haldia (autoregressive mode)
uv run python src/forecast_combined.py --port haldia \
    --start 2025-01-15 --hours 72 --mode forecast \
    --out output/forecast_next_72h.csv

# View results
type output\forecast_next_72h.csv
```

### Use Case 2: Hindcast Historical Period
```bash
# Hindcast for Haldia (using observed history)
uv run python src/forecast_combined.py --port haldia \
    --start 2024-12-15 --days 7 --mode hindcast \
    --out output/hindcast_dec15.csv

# Compare with observations
type output\hindcast_dec15.csv
```

### Use Case 3: Validate Model on New Date Range
```bash
# Test model on a new period
uv run python src/validate_forecast.py --port haldia --start 2024-11-01 --days 14

# View validation results
type output\validation_forecast_haldia_2024-11-01_to_2024-11-15.csv
```

### Use Case 4: Generate Long-Term Forecast
```bash
# 1-year forecast (LSM-only, deterministic)
uv run python src/forecast.py --port haldia --start 2026-01-01 --days 365 \
    --out output/forecast_2026_full_year.csv

# View first 100 rows
powershell -Command "Get-Content output\forecast_2026_full_year.csv -Head 100"
```

---

## Output Files

### Forecast Files
- `output/forecast_*.csv` — LSM-only forecasts
- `output/hindcast_*.csv` — LSM+LSTM hindcasts
- `output/validation_forecast_*.csv` — Validation results

### Model Files
- `output/lstm_models/haldia.pt` — Trained LSTM weights
- `output/lstm_models/diamond_harbour.pt` — Trained LSTM weights

### Metrics Files
- `output/lstm_summary.json` — LSTM test-set metrics
- `output/summary.json` — LSM metrics and datums

### Plot Files
- `output/plots/haldia_*.png` — LSM diagnostic plots
- `output/plots/diamond_harbour_*.png` — LSM diagnostic plots
- `output/validation_comparison.png` — Validation comparison
- `output/horizon_degradation.png` — Horizon accuracy
- `output/error_distribution.png` — Error distributions

---

## Troubleshooting

### Issue: "No GPU found"
**Solution**: The system will automatically fall back to CPU. GPU is optional but recommended for faster training.

### Issue: "No observations in range"
**Solution**: Check that the date range has data:
- Haldia: 2000-01-01 to 2024-12-31
- Diamond Harbour: 2000-01-01 to 2023-04-15

### Issue: "Need 168 hours of context"
**Solution**: LSTM requires 168 hours (1 week) of history before the forecast start date. Use a later start date.

### Issue: "ModuleNotFoundError"
**Solution**: Run `uv sync` to install dependencies, then try again.

---

## Performance Expectations

### Test-Set Accuracy (Held-out 10% of data)
| Port            | Method      | RMSE  | R      |
|-----------------|-------------|-------|--------|
| Haldia          | LSM only    | 0.268 | 0.9795 |
| Haldia          | LSM + LSTM  | 0.074 | 0.9984 |
| Diamond Harbour | LSM only    | 0.359 | 0.9679 |
| Diamond Harbour | LSM + LSTM  | 0.117 | 0.9966 |

### Validation Results (Hindcast Mode)
| Period                    | Port            | Improvement |
|---------------------------|-----------------|-------------|
| Dec 2024 (post-monsoon)   | Haldia          | +6.5%       |
| Sep 2024 (monsoon)        | Haldia          | +16.1%      |
| Feb 2023 (dry season)     | Diamond Harbour | +10.2%      |
| Mar 2023 (dry season)     | Diamond Harbour | +42.9%      |

---

## Documentation

- **README.md** — Full technical documentation
- **VALIDATION_REPORT.md** — Detailed validation findings
- **IMPLEMENTATION_SUMMARY.md** — Architecture and design
- **PROJECT_COMPLETION_REPORT.md** — Project overview
- **QUICK_START.md** — This file

---

## Next Steps

1. **Generate your first forecast**: Run the "Use Case 1" example above
2. **Validate on historical data**: Run the "Use Case 3" example
3. **Review results**: Check the generated CSV files and plots
4. **Read documentation**: Review README.md for detailed information
5. **Deploy to production**: Integrate forecasts into your application

---

## Support

For detailed information:
- Check `README.md` for technical details
- Review `VALIDATION_REPORT.md` for validation findings
- See `IMPLEMENTATION_SUMMARY.md` for architecture
- Inspect output plots for visual diagnostics

---

**Status**: ✅ Production-ready for 6-72 hour tidal forecasting
