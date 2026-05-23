# Tide Backend — Hybrid LSM + LSTM Tidal Prediction

Predicts tides at Diamond Harbour and Haldia (Hooghly Estuary, India) using a
two-stage hybrid model:

1. **Least Squares Method (LSM)** — deterministic harmonic model from
   *Rusdin et al. (2024), "Analysis and Prediction of Tidal Measurement Data
   from Temporary Stations using the Least Squares Method"* (Civil Engineering
   Journal). Captures astronomical tides via 19 constituents.

2. **LSTM residual model** — recurrent neural network trained on
   `(observed − LSM_prediction)` to learn non-astronomical patterns: storm
   surge, monsoon discharge, atmospheric pressure variation.

Source data: hourly tidal heights from Survey of India PDFs (2000-2023 for
Diamond Harbour, 2000-2024 for Haldia).

## Results

Test set is the most recent 10% of the observed series (chronological split,
no leakage). Metrics computed against held-out observations.

| Port            | Method      | Test RMSE | Test R | Δ vs LSM |
|-----------------|-------------|----------:|-------:|---------:|
| Haldia          | LSM only    |  0.268 m  | 0.9795 |        — |
| Haldia          | LSM + LSTM  |  0.073 m  | 0.9985 |   −72.9% |
| Diamond Harbour | LSM only    |  0.359 m  | 0.9679 |        — |
| Diamond Harbour | LSM + LSTM  |  0.117 m  | 0.9966 |   −67.3% |

For reference, the paper reports 0.05-0.06 m RMSE on Palu Bay (Indonesia,
deep open bay). Our LSM+LSTM at 0.07 m on the Hooghly estuary — which has
much larger non-astronomical variability — is in the same range.

19-year predicted tidal datums (m, relative to mean sea level):

| Datum | Diamond Harbour | Haldia |
|-------|-----------------|--------|
| HAT   |           +7.00 |  +6.64 |
| MHWS  |           +6.57 |  +6.30 |
| MHW   |           +5.42 |  +5.26 |
| MSL   |           +3.41 |  +3.29 |
| MLW   |           +1.54 |  +1.47 |
| MLWS  |           +0.61 |  +0.52 |
| LAT   |           +0.14 |  +0.06 |
| RA    |            6.85 |   6.58 |

## Pipeline

```
PDFs                        data/                 output/
─────                       ─────                 ───────
DIAMOND HARBOUR/*.pdf  ──►  diamond_harbour.csv  ──►  harmonic_constants_*.csv
HALDIA/*.pdf           ──►  haldia.csv           ──►  validation_*.csv
                                                      lstm_models/*.pt
                                                      lstm_test_*.csv
                                                      summary.json
                                                      lstm_summary.json
                                                      plots/*.png
```

## How to run

Install dependencies (one time, uses uv):
```
uv sync
```

This installs torch with CUDA 12.8 support (Blackwell sm_120 — RTX 50 series),
which auto-falls-back to CPU on systems without a compatible GPU.

Phase 1 — Extract hourly heights from PDFs:
```
uv run python src/extract_pdfs.py
```

Phase 2-5 — Fit LSM (90/10 chronological split), compute datums:
```
uv run python src/train_predict.py
```

Phase 6 — Train LSTM on residuals (uses GPU automatically):
```
uv run python src/residual_lstm.py
```

Plots:
```
uv run python src/plot_results.py    # LSM diagnostic plots
uv run python src/plot_lstm.py       # LSM vs LSM+LSTM comparison
```

Forecast any future window using LSM only (fast, deterministic, far-future capable):
```
uv run python src/forecast.py --port haldia --start 2026-01-01 --days 30 \
    --out output/jan_2026_haldia.csv
```

Combined LSM+LSTM forecast for the next few days, or hindcast over a historical
range:
```
# Hindcast (using observed history as LSTM context — most accurate)
uv run python src/forecast_combined.py --port haldia \
    --start 2024-12-01 --days 7 --mode hindcast --out output/hindcast.csv

# Forecast (autoregressive into the future — useful for ~24-72 hour horizons)
uv run python src/forecast_combined.py --port haldia \
    --start 2025-01-01 --hours 72 --mode forecast --out output/forecast.csv
```

## Method

### LSM (paper eqs. 1-22)

The harmonic model

    h(t) = h_o + Σ H_i cos(ω_i t − g_i)

is rewritten as a linear fit in `A_i = H_i cos(g_i)`, `B_i = H_i sin(g_i)`:

    h(t) = h_o + Σ A_i cos(ω_i t) + Σ B_i sin(ω_i t)

With observations at times `t_m`, this is an over-determined linear system
solved by `numpy.linalg.lstsq`. Amplitudes and phases are recovered as
`H_i = √(A_i² + B_i²)` and `g_i = arctan2(B_i, A_i)`.

We fit 19 constituents:
- Paper's 11: M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Msf, Mm
- Shallow-water: M4, MS4, MN4, M6, S4, 2N2 (estuarine nonlinearity)
- Seasonal: Sa, Ssa (annual / semi-annual cycles)

### LSTM residual model

After LSM is fitted, residuals = `observed − LSM_predicted` are computed for
the entire series. These are fed to a 2-layer LSTM (hidden=96) with a 168-hour
(1 week) lookback window. Per-timestep features:

```
[ residual_t,
  hour_sin, hour_cos,           # hour of day
  day_sin,  day_cos,            # day of year
  month_sin, month_cos ]        # month
```

Training: chronological 80/10/10 train/val/test split, Adam optimizer,
gradient clipping, early stopping on validation MSE. Residuals are
standardized using train-segment statistics.

At inference time, the LSTM is run autoregressively in forecast mode
(its own predicted residuals feed back in), or with real residuals as
context in hindcast mode.

## File layout

```
TIDE BACKEND/
├── TIDAL_DATA/                 # source PDFs (input)
│   ├── DIAMOND HARBOUR/*.pdf
│   └── HALDIA/*.pdf
├── data/                       # extracted hourly CSVs (Phase 1 output)
├── output/                     # models, predictions, plots
│   ├── harmonic_constants_*.csv
│   ├── validation_*.csv
│   ├── lstm_test_*.csv
│   ├── lstm_models/*.pt
│   ├── summary.json
│   ├── lstm_summary.json
│   └── plots/
└── src/
    ├── extract_pdfs.py         # Phase 1: PDF -> CSV
    ├── constituents.py         # tidal constituent definitions
    ├── lsm.py                  # LSM solver
    ├── evaluate.py             # bias / RMSE / R / SR metrics
    ├── datums.py               # HAT / MHWS / MHW / MSL / MLW / MLWS / LAT
    ├── train_predict.py        # LSM pipeline driver (Phase 2-5)
    ├── plot_results.py         # LSM diagnostic plots
    ├── residual_lstm.py        # Phase 6: LSTM training
    ├── plot_lstm.py            # LSM vs LSM+LSTM comparison plots
    ├── forecast.py             # LSM-only CLI (deterministic, far future)
    └── forecast_combined.py    # LSM+LSTM CLI (hindcast or short-term forecast)
```

## Notes & caveats

- Times are in IST (Survey of India publication standard).
- Diamond Harbour 2021-2023 source PDFs are sparse; both LSM and LSTM are
  trained on whatever is available. LSM tolerates gaps natively, LSTM is
  trained over the union of dense intervals.
- LSM-only forecasts are valid for arbitrary future horizons (years, the full
  19-year nodal cycle). LSTM contributions decay quickly when run
  autoregressively beyond ~24-72 hours, since we don't have future weather
  data; for far-future predictions, use `forecast.py` (LSM only).
- For real-time operational use, the LSTM should be re-fed with fresh
  observations and re-evaluated frequently. Data assimilation would be a
  natural next step.
- All harmonic constants and LSTM weights are persisted, so forecasts don't
  need re-training.
