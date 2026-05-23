# Tide Backend — LSM-based Tidal Prediction

Predicts future tides at Diamond Harbour and Haldia (Hooghly Estuary, India)
using the Least Squares Method described in
*Rusdin et al. (2024), "Analysis and Prediction of Tidal Measurement Data from
Temporary Stations using the Least Squares Method"*, Civil Engineering Journal.

Source data is hourly observed tidal heights from Survey of India PDFs
(2000-2023 for Diamond Harbour, 2000-2024 for Haldia).

## Pipeline

```
PDFs                        data/                 output/
─────                       ─────                 ───────
DIAMOND HARBOUR/*.pdf  ──►  diamond_harbour.csv  ──►  harmonic_constants_*.csv
HALDIA/*.pdf           ──►  haldia.csv           ──►  validation_*.csv
                                                      forecast_30days_*.csv
                                                      summary.json
                                                      plots/*.png
```

## How to run

Install dependencies (one time):
```
pip install pypdf pdfplumber numpy pandas matplotlib scikit-learn
```

Phase 1 — Extract hourly heights from PDFs:
```
python src/extract_pdfs.py
```

Phase 2-5 — Fit LSM (90% train / 10% validate), compute datums, save constants:
```
python src/train_predict.py
```

Plots:
```
python src/plot_results.py
```

Forecast any future window from saved harmonic constants:
```
python src/forecast.py --port haldia --start 2026-01-01 --days 30 \
    --out output/jan_2026_haldia.csv
```

## Results (90/10 chronological split, hourly data)

| Port            | Train size | Valid size | Train RMSE | Valid RMSE | Valid R |
|-----------------|-----------:|-----------:|-----------:|-----------:|--------:|
| Diamond Harbour |    140,248 |     15,584 |     0.31 m |     0.37 m |   0.967 |
| Haldia          |    196,581 |     21,843 |     0.27 m |     0.27 m |   0.979 |

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

## Method

The harmonic model (paper eq. 1):

    h(t) = h_o + Σ H_i cos(ω_i t − g_i)

is rewritten as a linear fit (paper eq. 3):

    h(t) = h_o + Σ A_i cos(ω_i t) + Σ B_i sin(ω_i t)

where `A_i = H_i cos(g_i)`, `B_i = H_i sin(g_i)`. With observations at times
`t_m`, this becomes an over-determined linear system solved by least squares.
Amplitudes and phases are recovered as `H_i = √(A_i² + B_i²)` and
`g_i = arctan2(B_i, A_i)`.

We fit 19 constituents: 11 from the paper (M2, S2, N2, K2, K1, O1, P1, Q1,
Mf, Msf, Mm) plus shallow-water (M4, MS4, MN4, M6, S4, 2N2) and seasonal
(Sa, Ssa) terms — important in the Hooghly estuary because of nonlinear
shallow-water effects, monsoon discharge, and storm surge contributions.

## File layout

```
TIDE BACKEND/
├── TIDAL_DATA/                # source PDFs (input)
│   ├── DIAMOND HARBOUR/*.pdf
│   └── HALDIA/*.pdf
├── data/                      # extracted hourly CSVs (Phase 1 output)
├── output/                    # models, predictions, plots
│   ├── harmonic_constants_*.csv
│   ├── validation_*.csv
│   ├── forecast_30days_*.csv
│   ├── summary.json
│   └── plots/
└── src/
    ├── extract_pdfs.py        # Phase 1: PDF -> CSV
    ├── constituents.py        # tidal constituent definitions
    ├── lsm.py                 # Least Squares Method solver
    ├── evaluate.py            # bias / RMSE / R / SR metrics
    ├── datums.py              # HAT / MHWS / MHW / MSL / MLW / MLWS / LAT
    ├── train_predict.py       # full pipeline driver
    ├── plot_results.py        # diagnostic plots
    └── forecast.py            # CLI for arbitrary future predictions
```

## Notes & caveats

- Times are in IST as published by Survey of India. The paper used UTC.
- Diamond Harbour is sparse for 2021-2023 (incomplete source PDFs); fit still
  works since LSM tolerates gaps.
- The astronomical model captures only the tidal signal. Storm surges, monsoon
  river discharge, and atmospheric pressure variations show up as residuals
  (~0.3 m RMSE). For higher accuracy, train an ML model on residuals (Phase 6,
  not implemented yet).
- All harmonic constants and the mean level are saved as CSV/JSON, so future
  forecasts don't need to refit.
