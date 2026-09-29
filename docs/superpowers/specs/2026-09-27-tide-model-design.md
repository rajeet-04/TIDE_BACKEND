# Tide Model Design: Haldia and Diamond Harbour

**Date:** 2026-09-27
**Status:** Approved 2026-09-27; refined while planning 1a and 1b
**Supporting material:** `bmad-output/brainstorming-report.md`, `bmad-output/decision-log.md`

## 1. Goal

Predict **real water levels** at Haldia and Diamond Harbour (Hooghly estuary) as accurately and reliably as the data allows, for any requested time frame. The model learns from Survey of India tide-gauge readings. Official prediction tables are a benchmark, never a training target.

### 1.1 Outputs (same types as the current model, plus reliability information)

- **Tide peaks:** time and height of every high and low water.
- **Hourly levels**, and levels at any finer step down to 1 minute.
- **Tide frequency summary:** count and mean interval of highs and lows.
- **For every value:** a calibrated 90% error range, the model version, the datum label, and any warning flags.

### 1.2 Success criteria

Measured by the backtest in section 5 against gauge readings the model never saw.

1. On the final test years, beat the **current pipeline** (UTide plus ExtraTrees event calibration) on the share of tides within ±30 min and ±0.30 m. The 95% confidence interval of the difference must be above zero.
2. Beat the **official tables** wherever they overlap gauge data. Today that is Haldia 2024: tables 79.4%, current pipeline 87.8%.
3. **No season gets worse** by more than 1 percentage point. The seasons are dry (Jan–Mar), pre-monsoon (Apr–May), monsoon (Jun–Sep) and post-monsoon (Oct–Dec).
4. **Error ranges are honest.** On the final test years, 90% ranges contain 88–92% of outcomes.
5. **Publishing horizon.** A horizon (1, 2 or 3 years ahead) may be published only if:
   - its score is within 2 percentage points of the 1-year-ahead score, and
   - it meets criterion 4.

### 1.3 Non-goals

- Reproducing official tables.
- Predicting storms beyond the range of weather forecasts.
- Ports without gauge history (INCOIS stations join later; see section 10).
- Tidal currents.
- Tidal bore timing.

## 2. Evidence that shaped this design

| Finding | Source |
|---|---|
| Historical `TIDAL_DATA` PDFs are observed gauge readings. The API and monthly PDFs are predicted tables. | Analysis on 2026-09-27 |
| Against the Haldia 2024 gauge: tables 79.4% (16 min late, 0.12 m low); current pipeline trained on 2000–2023 scored 87.8%. | `tables_vs_gauge_2024` analysis |
| The API's timestamps are labelled UTC but are IST minus 11 h. It serves only 2024 and 2026. Its minute data is interpolated between events. | API probes |
| Training data contains impossible values. Examples: Haldia 2021-12-05 reads 9.91/9.03/8.05 m; Haldia 2006-04-26 has six hours of 0.00; Diamond Harbour 2007-04-08 has an 8.57 m spike; Haldia 2007-08-25 drops to 1.54 m between 5.01 and 3.90 m. | Residual scan |
| Mean level swings about 0.55 m every year (February low, August high). Year-to-year monthly variation is 0.04–0.13 m. Long-term rise is +3.7 mm/yr at Haldia and +3.0 mm/yr at Diamond Harbour. | Monthly statistics |
| Haldia and Diamond Harbour residuals are strongly linked: correlation 0.745 with Haldia 1 h ahead; 0.864 for 25-hour means. | Cross-port analysis |
| INCOIS has live 1-minute data for 50 stations. On the Hooghly only Garden Reach is covered. Archive starts 2026-08-28. | `INCOIS_api.txt`, archiver |

## 3. Data

### 3.1 Sources and roles

| Source | Coverage | Role |
|---|---|---|
| Survey of India observed hourly heights (`TIDAL_DATA/` parsed to `data/*.csv`) | Haldia 2000–2024. Diamond Harbour 2000–2016; 2017–2020 missing; 2021–2023 sparse. | Training and scoring truth |
| INCOIS 1-minute archive (`incois-data` branch) | 50 stations from 2026-08-28. UTC. Datum unknown. | Monitoring; future ports |
| Official predictions: Survey of India monthly PDFs, API 2024 and 2026 | 2024; April–September 2026, growing monthly | Benchmark and monitoring only |
| PSMSL monthly mean sea level, Haldia | 1970–2024 | Checking the long-term trend |
| ERA5 hourly reanalysis (Copernicus data store, CC BY 4.0) | 1940 onward | Training S: 10 m wind, sea-level pressure, precipitation |
| GloFAS historical river discharge (Copernicus early-warning store, CEMS-FLOODS licence) | 1979 onward, daily | Training S; seasonal diagnostics |
| NASA GPM IMERG precipitation (Earthdata login; user has credentials) | 2000 onward, half-hourly; near-real-time feed | Challenger rainfall input for S, compared against ERA5 rainfall in the backtest |
| Forecasts at prediction time: ECMWF open data (CC BY 4.0) or NOAA GFS (public domain), plus GloFAS forecasts | Next 10–30 days | Running S |
| Astronomy: Skyfield with JPL DE440s ephemeris | 1849–2150, computed locally | Moon phases, sunrise and sunset (1c). B's features use UTide's astronomical arguments, which trees only need as smooth proxies (decision log 2026-09-29) |

ERA5 and GloFAS grid points are chosen by their correlation with model residuals **in training years only**, so that choice doesn't leak test information.

### 3.2 Canonical store and conventions

- **One tidy table per port and source:** `time_utc`, `height_m`, `source`, `qc_flag`.
  - Times are timezone-aware UTC throughout the code.
  - IST is UTC+05:30 with no daylight saving, and appears only in outputs.
- **Datum registry** (one entry per source and port) records:
  - the datum name
  - the offset to chart datum where known, otherwise `unknown`
  - the evidence for it
  - Survey of India gauges and tables are recorded as chart datum, to be confirmed with Survey of India. INCOIS stations are recorded as `unknown` until INCOIS supplies datums.
  - Every output states its datum. Sources with an `unknown` datum are never mixed into training.

### 3.3 Quality control (first implementation stage)

Readings are flagged, never deleted. Flagged readings are left out of fitting and scoring.

**Rules:**

1. **Physically impossible values:** zero or negative readings, and values more than 1 m outside the port's 0.01–99.99% quantile range.
2. **Spikes:** take the residual against a robust harmonic fit and subtract its 7-hour centred median, so slow surges are not treated as spikes. A reading is flagged when this exceeds both 0.5 m and 5 times the larger of two spreads: the robust standard deviation of its 7-day neighbourhood, and the spread at the same tidal phase (whole hours since the predicted low water × spring/neap tercile of the predicted 25-hour range; 90th percentile of the size ÷ 1.645, the largest over neighbouring phase classes). The phase spread keeps the spring-tide bore, which the harmonic fit smooths, from being flagged (decision log 2026-09-27). The fit is **cross-fitted**: it excludes the year being checked, so "truth" in a test year is never defined by a model that saw that year.
3. **Impossible rate of change:** the hour-to-hour change exceeds 1.2 × the 99.99th percentile of clean-year changes. Of the two readings, the one further from the harmonic prediction is flagged.
4. **Flat stretches:** 4 or more identical consecutive readings while the harmonic prediction changes by more than 0.3 m.
5. **Year-level problems:** found by the audit in 3.4.

**Review file:** each port has `data/qc/<port>_review.csv` for manual decisions, and each row records its evidence:

- `exclude` stretches: rule 5 decisions from the audit.
- `keep` stretches: genuine extremes that a rule flagged, such as cyclone surges.

Planning checks on the gauge record found that rule 2 flags the steepest real surge onsets, such as Amphan on 2020-05-20. The `keep` rows exist for these.

**Outputs:**

- a QC report (flag counts per year and rule)
- a review list of flagged stretches for manual inspection

**Acceptance:**

- All four known bad examples in section 2 are flagged.
- At most 1% of readings in any year are flagged, unless the audit explains why.

### 3.4 Per-year audit

Fit a basic harmonic model to each calendar year separately. Track:

- mean level
- M2, S2, K1, O1 and M4 amplitude and phase

A year that departs from its neighbours by more than 3 robust standard deviations is reported as a suspected re-levelling, clock error or channel change. Correction or exclusion needs evidence and is recorded in the decision log. The same audit runs whenever new gauge data arrives.

The Diamond Harbour 2022 gauge PDFs are headed "DIAMOND HARBOUR (ROY CHAK)", which may mean a different gauge site. The audit therefore compares 2021–2023 with 2011–2016 before those years are used as the final test.

### 3.5 Diamond Harbour gap filling

- **Transfer model:** predicts Diamond Harbour from Haldia. Inputs are Haldia readings lagged 0–3 h, Haldia's residual, and Diamond Harbour's harmonic prediction.
- **Training:** overlapping QC-passed years **before the test period only**. For the final Diamond Harbour test (2021–2023), it trains on 2016 and earlier; otherwise the reconstruction would carry information from the test years.
- **Validation:** hide real Diamond Harbour years 2014–2016 and score the reconstruction.
- **Use:** reconstructed 2017–2020 (and gaps in 2021–2023) are marked `derived`. They enter training only if the backtest shows Diamond Harbour improves with them. They are **never** used for scoring.

### 3.6 Official-table archiver

The Survey of India tidal page keeps only the latest three months, so a monthly job:

- downloads the new archive
- keeps the Haldia, Diamond Harbour and Garden Reach PDFs
- extracts events with `src/extract_reference_events.py`
- appends them to the benchmark store

## 4. Models

### 4.1 Common interface

Every model implements:

- `fit(observations, context) -> fitted model`
- `predict(times_utc) -> levels`, optionally with components

High and low tides are extracted by one shared routine (4.6). This lets any model family, including ones found later, compete in the same backtest.

### 4.2 Model A: harmonic core (champion candidate)

Weighted least squares on QC-passed hourly data. It extends `src/lsm.py` with nodal corrections from UTide's routines and is tested against UTide reconstructions.

**Terms:**

- Tidal constituents. Candidate sets are UTide's automatic selection for the record length, and that set plus extra shallow-water constituents.
- Mean-level seasonal terms: Sa, Ssa, Sta.
- Seasonal side-terms: ±1 cycle/year modulation of M2, S2, N2, K1, O1, M4 and MS4.
- An optional linear trend.

**Fitting:** iteratively reweighted least squares with a bisquare weight function, which down-weights surges, and a small ridge penalty for numerical stability.

**Settings chosen in the selection folds:**

- training window: last 5, 8 or 12 years, or all years
- constituent set
- side-term set
- trend on or off

**Comparison engines in the same slot:** UTide (the current method) and hatyan.

### 4.3 Model B: long-range correction (challenger)

LightGBM predicts A's hourly residual.

**Training target:** residuals are **cross-fitted by year**. For each training year, A is refit without that year. Otherwise B learns from A's in-sample errors, which are smaller than its real forecast errors.

**Features (all known in advance):**

- A's level, slope and curvature
- time since and until the predicted high and low
- the predicted range of the current tide
- tidal-stage phase
- lunar phase angle, declination and distance
- solar declination and Earth–Sun distance
- lunar node angle and lunar perigee longitude
- day of year and hour of day (sine and cosine)

**Settings:** tuned in inner folds; training is deterministic.

**Other challengers in the same slot:**

- XGBoost (CUDA)
- a small PyTorch network on the same features
- the existing ExtraTrees event calibrator

**Structure chosen while planning 1b** (selection folds only; decision log 2026-09-29):

- **Cross-fitting:** A's weighted normal equations are summed per year; dropping one year's share and solving refits without it. This matches a refit from scratch to 1.5 mm RMS and takes under a second.
- **Two independent corrections** learn from the same cross-fitted material:
  - a **level correction** (the hourly residual above), evaluated on whole IST hours and joined by a cubic spline;
  - an **event correction**: two learners predict the timing and height errors of A's 1-minute events from each event's context and the astronomical state, and move the event by them. This is the ExtraTrees calibrator's idea applied to A, open to any learner.
- **Where each is used:** events come from the event correction and hourly levels from the level correction. The two are not stacked: training the event correction on level-corrected events made it worse on every planning fold. The chosen stack is the best of A alone, A with one correction, and A with both, by joint share and then hourly RMSE.
- **Evidence:** mean joint share over nine selection fold-years was 88.8% for A, 91.3% for the current pipeline, 92.4% with the level correction and 93.8% with a LightGBM event correction.
- **Compute:** every learner trains on the CPU with one thread. Results are bit-for-bit reproducible, and a model chosen here refits unchanged on the CPU VM.

### 4.4 Layer S: short-range correction

- **Target:** the residual of the chosen long-range model (A or A+B).
- **Lead times:** 0–10 days. This extends to 30 days only where the backtest shows skill.
- **Features:**
  - wind (u, v) and sea-level pressure at the selected points
  - basin precipitation summed over 1–10 days
  - GloFAS discharge and its anomaly against climatology
  - lead time
  - B's known-in-advance features
- **Validation** (both required):
  1. Driven by weather records. This gives an upper bound on skill.
  2. Driven by **archived forecasts** where an archive covers the test period, such as NOAA GFS on AWS Open Data and the GloFAS forecast archive.

  S ships for a lead time only if test 2 beats A/B at that lead time.
- **Fallback:** if inputs are missing or stale, S is skipped and outputs carry the flag `no_weather_correction`.

### 4.5 Ensembles

A weighted average of the top models, with weights chosen in the selection folds, competes as its own candidate. In 1b the blends are made within each correction: the top two learners at weights 0.25, 0.5 and 0.75.

### 4.6 Event extraction

Peaks are found on the final 1-minute curve:

- prominence ≥ 0.5 m
- separation ≥ 4 h
- quadratic refinement, as today

**Long low-water stands:** on the 1-minute curve, a low water's time is the centre of the interval within 1 cm of the minimum. A high water's time is its maximum. This definition is documented in outputs.

### 4.7 Error ranges

Split-conformal ranges are computed for each combination of:

- horizon bucket
- season
- output type: level, high-water time and height, low-water time and height

**Calibration and checking:**

- **Backtest:** ranges are calibrated on the selection folds, and their coverage is checked on the final folds.
- **Production:** ranges are recalibrated on all folds.

**Details fixed while planning 1b:**

- The horizon buckets are the horizons 1, 2 and 3.
- A range bounds observed minus predicted, from split-conformal quantiles.
- A cell with fewer than 100 errors borrows from its horizon, then from the whole output.
- Coverage counts a missed event as outside its range. It is reported pooled, per horizon and per season.
- Forecasts beyond horizon 3 use horizon 3's ranges and are flagged.

## 5. Backtest (train/test split and cross-validation)

### 5.1 Folds

- **Rolling by year:** for each origin year Y, train on data before Y and test years Y, Y+1 and Y+2 (horizons 1–3).
- **Haldia:** origins 2008–2024.
- **Diamond Harbour:** origins 2008–2016. The final Diamond Harbour test trains on data before 2021 (real up to 2016, plus derived 2017–2020 if enabled) and tests 2021–2023.

### 5.2 Selection set vs final set

- **Selection set:** folds whose test year is 2019 or earlier. All settings, feature choices and ensemble weights are chosen here.
- **Final set:** Haldia test years 2020–2024 and Diamond Harbour 2021–2023. Each candidate is scored on it once, and nothing is tuned against it.
- **When new gauge years arrive,** both boundaries move forward by the same number of years, so the final set is always the most recent years.

### 5.3 Metrics

Reported per port × horizon × season × high/low:

- share of events within ±30 min and ±0.30 m
- timing and height: mean absolute error, 95th-percentile error and bias
- missed and extra events
- hourly RMSE
- coverage of the 90% ranges

**Truth:** QC-passed readings only. Observed events come from hourly readings with quadratic refinement; this step alone has a known timing error of about 2.4 minutes.

### 5.4 Confidence

Differences between candidates are tested with a block bootstrap by calendar month (1,000 resamples).

### 5.5 Benchmarks shown alongside

- the current pipeline
- the official tables, where they overlap

### 5.6 Leakage guards

- Everything is fitted inside folds. That covers QC spike fits (cross-fitted), grid-point selection, feature scaling and settings.
- Feature functions accept only times and known-in-advance inputs; a unit test enforces this.

### 5.7 Promotion rule

A candidate replaces the champion only if it meets all of section 1.2 on the final set:

1. higher joint score, with the interval of the difference above zero
2. timing and height MAE not worse beyond their intervals
3. no season worse by more than 1 percentage point
4. range coverage within 88–92% for every output type, pooled over the final set

Criterion 2 of section 1.2 (beat the official tables wherever they overlap) is checked with them. A horizon is published only if it meets criterion 5, including coverage within 88–92% at that horizon.

## 6. Interface and outputs

### 6.1 Commands

Available from the CLI and as a Python module.

- `predict(port, start, end | hours, step_minutes=60)` returns:
  - events: state, `time_utc`, `time_ist`, `height_m`, 90% timing range, 90% height range
  - levels at the step: `time_utc`, `time_ist`, `height_m`, `lower_90`, `upper_90`, and components (astronomical, seasonal, correction)
  - the frequency summary
  - moon phases: times of new moon, first quarter, full moon and last quarter
  - a spring/neap label per day: **spring** if the day's predicted range is in the top quarter of its ±7-day window, **neap** if in the bottom quarter, otherwise **mid**
  - sunrise and sunset at the port's coordinates
  - the datum label, model version and flags
- **Input times:** `start` and `end` may carry a timezone. Times without one are read as IST, the local convention for these ports.
- `train(port)` runs: refresh data → QC and audit → backtest candidates → promotion rule → report.
- `backtest(port, candidate)` runs the backtest alone.

### 6.2 Files

The files keep the current forecast names so existing users keep working. New columns are appended; existing ones are not renamed.

- `hourly_water_levels.csv`
- `predicted_tide_events.csv`
- `tide_frequency.csv`
- the chart PNG

### 6.3 Other outputs

- **Navigation windows:** `windows(port, start, end, threshold_m, confidence=0.9)` returns the intervals where the lower range edge stays at or above the threshold.
- **Tidal datums per port:** HAT, MHWS, MHW, MSL, MLW, MLWS and LAT, from a 19-year prediction (extends `src/datums.py`).
- **Observed data:** `observed(port, start, end)` from the gauge and INCOIS stores, with an observed-versus-predicted view.

## 7. Pipeline and operations

### 7.1 Model registry

- Each version lives in `models/<port>/<version>/` and contains:
  - A's constants (JSON)
  - the B and S model files
  - the error ranges (JSON)
  - `metadata.json`: data hashes, code commit and backtest summary
  - a model card
- A `current` pointer file marks the live version. Promotion rewrites it atomically, and rolling back restores the previous pointer.

### 7.2 Schedules

On the CPU VM; the INCOIS archive runs in GitHub Actions.

- **Daily:**
  - fetch forecasts for S (ECMWF or GFS, GloFAS)
  - run the monitoring checks
- **Monthly:**
  - archive the official tables
  - retrain → backtest → promotion rule → report
- **Retrain:** also runs as soon as new gauge data arrives.

### 7.3 Monitoring without live truth

Alerts fire when any of these moves outside its normal range:

1. **Divergence from the official tables:** the monthly mean timing or height difference exceeds the 99th percentile of past divergence. This catches bugs; it is not a measure of truth.
2. **Garden Reach anomaly:** the INCOIS Garden Reach daily mean level exceeds its trailing 30-day median by more than 3 robust standard deviations. This flags high-flow periods and works from the first month of the archive.
3. **Constants drift:** the per-year audit on newly arrived gauge data.

### 7.4 Compute

- **CPU:** training and serving A, B and S.
- **Local RTX 5050:** XGBoost and neural challengers.
- **Modal A100:** only with the user's approval.

### 7.5 Secrets

- **Locally:** the Copernicus token in `~/.cdsapirc`.
- **VM and CI:** an environment secret.
- **Never committed.**

### 7.6 Licences and attribution

A `NOTICE` file covers:

| Item | Terms |
|---|---|
| ERA5 | CC BY 4.0 |
| ECMWF open data | CC BY 4.0 |
| NOAA GFS | Public domain |
| GloFAS | Credit "Generated using / Contains modified Copernicus Emergency Management Service information [Year]". River-flow flags are labelled informational, not official flood warnings. |
| hatyan | LGPL-3.0 |
| UTide, pyTMD, Skyfield, LightGBM | MIT |
| XGBoost | Apache-2.0 |
| Survey of India tables | Benchmark only, not redistributed |
| INCOIS data | Kept private; terms to be confirmed with INCOIS |

## 8. Testing

- **Unit tests:**
  - nodal corrections against UTide
  - A recovers known synthetic constituents
  - event extraction on synthetic curves, including low-water stands
  - QC flags the four known bad examples
  - IST/UTC invariants and datum labels
  - the leakage guard
  - the archiver (already exists)
- **Regression snapshots:** each model version's predictions for fixed dates are stored. CI fails on any unexplained change.
- **Reproducibility:** fixed seeds; identical inputs give identical backtest metrics.
- **Runner:** pytest, added as a development dependency. The existing plain-Python test file stays runnable on its own.

## 9. Implementation split

The work is too large for one plan, so it is split into three plans run in order:

1. **Core** (sections 3, 4.1–4.3, 4.5–4.7, 5, 6, 8), in three parts:
   - **1a** (done):
     - data store, QC, audit, datum registry
     - backtest harness, with the current pipeline and tables as baselines
     - model A variants, selected and scored
     - `predict` and `backtest` commands serving model A in the current file formats
   - **1b** (`docs/superpowers/plans/2026-09-29-tide-core-1b.md`):
     - B and its challengers, ensembles and error ranges
     - `train` with the full promotion rule
     - versions holding B and ranges; forecasts with ranges
     - regression snapshots
   - **1c:**
     - hatyan as a comparison engine
     - Diamond Harbour gap filling and the official-table archiver
     - moon phases, spring/neap labels, sunrise and sunset
     - windows, datums and `observed()`
     - the PSMSL trend check

   1b was planned after 1a ran, and 1c is planned after 1b runs, so each design can use the previous part's backtest results.
2. **Short range** (section 4.4): ERA5 and GloFAS ingestion, layer S, archived-forecast validation, fallback.
3. **Operations** (section 7): registry and promotion, schedules on the VM, monitoring, NOTICE file.

## 10. Later (recorded, not in scope)

- cyclone catalogue and storm scoring for S
- shared seasonal terms across both ports
- satellite sea-level anomaly and climate indices
- ensemble weather forecasts for surge ranges
- Garden Reach nowcast input
- return levels
- risk flags
- tidal bore timing
- INCOIS stations as new ports once they have about a year of history
- route compatibility with the frontend from the API capture, which needs its own spec if wanted

## 11. Dependencies on the user

1. ~~Accept the ERA5 CC BY licence and the GloFAS CEMS-FLOODS licence on their dataset pages.~~ Done 2026-09-27; both show as accepted on both Copernicus stores.
2. Send formal data requests:
   - **Survey of India:** Haldia 2025–26, Diamond Harbour 2017–26, Sagar and Garden Reach history, and gauge datums
   - **INCOIS:** archives, datums, and whether Haldia records exist
3. ~~Merge PR #5 so the INCOIS archiver runs, and make the repository private.~~ Done.
