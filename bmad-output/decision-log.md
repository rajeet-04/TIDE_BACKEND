# Decision Log

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2026-09-27 | The model targets real gauge water levels. Survey of India tables are a benchmark only. | Against the Haldia 2024 gauge, the tables scored 79% within ±30 min and ±0.30 m; our model scored 88%. | Training data, evaluation |
| 2026-09-27 | Model stack: harmonic core A, LightGBM correction B as challenger, short-range weather and river layer S. Any future model can replace the current one through the backtest. | The user wants the most accurate and reliable model and to stay open to better methods. | Model architecture |
| 2026-09-27 | Evaluation is a rolling yearly backtest. Settings are chosen on test years up to 2019. Final scores: Haldia 2020–2024, Diamond Harbour 2021–2023. Promotion requires no season to get worse, with bootstrap confidence ranges. | Honest out-of-sample accuracy across horizons and seasons. | Evaluation, promotion |
| 2026-09-27 | Seasonal handling: annual, semi-annual and ter-annual terms plus seasonal side-terms in A. S handles anomalies at short range. A trend term is tested. | Mean level swings about 0.55 m every year; year-to-year variation is 0.04–0.13 m. | Model A, S |
| 2026-09-27 | Moon and sun are computed locally (Skyfield, JPL ephemeris). Weather and river flow come from Copernicus ERA5 and GloFAS, and ECMWF open or NOAA GFS forecasts. The Open-Meteo free tier is not used. | Reliability and commercial-use licensing. | External data |
| 2026-09-27 | No porting of JS, Rust or C++ tide libraries. hatyan is used as a second harmonic engine. | Those libraries predict only and cannot learn; Python already has the best open tools. | Dependencies |
| 2026-09-27 | The INCOIS archiver runs twice daily on GitHub Actions into the `incois-data` branch. The repo will be made private. | INCOIS serves only rolling 30-day windows. | Data pipeline |
| 2026-09-27 | Production runs on a CPU VM. The RTX 5050 is for challengers; Modal A100 only with approval. | A and B train in minutes on CPU. | Infrastructure |
