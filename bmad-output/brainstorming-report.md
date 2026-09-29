# Brainstorming Session Report

**Date:** 2026-09-27
**Session Duration:** ~1 hour (inline, single facilitator, with data checks)
**BMAD Track:** bmad-method
**Topic / Problem:** What features are missing from the tide model design for Haldia and Diamond Harbour?

---

## Session Objective

**Goal:** Find the features, data steps and safeguards the current design lacks before it is written up as a spec.

**Context:** The design is agreed in outline. The model targets real gauge water levels. It stacks a harmonic core (A), a LightGBM correction (B) and a short-range weather and river layer (S). It handles seasonal effects, is scored by a rolling yearly backtest, and runs predict and train commands on a CPU VM. An INCOIS archiver runs on GitHub Actions. Outputs are tide peak times, hourly levels and a frequency summary.

**Constraints:**
- CPU-only production VM.
- GPUs (local RTX 5050; Modal A100 with approval) for offline experiments only.
- Open and free data sources that allow commercial use.
- No live gauge feed for Haldia or Diamond Harbour.

**Success Criteria:** Every missing item is named, rated for impact and feasibility, and sorted into "put in the spec now" or "later".

**Related BMAD Artifacts:**
- Decision log: `bmad-output/decision-log.md`
- Design spec (next): `docs/superpowers/specs/2026-09-27-tide-model-design.md`

---

## Techniques Used

### Primary Technique: Reverse Brainstorming
**Rationale:** "Most accurate and most reliable" is the goal, so the fastest way to find gaps is to ask how the model could give wrong answers. Each failure mode was checked against the data where possible.
**Duration:** ~25 min

### Secondary Technique: Starbursting (Who / What / When / Where / Why / How)
**Rationale:** Surfaces user-facing features. Port users plan ship movements around water depth and need more than a raw curve.
**Duration:** ~20 min

### Additional Technique: SCAMPER
**Rationale:** Generates variations on the model and data: combining ports and engines, substituting sources, adapting practice from other tide agencies.
**Duration:** ~15 min

---

## Evidence Gathered During the Session

| Check | Result |
|---|---|
| Largest hourly residuals (observed − harmonic model) | Dominated by **data errors**, not storms. Examples: Haldia 2021-12-05 reads 9.91, 9.03 and 8.05 m (neighbours 1.17 and 3.06 m). Haldia 2006-04-26 has six hours of 0.00. Diamond Harbour 2007-04-08 has an 8.57 m spike. Haldia 2007-08-25 drops to 1.54 m between 5.01 and 3.90 m. |
| Residual spread | Hourly SD: Haldia 0.195 m, Diamond Harbour 0.223 m. 99.9th percentile: 0.94 m and 1.12 m. |
| Haldia vs Diamond Harbour residuals (155,088 shared hours) | Correlation 0.745 with Haldia leading by 1 h. Correlation 0.864 for 25-hour mean residuals (the surge and river part). |
| Seasonal cycle (from the prior step) | Mean level swings about 0.55 m every year. Year-to-year monthly variation is only 0.04–0.13 m. |

---

## Ideas Generated

### Category 1: Data quality and coverage

1. **Gauge data quality control with flags**
   - Description: Detect zero-filled gaps, flat stretches, digit-typo spikes, impossible rise rates and jumps. Flag bad readings rather than deleting them, and exclude flagged values from fitting and scoring.
   - Source Technique: Reverse Brainstorming (confirmed in the data)
   - Potential Impact: High
   - Feasibility: High
2. **Per-year audit of tidal constants and mean level**
   - Description: Fit each year separately and compare M2, S2, K1 and O1 amplitudes and phases and the mean level. This catches gauge re-levelling, clock errors and gradual channel change.
   - Source Technique: Reverse Brainstorming
   - Potential Impact: High
   - Feasibility: High
3. **PDF extraction verification**
   - Description: Plausibility checks on every parsed value (neighbour continuity, digit-swap patterns), plus a review list for suspects.
   - Source Technique: Reverse Brainstorming
   - Potential Impact: Medium
   - Feasibility: High
4. **Datum registry**
   - Description: Record the reference level for each source and port. Survey of India gauges and tables use chart datum; the INCOIS radar zero is unknown. Convert to MSL, and state the datum in every output.
   - Source Technique: Starbursting / Reverse Brainstorming
   - Potential Impact: High
   - Feasibility: Medium
5. **Fill Diamond Harbour's gaps from Haldia**
   - Description: Build a transfer model from Haldia (1 h lag; residual correlation 0.745 hourly and 0.864 for 25-hour means). Use it to reconstruct Diamond Harbour 2017–2020 and the sparse 2021–2023, so Diamond Harbour's model can use recent years.
   - Source Technique: SCAMPER (Combine)
   - Potential Impact: High
   - Feasibility: Medium
6. **Formal data requests**
   - Description: Ask Survey of India for Haldia 2025–26 and Diamond Harbour 2017–26 (digital, sub-hourly if possible), plus Sagar and Garden Reach. Ask INCOIS for archives, gauge datums and Haldia records.
   - Source Technique: Starbursting (Where)
   - Potential Impact: High
   - Feasibility: Medium
7. **Monthly archiver for official tables**
   - Description: The Survey of India page keeps only three months, so save each month as it appears.
   - Source Technique: Reverse Brainstorming
   - Potential Impact: Medium
   - Feasibility: High
8. **PSMSL mean sea level, 1970–2024**
   - Description: Longer record to test the trend term.
   - Source Technique: SCAMPER (Adapt)
   - Potential Impact: Low
   - Feasibility: High
9. **Cyclone and extreme-event catalogue**
   - Description: IMD best tracks (to verify). Used to label events for scoring S and to down-weight them in harmonic fitting.
   - Source Technique: Starbursting (When)
   - Potential Impact: Medium
   - Feasibility: Medium

### Category 2: Model accuracy

10. **Ensemble of harmonic engines**
    - Description: Our least-squares fit, UTide and hatyan, averaged or stacked. Adopted only if it wins the backtest.
    - Source Technique: SCAMPER (Combine)
    - Potential Impact: Medium
    - Feasibility: High
11. **Direct event-correction challenger**
    - Description: Keep the existing ExtraTrees event calibrator as a backtest option alongside curve-level correction.
    - Source Technique: SCAMPER (Reverse)
    - Potential Impact: Medium
    - Feasibility: High
12. **Consistent event definition**
    - Description: Handle long low-water stands and small secondary peaks, and document how peak times are defined.
    - Source Technique: Reverse Brainstorming
    - Potential Impact: Medium
    - Feasibility: Medium
13. **Joint Haldia and Diamond Harbour seasonal terms**
    - Description: Share the seasonal and river response across both ports so the sparse Diamond Harbour record borrows strength.
    - Source Technique: SCAMPER (Combine)
    - Potential Impact: Medium
    - Feasibility: Medium
14. **Offshore and remote predictors**
    - Description: CMEMS satellite sea-level anomaly for the northern Bay (daily, 1993 onward) and ENSO/IOD indices for 10–90 day anomalies.
    - Source Technique: SCAMPER (Adapt)
    - Potential Impact: Medium
    - Feasibility: Low
15. **Garden Reach live residual as a nowcast input**
    - Description: Only possible once Haldia and Diamond Harbour truth overlaps the INCOIS archive, which starts 2026-08-28.
    - Source Technique: SCAMPER (Put to other use)
    - Potential Impact: Medium
    - Feasibility: Low
16. **Probabilistic S from ECMWF ensemble forecasts**
    - Description: Surge uncertainty bands.
    - Source Technique: SCAMPER (Modify)
    - Potential Impact: Medium
    - Feasibility: Medium
17. **Live Haldia and Diamond Harbour gauge feed**
    - Description: Via an INCOIS or Kolkata port agreement. Enables nowcasting and true live monitoring.
    - Source Technique: Starbursting (Where)
    - Potential Impact: High
    - Feasibility: Low

### Category 3: Reliability and trust

18. **Calibrated error bands (conformal)**
    - Description: Derived from backtest residuals by horizon and season. A coverage test checks that 90% bands really contain about 90%.
    - Source Technique: SCAMPER (Magnify)
    - Potential Impact: High
    - Feasibility: High
19. **Canary monitoring without live truth**
    - Description: Watch for divergence from the official tables (a sanity check, not truth), the INCOIS upstream anomaly and drift in the per-year constants. Alert when any of them moves.
    - Source Technique: Reverse Brainstorming
    - Potential Impact: High
    - Feasibility: High
20. **Graceful fallback when S inputs fail**
    - Description: Serve A/B with a visible flag and use cached forecasts when an external API is down.
    - Source Technique: Reverse Brainstorming
    - Potential Impact: High
    - Feasibility: High
21. **Backtest leakage guards**
    - Description: All preprocessing is fitted inside folds, feature functions accept only known-ahead inputs, and unit tests enforce both.
    - Source Technique: Reverse Brainstorming
    - Potential Impact: High
    - Feasibility: High
22. **Timezone and datum invariant tests, plus regression snapshots**
    - Description: Fixed dates are predicted and compared so outputs can't change silently.
    - Source Technique: Reverse Brainstorming
    - Potential Impact: High
    - Feasibility: High
23. **Model card and auto-generated backtest report per candidate**
    - Source Technique: SCAMPER (Adapt)
    - Potential Impact: Medium
    - Feasibility: High
24. **Extreme-event scoring**
    - Description: Score S on catalogued cyclones after QC: peak surge height and timing.
    - Source Technique: Starbursting (Why)
    - Potential Impact: Medium
    - Feasibility: Medium
25. **Licence and attribution register**
    - Description: Copernicus, ECMWF CC BY 4.0, hatyan LGPL, Survey of India copyright (benchmark only), INCOIS terms.
    - Source Technique: Starbursting (How)
    - Potential Impact: Medium
    - Feasibility: High
26. **Secrets handling**
    - Description: Copernicus API key in environment or secret store, never in the repo.
    - Source Technique: Reverse Brainstorming
    - Potential Impact: High
    - Feasibility: High

### Category 4: User-facing outputs

27. **Datum-labelled outputs plus tidal datums per port**
    - Description: HAT, MHWS, MHW, MSL, MLW, MLWS and LAT, using the existing `src/datums.py`.
    - Source Technique: Starbursting (What)
    - Potential Impact: High
    - Feasibility: High
28. **Navigation windows**
    - Description: "When is the water above X m?" for ship draft planning at Haldia, with safe windows that account for the error bands.
    - Source Technique: Starbursting (Who)
    - Potential Impact: High
    - Feasibility: High
29. **Component breakdown**
    - Description: Astronomical tide + seasonal + weather/river correction, shown per time step.
    - Source Technique: Starbursting (Why)
    - Potential Impact: Medium
    - Feasibility: High
30. **Moon phases, spring/neap labels, sunrise and sunset computed locally**
    - Description: Replaces the external sunrise API the old frontend used.
    - Source Technique: Starbursting (What)
    - Potential Impact: Medium
    - Feasibility: High
31. **Observed data and observed-vs-predicted views**
    - Description: Gauge history plus the INCOIS archive.
    - Source Technique: Starbursting (What)
    - Potential Impact: Medium
    - Feasibility: High
32. **Drop-in compatibility with the captured frontend API**
    - Description: The port-list, port-details, predicted-tide-data, predicted-one-minute-data and moon-data routes.
    - Source Technique: Starbursting (Who)
    - Potential Impact: High
    - Feasibility: Medium
33. **Risk flags in outputs**
    - Description: Surge risk, high river flow, low-confidence periods.
    - Source Technique: Starbursting (Why)
    - Potential Impact: Medium
    - Feasibility: Medium
34. **Rate of rise/fall and flood/ebb stage per hour**
    - Source Technique: Starbursting (What)
    - Potential Impact: Low
    - Feasibility: High
35. **Extreme water level statistics (return levels)**
    - Source Technique: SCAMPER (Put to other use)
    - Potential Impact: Low
    - Feasibility: Medium
36. **Tidal bore ("Baan") timing**
    - Source Technique: Starbursting (When)
    - Potential Impact: Low
    - Feasibility: Low

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total Ideas Generated | 36 |
| Categories | 4 |
| High-Impact Ideas | 17 |
| Quick Wins (High Impact + High Feasibility) | 12 (#1, 2, 18, 19, 20, 21, 22, 26, 27, 28, plus #4 and #5 at Medium feasibility just below) |
| Moon Shots (High Impact + Low Feasibility) | 1 (#17 live Haldia/Diamond Harbour gauge feed) |

---

## Top Actionable Insights

### 1. Clean the training data before any modelling

**Description:** The historical gauge data contains impossible values: 9.91 m spikes, zero-filled hours and digit typos. The current model was trained on them, and a backtest would score against them. Genuine storm signals are buried under these errors.

**Supporting Ideas:**
- #1 quality control with flags
- #2 per-year constants audit
- #3 PDF extraction verification

**Why It Matters:** Every model, metric and error band inherits these errors. This is the cheapest, largest accuracy and reliability gain available.

**Recommended Action:** Make QC and the audit the first implementation stage. Use a robust harmonic fit as a second line of defence.

**Feeds Into:** Spec section "Data"; first task in the implementation plan.

---

### 2. Build trust without live truth

**Description:** Haldia and Diamond Harbour have no live gauge, so drift cannot be seen directly.

**Supporting Ideas:**
- #18 calibrated error bands with a coverage test
- #19 canary monitoring (tables divergence, INCOIS upstream, constants drift)
- #20 graceful fallback
- #21 and #22 leakage and invariant tests

**Why It Matters:** "Most reliable" means knowing when the model is wrong and saying so.

**Recommended Action:** Add these to the spec's backtest and pipeline sections as requirements.

**Feeds Into:** Spec sections "Backtest" and "Pipeline".

---

### 3. Make datums and time zones explicit everywhere

**Description:** We have already hit an 11-hour clock bug (the API) and an unknown datum (INCOIS radar). A datum registry, timezone-aware data and invariant tests stop this class of error.

**Supporting Ideas:**
- #4 datum registry
- #22 invariant tests
- #27 datum-labelled outputs

**Why It Matters:** A correct model with the wrong datum or clock is wrong by metres or hours.

**Recommended Action:** Add a datum and time section to the spec.

**Feeds Into:** Spec sections "Data" and "Outputs".

---

### 4. Use Haldia to strengthen Diamond Harbour

**Description:** Residuals are strongly linked: correlation 0.745 hourly with a 1 h lag, and 0.864 for the slow part. A transfer model can fill Diamond Harbour's 2017–2023 gaps so its model can use recent years.

**Supporting Ideas:**
- #5 gap filling
- #13 joint seasonal terms

**Why It Matters:** Diamond Harbour's record stops at a full 2016, which weakens recent-year accuracy.

**Recommended Action:** Include gap filling as a data step. Test it in the backtest by hiding real Diamond Harbour years and checking the reconstruction.

**Feeds Into:** Spec section "Models".

---

### 5. Deliver what port users actually act on

**Description:** Ship planning needs depth windows, datums and confidence, not just curves.

**Supporting Ideas:**
- #28 navigation windows
- #27 datums
- #29 component breakdown
- #30 moon, sun and spring/neap
- #32 frontend API compatibility

**Why It Matters:** These turn predictions into decisions, and most are cheap once the curve exists.

**Recommended Action:** Put #27, #28 and #30 in scope. Keep #32 if the existing frontend will call this backend.

**Feeds Into:** Spec section "Outputs".

---

## Risk Considerations

1. **Undetected bad data**
   - Description: Errors bias constants, calibrators and metrics.
   - Impact: High
   - Probability: High (already present)
   - Planning Response: QC, audit, robust fit, review list.
2. **B or S learning past weather as if it were a pattern**
   - Description: The correction layers treat one-off weather in past years as repeatable.
   - Impact: High
   - Probability: Medium
   - Planning Response: Cross-fitting, rolling backtest, per-season gate, bootstrap confidence ranges.
3. **S looks better on reanalysis than on real forecasts**
   - Description: Training on weather records overstates the skill S will have with forecasts.
   - Impact: Medium
   - Probability: High
   - Planning Response: Validate with archived forecasts; report S accuracy by lead time.
4. **External data access and licensing**
   - Description: Copernicus key, API outages, INCOIS terms, Survey of India copyright.
   - Impact: Medium
   - Probability: Medium
   - Planning Response: Fallbacks, licence register, formal data requests.
5. **Datum or timezone mix-ups**
   - Impact: High
   - Probability: Medium
   - Planning Response: Registry, timezone-aware types, invariant tests.
6. **Scope creep (36 ideas)**
   - Impact: Medium
   - Probability: High
   - Planning Response: Split the spec into "core now" and "later"; ship core first.
7. **Diamond Harbour data scarcity**
   - Impact: Medium
   - Probability: High
   - Planning Response: Transfer-model gap filling, wider error bands, explicit caveat.
8. **Channel change making old years misleading**
   - Impact: Medium
   - Probability: Medium
   - Planning Response: Recency window chosen by backtest; constants-drift audit.

---

## Ideas Requiring Further Research

1. **ERA5 and GloFAS access details**
   - Open Questions: Variables, grid points, latency, and whether archived forecasts exist for honest S validation.
   - Priority: High
   - Suggested Skill: bmad-research
2. **Gauge datums**
   - Open Questions: Chart datum definition for the Haldia and Diamond Harbour gauges; the INCOIS radar zero.
   - Priority: High
   - Suggested Skill: bmad-research, plus formal requests to Survey of India and INCOIS.
3. **Frontend compatibility**
   - Open Questions: Will the captured frontend call this backend directly?
   - Priority: High
   - Suggested Skill: ask the user.
4. **IMD best-track data**
   - Open Questions: Availability and format for the cyclone catalogue.
   - Priority: Medium
   - Suggested Skill: bmad-research
5. **CMEMS sea-level anomaly and ENSO/IOD**
   - Open Questions: Do they add skill for 10–90 day horizons?
   - Priority: Low
   - Suggested Skill: bmad-research

---

## Recommended Next Steps

### Immediate (handoff to next skill)

1. **Pick which ideas enter the spec**
   - Skill: superpowers:brainstorming (spec at `docs/superpowers/specs/2026-09-27-tide-model-design.md`)
   - Key Input: the "core now" list from this report.
2. **Write the implementation plan**
   - Skill: superpowers:writing-plans
   - Key Input: the approved spec; first task is data QC and audit.

### Short-term Planning Actions

1. **Send the formal data requests** (#6)
   - Rationale: Newer Haldia and Diamond Harbour truth is the biggest lever for both accuracy and monitoring.
2. **Create a Copernicus data-service account and API key**
   - Rationale: Needed for ERA5 and GloFAS before S can be built.

### Deferred Considerations

1. **Bore timing (#36), return levels (#35), offshore and remote predictors (#14), probabilistic S (#16), Garden Reach nowcast (#15)**
   - Rationale: Useful but not needed for an accurate core. Revisit after the first backtest.

---

## Follow-up Sessions

- [ ] Reverse Brainstorming on the finished spec before implementation
- [ ] Architecture ideation for serving and API compatibility, if #32 is kept
- [ ] Other: review data-request responses from Survey of India and INCOIS

---

## Decisions to Log

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2026-09-27 | Proposed: data QC and audit is the first implementation stage | Impossible values found in training data | Data pipeline, all models |
| 2026-09-27 | Proposed: calibrated error bands, canary monitoring and fallbacks are required | No live truth for Haldia and Diamond Harbour | Backtest, pipeline, outputs |

---

## Appendix

### Full Idea List

| # | Idea | Category | Impact | Feasibility | Source Technique |
|---|------|----------|--------|-------------|-----------------|
| 1 | Gauge data QC with flags | Data | H | H | Reverse |
| 2 | Per-year constants and mean-level audit | Data | H | H | Reverse |
| 3 | PDF extraction verification | Data | M | H | Reverse |
| 4 | Datum registry | Data | H | M | Starbursting |
| 5 | Fill Diamond Harbour gaps from Haldia | Data | H | M | SCAMPER |
| 6 | Formal data requests | Data | H | M | Starbursting |
| 7 | Monthly official-table archiver | Data | M | H | Reverse |
| 8 | PSMSL 1970–2024 trend check | Data | L | H | SCAMPER |
| 9 | Cyclone catalogue | Data | M | M | Starbursting |
| 10 | Ensemble of harmonic engines | Model | M | H | SCAMPER |
| 11 | Direct event-correction challenger | Model | M | H | SCAMPER |
| 12 | Consistent event definition | Model | M | M | Reverse |
| 13 | Joint seasonal terms across ports | Model | M | M | SCAMPER |
| 14 | Offshore sea-level anomaly, ENSO/IOD | Model | M | L | SCAMPER |
| 15 | Garden Reach live nowcast input | Model | M | L | SCAMPER |
| 16 | Probabilistic S (ensemble forecasts) | Model | M | M | SCAMPER |
| 17 | Live Haldia/Diamond Harbour gauge feed | Model | H | L | Starbursting |
| 18 | Conformal error bands with coverage test | Reliability | H | H | SCAMPER |
| 19 | Canary monitoring | Reliability | H | H | Reverse |
| 20 | Graceful fallback for S | Reliability | H | H | Reverse |
| 21 | Backtest leakage guards | Reliability | H | H | Reverse |
| 22 | Timezone/datum invariant tests, regression snapshots | Reliability | H | H | Reverse |
| 23 | Model card and backtest report | Reliability | M | H | SCAMPER |
| 24 | Extreme-event scoring | Reliability | M | M | Starbursting |
| 25 | Licence and attribution register | Reliability | M | H | Starbursting |
| 26 | Secrets handling | Reliability | H | H | Reverse |
| 27 | Datum-labelled outputs and tidal datums | Outputs | H | H | Starbursting |
| 28 | Navigation windows | Outputs | H | H | Starbursting |
| 29 | Component breakdown | Outputs | M | H | Starbursting |
| 30 | Local moon, sun, spring/neap | Outputs | M | H | Starbursting |
| 31 | Observed and observed-vs-predicted views | Outputs | M | H | Starbursting |
| 32 | Frontend API compatibility | Outputs | H | M | Starbursting |
| 33 | Risk flags | Outputs | M | M | Starbursting |
| 34 | Rate of rise/fall, flood/ebb | Outputs | L | H | Starbursting |
| 35 | Return levels | Outputs | L | M | SCAMPER |
| 36 | Tidal bore timing | Outputs | L | L | Starbursting |

### Sources Referenced

- `data/haldia.csv`, `data/diamond_harbour.csv` (observed gauge heights) and `output/utide_models/*.pkl`
- `API_CONTEXT.md`, `INCOIS_api.txt`, `src/utide_event_model.py`, `src/datums.py`
- This session's analyses: tables vs Haldia 2024 gauge, INCOIS Garden Reach vs tables, seasonal cycle, residual extremes, cross-port residual correlation

### Session Notes

Ideas #10–12 overlap the existing design and were kept as backtest options, not new scope. Techniques ran inline rather than as parallel subagents.

---

**Report Generated By:** BMAD Brainstorm Skill (`/bmad-planning-orchestrator:bmad-brainstorm`)
**Output Path:** `bmad-output/brainstorming-report.md`
**Related Artifacts:** `bmad-output/decision-log.md`
