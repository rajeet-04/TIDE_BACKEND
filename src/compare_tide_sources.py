"""Reproduce the July-September 2026 model/PDF/live-API comparison.

Run with uv; stages are local, api, compare, or all. Original API JSON and
declared UTC timestamps are preserved. PDF-aligned API timestamps are a
separate diagnostic, never a silent correction of the source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output/comparison_2026_jul_sep"
START, END = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-10-01")
API = "http://117.250.29.124:5008"
PORTS = {"haldia": "HALDIA", "diamond_harbour": "DIAMOND HARBOUR"}
TIME_GATE, HEIGHT_GATE = 30.0, 0.30


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def in_period(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame.datetime_ist.ge(START) & frame.datetime_ist.lt(END)].copy()


def local() -> None:
    import extract_reference_events as extraction
    import utide_event_model as model
    import sklearn
    import utide

    OUT.mkdir(parents=True, exist_ok=True)
    pdf_frames, models, series, metadata, checks = [], [], [], {}, []
    for slug, name in PORTS.items():
        for month, folder in [(7, "JULY 2026"), (8, "AUGUST 2026"), (9, "PDF/INDIAN RESPONSIBILITY_2026")]:
            path = ROOT / "data" / folder / f"{name}.pdf"
            frame = extraction.extract_pdf(path)
            extraction._validate_month(frame, path)
            if not frame.datetime_ist.dt.month.eq(month).all() or not frame.datetime_ist.dt.year.eq(2026).all():
                raise ValueError(f"Unexpected PDF period: {path}")
            frame["port"] = slug
            frame["month"] = month
            frame["source_pdf"] = str(path.relative_to(ROOT))
            pdf_frames.append(frame)
            checks.append({"source_pdf": str(path.relative_to(ROOT)), "sha256": digest(path), "events": len(frame), "days": frame.datetime_ist.dt.day.nunique(), "expected_month": month})
            print(f"PDF {slug} month {month}: {len(frame)} events", flush=True)

        artifact = model.load_model(slug)
        # Reconstruct once; the saved model calibrates turning points, not the curve.
        timestamps = pd.date_range(START - pd.Timedelta(hours=12), END + pd.Timedelta(hours=12), freq="1min")
        heights = model._reconstruct_chunked(timestamps, artifact["coefficients"], chunk_size=20_000)
        raw = model.find_events(timestamps, heights, expected_step_minutes=1)
        events = in_period(model._apply_event_calibration(raw, artifact))
        events.insert(0, "port", slug)
        events.insert(3, "datetime_utc", events.datetime_ist - model.IST_OFFSET)
        models.append(events)
        minute = in_period(pd.DataFrame({"port": slug, "datetime_ist": timestamps, "height_m": heights}))
        minute.insert(2, "datetime_utc", minute.datetime_ist - model.IST_OFFSET)
        series.append(minute)
        record = {k: v for k, v in artifact.items() if k in ("version", "trained_at", "history_start", "history_end", "history_rows", "constituent_count", "calibrator", "calibration", "latitude")}
        record["sha256"] = digest(model.MODEL_DIR / f"{slug}.pkl")
        record["hyperparameters"] = artifact["time_calibrator"].get_params()
        # Independently compare the restored runtime to an existing saved forecast.
        old_path = ROOT / f"output/utide_forecast_{slug}_2026-06-01/hourly_water_levels.csv"
        old = pd.read_csv(old_path, parse_dates=["datetime_ist"])
        predicted = model._reconstruct_chunked(old.datetime_ist, artifact["coefficients"])
        record["saved_hourly_forecast_max_abs_difference_m"] = float(np.max(np.abs(predicted - old.predicted_height_m)))
        old_events_path = old_path.parent / "predicted_tide_events.csv"
        old_events = pd.read_csv(old_events_path, parse_dates=["datetime_ist"])
        regenerated = model.predict_events(artifact, old.datetime_ist.min().to_pydatetime(), (old.datetime_ist.max() + pd.Timedelta(hours=1)).to_pydatetime())
        matched = model.match_events(old_events, regenerated)
        record["saved_event_forecast_check"] = {"expected_events": len(old_events), "matched_events": len(matched), "max_time_difference_minutes": float(matched.absolute_time_error_minutes.max()), "max_height_difference_m": float(matched.absolute_height_error_m.max())}
        metadata[slug] = record
        print(f"MODEL {slug}: {len(minute)} minute rows, {len(events)} events; saved forecast curve difference {record['saved_hourly_forecast_max_abs_difference_m']:.3g}m", flush=True)

    pdf = pd.concat(pdf_frames, ignore_index=True).sort_values(["port", "datetime_ist"])
    pdf.to_csv(OUT / "pdf_events.csv", index=False)
    pd.concat(models, ignore_index=True).to_csv(OUT / "model_events.csv", index=False)
    minutes = pd.concat(series, ignore_index=True)
    minutes.to_csv(OUT / "model_minute.csv", index=False, float_format="%.8f")
    minutes.loc[minutes.datetime_ist.dt.minute.eq(0)].to_csv(OUT / "model_hourly.csv", index=False, float_format="%.8f")
    save_json(OUT / "local_manifest.json", {"period_ist": [START, END], "end_exclusive": True, "runtime": {"python": platform.python_version(), "sklearn": sklearn.__version__, "utide": utide.__version__, "numpy": np.__version__, "pandas": pd.__version__}, "models": metadata, "pdfs": checks, "script_sha256": digest(Path(__file__)), "model_source_sha256": digest(ROOT / "src/utide_event_model.py"), "extractor_source_sha256": digest(ROOT / "src/extract_reference_events.py")})


def api_fetch() -> None:
    raw_dir = OUT / "api_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    port_path = raw_dir / "port-list.json"
    if not port_path.exists():
        with urllib.request.urlopen(API + "/api/port/port-list", timeout=60) as response:
            port_path.write_bytes(response.read())
    directory = json.loads(port_path.read_text(encoding="utf-8-sig"))["data"]
    ids = {}
    for slug, name in PORTS.items():
        candidates = [p for p in directory if p["port_name"].startswith(name)]
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous API identity for {slug}: {candidates}")
        ids[slug] = candidates[0]["port_id"]

    jobs = []
    for slug in PORTS:
        # Extra dates cover the API's unusual range boundaries and both interpretations.
        start = START - pd.Timedelta(days=1)
        stop = END + pd.Timedelta(days=1)
        while start < stop:
            days = min(7, (stop - start).days)
            for route in ("predicted-tide-data", "predicted-one-minute-data"):
                jobs.append((slug, route, start.strftime("%Y-%m-%d"), days))
            start += pd.Timedelta(days=days)

    def fetch(job):
        slug, route, start, days = job
        body = {"port_id": ids[slug], "start_date": start, "days": days}
        path = raw_dir / f"{route}-{slug}-{start}-{days}.json"
        reused = path.exists()
        if not reused:
            request = urllib.request.Request(API + "/api/tide/" + route, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=90) as response:
                        contents = response.read()
                        status = response.status
                    parsed = json.loads(contents)
                    if not isinstance(parsed.get("data"), list):
                        raise ValueError(f"No data array: {str(parsed)[:250]}")
                    path.write_bytes(contents)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = parsed["data"]
        if parsed.get("count", len(rows)) != len(rows):
            raise ValueError(f"Declared count mismatch: {path}")
        if route == "predicted-one-minute-data" and len(rows) != days * 1440:
            raise ValueError(f"Incomplete minute response {path}: {len(rows)} != {days * 1440}")
        if any(r["port_id"] != ids[slug] for r in rows):
            raise ValueError(f"Wrong port returned: {path}")
        return job, rows, {"url": API + "/api/tide/" + route, "request": body, "port": slug, "rows": len(rows), "raw_file": str(path.relative_to(ROOT)), "sha256": digest(path), "http_status": 200, "reused_current_run_cache": reused}

    events, minutes, receipts, failures = [], [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, job): job for job in jobs}
        for future in as_completed(futures):
            try:
                job, rows, receipt = future.result()
                frame = pd.DataFrame(rows)
                frame.insert(0, "port", job[0])
                (minutes if job[1] == "predicted-one-minute-data" else events).append(frame)
                receipts.append(receipt)
                print(f"API {job}: {len(rows)} rows", flush=True)
            except Exception as exc:
                failures.append({"job": futures[future], "error": str(exc)})
                print(f"API FAILURE {futures[future]}: {exc}", flush=True)
    audit = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(), "port_ids": ids, "requests": receipts, "failures": failures, "port_list_sha256": digest(port_path), "period_ist": [START, END], "end_exclusive": True}
    for label, frames in (("events", events), ("minute", minutes)):
        if not frames:
            continue
        data = pd.concat(frames, ignore_index=True)
        data["height_m"] = pd.to_numeric(data.tide_height_m, errors="raise")
        data["datetime_utc"] = pd.to_datetime(data.date_time, utc=True).dt.tz_localize(None)
        data["datetime_ist"] = data.datetime_utc + pd.Timedelta(hours=5, minutes=30)
        if label == "events":
            data["state"] = data.tide_type.str.title()
            if not data.state.isin(["High", "Low"]).all():
                raise ValueError("Unrecognized API tide states")
        keys = ["port", "datetime_utc"] + (["state"] if label == "events" else [])
        if data.groupby(keys).height_m.nunique().gt(1).any():
            raise ValueError("Conflicting API duplicates")
        audit[label + "_duplicate_rows"] = int(data.duplicated(keys).sum())
        data = data.drop_duplicates(keys).sort_values(["port", "datetime_utc"])
        data.to_csv(OUT / f"api_{label}_with_boundaries.csv", index=False)
        in_period(data).to_csv(OUT / f"api_{label}.csv", index=False)
    save_json(OUT / "api_manifest.json", audit)
    if failures:
        raise SystemExit(f"{len(failures)} API requests failed; saved all successful data and failure ledger")


def load(name: str) -> pd.DataFrame:
    frame = pd.read_csv(OUT / name)
    for col in frame:
        if "datetime" in col:
            frame[col] = pd.to_datetime(frame[col])
    return frame


def event_match(reference: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    import utide_event_model as model
    frames = []
    for slug in PORTS:
        obs, pred = reference[reference.port.eq(slug)], candidate[candidate.port.eq(slug)].reset_index(drop=True)
        errors = model.match_events(obs, pred)
        if errors.empty:
            continue
        errors.insert(0, "port", slug)
        frames.append(errors)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summary(errors: pd.DataFrame, count: int) -> dict:
    if errors.empty:
        return {"reference_events": count, "matched_events": 0, "match_rate_percent": 0.0}
    time_error, height_error = errors.time_error_minutes, errors.height_error_m
    return {"reference_events": count, "matched_events": len(errors), "unmatched_events": count-len(errors), "match_rate_percent": 100*len(errors)/count, "time_bias_minutes": float(time_error.mean()), "time_mae_minutes": float(time_error.abs().mean()), "time_rmse_minutes": float(np.sqrt((time_error**2).mean())), "height_bias_m": float(height_error.mean()), "height_mae_m": float(height_error.abs().mean()), "height_rmse_m": float(np.sqrt((height_error**2).mean())), "time_within_30_minutes_percent": 100*int(time_error.abs().le(TIME_GATE).sum())/count, "height_within_0_30m_percent": 100*int(height_error.abs().le(HEIGHT_GATE).sum())/count, "joint_pass_percent": 100*int((time_error.abs().le(TIME_GATE)&height_error.abs().le(HEIGHT_GATE)).sum())/count, "exact_time_and_height_events": int((time_error.abs().lt(1e-6)&height_error.abs().lt(1e-6)).sum())}


def grouped_summary(reference, errors, label):
    rows = []
    for slug in PORTS:
        for month in (0, 7, 8, 9):
            refs = reference[reference.port.eq(slug)]
            selected = errors[errors.port.eq(slug)] if not errors.empty else errors
            if month:
                refs = refs[refs.datetime_ist.dt.month.eq(month)]
                selected = selected[selected.observed_datetime_ist.dt.month.eq(month)] if not selected.empty else selected
            rows.append({"comparison": label, "port": slug, "month": month or "all", **summary(selected, len(refs))})
    return rows


def tuning(reference, errors):
    """Select on August only; refit July+August and freeze for September."""
    rows, parameters, corrected = [], {}, []
    for slug in PORTS:
        data = errors[errors.port.eq(slug)].copy()
        july = data[data.observed_datetime_ist.dt.month.eq(7)]
        august = data[data.observed_datetime_ist.dt.month.eq(8)]
        train = data[data.observed_datetime_ist.dt.month.isin([7, 8])]
        holdout = data[data.observed_datetime_ist.dt.month.eq(9)]
        candidates = [("unchanged", 0), ("global_bias", 0)] + [("state_bias", n) for n in (0, 20, 50, 100)]

        def fit(frame, kind, shrink):
            values = {}
            for state in ("High", "Low"):
                segment = frame if kind == "global_bias" else frame[frame.state.eq(state)]
                scale = 0 if kind == "unchanged" else len(segment)/(len(segment)+shrink)
                values[state] = {"time_subtract_minutes": float(segment.time_error_minutes.mean()*scale), "height_subtract_m": float(segment.height_error_m.mean()*scale), "fit_events": len(segment)}
            return values

        def apply(frame, values):
            out = frame.copy()
            out["time_error_minutes"] -= out.state.map({k:v["time_subtract_minutes"] for k,v in values.items()})
            out["height_error_m"] -= out.state.map({k:v["height_subtract_m"] for k,v in values.items()})
            return out

        scores = []
        for kind, shrink in candidates:
            values = fit(july, kind, shrink)
            metrics = summary(apply(august, values), len(reference[reference.port.eq(slug)&reference.datetime_ist.dt.month.eq(8)]))
            score = (metrics["time_rmse_minutes"]/TIME_GATE)**2 + (metrics["height_rmse_m"]/HEIGHT_GATE)**2
            scores.append((score, kind, shrink))
            rows.append({"port": slug, "evaluation": "August selection (fit July only)", "candidate": kind, "shrinkage_events": shrink, "normalized_rmse_score": score, **metrics})
        _, kind, shrink = min(scores)
        values = fit(train, kind, shrink)
        tuned = apply(holdout, values)
        sept_count = len(reference[reference.port.eq(slug)&reference.datetime_ist.dt.month.eq(9)])
        baseline, result = summary(holdout, sept_count), summary(tuned, sept_count)
        # Acceptance requires both RMSE measures and joint coverage not to worsen.
        accepted = result["time_rmse_minutes"] <= baseline["time_rmse_minutes"] and result["height_rmse_m"] <= baseline["height_rmse_m"] and result["joint_pass_percent"] >= baseline["joint_pass_percent"]
        parameters[slug] = {"selected_on_august": kind, "shrinkage_events": shrink, "fit_period": "2026-07-01 through 2026-08-31", "evaluation_period": "2026-09-01 through 2026-09-30", "candidate_corrections": values, "september_baseline": baseline, "september_candidate": result, "accepted_for_followup": bool(accepted), "production_corrections": values if accepted else {"High": {"time_subtract_minutes": 0.0,"height_subtract_m":0.0},"Low": {"time_subtract_minutes":0.0,"height_subtract_m":0.0}}, "note": "Diagnostic correction only; saved model weights are not changed. September has now been evaluated and is not an untouched holdout for later iterations."}
        rows.append({"port": slug, "evaluation": "September holdout (fit July+August)", "candidate": kind, "shrinkage_events": shrink, **result})
        all_tuned = apply(data, values)
        all_tuned["candidate_datetime_ist"] = all_tuned.predicted_datetime_ist - all_tuned.state.map({k:pd.Timedelta(minutes=v["time_subtract_minutes"]) for k,v in values.items()})
        all_tuned["candidate_height_m"] = all_tuned.predicted_height_m - all_tuned.state.map({k:v["height_subtract_m"] for k,v in values.items()})
        all_tuned["is_september_holdout"] = all_tuned.observed_datetime_ist.dt.month.eq(9)
        corrected.append(all_tuned)
    pd.DataFrame(rows).to_csv(OUT / "tuning_evaluation.csv", index=False)
    pd.concat(corrected, ignore_index=True).to_csv(OUT / "tuning_candidate_events.csv", index=False)
    save_json(OUT / "tuning_parameters.json", {"selection_score": "(time_RMSE/30min)^2+(height_RMSE/0.30m)^2", "pdf_reference": "Published prediction tables, not physical gauge truth", "ports": parameters})
    return parameters


def compare() -> None:
    pdf, model = load("pdf_events.csv"), load("model_events.csv")
    api_full = load("api_events_with_boundaries.csv")
    model_errors = event_match(pdf, model)
    model_errors.to_csv(OUT / "model_vs_pdf.csv", index=False)
    raw_model = model.copy()
    raw_model["datetime_ist"], raw_model["height_m"] = raw_model.raw_datetime_ist, raw_model.raw_height_m
    raw_errors = event_match(pdf, raw_model)
    raw_errors.to_csv(OUT / "raw_utide_vs_pdf.csv", index=False)
    api_declared = in_period(api_full)
    api_errors = event_match(pdf, api_declared)
    api_errors.to_csv(OUT / "api_declared_timezone_vs_pdf.csv", index=False)
    shifts = []
    for offset in (-660, -330, 0, 330, 660):
        adjusted = api_full.copy()
        adjusted["datetime_ist"] += pd.Timedelta(minutes=offset)
        errors = event_match(pdf, in_period(adjusted))
        shifts.append({"additional_minutes_after_UTC_to_IST": offset, **summary(errors, len(pdf))})
    pd.DataFrame(shifts).to_csv(OUT / "api_timezone_diagnostics.csv", index=False)
    best = min(shifts, key=lambda x: (-x["matched_events"], x.get("time_mae_minutes", float("inf"))))
    offset = best["additional_minutes_after_UTC_to_IST"]
    api_aligned = api_full.copy()
    api_aligned["datetime_ist_declared"] = api_aligned.datetime_ist
    api_aligned["datetime_ist"] += pd.Timedelta(minutes=offset)
    api_aligned["empirical_additional_shift_minutes"] = offset
    api_aligned = in_period(api_aligned)
    api_aligned.to_csv(OUT / "api_events_pdf_aligned.csv", index=False)
    aligned_errors = event_match(pdf, api_aligned)
    aligned_errors.to_csv(OUT / "api_pdf_aligned_vs_pdf.csv", index=False)
    rows = grouped_summary(pdf, model_errors, "calibrated model vs PDF") + grouped_summary(pdf, raw_errors, "raw UTide vs PDF") + grouped_summary(pdf, api_errors, "API declared UTC vs PDF") + grouped_summary(pdf, aligned_errors, "API empirically aligned vs PDF")
    event_summary = pd.DataFrame(rows)
    event_summary.to_csv(OUT / "event_metrics.csv", index=False)
    # Keep every PDF reference event, including unmatched values, in reconciliation.
    reconciliation = pdf.rename(columns={"datetime_ist":"reference_datetime_ist","height_m":"reference_height_m"}).copy()
    for label, errors in [("model", model_errors), ("api", aligned_errors)]:
        columns = ["port","state","observed_datetime_ist","predicted_datetime_ist","predicted_height_m","time_error_minutes","height_error_m"]
        selected = errors[columns].rename(columns={"observed_datetime_ist":"reference_datetime_ist", **{c:f"{label}_{c}" for c in columns[3:]}})
        reconciliation = reconciliation.merge(selected, on=["port","state","reference_datetime_ist"], how="left", validate="one_to_one")
    reconciliation["model_time_inconsistent"] = reconciliation.model_time_error_minutes.abs().gt(TIME_GATE)
    reconciliation["model_height_inconsistent"] = reconciliation.model_height_error_m.abs().gt(HEIGHT_GATE)
    reconciliation["model_missing"] = reconciliation.model_predicted_datetime_ist.isna()
    reconciliation["api_time_inconsistent_after_alignment"] = reconciliation.api_time_error_minutes.abs().gt(1.0)
    reconciliation["api_height_inconsistent"] = reconciliation.api_height_error_m.abs().gt(0.011)
    reconciliation["api_missing"] = reconciliation.api_predicted_datetime_ist.isna()
    flags = [c for c in reconciliation if c.endswith("inconsistent") or c.endswith("after_alignment") or c.endswith("missing")]
    reconciliation.to_csv(OUT / "event_reconciliation.csv", index=False)
    inconsistencies = reconciliation[reconciliation[flags].any(axis=1)].copy()
    inconsistencies.to_csv(OUT / "event_inconsistencies.csv", index=False)

    minute_full, model_minute = load("api_minute_with_boundaries.csv"), load("model_minute.csv")
    minute_rows = []
    for label, shift in [("declared_UTC",0), ("PDF_aligned",offset)]:
        minute_api = minute_full.copy()
        minute_api["datetime_ist_declared"] = minute_api.datetime_ist
        minute_api["datetime_ist"] += pd.Timedelta(minutes=shift)
        minute_api = in_period(minute_api)
        if shift:
            minute_api["empirical_additional_shift_minutes"] = shift
            minute_api.to_csv(OUT / "api_minute_pdf_aligned.csv", index=False)
        joined = model_minute.merge(minute_api[["port","datetime_ist","height_m"]], on=["port","datetime_ist"], suffixes=("_model","_api"), how="left", validate="one_to_one")
        joined["height_difference_m"] = joined.height_m_model - joined.height_m_api
        if label == "PDF_aligned":
            joined.to_csv(OUT / "minute_comparison.csv", index=False, float_format="%.8f")
            joined[joined.height_difference_m.abs().gt(HEIGHT_GATE)|joined.height_m_api.isna()].to_csv(OUT / "minute_inconsistencies.csv", index=False, float_format="%.8f")
        for (slug, month), group in joined.groupby(["port",joined.datetime_ist.dt.month]):
            d = group.height_difference_m.dropna()
            minute_rows.append({"timestamp_interpretation":label,"port":slug,"month":int(month),"expected_rows":len(group),"matched_rows":len(d),"missing_rows":int(group.height_m_api.isna().sum()),"bias_m":float(d.mean()),"mae_m":float(d.abs().mean()),"rmse_m":float(np.sqrt((d**2).mean())),"difference_above_0_30m_rows":int(d.abs().gt(HEIGHT_GATE).sum()),"difference_above_0_30m_percent":100*int(d.abs().gt(HEIGHT_GATE).sum())/len(group)})
    pd.DataFrame(minute_rows).to_csv(OUT / "minute_metrics.csv", index=False)
    parameters = tuning(pdf, model_errors)

    quality = []
    for name, frame, grain in [("PDF events",pdf,["port","datetime_ist"]),("model events",model,["port","datetime_ist"]),("API aligned events",api_aligned,["port","datetime_ist"]),("model minute",model_minute,["port","datetime_ist"]),("API aligned minute",minute_api,["port","datetime_ist"])]:
        for slug in PORTS:
            segment=frame[frame.port.eq(slug)]
            quality.append({"source":name,"port":slug,"rows":len(segment),"duplicates":int(segment.duplicated(grain).sum()),"null_times":int(segment.datetime_ist.isna().sum()),"null_heights":int(segment.height_m.isna().sum()),"first_ist":str(segment.datetime_ist.min()),"last_ist":str(segment.datetime_ist.max()),"covered_days":int(segment.datetime_ist.dt.normalize().nunique())})
    pd.DataFrame(quality).to_csv(OUT / "source_quality.csv", index=False)
    plot_results(model_errors, model_minute, minute_api)
    result={"period_ist":[START,END],"end_exclusive":True,"pdf_events":len(pdf),"model_events":len(model),"api_events_pdf_aligned":len(api_aligned),"model_minute_rows":len(model_minute),"api_minute_rows_pdf_aligned":len(minute_api),"inconsistent_reference_events":len(inconsistencies),"api_timezone_additional_shift_minutes":offset,"api_declared_timezone":summary(api_errors,len(pdf)),"api_pdf_aligned":summary(aligned_errors,len(pdf)),"model_vs_pdf":summary(model_errors,len(pdf)),"source_authority":"PDFs are the local-calendar reference for published tide predictions; none of the three sources establishes physical gauge truth.","tuning":parameters}
    save_json(OUT / "comparison_summary.json",result)
    report(result,event_summary,minute_rows)
    notebook()
    print(json.dumps({k:v for k,v in result.items() if k not in ("tuning",)},indent=2,default=str),flush=True)


def plot_results(errors, model_minute, api_minute):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,7),sharex="col")
    for row,slug in enumerate(PORTS):
        segment=errors[errors.port.eq(slug)]
        for col,(field,gate,unit) in enumerate([("time_error_minutes",TIME_GATE,"minutes"),("height_error_m",HEIGHT_GATE,"metres")]):
            for state,color in [("High","#c04b46"),("Low","#2a788e")]:
                s=segment[segment.state.eq(state)]
                axes[row,col].scatter(s.observed_datetime_ist,s[field],s=9,label=state,color=color,alpha=.8)
            for value in [-gate,0,gate]: axes[row,col].axhline(value,color="grey",ls="--",lw=.8)
            axes[row,col].set_title(f"{PORTS[slug]} model minus PDF: {unit}")
            axes[row,col].legend(); axes[row,col].grid(alpha=.2)
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/"event_errors.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(13,7))
    for ax,slug in zip(axes,PORTS):
        for data,label,color in [(model_minute,"Raw UTide minute curve","#c04b46"),(api_minute,"API minute curve, empirical timestamp alignment","#2a788e")]:
            s=data[data.port.eq(slug)&data.datetime_ist.ge("2026-09-01")&data.datetime_ist.lt("2026-09-04")]
            ax.plot(s.datetime_ist,s.height_m,label=label,color=color,lw=1.3)
        ax.set_title(f"{PORTS[slug]}: September 1-3 (IST)");ax.set_ylabel("Height (m)");ax.legend();ax.grid(alpha=.2)
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/"minute_curve_comparison.png",dpi=160);plt.close(fig)


def report(result, event_summary, minute_rows):
    local_manifest=json.loads((OUT/"local_manifest.json").read_text())
    lines=["# July-September 2026 tide source comparison","","Period: 2026-07-01 00:00 IST through 2026-10-01 00:00 IST (exclusive). Ports: Haldia and Diamond Harbour, the two ports supported by the saved model.","","## Source selection and timestamp finding","",result["source_authority"],"","The API raw JSON and declared UTC interpretation are retained. `api_*_pdf_aligned.csv` is a separate empirical correction, not an assertion about the API's intended timezone contract.",f"Best tested additional shift after ordinary UTC-to-IST conversion: **{result['api_timezone_additional_shift_minutes']} minutes**. API/PDF exact event agreement after this shift: **{result['api_pdf_aligned']['exact_time_and_height_events']} / {len(load('pdf_events.csv'))}**.","","PDFs contain event times and heights only. They cannot independently validate the minute curve; minute comparisons measure disagreement between two predictions.","","The UTide minute/hourly curve is uncalibrated. Saved Extra Trees calibration changes event times and heights only, so calibrated event markers need not lie on the minute curve.","","## Model versus PDF","","| Port | Month | Events matched / reference | Time MAE (min) | Height MAE (m) | Joint pass (%) |","|---|---|---:|---:|---:|---:|"]
    for row in event_summary[event_summary.comparison.eq("calibrated model vs PDF")].to_dict("records"):
        lines.append(f"| {row['port']} | {row['month']} | {row['matched_events']} / {row['reference_events']} | {row['time_mae_minutes']:.3f} | {row['height_mae_m']:.4f} | {row['joint_pass_percent']:.2f} |")
    lines += ["","Joint gate: absolute event timing error <=30 minutes and absolute event height error <=0.30m. Unmatched reference events count as failures.","","## Minute curve disagreement after empirical API alignment","","| Port | Month | Matched / expected minutes | RMSE (m) | Difference >0.30m (%) |","|---|---|---:|---:|---:|"]
    for row in minute_rows:
        if row["timestamp_interpretation"]=="PDF_aligned": lines.append(f"| {row['port']} | {row['month']} | {row['matched_rows']} / {row['expected_rows']} | {row['rmse_m']:.4f} | {row['difference_above_0_30m_percent']:.2f} |")
    lines += ["","## Inconsistencies and model tuning","",f"`event_inconsistencies.csv` contains {result['inconsistent_reference_events']} PDF events with missing matches or timing/height disagreement. `event_reconciliation.csv` retains every reference event. Minute disagreements are separately recorded in `minute_inconsistencies.csv`.","","Bias candidates were fitted on July, selected on August using normalized timing/height RMSE, refitted on July+August, and evaluated once on September. September was not used for candidate selection. Candidates and rejected results are retained; original model files are not modified.",""]
    for slug,p in result["tuning"].items():
        b,c=p["september_baseline"],p["september_candidate"]
        lines += [f"### {PORTS[slug]}",f"Selected candidate: `{p['selected_on_august']}`, shrinkage={p['shrinkage_events']}; follow-up acceptance={p['accepted_for_followup']}.",f"September time RMSE {b['time_rmse_minutes']:.3f} -> {c['time_rmse_minutes']:.3f} minutes; height RMSE {b['height_rmse_m']:.4f} -> {c['height_rmse_m']:.4f}m; joint pass {b['joint_pass_percent']:.2f} -> {c['joint_pass_percent']:.2f}%.",f"Candidate parameters: `{json.dumps(p['candidate_corrections'])}`.",""]
    lines += ["Further tuning experiments: compare UTide trend enabled/disabled using rolling historical holdouts; test calibration `min_samples_leaf` in [15,30,60], `max_features` in [0.7,0.9,1.0], and recent calibration windows of [2,4,6] years, preserving `random_state=1` and `n_estimators=160` for the initial comparison. These are proposed experiments, not measured winners. Inspect high/low bias before adding model complexity. Resolve the API serialization offset before tuning a model to API timestamps.","","## Reproducibility and evidence","","Generated from unchanged saved `.pkl` models. Exact hashes, training metadata, dependency versions, and comparisons to existing June forecast artifacts are in `local_manifest.json`. HTTP request bodies, fetch time, response counts, hashes, and failure ledger are in `api_manifest.json`; every response is stored under `api_raw/`.","","Artifacts: `pdf_events.csv`, `model_events.csv`, `model_minute.csv`, `model_hourly.csv`, `api_events.csv`, `api_minute.csv`, empirical alignment CSVs, `event_metrics.csv`, `minute_metrics.csv`, `source_quality.csv`, `tuning_parameters.json`, `tuning_evaluation.csv`, and `comparison.ipynb`.","","Run from the repository root:","```powershell","$env:UV_CACHE_DIR = Join-Path (Get-Location) '.runtime/cache'","$env:UV_PYTHON_INSTALL_DIR = Join-Path (Get-Location) '.runtime/python'","$env:PYTHONUTF8 = '1'","uv run --no-sync --with scikit-learn==1.8.0 --with utide==0.3.1 python src/compare_tide_sources.py --stage local","uv run --no-sync python src/compare_tide_sources.py --stage api","uv run --no-sync --with scikit-learn==1.8.0 --with utide==0.3.1 python src/compare_tide_sources.py --stage compare","```","","Cached API responses are reused on rerun; remove explicitly selected response files if a fresh fetch is needed. Model and PDF source hashes are recorded for this run."]
    for slug,m in local_manifest["models"].items(): lines.append(f"Runtime check {slug}: saved June hourly maximum difference={m['saved_hourly_forecast_max_abs_difference_m']:.3g}m; saved event check={json.dumps(m['saved_event_forecast_check'])}.")
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def notebook():
    cells=[{"cell_type":"markdown","metadata":{},"source":["# July-September 2026 tide comparison\n","PDFs are published prediction references, not physical gauge truth. API timestamp alignment is an empirical diagnostic; raw responses are preserved. See REPORT.md for methodology and interpretation."]}, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["from pathlib import Path\n","import json, pandas as pd, numpy as np\n","base = Path.cwd()\n","if not (base / 'event_metrics.csv').exists():\n","    base = base / 'output/comparison_2026_jul_sep'\n","pd.read_csv(base / 'source_quality.csv')"]}, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["pd.read_csv(base / 'api_timezone_diagnostics.csv')"]}, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["pd.read_csv(base / 'event_metrics.csv')"]}, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["errors = pd.read_csv(base / 'model_vs_pdf.csv')\n","errors.groupby(['port','state']).agg(time_bias_minutes=('time_error_minutes','mean'), height_bias_m=('height_error_m','mean'), events=('state','size'))"]}, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["# Independent recomputation of important model metrics from event-level rows.\n","for port, e in errors.groupby('port'):\n","    joint = (e.time_error_minutes.abs() <= 30) & (e.height_error_m.abs() <= .30)\n","    print(port, 'time RMSE', np.sqrt(np.mean(e.time_error_minutes**2)),\n","          'height RMSE', np.sqrt(np.mean(e.height_error_m**2)), 'joint hits', joint.sum())"]}, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["pd.read_csv(base / 'minute_metrics.csv')"]}, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["pd.read_csv(base / 'tuning_evaluation.csv')"]}, {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":["json.loads((base / 'tuning_parameters.json').read_text())"]}]
    save_json(OUT/"comparison.ipynb",{"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":platform.python_version()}},"nbformat":4,"nbformat_minor":5})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["local","api","compare","all"],default="all")
    args=parser.parse_args()
    for stage,operation in [("local",local),("api",api_fetch),("compare",compare)]:
        if args.stage in (stage,"all"): operation()
