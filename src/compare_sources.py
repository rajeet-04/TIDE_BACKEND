"""Compare Jul-Sep 2026 tide events from three sources: our UTide model,
the monthly reference-table PDFs, and the remote tide API.

Writes everything to output/source_comparison_2026q3/:
  model_events.csv, pdf_events.csv, api_events.csv   raw per-source events
  pairs_<a>_vs_<b>.csv                               matched event pairs
  inconsistencies.csv                                every event that fails a gate or is unmatched
  metrics.json, tuning.json                          summary + proposed model corrections

Usage:
  uv run python src/compare_sources.py
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from extract_reference_events import extract_pdf
from utide_event_model import (
    HEIGHT_GATE_M,
    IST_OFFSET,
    PORTS,
    TIME_GATE_MINUTES,
    load_model,
    match_events,
    predict_events,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "source_comparison_2026q3"
START, END = datetime(2026, 7, 1), datetime(2026, 10, 1)
PDF_MONTHS = ("JULY", "AUGUST", "SEPTEMBER")
API = "http://117.250.29.124:5008/api/tide/predicted-tide-data"
API_CLOCK_SHIFT = pd.Timedelta(hours=5, minutes=30)  # re-measured every run, see measured_api_shift()
API_PORT_IDS = {
    "haldia": "ef43eb28-8fe3-400d-a837-6d5a69e5ab59",
    "diamond_harbour": "6fdb84c4-1459-4f5a-960e-a5c50acfdeba",
}


def model_events() -> pd.DataFrame:
    frames = []
    for slug, port in PORTS.items():
        ev = predict_events(load_model(slug), START, END)
        frames.append(ev.assign(port=port["name"]))
    return pd.concat(frames, ignore_index=True)


def pdf_events() -> tuple[pd.DataFrame, list[str]]:
    frames, missing = [], []
    for month in PDF_MONTHS:
        for port in PORTS.values():
            path = ROOT / "data" / f"{month} 2026" / f"{port['name']}.pdf"
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            frames.append(extract_pdf(path))
    return pd.concat(frames, ignore_index=True), missing


def api_events() -> pd.DataFrame:
    frames = []
    for slug, port_id in API_PORT_IDS.items():
        rows = []
        for chunk_start in pd.date_range(START, END, freq="30D", inclusive="left"):  # API caps days at 30
            body = json.dumps({"port_id": port_id, "start_date": chunk_start.date().isoformat(), "days": "30"})
            req = urllib.request.Request(API, body.encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                rows += json.load(resp)["data"]
        df = pd.DataFrame(rows).drop_duplicates("date_time")
        # API labels timestamps 'Z', but they sit 11h before the PDF's IST times (IST offset
        # subtracted twice). Keep the as-labelled value; datetime_ist adds the shift measured in main().
        labelled = pd.to_datetime(df["date_time"]).dt.tz_localize(None) + IST_OFFSET
        frames.append(pd.DataFrame({
            "port": PORTS[slug]["name"],
            "state": df["tide_type"].str.title(),
            "datetime_ist_as_labelled": labelled,
            "datetime_ist": labelled + API_CLOCK_SHIFT,
            "height_m": df["tide_height_m"].astype(float),
        }))
    out = pd.concat(frames, ignore_index=True)
    return out[(out["datetime_ist"] >= START) & (out["datetime_ist"] < END)]


def measured_api_shift(pdf: pd.DataFrame, api: pd.DataFrame) -> dict:
    """Offset between PDF IST and API as-labelled IST, from events with identical port/state/height."""
    j = pd.merge_asof(pdf.sort_values("datetime_ist"), api.sort_values("datetime_ist_as_labelled"),
                      left_on="datetime_ist", right_on="datetime_ist_as_labelled",
                      by=["port", "state", "height_m"], direction="nearest",
                      tolerance=pd.Timedelta(hours=12), suffixes=("_pdf", "_api"))
    d = (j.datetime_ist_pdf - j.datetime_ist_as_labelled).dropna().dt.total_seconds() / 60
    return {"events_joined": int(d.size), "median_shift_min": float(d.median()),
            "share_exactly_median_pct": round(100 * float((d == d.median()).mean()), 1)}


def compare(ref: pd.DataFrame, cand: pd.DataFrame, ref_name: str, cand_name: str) -> pd.DataFrame:
    """Match cand against ref (ref is treated as 'observed')."""
    frames = []
    for port in ref["port"].unique():
        r = ref[ref["port"] == port].reset_index(drop=True)
        c = cand[cand["port"] == port].reset_index(drop=True)
        # only compare the overlapping window
        lo, hi = max(r.datetime_ist.min(), c.datetime_ist.min()), min(r.datetime_ist.max(), c.datetime_ist.max())
        r = r[(r.datetime_ist >= lo) & (r.datetime_ist <= hi)]
        c = c[(c.datetime_ist >= lo - pd.Timedelta(hours=3)) & (c.datetime_ist <= hi + pd.Timedelta(hours=3))]
        m = match_events(r[["state", "datetime_ist", "height_m"]], c[["state", "datetime_ist", "height_m"]].reset_index(drop=True))
        m.insert(0, "port", port)
        m["unmatched_ref"] = False
        unmatched = r[~r.datetime_ist.isin(m.observed_datetime_ist)]
        um = pd.DataFrame({"port": port, "state": unmatched.state, "observed_datetime_ist": unmatched.datetime_ist,
                           "observed_height_m": unmatched.height_m, "unmatched_ref": True})
        frames += [m, um]
    out = pd.concat(frames, ignore_index=True).drop(columns="predicted_index", errors="ignore")
    out = out.rename(columns=lambda k: k.replace("observed", ref_name).replace("predicted", cand_name))
    out.insert(1, "comparison", f"{cand_name}_vs_{ref_name}")
    return out


def summarize(pairs: pd.DataFrame) -> dict:
    res = {}
    for port, g in pairs.groupby("port"):
        m = g[~g.unmatched_ref]
        t, h = m.time_error_minutes, m.height_error_m
        res[port] = {
            "reference_events": len(g), "matched": len(m), "unmatched": int(g.unmatched_ref.sum()),
            "time_bias_min": round(t.mean(), 2), "time_mae_min": round(t.abs().mean(), 2),
            "time_rmse_min": round(float(np.sqrt((t**2).mean())), 2),
            "height_bias_m": round(h.mean(), 3), "height_mae_m": round(h.abs().mean(), 3),
            "height_rmse_m": round(float(np.sqrt((h**2).mean())), 3),
            "within_time_gate_pct": round(100 * (t.abs() <= TIME_GATE_MINUTES).sum() / len(g), 1),
            "within_height_gate_pct": round(100 * (h.abs() <= HEIGHT_GATE_M).sum() / len(g), 1),
            "joint_pct": round(100 * ((t.abs() <= TIME_GATE_MINUTES) & (h.abs() <= HEIGHT_GATE_M)).sum() / len(g), 1),
        }
    return res


def tuning(pairs: pd.DataFrame) -> dict:
    """Post-hoc corrections for model vs PDF. Fit on July, scored on August (no leakage)."""
    m = pairs[~pairs.unmatched_ref].copy()
    m["month"] = m.pdf_datetime_ist.dt.month
    out = {}
    for port, g in m.groupby("port"):
        fit, test = g[g.month == 7], g[g.month == 8]
        per_state = {}
        for state, s in fit.groupby("state"):
            a, b = np.polyfit(s.model_height_m, s.pdf_height_m, 1)  # pdf ≈ a*model + b
            per_state[state] = {
                "time_offset_min": round(-s.time_error_minutes.mean(), 1),  # add to model time
                "height_offset_m": round(-s.height_error_m.mean(), 3),      # add to model height
                "height_scale_a": round(a, 4), "height_intercept_b": round(b, 3),
            }
        # evaluate on August
        def corrected(row, kind):
            p = per_state[row.state]
            if kind == "offset":
                return row.model_height_m + p["height_offset_m"]
            return p["height_scale_a"] * row.model_height_m + p["height_intercept_b"]
        t_corr = test.apply(lambda r: r.time_error_minutes + per_state[r.state]["time_offset_min"], axis=1)
        h_off = test.apply(lambda r: corrected(r, "offset"), axis=1) - test.pdf_height_m
        h_lin = test.apply(lambda r: corrected(r, "linear"), axis=1) - test.pdf_height_m
        rmse = lambda x: round(float(np.sqrt((x**2).mean())), 3)
        out[port] = {
            "fit_month": "2026-07", "test_month": "2026-08", "per_state": per_state,
            "august_time_rmse_min": {"raw": rmse(test.time_error_minutes), "offset": rmse(t_corr)},
            "august_height_rmse_m": {"raw": rmse(test.height_error_m), "offset": rmse(h_off), "linear": rmse(h_lin)},
            "august_joint_pct": {
                "raw": round(100 * ((test.time_error_minutes.abs() <= TIME_GATE_MINUTES) & (test.height_error_m.abs() <= HEIGHT_GATE_M)).mean(), 1),
                "corrected": round(100 * ((t_corr.abs() <= TIME_GATE_MINUTES) & (h_lin.abs() <= HEIGHT_GATE_M)).mean(), 1),
            },
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = model_events()
    pdf, missing = pdf_events()
    api = api_events()
    for name, df in (("model", model), ("pdf", pdf), ("api", api)):
        df.to_csv(OUT / f"{name}_events.csv", index=False)
        print(f"{name}: {len(df)} events  {df.datetime_ist.min()} -> {df.datetime_ist.max()}")
    if missing:
        print("missing PDFs:", missing)
    shift = measured_api_shift(pdf, api)
    print("API clock shift vs PDF:", shift)
    if shift["median_shift_min"] != API_CLOCK_SHIFT.total_seconds() / 60:
        raise SystemExit(f"API_CLOCK_SHIFT is stale; measured {shift}")

    pairs = {
        "model_vs_pdf": compare(pdf, model, "pdf", "model"),
        "api_vs_pdf": compare(pdf, api, "pdf", "api"),
        "model_vs_api": compare(api, model, "api", "model"),
    }
    for k, v in pairs.items():
        v.to_csv(OUT / f"pairs_{k}.csv", index=False)

    bad = []
    for k, v in pairs.items():
        ref, cand = k.split("_vs_")[1], k.split("_vs_")[0]
        flag = v.unmatched_ref | (v[f"absolute_time_error_minutes"].abs() > TIME_GATE_MINUTES) | (v["absolute_height_error_m"] > HEIGHT_GATE_M)
        bad.append(v[flag].assign(reason=np.where(v[flag].unmatched_ref, "unmatched",
                   np.where(v[flag].absolute_time_error_minutes > TIME_GATE_MINUTES, "time>30min", "height>0.30m"))))
    pd.concat(bad, ignore_index=True).to_csv(OUT / "inconsistencies.csv", index=False)

    metrics = {"window": [START.isoformat(), END.isoformat()], "missing_pdfs": missing, "api_clock_shift": shift,
               **{k: summarize(v) for k, v in pairs.items()}}
    tune = tuning(pairs["model_vs_pdf"])
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (OUT / "tuning.json").write_text(json.dumps(tune, indent=2))
    print(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(tune, indent=2))


if __name__ == "__main__":
    main()
