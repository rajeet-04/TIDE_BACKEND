"""Command line: `uv run tide <command> --help`."""
from __future__ import annotations

import argparse
from pathlib import Path

from tide.ports import PORTS

def _port(command: argparse.ArgumentParser) -> argparse.ArgumentParser:
    command.add_argument("--port", choices=sorted(PORTS), required=True)
    return command

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tide", description="Tide model for Haldia and Diamond Harbour.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    _port(commands.add_parser("qc", help="flag suspect gauge readings; write data/qc and output/qc")).set_defaults(handler=_qc)
    _port(commands.add_parser("audit", help="fit each year separately; report years that depart from their neighbours")).set_defaults(handler=_audit)

    back = _port(commands.add_parser("backtest", help="score one candidate on the rolling yearly folds"))
    back.add_argument("--candidate", required=True,
                      help="current_pipeline, utide_only, official_tables or a model A name such as A-w8-auto-side-trend")
    back.add_argument("--set", dest="fold_set", choices=["selection", "final", "all"], default="all")
    back.add_argument("--jobs", type=int, default=4)
    back.set_defaults(handler=_backtest)

    api = _port(commands.add_parser("tables-api", help="store one year of official table events from the prediction API"))
    api.add_argument("--year", type=int, required=True)
    api.add_argument("--from-json", type=Path, help="a saved API response (list of rows) instead of a live request")
    api.set_defaults(handler=_tables_api)

    choose = _port(commands.add_parser("select", help="choose model A's settings and model B's corrections on the "
                                                      "selection folds, then score the final folds"))
    choose.add_argument("--jobs", type=int, default=4)
    choose.add_argument("--skip-grid", action="store_true", help="reuse model A's settings from models/<port>/selected.json")
    choose.set_defaults(handler=_select)

    fit = _port(commands.add_parser("fit", help="fit the selected model on all QC-passed readings and save a version"))
    fit.add_argument("--promote", action="store_true",
                     help="re-run the final backtest and make the version current only if it passes")
    fit.add_argument("--jobs", type=int, default=4)
    fit.set_defaults(handler=_fit)

    training = _port(commands.add_parser("train", help="QC, audit, select A and B, score the final folds, save a "
                                                       "version and promote it if it passes"))
    training.add_argument("--jobs", type=int, default=4)
    training.add_argument("--skip-qc", action="store_true", help="keep the committed QC flags and audit")
    training.set_defaults(handler=_train)

    pred = _port(commands.add_parser("predict", help="forecast levels and high and low waters"))
    pred.add_argument("--start", required=True, help="ISO time; without a zone it is read as IST")
    span = pred.add_mutually_exclusive_group(required=True)
    span.add_argument("--end", help="ISO time; without a zone it is read as IST")
    span.add_argument("--hours", type=float)
    pred.add_argument("--step", type=int, default=60, help="minutes between levels (default 60)")
    pred.add_argument("--out-dir", type=Path)
    pred.set_defaults(handler=_predict)
    return parser

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)

def _qc(args: argparse.Namespace) -> None:
    from tide.ports import DATA_DIR
    from tide.qc import run_qc, write_qc
    from tide.store import read_gauge_csv

    table = run_qc(read_gauge_csv(DATA_DIR / f"{args.port}.csv"), args.port)
    print(write_qc(args.port, table).to_string(index=False))

def _audit(args: argparse.Namespace) -> None:
    from tide.audit import audit, write_audit, year_constants
    from tide.store import load_gauge

    table = audit(year_constants(load_gauge(args.port), args.port))
    write_audit(args.port, table)
    shown = ["year", "hours", "sparse", "mean_level_m", "M2_amp_m", "M2_phase_deg", "M4_amp_m", "suspect"]
    print(table[shown].round(3).to_string(index=False))

def _backtest(args: argparse.Namespace) -> None:
    import pandas as pd

    from tide.backtest import folds, run, score_fixed, summary
    from tide.candidates import candidate
    from tide.ports import OUTPUT_DIR, ist_years
    from tide.store import load_gauge
    from tide.tables import load_tables

    if args.candidate == "official_tables":
        events, hours, by = score_fixed(args.port, load_tables(args.port)), pd.DataFrame(), ("test_year",)
    else:
        years = set(ist_years(load_gauge(args.port, passed_only=True)["time_utc"]))
        chosen = [f for f in folds(args.port, years) if args.fold_set == "all" or f.final == (args.fold_set == "final")]
        events, hours = run(args.port, candidate(args.candidate), chosen, n_jobs=args.jobs)
        by = ("final", "test_year", "horizon")
    if events.empty:
        print("nothing to score: no overlap with QC-passed gauge years")
        return
    table = summary(events, hours, by=by)
    out = OUTPUT_DIR / "backtest" / args.port
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"{args.candidate}.csv", index=False)
    print(table.round(3).to_string(index=False))

def _tables_api(args: argparse.Namespace) -> None:
    import json

    from tide.ports import ist_year_start
    from tide.tables import fetch_api, parse_api_rows, save_tables

    start, end = ist_year_start(args.year), ist_year_start(args.year + 1)
    events = (parse_api_rows(json.loads(args.from_json.read_text()), start, end) if args.from_json
              else fetch_api(args.port, start, end))
    merged = save_tables(args.port, events)
    print(f"{len(events)} events in {args.year}; {len(merged)} stored in data/tables/{args.port}.csv")

def _select(args: argparse.Namespace) -> None:
    import json

    from tide.harmonic import HarmonicConfig
    from tide.report import final_report, select, selected_path
    from tide.selection import select_b

    if args.skip_grid:
        saved = json.loads(selected_path(args.port).read_text())
        a = HarmonicConfig.from_name(saved.get("a_candidate", saved["candidate"]))
    else:
        a = select(args.port, args.jobs)
    config = select_b(args.port, a, args.jobs)
    result = final_report(args.port, config, args.jobs)
    print(json.dumps({"candidate": config.name, **result["promotion"]["checks"],
                      "passed": result["promotion"]["passed"]}, indent=2))
    print(f"report: output/backtest/{args.port}/report.md")

def _fit(args: argparse.Namespace) -> None:
    from tide.training import fit_version

    version = fit_version(args.port, promote=args.promote, n_jobs=args.jobs)
    print(f"saved models/{args.port}/{version}{' and made it current' if args.promote else ''}")

def _train(args: argparse.Namespace) -> None:
    import json

    from tide.training import train

    print(json.dumps(train(args.port, n_jobs=args.jobs, quality_control=not args.skip_qc), indent=2))

def _predict(args: argparse.Namespace) -> None:
    from tide.forecast import predict, write_outputs
    from tide.ports import OUTPUT_DIR, to_utc

    result = predict(args.port, args.start, end=args.end, hours=args.hours, step_minutes=args.step)
    out = args.out_dir or OUTPUT_DIR / "forecast" / f"{args.port}_{to_utc(args.start):%Y%m%dT%H%MZ}"
    write_outputs(result, out)
    print(result.events[["state", "time_ist", "height_m", "flags"]].to_string(index=False))
    print(f"model {result.version}; heights above {result.datum}; files in {out}")
