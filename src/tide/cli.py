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
