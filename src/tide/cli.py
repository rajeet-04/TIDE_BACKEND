"""Command line: `uv run tide <command> --help`."""
from __future__ import annotations

import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tide", description="Tide model for Haldia and Diamond Harbour.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    return parser

def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)
