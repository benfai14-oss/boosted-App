"""
Command-line interface (CLI) 
"""

from __future__ import annotations

import argparse
import subprocess
from typing import List, Optional


def run_ingest(args: argparse.Namespace) -> None:
  
    cmd = [
        "python", "-m", "scripts.pull_all",
        "--commodity", args.commodity,
        "--regions", args.regions,
        "--start", args.start,
        "--end", args.end,
    ]

    if args.output:
        cmd.extend(["--output", args.output])

    subprocess.run(cmd, check=True)
    print(f"Ingestion completed. Data saved to {args.output or 'data/silver'}")


def main(argv: Optional[List[str]] = None) -> None:
    """Entrypoint for the ingestion CLI."""
    parser = argparse.ArgumentParser(description="Global Climate Hedging CLI – Ingestion only")
    sub = parser.add_subparsers(dest="command")

    # Ingest command
    ingest_parser = sub.add_parser("ingest", help="Download raw data into silver layer")
    ingest_parser.add_argument("--commodity", required=True, help="Name of the commodity (e.g. wheat)")
    ingest_parser.add_argument("--regions", required=True, help="Comma-separated region identifiers")
    ingest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    ingest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    ingest_parser.add_argument("--output", help="Output file path (defaults to data/silver)")
    ingest_parser.set_defaults(func=run_ingest)

    parsed = parser.parse_args(argv)
    if not hasattr(parsed, "func"):
        parser.print_help()
        return
    parsed.func(parsed)


if __name__ == "__main__":
    main()
