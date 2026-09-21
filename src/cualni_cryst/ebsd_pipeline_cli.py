from __future__ import annotations

"""Command line entry point for the strict real-map EBSD pipeline."""

import argparse
import json
from pathlib import Path
import sys

from .ebsd_pipeline import (
    PipelineConfigError,
    run_pipeline,
    validate_config_only,
)


def _print_json(value) -> None:
    print(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cualni_cryst.ebsd_pipeline_cli",
        description=(
            "Run the reproducible vendor-neutral EBSD transformation-"
            "crystallography pipeline."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate-config",
        help="validate the scientific configuration without reading the EBSD map",
    )
    validate.add_argument("config", type=Path)

    run = sub.add_parser(
        "run",
        help="execute the complete EBSD pipeline and write an atomic run directory",
    )
    run.add_argument("config", type=Path)

    args = parser.parse_args(argv)

    try:
        if args.command == "validate-config":
            _print_json(validate_config_only(args.config))
            return 0

        result = run_pipeline(args.config)
        _print_json(
            {
                "status": "PASS",
                "run_directory": str(result.run_directory),
                "summary_file": str(
                    result.run_directory / "summary.json"
                ),
            }
        )
        return 0

    except (PipelineConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
