from __future__ import annotations

import argparse
from pathlib import Path

from .report import write_all_reports
from .symbolic import pretty_symbolic_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Cu-Al-Ni crystallography research toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    report_parser = sub.add_parser("report", help="generate exact/reference markdown reports")
    report_parser.add_argument("--out", default="reports")
    sub.add_parser("symbolic", help="print symbolic pulled-back metrics")
    args = parser.parse_args()
    if args.cmd == "report":
        for report in write_all_reports(Path(args.out)):
            print(report)
    else:
        print(pretty_symbolic_reference())


if __name__ == "__main__":
    main()
