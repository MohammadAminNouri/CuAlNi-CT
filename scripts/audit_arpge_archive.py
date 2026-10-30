#!/usr/bin/env python3
"""Non-executing archive-scale ARPGE 2021 source geometry audit.

Usage:
  python scripts/audit_arpge_archive.py '/path/to/ARPGE_Sept2021.zip' \
    --output archive_science_audit.json

Only reads static protocol-0 numeric/string records; neither original source
nor potentially executable NumPy pickle objects are executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from cualni_cryst.arpge_archive import (
    inspect_arpge_archive,audit_legacy_orientation_presets,
)
from cualni_cryst.arpge_genova_or import audit_archived_or_groupoids


def run(zip_path:Path,output:Path)->dict:
    inventory=inspect_arpge_archive(str(zip_path))
    geometry=audit_legacy_orientation_presets(inventory)
    groupoid=audit_archived_or_groupoids(inventory,geometry)
    digest=hashlib.sha256()
    with zip_path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    report={
        "archive_filename":zip_path.name,
        "archive_sha256":digest.hexdigest(),
        "source_inventory":inventory,
        "transition_geometry":geometry,
        "numerical_genova_or_groupoids":groupoid,
        "original_python2_code_executed":False,
        "original_numpy_pickle_executed":False,
        "physical_or_validated":False,
        "source_byte_redistribution":False,
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    return report


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive",type=Path)
    cli.add_argument("--output",type=Path,default=Path("arpge_2021_safe_audit.json"))
    a=cli.parse_args()
    report=run(a.archive,a.output)
    inv=report["source_inventory"]
    print("Decoded transitions:",inv["transition_count"])
    print("Decoded crystal metrics:",inv["crystal_metric_count"])
    print("Rejected source records:",len(inv["rejected_records"]))
    print("Geometry statuses:",report["transition_geometry"]["status_counts"])
    print("OR groupoid statuses:",report["numerical_genova_or_groupoids"]["status_counts"])
    print("SHA-256:",report["archive_sha256"])
    print("Results:",a.output)
    if inv["rejected_records"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
