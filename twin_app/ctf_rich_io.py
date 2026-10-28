from __future__ import annotations

"""Maximal lossless-ish numeric inspection of native Oxford/HKL CTF tables.

A CTF is a vendor-indexing output, not raw Kikuchi patterns. Preserve original
reported Euler values/quality columns and the declared acquisition metadata;
never infer missing columns, space-group evidence, sample-axis corrections,
a parent OR or the ground-truth identity of a phase.

Only the native CTF reader is used for orientation/segmentation downstream.
This supplementary catalog is for fast raw maps, quality distributions,
per-phase summaries, validity diagnostics and reproducible exports.
"""

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cualni_cryst.ebsd_io import _ctf_header_and_table, _header_key_values
from twin_app.ctf_independent import MAX_CTF_BYTES


def inspect_ctf_measurements(file_bytes: bytes, filename: str) -> dict[str, Any]:
    if (not file_bytes or len(file_bytes) > MAX_CTF_BYTES
            or Path(filename).name != filename
            or Path(filename).suffix.lower() != ".ctf"):
        raise ValueError("Use a simple .ctf filename, maximum 35 MB")
    try:
        lines = file_bytes.decode("utf-8-sig", errors="strict").splitlines()
    except UnicodeError as exc:
        raise ValueError(
            "Unsupported CTF text encoding; re-export as Unicode/UTF-8"
        ) from exc
    if not lines or lines[0].strip() != "Channel Text File":
        raise ValueError("CTF header not found")
    header, columns, records = _ctf_header_and_table(lines)
    if not records:
        raise ValueError("CTF has no point data")
    if len(set(c.lower() for c in columns)) != len(columns):
        raise ValueError("CTF contains ambiguous duplicate column names")
    lookup = {name.lower(): name for name in columns}
    required = {"phase", "x", "y", "euler1", "euler2", "euler3"}
    missing = required - set(lookup)
    if missing:
        raise ValueError(f"CTF missing required fields: {sorted(missing)}")
    # Native loader uses the same table header and rejects invalid orientations.
    # Here every field is kept as reported (including vendor-specific columns).
    df = pd.read_csv(
        io.StringIO("\n".join(records)),
        sep=r"\s+", names=columns, engine="python", header=None,
    )
    if len(df) != len(records):
        raise ValueError("CTF numeric row count mismatch")
    converted: dict[str, np.ndarray] = {}
    unused_columns: list[str] = []
    required_numeric = {"phase","x","y","z","euler1","euler2","euler3"}
    for field in columns:
        numeric = pd.to_numeric(df[field], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(numeric)):
            if field.lower() in required_numeric:
                raise ValueError(
                    f"CTF required field {field} contains missing/nonnumeric/nonfinite values"
                )
            # Optional vendor annotations can be textual. Keep the source CTF
            # intact in the ZIP but do not invent a numeric map for that field.
            unused_columns.append(field)
            continue
        converted[field] = numeric
    phase_raw = converted[lookup["phase"]]
    if np.any(phase_raw != np.rint(phase_raw)) or np.any(phase_raw < 0):
        raise ValueError("CTF phase IDs must be nonnegative integers")
    phase = phase_raw.astype(int)
    indexed = phase > 0
    x, y = converted[lookup["x"]], converted[lookup["y"]]
    z = converted[lookup["z"]] if "z" in lookup else np.zeros(len(x))
    if not (len(x) == len(y) == len(z) == len(phase)):
        raise ValueError("CTF coordinate and phase lengths differ")
    meta = _header_key_values(header)
    maps: dict[str, np.ndarray] = {
        "x": x, "y": y, "z": z,
        "phase": phase.astype(float),
        "indexed": indexed.astype(float),
    }
    for field in columns:
        canonical = field.upper()
        if field not in converted:
            continue
        if field.lower() in {"phase", "x", "y", "z"}:
            continue
        value = converted[field].copy()
        if field.lower() in {"euler1", "euler2", "euler3", "mad"}:
            # Unindexed rows frequently contain placeholder zero angles/MAD.
            # Avoid plotting these as evidence of a measured orientation.
            value[~indexed] = np.nan
        maps[canonical] = value

    active = np.array(sorted(set(int(x) for x in phase if x > 0)),dtype=int)
    declared_phase_counts = {
        int(p): int(np.count_nonzero(phase == p)) for p in active
    }
    quality_summary = []
    for field in columns:
        if field not in converted or field.lower() in {
            "phase", "x", "y", "z","euler1", "euler2", "euler3"
        }:
            continue
        observed = converted[field][indexed]
        if not len(observed):
            continue
        quality_summary.append({
            "column": field,
            "count_indexed": int(len(observed)),
            "min": float(np.min(observed)),
            "p05": float(np.percentile(observed, 5)),
            "median": float(np.median(observed)),
            "p95": float(np.percentile(observed, 95)),
            "max": float(np.max(observed)),
            "mean": float(np.mean(observed)),
        })
    phase_rows = []
    for p in sorted(set(int(v) for v in phase)):
        mask = phase == p
        row = {"phase_id":p,"points":int(np.count_nonzero(mask)),
               "fraction_percent":float(100*np.mean(mask))}
        for field in ("MAD","BANDS","BC","BS","ERROR"):
            if field in maps:
                subset=maps[field][mask]
                subset=subset[np.isfinite(subset)]
                row[f"{field}_median"]=(
                    float(np.median(subset)) if subset.size else None
                )
        phase_rows.append(row)

    # Infer 2-D completeness from coordinates, never blindly reshape by
    # XCells*YCells. 3-D maps must be browsed by physical z-slice.
    rounded_x, xi=np.unique(np.round(x,6),return_inverse=True)
    rounded_y, yi=np.unique(np.round(y,6),return_inverse=True)
    rounded_z, zi=np.unique(np.round(z,6),return_inverse=True)
    possible = int(len(rounded_x)*len(rounded_y)*len(rounded_z))
    key=np.ravel_multi_index((zi,yi,xi),
                             (len(rounded_z),len(rounded_y),len(rounded_x)))
    unique_points=int(np.unique(key).size)
    grid = {
        "n_x": int(len(rounded_x)), "n_y": int(len(rounded_y)),
        "n_z": int(len(rounded_z)),
        "complete_unique_cartesian_grid": bool(
            possible==len(x)==unique_points
        ),
        "duplicate_coordinates": int(len(x)-unique_points),
        "header_xcells": meta.get("XCELLS"),
        "header_ycells": meta.get("YCELLS"),
        "header_zcells": meta.get("ZCELLS", "1"),
        "coordinate_units": "CTF units (typically micrometres; verify acquisition)",
        "min_x": float(np.min(x)), "max_x": float(np.max(x)),
        "min_y": float(np.min(y)), "max_y": float(np.max(y)),
        "min_z": float(np.min(z)), "max_z": float(np.max(z)),
    }
    header_dimensions = (
        meta.get("XCELLS"), meta.get("YCELLS"), meta.get("ZCELLS","1")
    )
    if all(item is not None for item in header_dimensions):
        try:
            grid["header_point_count_matches"] = (
                int(float(header_dimensions[0]))*
                int(float(header_dimensions[1]))*
                int(float(header_dimensions[2])) == len(x)
            )
        except ValueError:
            grid["header_point_count_matches"] = False
    indexed_count=int(np.count_nonzero(indexed))
    return {
        "filename": filename,
        "file_sha256": hashlib.sha256(file_bytes).hexdigest(),
        "bytes": len(file_bytes),
        "n_points": int(len(x)),
        "indexed_count": indexed_count,
        "indexed_fraction": indexed_count/len(x),
        "unindexed_count": int(len(x)-indexed_count),
        "indexed_phase_counts": declared_phase_counts,
        "phase_rows": phase_rows,
        "numeric_columns": list(converted),
        "non_numeric_optional_columns": unused_columns,
        "quality_summary": quality_summary,
        "header": meta,
        "raw_header": header,
        "grid": grid,
        "maps": maps,
        "interpretation": (
            "Vendor-reported indexing and quality, not independent "
            "Kikuchi reindexing, a unique orientation relationship, "
            "or a physical twin/phase certificate."
        ),
    }


def plot_ctf_numeric_map(catalog: dict, field: str, *,
                         z_value: float | None = None):
    """Render CTF data by real coordinates. Nonrectangular -> points, not pixels."""
    import plotly.graph_objects as go
    maps = catalog["maps"]
    if field in {"x","y","z"} or field not in maps:
        raise ValueError("Select a measured numeric map field")
    x = np.asarray(maps["x"],float)
    y = np.asarray(maps["y"],float)
    z = np.asarray(maps["z"],float)
    values = np.asarray(maps[field],float)
    if z_value is not None:
        mask=np.isclose(z,float(z_value),atol=1e-5,rtol=0)
        if not np.any(mask):
            raise ValueError("Requested Z slice has no data")
        x,y,values=x[mask],y[mask],values[mask]
    if len(x)!=len(values):
        raise ValueError("CTF coordinate and map lengths differ")
    xs,xi=np.unique(np.round(x,6),return_inverse=True)
    ys,yi=np.unique(np.round(y,6),return_inverse=True)
    complete=(len(xs)*len(ys)==len(values))
    if complete:
        keys=yi.astype(np.int64)*len(xs)+xi.astype(np.int64)
        complete=np.unique(keys).size==len(values)
    if complete:
        image=np.full((len(ys),len(xs)),np.nan)
        image[yi,xi]=values
        data=go.Heatmap(z=image,x=xs,y=ys,colorscale="Viridis",
                        colorbar=dict(title=field),hoverongaps=False)
        kind="rectangular"
    else:
        data=go.Scattergl(
            x=x,y=y,mode="markers",
            marker=dict(size=4,color=values,colorscale="Viridis",
                        showscale=True,colorbar=dict(title=field)),
        )
        kind="irregular"
    fig=go.Figure(data=[data])
    fig.update_layout(
        title=f"{field} · {kind} CTF coordinates",
        height=560,margin=dict(l=20,r=20,t=48,b=35),
        xaxis_title="X (CTF units)",yaxis_title="Y (CTF units)",
    )
    fig.update_yaxes(autorange="reversed",scaleanchor="x",scaleratio=1)
    return fig


def ctf_field_histogram(catalog: dict, field: str, *,
                        indexed_only: bool = True):
    """Vendor quality/Euler distributions, using original CTF table values."""
    import plotly.graph_objects as go
    if field not in catalog["maps"] or field in {"x","y","z"}:
        raise ValueError("CTF numeric field not found")
    values=np.asarray(catalog["maps"][field],dtype=float)
    mask=np.isfinite(values)
    if indexed_only:
        mask &= np.asarray(catalog["maps"]["indexed"],bool)
    fig=go.Figure(data=[
        go.Histogram(x=values[mask],nbinsx=60,name=field)
    ])
    fig.update_layout(height=250,margin=dict(l=20,r=20,t=8,b=28),
                      xaxis_title=field,yaxis_title="Pixels")
    return fig


def raw_catalog_json(catalog: dict) -> str:
    """All metadata and summary numbers, without huge coordinate arrays."""
    import json
    return json.dumps(
        {key:value for key,value in catalog.items() if key!="maps"},
        indent=2,default=str,ensure_ascii=False,
    )


def raw_catalog_npz(catalog: dict) -> bytes:
    output=io.BytesIO()
    np.savez_compressed(output,**catalog["maps"])
    return output.getvalue()



def ctf_evidence_bundle(
    catalog: dict, original_file: bytes,
    *, independent: dict | None = None,
) -> bytes:
    """One complete self-contained research export, preserving original bytes.

    Original CTF is authoritative. NPZ maps mask unindexed orientation values
    on purpose, but untouched vendor columns remain in source.ctf. No
    transformations, OR or physical twin claims are added in the bundle.
    """
    import json
    import zipfile

    if hashlib.sha256(original_file).hexdigest()!=catalog["file_sha256"]:
        raise ValueError("Original file does not match measured CTF audit")
    stream=io.BytesIO()
    with zipfile.ZipFile(
        stream,"w",compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as z:
        z.writestr("source.ctf",original_file)
        z.writestr("raw_ctf_header_and_quality.json",raw_catalog_json(catalog))
        z.writestr("raw_numeric_pixel_fields.npz",raw_catalog_npz(catalog))
        z.writestr("raw_phase_statistics.csv",
                   pd.DataFrame(catalog["phase_rows"]).to_csv(index=False))
        z.writestr("raw_quality_statistics.csv",
                   pd.DataFrame(catalog["quality_summary"]).to_csv(index=False))
        if independent is not None:
            maps=io.BytesIO()
            np.savez_compressed(
                maps,**{key:np.asarray(value)
                        for key,value in independent["maps"].items()},
            )
            z.writestr("derived_orientation_grain_kam_maps.npz",
                       maps.getvalue())
            z.writestr("derived_grain_table.csv",
                       pd.DataFrame(independent["grains"]).to_csv(index=False))
            z.writestr("derived_grain_adjacency.csv",
                       pd.DataFrame(
                           independent["adjacent_grain_pairs"]
                       ).to_csv(index=False))
            z.writestr(
                "derived_method_and_limits.json",
                json.dumps(
                    {k:v for k,v in independent.items() if k!="maps"},
                    indent=2,default=str,ensure_ascii=False,
                ),
            )
    return stream.getvalue()
