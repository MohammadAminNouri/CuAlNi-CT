from __future__ import annotations

"""Strict Oxford H5OINA EBSD reader. Do not infer a parent orientation from
single-phase daughter EBSD and never execute/unpickle uploaded datasets.
A deterministic sampled EBSDMap is returned; full-grid maps remain separate.
"""

from io import BytesIO
from pathlib import Path

import h5py
import numpy as np

from .ebsd_map import EBSDMap, OrientationConvention

MAX_H5OINA_BYTES = 120 * 1024 * 1024


def _one(handle, key):
    value = np.asarray(handle[key])
    if value.size != 1:
        raise ValueError(f"Expected scalar H5OINA field: {key}")
    obj = value.reshape(-1)[0]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="strict")
    return obj.item() if hasattr(obj, "item") else obj


def load_h5oina_sample(filename: str, payload: bytes, *, maximum_sample: int = 12000):
    """Return (sampled EBSDMap, full-scan summary) with explicit raw frames."""
    if Path(filename).suffix.lower() != ".h5oina":
        raise ValueError("Input must have the .h5oina suffix")
    if not isinstance(payload, bytes) or len(payload) < 100:
        raise ValueError("Expected nonempty H5OINA bytes")
    if len(payload) > MAX_H5OINA_BYTES:
        raise ValueError("H5OINA exceeds the 120 MiB upload limit")
    if not 2 <= maximum_sample <= 25000:
        raise ValueError("maximum_sample must be within 2..25000")
    try:
        handle = h5py.File(BytesIO(payload), "r")
    except (OSError, ValueError) as exc:
        raise ValueError("Not a readable HDF5 container") from exc
    with handle:
        try:
            if _one(handle, "Manufacturer") != "Oxford Instruments":
                raise ValueError("HDF5 file is not from Oxford Instruments")
            base = "1/EBSD"
            h = handle[f"{base}/Header"]
            d = handle[f"{base}/Data"]
            nx = int(_one(handle, f"{base}/Header/X Cells"))
            ny = int(_one(handle, f"{base}/Header/Y Cells"))
            n = nx * ny
            if not 0 < nx <= 100000 or not 0 < ny <= 100000 or not 0 < n <= 20000000:
                raise ValueError("Invalid H5OINA scan dimensions")
            for key, shape in (
                ("Phase", (n,)), ("Euler", (n, 3)),
                ("X", (n,)), ("Y", (n,)),
                ("Mean Angular Deviation", (n,)),
            ):
                if d[key].shape != shape:
                    raise ValueError(f"Unexpected {key} shape: {d[key].shape}")
            if str(d["Euler"].attrs.get("Unit")) != "rad":
                raise ValueError("H5OINA Euler must explicitly have Unit=rad")
            phase = np.asarray(d["Phase"], dtype=int)
            if np.any(phase < 0):
                raise ValueError("Invalid negative phase ID")
            indexed = phase > 0
            ids, counts = np.unique(phase, return_counts=True)
            indexed_ids = np.flatnonzero(indexed)
            if len(indexed_ids) == 0:
                raise ValueError("H5OINA contains no indexed orientations")
            phase_descriptions = []
            known_ids = set()
            for ident in h["Phases"]:
                item = h["Phases"][ident]
                if str(item["Lattice Dimensions"].attrs.get("Unit")) != "angstrom":
                    raise ValueError("Phase lengths require angstrom units")
                if str(item["Lattice Angles"].attrs.get("Unit")) != "rad":
                    raise ValueError("Phase angles require radian units")
                abc = np.asarray(item["Lattice Dimensions"]).reshape(-1)
                deg = np.rad2deg(np.asarray(item["Lattice Angles"]).reshape(-1))
                if abc.shape != (3,) or deg.shape != (3,) or not np.isfinite(abc).all() or not np.isfinite(deg).all():
                    raise ValueError("Invalid indexed phase cell")
                known_ids.add(int(ident))
                phase_descriptions.append({
                    "phase_id": int(ident),
                    "name": str(_one(handle, f"{base}/Header/Phases/{ident}/Phase Name")),
                    "abc_angstrom": abc.astype(float).tolist(),
                    "angles_deg": deg.astype(float).tolist(),
                    "laue": str(item["Laue Group"].attrs.get("Symbol", "")),
                    "space_group": str(item["Space Group"].attrs.get("Symbol", "")),
                })
            if set(int(x) for x in ids if x > 0) - known_ids:
                raise ValueError("Indexed phase IDs are missing phase metadata")
            count = min(maximum_sample, len(indexed_ids))
            selection = indexed_ids[np.linspace(0, len(indexed_ids)-1, count, dtype=int)]
            euler = np.asarray(d["Euler"][selection], dtype=float)
            x = np.asarray(d["X"][selection], dtype=float)
            y = np.asarray(d["Y"][selection], dtype=float)
            mad = np.asarray(d["Mean Angular Deviation"], dtype=float)
            if not np.isfinite(euler).all() or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("Indexed orientation/coordinates contain nonfinite values")
            if not np.isfinite(mad[indexed]).all():
                raise ValueError("Invalid indexed MAD")
            sampled = EBSDMap.from_eulers(
                euler, phase[selection], x, y,
                indexed=np.ones(count, dtype=bool),
                quality={"MAD_rad": mad[selection]},
                convention=OrientationConvention.bunge_crystal_to_sample_radians(
                    label="Oxford H5OINA Bunge ZXZ radians: crystal to sample-surface"
                ),
                metadata={
                    "source_format": "h5oina",
                    "full_pixel_count": int(n),
                    "sampled_pixel_count": int(count),
                    "reference_frame_normalization_applied": False,
                    "spatial_full_resolution": False,
                },
            )
            return sampled, {
                "source_format": "h5oina",
                "points": int(n), "indexed_points": int(indexed.sum()),
                "indexed_pct": 100.0 * float(indexed.mean()),
                "phase_counts": {str(int(k)): int(v) for k, v in zip(ids, counts)},
                "phase_geometry": phase_descriptions,
                "mean_angular_deviation_median_deg": float(np.median(np.rad2deg(mad[indexed]))),
                "selected_orientations": int(count),
                "orientation_convention": "Oxford Bunge ZXZ radians, crystal to sample-surface",
                "frame_corrected": False,
                "warning": "No experimental parent OR is inferred; sampled points are not full-map segmentation",
            }
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"Incomplete Oxford H5OINA EBSD schema: {exc}") from exc
