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
from .orientation_kernel import require_so3

MAX_H5OINA_BYTES = 120 * 1024 * 1024


def _one(handle, key):
    value = np.asarray(handle[key])
    if value.size != 1:
        raise ValueError(f"Expected scalar H5OINA field: {key}")
    obj = value.reshape(-1)[0]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="strict")
    return obj.item() if hasattr(obj, "item") else obj


def load_h5oina_sample(
    filename: str, payload: bytes, *, maximum_sample: int = 12000,
    sample_frame_correction: np.ndarray | None = None,
):
    """Return sampled EBSDMap and scan/frame audit.

    The Oxford scan angle describes relative tilt axes; it is not itself an
    arbitrary active z or y rotation to apply to crystal->sample orientations.
    An independent specimen calibration may supply a proper 3x3 correction.
    We never silently infer or apply it.
    """
    if Path(filename).suffix.lower() != ".h5oina":
        raise ValueError("Input must have the .h5oina suffix")
    if not isinstance(payload, bytes) or len(payload) < 100:
        raise ValueError("Expected nonempty H5OINA bytes")
    if len(payload) > MAX_H5OINA_BYTES:
        raise ValueError("H5OINA exceeds the 120 MiB upload limit")
    if not 2 <= maximum_sample <= 25000:
        raise ValueError("maximum_sample must be within 2..25000")
    if sample_frame_correction is not None:
        sample_frame_correction = require_so3(
            np.asarray(sample_frame_correction, float),
            tolerance=2e-8, name="explicit specimen-frame correction",
        )
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
            scan_angle = float(_one(handle, f"{base}/Header/Scanning Rotation Angle"))
            surface_euler = np.asarray(h["Specimen Orientation Euler"]).reshape(-1).astype(float)
            if surface_euler.shape != (3,) or not np.isfinite(surface_euler).all():
                raise ValueError("Invalid specimen-surface orientation metadata")
            tilt = float(_one(handle, f"{base}/Header/Tilt Angle"))
            tilt_axis = float(_one(handle, f"{base}/Header/Tilt Axis"))
            if not np.isfinite(scan_angle) or not np.isfinite(tilt) or not np.isfinite(tilt_axis):
                raise ValueError("Invalid Oxford scan/tilt frame metadata")
            frame_metadata = {
                "scan_rotation_angle_rad": scan_angle,
                "scan_rotation_angle_deg": float(np.rad2deg(scan_angle)),
                "specimen_surface_euler_rad": surface_euler.tolist(),
                "specimen_surface_euler_deg": np.rad2deg(surface_euler).tolist(),
                "specimen_tilt_rad": tilt,
                "specimen_tilt_deg": float(np.rad2deg(tilt)),
                "tilt_axis_rad": tilt_axis,
                "frame_note": "Oxford scan rotation is the angle between specimen and scanning tilt axes; do not equate it to an unverified active specimen rotation.",
            }
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
                convention=OrientationConvention(
                    scipy_sequence="ZXZ",
                    angle_unit=OrientationConvention.bunge_crystal_to_sample_radians().angle_unit,
                    sample_correction=tuple(tuple(float(v) for v in row)
                        for row in (
                            sample_frame_correction if sample_frame_correction is not None
                            else np.eye(3)
                        )),
                    label=("Oxford CS1 raw Bunge ZXZ radians" if sample_frame_correction is None
                           else "Oxford Bunge ZXZ radians with EXPLICIT user-calibrated sample correction"),
                ),
                metadata={
                    "source_format": "h5oina",
                    "full_pixel_count": int(n),
                    "sampled_pixel_count": int(count),
                    "reference_frame_normalization_applied": sample_frame_correction is not None,
                    **frame_metadata,
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
                "frame_corrected": sample_frame_correction is not None,
                "sample_frame_correction_matrix": (
                    None if sample_frame_correction is None
                    else sample_frame_correction.tolist()
                ),
                **frame_metadata,
                "warning": "No experimental parent OR or specimen frame is inferred; sampled points are not full-map segmentation",
            }
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"Incomplete Oxford H5OINA EBSD schema: {exc}") from exc
