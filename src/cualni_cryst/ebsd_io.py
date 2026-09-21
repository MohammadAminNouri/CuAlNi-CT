from __future__ import annotations

"""Strict EBSD I/O adapters.

Native text support:
- EDAX/TSL ``.ang`` (standard 10-column layout);
- Oxford/HKL ``.ctf`` (2-D degrees by default; 3-D angle unit must be explicit);
- generic CSV/TSV through an explicit schema.

HDF5 support is intentionally schema-explicit because H5EBSD, H5OINA, EDAX H5
and research files do not share one universal dataset layout.  If ``h5py`` is
available, ``load_hdf5_explicit`` can ingest any such file once dataset paths
are provided.

The parser never infers sample/crystal reference-frame corrections from data.
Those belong in :class:`OrientationConvention`.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd

from .ebsd_map import (
    AngleUnit,
    EBSDMap,
    OrientationConvention,
    eulers_to_matrices,
    quaternions_to_matrices,
    convert_raw_matrices,
)


def _header_key_values(lines: list[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for raw in lines:
        text = raw.lstrip("#").strip()
        if not text:
            continue
        if ":" in text:
            key, value = text.split(":", 1)
        elif "=" in text:
            key, value = text.split("=", 1)
        else:
            parts = text.split(None, 1)
            if len(parts) != 2:
                continue
            key, value = parts
        output[key.strip().upper()] = value.strip()
    return output


def load_ang(
    path: str | Path,
    *,
    convention: OrientationConvention | None = None,
) -> EBSDMap:
    """Read standard EDAX/TSL ANG text data.

    Standard columns are:
      phi1 Phi phi2 x y IQ CI phase SEM fit

    ANG Euler angles are stored in radians in the standard EDAX layout.  The
    default conversion preserves the raw sample/crystal frame; no hidden DREAM3D
    or MTEX frame correction is applied.
    """

    path = Path(path)
    lines = path.read_text(errors="strict").splitlines()
    header = [line for line in lines if line.lstrip().startswith("#")]
    data_lines = [
        line for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not data_lines:
        raise ValueError("ANG file contains no data rows")

    rows = []
    for number, line in enumerate(data_lines, start=1):
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(
                f"ANG data row {number} has {len(parts)} columns; expected >= 8"
            )
        try:
            rows.append([float(value) for value in parts])
        except ValueError as exc:
            raise ValueError(f"ANG row {number} contains non-numeric data") from exc

    array = np.asarray(rows, dtype=float)
    eulers = array[:, 0:3]
    x = array[:, 3]
    y = array[:, 4]
    phase = array[:, 7].astype(int)
    indexed = phase > 0

    quality: dict[str, np.ndarray] = {}
    if array.shape[1] >= 6:
        quality["IQ"] = array[:, 5]
    if array.shape[1] >= 7:
        quality["CI"] = array[:, 6]
    if array.shape[1] >= 9:
        quality["SEM"] = array[:, 8]
    if array.shape[1] >= 10:
        quality["Fit"] = array[:, 9]

    if convention is None:
        convention = OrientationConvention.bunge_crystal_to_sample_radians(
            label="EDAX ANG raw Bunge ZXZ; vendor/sample frame uncorrected"
        )

    kv = _header_key_values(header)
    metadata: dict[str, object] = {
        "source_format": "ang",
        "source_path": str(path),
        "orientation_convention": convention.label,
        "reference_frame_normalization_applied": False,
        "raw_header": tuple(header),
    }
    for key in (
        "GRID",
        "XSTEP",
        "YSTEP",
        "NCOLS_ODD",
        "NCOLS_EVEN",
        "NROWS",
        "OPERATOR",
        "SAMPLEID",
        "SCANID",
    ):
        if key in kv:
            metadata[key] = kv[key]

    return EBSDMap.from_eulers(
        eulers,
        phase,
        x,
        y,
        z=np.zeros(len(phase)),
        indexed=indexed,
        quality=quality,
        metadata=metadata,
        convention=convention,
    )


def _ctf_header_and_table(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    header: list[str] = []
    columns: list[str] | None = None
    data: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if columns is None:
                header.append(line)
            continue
        tokens = stripped.split()
        if columns is None and len(tokens) >= 8 and tokens[0].lower() == "phase":
            lowered = {item.lower() for item in tokens}
            if {"x", "y", "euler1", "euler2", "euler3"}.issubset(lowered):
                columns = tokens
                continue
        if columns is None:
            header.append(line)
        else:
            data.append(line)
    if columns is None:
        raise ValueError("CTF column header was not found")
    return header, columns, data


def load_ctf(
    path: str | Path,
    *,
    convention: OrientationConvention | None = None,
    three_dimensional_angle_unit: AngleUnit | None = None,
) -> EBSDMap:
    """Read Oxford/HKL CTF.

    Standard 2-D CTF Euler angles are interpreted as Bunge ZXZ degrees.
    Historical 3-D CTF variants may store radians; if ``ZCells > 1`` the angle
    unit must therefore be supplied explicitly.
    """

    path = Path(path)
    lines = path.read_text(errors="strict").splitlines()
    header, columns, data_lines = _ctf_header_and_table(lines)
    if not data_lines:
        raise ValueError("CTF file contains no data rows")

    frame = pd.read_csv(
        path,
        sep=r"\s+",
        names=columns,
        comment=None,
        skiprows=len(header) + 1,
        engine="python",
    )
    # read_csv skiprows can be confused by blank lines in exotic files; fall
    # back to direct rows if row count is inconsistent.
    if len(frame) != len(data_lines):
        records = [line.split() for line in data_lines]
        if any(len(row) != len(columns) for row in records):
            raise ValueError("CTF data rows do not match the declared columns")
        frame = pd.DataFrame(records, columns=columns).apply(pd.to_numeric)

    lookup = {column.lower(): column for column in columns}
    required = ("phase", "x", "y", "euler1", "euler2", "euler3")
    missing = [name for name in required if name not in lookup]
    if missing:
        raise ValueError(f"CTF missing required columns: {missing}")

    kv = _header_key_values(header)
    zcells = int(float(kv.get("ZCELLS", "1"))) if "ZCELLS" in kv else 1

    if convention is None:
        if zcells > 1:
            if three_dimensional_angle_unit is None:
                raise ValueError(
                    "3-D CTF angle unit is ambiguous across historical formats. "
                    "Pass three_dimensional_angle_unit explicitly."
                )
            unit = three_dimensional_angle_unit
        else:
            unit = AngleUnit.DEGREE
        convention = OrientationConvention(
            scipy_sequence="ZXZ",
            angle_unit=unit,
            label=f"Oxford CTF raw Bunge ZXZ ({unit.value}); frame uncorrected",
        )

    eulers = frame[
        [lookup["euler1"], lookup["euler2"], lookup["euler3"]]
    ].to_numpy(dtype=float)
    x = frame[lookup["x"]].to_numpy(dtype=float)
    y = frame[lookup["y"]].to_numpy(dtype=float)
    z = (
        frame[lookup["z"]].to_numpy(dtype=float)
        if "z" in lookup
        else np.zeros(len(frame), dtype=float)
    )
    phase = frame[lookup["phase"]].to_numpy(dtype=int)
    indexed = phase > 0

    quality = {}
    for canonical in (
        "bands",
        "error",
        "mad",
        "bc",
        "bs",
        "grainindex",
    ):
        if canonical in lookup:
            quality[canonical.upper()] = frame[lookup[canonical]].to_numpy()

    metadata: dict[str, object] = {
        "source_format": "ctf",
        "source_path": str(path),
        "orientation_convention": convention.label,
        "reference_frame_normalization_applied": False,
        "raw_header": tuple(header),
    }
    for key in ("XCELLS", "YCELLS", "ZCELLS", "XSTEP", "YSTEP", "ZSTEP"):
        if key in kv:
            metadata[key] = kv[key]

    return EBSDMap.from_eulers(
        eulers,
        phase,
        x,
        y,
        z=z,
        indexed=indexed,
        quality=quality,
        metadata=metadata,
        convention=convention,
    )


@dataclass(frozen=True)
class DelimitedSchema:
    phase: str
    x: str
    y: str
    euler: tuple[str, str, str] | None = None
    quaternion: tuple[str, str, str, str] | None = None
    matrix: tuple[str, ...] | None = None
    z: str | None = None
    indexed: str | None = None
    quaternion_order: str = "wxyz"
    quality: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modes = sum(
            item is not None
            for item in (self.euler, self.quaternion, self.matrix)
        )
        if modes != 1:
            raise ValueError(
                "DelimitedSchema requires exactly one of euler/quaternion/matrix"
            )
        if self.matrix is not None and len(self.matrix) != 9:
            raise ValueError("matrix schema must name exactly 9 columns")


def load_delimited(
    path: str | Path,
    schema: DelimitedSchema,
    *,
    convention: OrientationConvention,
    separator: str | None = None,
) -> EBSDMap:
    path = Path(path)
    if separator is None:
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=separator)

    required = {schema.phase, schema.x, schema.y}
    if schema.z:
        required.add(schema.z)
    if schema.indexed:
        required.add(schema.indexed)
    if schema.euler:
        required.update(schema.euler)
    if schema.quaternion:
        required.update(schema.quaternion)
    if schema.matrix:
        required.update(schema.matrix)
    required.update(schema.quality.values())
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"delimited EBSD file missing columns: {missing}")

    phase = frame[schema.phase].to_numpy(dtype=int)
    indexed = (
        frame[schema.indexed].to_numpy(dtype=bool)
        if schema.indexed
        else phase > 0
    )
    n = len(frame)
    orientations = np.full((n, 3, 3), np.nan, dtype=float)

    if schema.euler:
        raw = frame[list(schema.euler)].to_numpy(dtype=float)
        orientations[indexed] = eulers_to_matrices(raw[indexed], convention)
    elif schema.quaternion:
        raw = frame[list(schema.quaternion)].to_numpy(dtype=float)
        orientations[indexed] = quaternions_to_matrices(
            raw[indexed],
            convention,
            order=schema.quaternion_order,
        )
    else:
        assert schema.matrix is not None
        raw = frame[list(schema.matrix)].to_numpy(dtype=float).reshape(n, 3, 3)
        orientations[indexed] = convert_raw_matrices(raw[indexed], convention)

    return EBSDMap(
        orientations=orientations,
        phase_id=phase,
        indexed=indexed,
        x=frame[schema.x].to_numpy(dtype=float),
        y=frame[schema.y].to_numpy(dtype=float),
        z=(
            frame[schema.z].to_numpy(dtype=float)
            if schema.z
            else np.zeros(n, dtype=float)
        ),
        quality={
            name: frame[column].to_numpy()
            for name, column in schema.quality.items()
        },
        metadata={
            "source_format": "delimited",
            "source_path": str(path),
            "orientation_convention": convention.label,
            "reference_frame_normalization_applied": False,
        },
    )


@dataclass(frozen=True)
class HDF5Schema:
    phase: str
    x: str
    y: str
    euler: tuple[str, str, str] | None = None
    quaternion: str | None = None
    matrix: str | None = None
    z: str | None = None
    indexed: str | None = None
    quaternion_order: str = "wxyz"
    quality: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modes = sum(
            item is not None for item in (self.euler, self.quaternion, self.matrix)
        )
        if modes != 1:
            raise ValueError(
                "HDF5Schema requires exactly one orientation representation"
            )


def load_hdf5_explicit(
    path: str | Path,
    schema: HDF5Schema,
    *,
    convention: OrientationConvention,
) -> EBSDMap:
    """Load arbitrary HDF5 EBSD when dataset paths are explicitly declared."""

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "HDF5 EBSD loading requires optional dependency 'h5py'. "
            "Install h5py or convert the map to ANG/CTF/CSV."
        ) from exc

    path = Path(path)
    with h5py.File(path, "r") as handle:
        def read(dataset_path: str) -> np.ndarray:
            if dataset_path not in handle:
                raise ValueError(f"HDF5 dataset not found: {dataset_path}")
            return np.asarray(handle[dataset_path])

        phase = read(schema.phase).reshape(-1).astype(int)
        x = read(schema.x).reshape(-1).astype(float)
        y = read(schema.y).reshape(-1).astype(float)
        z = (
            read(schema.z).reshape(-1).astype(float)
            if schema.z
            else np.zeros(len(phase), dtype=float)
        )
        indexed = (
            read(schema.indexed).reshape(-1).astype(bool)
            if schema.indexed
            else phase > 0
        )
        n = len(phase)
        orientations = np.full((n, 3, 3), np.nan, dtype=float)

        if schema.euler:
            eulers = np.column_stack([read(item).reshape(-1) for item in schema.euler])
            orientations[indexed] = eulers_to_matrices(
                eulers[indexed], convention
            )
        elif schema.quaternion:
            quats = read(schema.quaternion).reshape(n, 4)
            orientations[indexed] = quaternions_to_matrices(
                quats[indexed],
                convention,
                order=schema.quaternion_order,
            )
        else:
            assert schema.matrix is not None
            matrices = read(schema.matrix).reshape(n, 3, 3)
            orientations[indexed] = convert_raw_matrices(
                matrices[indexed],
                convention,
            )

        quality = {
            name: read(dataset).reshape(-1)
            for name, dataset in schema.quality.items()
        }

    return EBSDMap(
        orientations=orientations,
        phase_id=phase,
        indexed=indexed,
        x=x,
        y=y,
        z=z,
        quality=quality,
        metadata={
            "source_format": "hdf5_explicit",
            "source_path": str(path),
            "orientation_convention": convention.label,
            "reference_frame_normalization_applied": False,
        },
    )
