from __future__ import annotations

"""Explicit lattice-length unit metadata.

Unit metadata is deliberately separate from numerical conversion. Attaching a
unit to a Lattice never rescales a, b or c.
"""

_UNIT_ALIASES = {
    "": "",
    "a": "angstrom",
    "å": "angstrom",
    "ångström": "angstrom",
    "angstrom": "angstrom",
    "angstroms": "angstrom",
    "nm": "nanometer",
    "nanometer": "nanometer",
    "nanometers": "nanometer",
    "pm": "picometer",
    "picometer": "picometer",
    "picometers": "picometer",
    "um": "micrometer",
    "µm": "micrometer",
    "μm": "micrometer",
    "micrometer": "micrometer",
    "micrometers": "micrometer",
    "m": "meter",
    "meter": "meter",
    "meters": "meter",
}

_UNIT_SYMBOLS = {
    "": "",
    "angstrom": "Å",
    "nanometer": "nm",
    "picometer": "pm",
    "micrometer": "µm",
    "meter": "m",
}


def normalize_length_unit(value: str | None) -> str:
    """Normalize a declared lattice-length unit without rescaling data."""

    if value is None:
        return ""
    key = value.strip().lower()
    if key not in _UNIT_ALIASES:
        allowed = "Å, nm, pm, µm, m"
        raise ValueError(
            f"Unsupported lattice-length unit {value!r}. "
            f"Supported units: {allowed}, or leave it empty."
        )
    return _UNIT_ALIASES[key]


def length_unit_symbol(value: str | None) -> str:
    """Display symbol for a declared lattice-length unit."""

    return _UNIT_SYMBOLS[normalize_length_unit(value)]


def reciprocal_length_unit_symbol(value: str | None) -> str:
    """Display symbol for reciprocal length, e.g. Å⁻¹."""

    symbol = length_unit_symbol(value)
    return f"{symbol}⁻¹" if symbol else ""
