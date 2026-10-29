from __future__ import annotations

"""Curated example cells and exact input adapters for the standalone workflow.

Numbers were transcribed from the original archived phase records using
pickletools static disassembly (NO pickle.load). Not independent experiments.
CIF parsing uses gemmi and never executes uploaded content.
"""

from dataclasses import dataclass
import re

import numpy as np
import sympy as sp

from .legacy_suite_bridge import Phase, exact_correspondence

ARCHIVE = "ARPGE Sept 2021 / Crystallography / PhasesCRYST"
SOURCE_FILES = {
    "NiTi_B2.cryst": "a0602ddfa6ff7a9eff803036620674bcc835b3a79adafca6cd043df74d77e4d1",
    "NiTi_B19'.cryst": "3e3ef58f114c5ce0d991cdf3134b7d6abaf5f638b7d295ddcf49de39ff346695",
    "AuTi_B2.cryst": "ea692e482da801ed7d87b1ce31ecbdc3b9c77238365b877433c5bc245741af2c",
    "AuTi_B19.cryst": "b21538afc23b4adea186e83cf3e0da660c574abfc6bc2ce2f156d84b08c306bc",
    "Fe_gamma.cryst": "790b4e97af22683ddf92d9971bdc55085b09827ce9a5fdcde735cd70a9266194",
    "Fe_alpha.cryst": "7d58dc18af7265382b70ed9dd991799eaf0ff6aafdf98a8513f3c2956093c147",
}
NI_TI_TRANSITION_SHA256 = "1a9aebf6ef50a62044d435cc62b2618b6bc45f98d332a8c930275599ad3a1229"
# The original NiTi.trans contains two raw paired entries. These are reference
# material only, NOT a complete/unique 3D correspondence or verified OR.
NI_TI_REFERENCE_PAIRS = (
    ((0, 1, 0), (1, 1, 0)),
    ((0, 0, 1), (1, -1, 0)),
)


@dataclass(frozen=True)
class Example:
    label: str
    description: str
    parent: Phase
    product: Phase
    files: tuple[str, str]
    status: str = "Archived phase-cell example. No verified experimental OR or C assigned."


EXAMPLES: dict[str, Example] = {
    "NiTi · B2 → B19′": Example(
        "NiTi · B2 → B19′", "Cubic B2 to monoclinic B19′",
        Phase("NiTi B2", "m-3m", 3.015, 3.015, 3.015),
        Phase("NiTi B19′", "2/m", 2.889, 4.120, 4.622, 90, 96.8, 90),
        ("NiTi_B2.cryst", "NiTi_B19'.cryst"),
    ),
    "AuTi · B2 → B19": Example(
        "AuTi · B2 → B19", "Cubic B2 to orthorhombic B19",
        Phase("AuTi B2", "m-3m", 3.250, 3.250, 3.250),
        Phase("AuTi B19", "mmm", 4.630, 2.940, 4.889),
        ("AuTi_B2.cryst", "AuTi_B19.cryst"),
    ),
    "Fe · γ → α": Example(
        "Fe · γ → α", "FCC austenite to BCC ferrite, conventional cells",
        Phase("Fe gamma (FCC)", "m-3m", 3.580, 3.580, 3.580),
        Phase("Fe alpha (BCC)", "m-3m", 2.870, 2.870, 2.870),
        ("Fe_gamma.cryst", "Fe_alpha.cryst"),
    ),
}


def provenance(label: str, *, edited: bool = False) -> dict:
    example = EXAMPLES[label]
    return {
        "kind": "edited_archived_example" if edited else "archived_example",
        "archive": ARCHIVE,
        "example": label,
        "files": [{ "path": fname, "sha256": SOURCE_FILES[fname]} for fname in example.files],
        "note": "Cell records only; correspondence and experimental OR are not certified by this preset.",
    }


_CIF_NUMBER = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\(\d+\))?$"
)


def parse_cif_phase(content: bytes, name: str, point_group: str) -> Phase:
    """Read CIF cell measurements only; symmetry setting MUST be user-selected."""
    if not content or len(content) > 2 * 1024 * 1024:
        raise ValueError("Provide a nonempty CIF smaller than 2 MiB")
    if b"\x00" in content:
        raise ValueError("Binary content is not a valid text CIF")
    try:
        import gemmi
        doc = gemmi.cif.read_string(content.decode("utf-8-sig"))
        block = doc.sole_block() if len(doc) == 1 else doc[0]
        def field(key: str) -> float:
            raw = block.find_value(key)
            if not raw or raw in ("?", "."):
                raise ValueError(f"Missing CIF field {key}")
            match = _CIF_NUMBER.fullmatch(str(raw).strip().strip("'\""))
            if not match:
                raise ValueError(f"Invalid CIF numeric value for {key}: {raw}")
            num = float(match.group(1))
            if not np.isfinite(num):
                raise ValueError(f"Nonfinite CIF field {key}")
            return num
        vals = [field("_cell_length_" + x) for x in "abc"] + [
            field("_cell_angle_" + x) for x in ("alpha", "beta", "gamma")
        ]
    except (UnicodeError, RuntimeError, IndexError, AttributeError) as exc:
        raise ValueError(f"Could not read CIF cell: {exc}") from exc
    if not name.strip():
        raise ValueError("Phase name must be nonempty")
    return Phase(name.strip(), point_group, *vals)


def _parse_rows(text: str) -> sp.Matrix:
    """Parse exactly three vector rows, with rational entries (no evaluation)."""
    rows = [row.strip() for row in text.splitlines() if row.strip()]
    if len(rows) != 3:
        raise ValueError("Enter exactly three independent direction rows")
    values: list[sp.Rational] = []
    for row in rows:
        parts = row.replace(",", " ").split()
        if len(parts) != 3:
            raise ValueError("Each direction must have three indices")
        for token in parts:
            if len(token) > 32 or not re.fullmatch(r"[+-]?\d+(?:/\d+)?", token):
                raise ValueError("Use integer or rational direction indices only")
            value = sp.Rational(token)
            values.append(value)
    # Each line represents one direction; columns needed for C @ u_A = u_M.
    return sp.Matrix(3, 3, values).T


def correspondence_from_pairs(parent_rows: str, product_rows: str) -> str:
    """Derive exact C from THREE linearly independent parent/product vector pairs.

    The inferred linear map is a user-specified *basis correspondence* only.
    Does not establish physical OR or the correctness of chosen direction pairs.
    """
    A = _parse_rows(parent_rows)
    M = _parse_rows(product_rows)
    if A.det() == 0 or M.det() == 0:
        raise ValueError("Both sets of directions must be linearly independent")
    C = sp.simplify(M * A.inv())
    if C.det() == 0:
        raise ValueError("Derived correspondence is singular")
    return "\n".join(" ".join(str(C[i,j]) for j in range(3)) for i in range(3))


def invert_correspondence(matrix_text: str) -> str:
    """Invert exact C when entered as A<-M rather than required M<-A."""
    C = exact_correspondence(matrix_text).C_M_from_A
    I = sp.simplify(C.inv())
    return "\n".join(" ".join(str(I[i,j]) for j in range(3)) for i in range(3))
