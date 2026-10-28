from __future__ import annotations

"""CTF-declared phase -> editable crystal input; no invented correspondence.

All CTF phase names/space groups are *indexing declarations*. A 2H or 18R
label alone is not experimental structure validation, and the austenite may
have zero indexed pixels.  No code in this module ever sets C or an OR.
"""
import hashlib
from typing import MutableMapping, Any

from .ctf_independent import declared_ctf_phases
from .input_explainer import cif_compatible_families, convert_cell_lengths


def ctf_crystal_handoff(file_bytes: bytes, *, parent_id: int,
                        product_id: int) -> dict[str, Any]:
    if int(parent_id) == int(product_id):
        raise ValueError("Parent and product phase IDs must differ")
    phases = {int(p["phase_id"]): p for p in declared_ctf_phases(file_bytes)}
    missing = [p for p in (int(parent_id), int(product_id)) if p not in phases]
    if missing:
        raise ValueError(f"CTF header does not define phase IDs {missing}")
    result = {}
    for role, pid in (("parent", parent_id), ("product", product_id)):
        item = phases[int(pid)]
        compatible = cif_compatible_families(
            item["lengths_angstrom"], item["angles_deg"]
        )
        if not compatible:
            raise ValueError(f"CTF {role} cell has no supported crystal system")
        # Highest-symmetry *metric family* is only a geometric suggestion.
        family = compatible[0]
        symbol = item["declared_point_group"]
        result[role] = {
            "ctf_phase_id":int(pid),
            "name":item["name"],
            "lengths_angstrom":item["lengths_angstrom"],
            "angles_deg":item["angles_deg"],
            "point_group":symbol,
            "crystal_family":family,
            "space_group_number":item["space_group_number"],
        }
    return {
        "sha256":hashlib.sha256(file_bytes).hexdigest(),
        "parent":result["parent"], "product":result["product"],
        "correspondence_status":"NOT_INFERRED",
        "orientation_relationship_status":"NOT_IDENTIFIED",
        "phase_provenance":"CTF_INDEXING_DECLARATIONS_UNVERIFIED",
    }


def apply_ctf_crystal_handoff(
    proposal: dict[str, Any], state: MutableMapping[str, Any], *,
    length_unit: str,
) -> None:
    """Apply both cells atomically before Streamlit widgets are instantiated.

    No correspondence editor keys are touched. Crystal symmetry/axis setting
    must still be verified; this function cannot create an experimental OR.
    """
    changes = {}
    for role in ("parent","product"):
        row = proposal[role]
        lengths = convert_cell_lengths(
            row["lengths_angstrom"],
            from_unit="angstrom",to_unit=length_unit,
        )
        family = row["crystal_family"]
        changes[f"{role}_family"] = family
        changes[f"{role}_point_group_{family}"] = row["point_group"]
        for key,value in zip(
            ("a","b","c","alpha","beta","gamma"),
            (*lengths,*row["angles_deg"]),
        ):
            changes[f"{role}_{key}"] = float(value)
        changes[f"{role}_label"] = row["name"]
    state.update(changes)
    state["tf_ctf_applied_sha256"] = proposal["sha256"]
    state["tf_ctf_applied_ids"] = (
        proposal["parent"]["ctf_phase_id"],
        proposal["product"]["ctf_phase_id"],
    )
