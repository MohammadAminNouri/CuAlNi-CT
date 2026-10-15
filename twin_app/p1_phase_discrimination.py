"""Cu-Al-Ni phase-identity evidence audit, distinct from tensor compatibility.

Stacking sequence evidence is required to discriminate thermal martensite
gamma1' (2H) from beta1' (18R/M18R). Metric shape, Al composition, named
symmetry, exact C, or a tiny twin residual ALONE do not identify the phase.
Even user-supplied indexed observations are NOT independently authenticated.

Sources:
Sari & Aksoy, J Alloys Compd 417 (2006) 138-142,
doi:10.1016/j.jallcom.2005.09.049 (2H and M18R diffraction).
Zarubova et al., in situ TEM studies of Cu-Al-Ni stress-dependent
gamma1' <-> beta1' <-> alpha1' transformation sequences.
Otsuka et al., Mater Trans 15 (1974) 200, doi:10.2320/matertrans1960.15.200.
"""
from __future__ import annotations
from typing import Mapping, Any

class PhaseDiscriminationError(ValueError):
    """Structural observations contradict a stated homogeneous phase."""

_ALLOWED = {
    "gamma1_prime_2H": "2H",
    "beta_prime_monoclinic": "18R",
    # Reduced 6M is a cell *representation*, not evidence of a 6R phase.
    "faulted_monoclinic_reduced_6M": "18R",
}
_SUPPORTED_INDEX_METHODS = frozenset({
    "TEM_SAED", "HRTEM_FFT", "XRD_REFINED_STACKING",
    "ATOMIC_RESOLUTION_STEM", "EBSD_PATTERN_INDEXING",
})
_STACKINGS = frozenset({"2H", "18R", "6R"})

def discriminate_cualni_martensite(declared_phase: str,
                                   indexed_stacking: Mapping[str, Any] | None) -> dict[str, Any]:
    """Audit indexed stacking: never infer it from lattice or correspondence.

    An indexed observation is a user-provided hypothesis with provenance,
    *not* third-party validation. Multiple indexings imply separate domains
    and prohibit a single homogeneous-twin physical interpretation.
    """
    if declared_phase not in _ALLOWED:
        raise PhaseDiscriminationError("Unsupported thermal martensite declaration")
    candidate = _ALLOWED[declared_phase]
    preliminary = {
        "declared_phase": declared_phase,
        "candidate_stacking": candidate,
        "basis_warning": ("Reduced 6M is a cell representation; do not equate "
                          "its numeric periodicity with stress-induced alpha1-prime 6R")
                         if "reduced_6M" in declared_phase else None,
        "independent_experimental_verification": False,
        "can_claim_phase_identified": False,
    }
    if indexed_stacking is None:
        return {**preliminary,
                "status": "PHASE_UNRESOLVED_NO_INDEXED_STACKING",
                "reason": "Metric, point group, labels and correspondence are insufficient to identify stacking"}
    if not isinstance(indexed_stacking, Mapping):
        raise PhaseDiscriminationError("Indexed stacking evidence must be a mapping")
    method = indexed_stacking.get("indexing_method")
    if method not in _SUPPORTED_INDEX_METHODS:
        raise PhaseDiscriminationError("Independent phase indexing method is absent or unrecognized")
    labels = indexed_stacking.get("indexed_sequences")
    if isinstance(labels, str):
        labels = [labels]
    if not isinstance(labels, (tuple, list)) or not labels:
        raise PhaseDiscriminationError("Provide a nonempty indexed_sequences list")
    labels = [str(x).strip() for x in labels]
    if any(x not in _STACKINGS for x in labels):
        raise PhaseDiscriminationError("Unknown stacking assignment; must be 2H, 18R or 6R")
    detected = sorted(set(labels))
    if len(detected) > 1:
        raise PhaseDiscriminationError(
            "MIXED_MARTENSITE_DOMAINS: 2H/18R/6R indexed together; "
            "model each identified domain separately, never as one homogeneous phase"
        )
    if detected == ["6R"]:
        raise PhaseDiscriminationError(
            "STRESS_INDUCED_6R_NOT_REDUCED_6M: alpha1-prime 6R needs a separate structural model"
        )
    if detected[0] != candidate:
        raise PhaseDiscriminationError(
            f"STACKING_PHASE_CONTRADICTION: declared {candidate} but indexing reports {detected[0]}"
        )
    provenance = str(indexed_stacking.get("observation_reference", "")).strip()
    if not provenance:
        return {**preliminary, "status": "CONSISTENT_BUT_OBSERVATION_PROVENANCE_MISSING",
                "indexing_method": method,
                "observed_stacking": detected[0]}
    if method == "EBSD_PATTERN_INDEXING":
        return {**preliminary, "status": "CONSISTENT_EBSD_INDEXING_PSEUDOSYMMETRY_NOT_RESOLVED",
                "indexing_method": method, "observed_stacking": detected[0],
                "observation_reference": provenance}
    return {**preliminary,
            "status": "REPORTED_STACKING_CONSISTENT_SOURCE_NOT_INDEPENDENTLY_VERIFIED",
            "indexing_method": method,
            "observed_stacking": detected[0], "observation_reference": provenance}

