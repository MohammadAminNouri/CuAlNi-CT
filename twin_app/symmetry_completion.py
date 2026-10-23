"""Proof-carrying classification completion across exact parent-symmetry orbits.

No material names, shear targets or literature classifications occur in this module.
It NEVER changes the geometrical solutions. Direct cross-locked classifications
are retained verbatim; missing types can be inferred only from a fully matching
parent-symmetry witness transporting *both ordered stretches AND rank-one tensor*.

For every witness: U'_i = Q U_i Q^T, U'_j = Q U_j Q^T,
                    (a'⊗n') = Q(a⊗n)Q^T.
The tensor identity is necessary because a stretch-pair orbit may contain TWO
physically distinct twin-interface branches.

The inferred status explicitly indicates symmetry transport (not a second direct
independent cross-lock). A contradiction between directly verified anchors
raises an error rather than arbitrarily selecting one.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
import numpy as np

LABELS = frozenset({"Type I", "Type II", "Compound"})

class SymmetryClassificationConflict(ValueError):
    """Inconsistent types for geometrically equivalent rank-one constructions."""


@dataclass(frozen=True)
class CompletionEvidence:
    source: str
    target: str
    classification: str
    parent_operation_index: int
    stretch_i_error: float
    stretch_j_error: float
    twin_tensor_error: float


@dataclass(frozen=True)
class CompletionAudit:
    direct_branches: int
    transported_branches: int
    unresolved_branches: int
    proofs: tuple[CompletionEvidence, ...]


def _tensor(a: Any, n: Any) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(3)
    n = np.asarray(n, dtype=float).reshape(3)
    if not np.isfinite(a).all() or not np.isfinite(n).all() or np.linalg.norm(a) < 1e-14 or np.linalg.norm(n) < 1e-14:
        raise ValueError("Non-finite or degenerate rank-one twin geometry")
    return np.outer(a, n)


def _relative(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y, 'fro') / max(np.linalg.norm(x, 'fro'), np.linalg.norm(y, 'fro'), 1e-14))


def _operations(parent_metric: Any, parent_group: Sequence[Any], tolerance: float) -> tuple[np.ndarray, ...]:
    metric = np.asarray(parent_metric, dtype=float)
    if metric.shape != (3,3) or not np.isfinite(metric).all() or not np.allclose(metric, metric.T, rtol=0, atol=tolerance):
        raise ValueError("Parent metric must be finite and symmetric")
    eigs, vectors = np.linalg.eigh(metric)
    if eigs[0] <= 0 or eigs[-1]/eigs[0] > 1e12:
        raise ValueError("Parent metric not positive definite or is critically ill-conditioned")
    B = (vectors*np.sqrt(eigs))@vectors.T
    Bi = np.linalg.inv(B)
    ops = []
    for index, g in enumerate(parent_group):
        crystallographic = np.asarray(g, dtype=float)
        if crystallographic.shape != (3,3) or not np.isfinite(crystallographic).all():
            raise ValueError(f"Invalid parent group operation #{index}")
        if _relative(crystallographic.T@metric@crystallographic, metric) > tolerance:
            raise ValueError(f"Parent group element #{index} violates lattice metric")
        Q = B@crystallographic@Bi
        if _relative(Q.T@Q, np.eye(3)) > tolerance or abs(abs(np.linalg.det(Q))-1.0) > tolerance:
            raise ValueError(f"Parent group operation #{index} is not orthogonal in Cartesian space")
        ops.append(Q)
    if not ops:
        raise ValueError('No parent symmetry operations')
    return tuple(ops)


def complete_classifications_by_symmetry(
    families: Sequence[Any], *, parent_metric: Any,
    parent_group: Sequence[Any], stretch_variants: Mapping[int, Any],
    tolerance: float=1e-7,
) -> tuple[tuple[Any,...], CompletionAudit]:
    """Return new immutable family records and a full trail of completion proofs.

    stretch_variants uses **1-based** stretch indices, matching VariantPairRecord.
    To avoid false positives, preserve the order of stretch_i/stretch_j. A swap
    is not sufficient to transport the same rank-one tensor representation.
    """
    if not (np.isfinite(tolerance) and 0 < tolerance <= 1e-5):
        raise ValueError('tolerance outside the allowed scientific range')
    operations=_operations(parent_metric, parent_group, tolerance)
    stretches={int(k):np.asarray(v,dtype=float) for k,v in stretch_variants.items()}
    for key,U in stretches.items():
        if U.shape != (3,3) or not np.isfinite(U).all() or _relative(U,U.T)>tolerance or np.linalg.eigvalsh(U)[0] <= 0:
            raise ValueError(f"Invalid positive stretch U_{key}")
    entries=[]
    for fi,family in enumerate(families):
        for pi,pair in enumerate(family.pair_records):
            if not pair.constructions:continue
            si,sj=int(pair.stretch_i),int(pair.stretch_j)
            if si not in stretches or sj not in stretches or si == sj:
                raise ValueError(f"Missing or identical stretch indices in pair {pair.pair_id}")
            for ci,branch in enumerate(pair.constructions):
                entries.append((fi,pi,ci,family,pair,branch,si,sj,_tensor(branch.a_parent_cartesian, branch.n_parent_cartesian)))
    # Symmetry transport must uniquely resolve already-distinct stretch states.
    # A fixed model tolerance is not an equivalence relation near degeneracy.
    variant_ids = sorted(stretches)
    gaps = [
        _relative(stretches[left], stretches[right])
        for k, left in enumerate(variant_ids)
        for right in variant_ids[k + 1:]
    ]
    if gaps:
        minimum_distinct_gap = min(gaps)
        if not np.isfinite(minimum_distinct_gap) or minimum_distinct_gap <= 128.0 * np.finfo(float).eps:
            raise ValueError(
                "UNRESOLVED_SYMMETRY_TRANSPORT: distinct stretches below reliable "
                f"binary64 classification resolution; minimum gap={minimum_distinct_gap:.3e}"
            )
        # One-eighth of minimum orbit separation: less than half, with margin.
        transport_tolerance = min(tolerance, minimum_distinct_gap / 8.0)
    else:
        transport_tolerance = tolerance
    originals=[x for x in entries if str(x[5].classification) in LABELS and "cross-locked" in str(x[5].classification_status)]
    proposals={}
    proofs=[]
    # Include all original direct anchors, including comparisons with one another,
    # so contradictory direct claims cannot be hidden by a single chosen witness.
    for target in entries:
        candidates=[]
        for source in originals:
            if source is target:continue
            for oi,Q in enumerate(operations):
                ui=_relative(Q@stretches[source[6]]@Q.T, stretches[target[6]])
                if ui>transport_tolerance:continue
                uj=_relative(Q@stretches[source[7]]@Q.T, stretches[target[7]])
                if uj>transport_tolerance:continue
                rt=_relative(Q@source[8]@Q.T, target[8])
                if rt>transport_tolerance:continue
                candidates.append(CompletionEvidence(
                    source=f"{source[3].family_id}:{source[4].pair_id}:{source[5].construction_id}",
                    target=f"{target[3].family_id}:{target[4].pair_id}:{target[5].construction_id}",
                    classification=str(source[5].classification),parent_operation_index=oi,
                    stretch_i_error=ui,stretch_j_error=uj,twin_tensor_error=rt
                ))
                break
        if str(target[5].classification) in LABELS:
            types={c.classification for c in candidates} | {str(target[5].classification)}
            if len(types)>1:
                raise SymmetryClassificationConflict(f"Contradictory direct classifications for {target[4].pair_id}: {types}")
        elif candidates:
            types={c.classification for c in candidates}
            if len(types)>1:
                raise SymmetryClassificationConflict(f"Conflicting witness types for {target[4].pair_id}: {types}")
            selected=min(candidates,key=lambda c:(max(c.stretch_i_error,c.stretch_j_error,c.twin_tensor_error),c.source,c.parent_operation_index))
            proposals[(target[0],target[1],target[2])]=selected
            proofs.append(selected)
    new_families=[]
    for fi,family in enumerate(families):
        new_pairs=[]
        for pi,pair in enumerate(family.pair_records):
            new_constructions=[]
            for ci,branch in enumerate(pair.constructions):
                evidence=proposals.get((fi,pi,ci))
                if evidence is None:
                    new_constructions.append(branch)
                else:
                    types=('I','II') if evidence.classification=='Compound' else (('I',) if evidence.classification=='Type I' else ('II',))
                    status=("symmetry-transported from directly cross-locked " + evidence.source +
                            f"; G-index {evidence.parent_operation_index}; ordered stretch and physical rank-one tensor " +
                            f"residuals {evidence.stretch_i_error:.3e}, {evidence.stretch_j_error:.3e}, {evidence.twin_tensor_error:.3e}; " +
                            "this is NOT an independent discrete cross-lock for the target pair")
                    # Leave classical_system_ids and source-specific residuals untouched:
                    # copying their provenance would misrepresent the evidence.
                    new_constructions.append(replace(branch,classification=evidence.classification,
                                                     representations=types,classification_status=status))
            new_pairs.append(replace(pair,constructions=tuple(new_constructions)))
        new_families.append(replace(family,pair_records=tuple(new_pairs)))
    unresolved=len(entries)-len(originals)-len(proofs)
    return tuple(new_families), CompletionAudit(len(originals),len(proofs),unresolved,tuple(proofs))
