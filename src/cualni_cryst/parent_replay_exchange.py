from __future__ import annotations

"""Cross-tool beta -> 18R -> beta validation with explicit frame provenance.

The OR Q maps martensite (indexed 18R Cartesian crystal) to beta (cubic
Cartesian crystal). All g map crystal Cartesian to a COMMON sample frame.
This checks rotation bookkeeping, not whether Q is a physical Cu-Al-Ni OR.
"""

from io import BytesIO, StringIO
import csv
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.spatial.transform import Rotation

from .orientation_kernel import require_so3
from .parent_variant_audit import parent_generated_variants
from .parent_or_interop import mtex_or_contract


def _stack_orientations(value: object, *, bound: int = 25000) -> np.ndarray:
    obs = np.asarray(value, float)
    if obs.ndim != 3 or obs.shape[1:] != (3, 3) or not 1 <= len(obs) <= bound:
        raise ValueError(f"Expected 1..{bound} (3,3) crystal-to-sample matrices")
    if not np.isfinite(obs).all():
        raise ValueError("Non-finite indexed orientation")
    if np.max(np.linalg.norm(obs @ np.transpose(obs,(0,2,1)) - np.eye(3), axis=(1,2))) > 2e-7:
        raise ValueError("Indexed orientations are not proper SO(3)")
    if np.max(np.abs(np.linalg.det(obs) - 1)) > 2e-7:
        raise ValueError("Improper indexed orientation")
    return obs


def forward_predict_daughter(
    parent_g: object, parent_from_daughter_q: object,
    parent_proper_symmetry: tuple[np.ndarray, ...],
    daughter_proper_symmetry: tuple[np.ndarray, ...],
) -> dict:
    """All distinct daughter variants, plus each indexer-equivalent C2 copy."""
    gA = require_so3(np.asarray(parent_g, float), tolerance=2e-8, name="beta parent g")
    Q = require_so3(np.asarray(parent_from_daughter_q, float), tolerance=2e-8, name="beta<-18R Q")
    p = tuple(require_so3(S, tolerance=2e-8, name="beta symmetry") for S in parent_proper_symmetry)
    m = tuple(require_so3(S, tolerance=2e-8, name="18R symmetry") for S in daughter_proper_symmetry)
    if not p or not m:
        raise ValueError("Complete proper symmetry groups required")
    family = np.asarray(parent_generated_variants(Q, p, m))
    prediction = np.einsum("ij,vjk->vik", gA, family)
    indexed_equivalents = np.einsum("vij,sjk->vsik", prediction, np.asarray(m))
    return {
        "variant_rotations_parent_from_daughter": family,
        "daughter_crystal_to_sample": prediction,
        "daughter_symmetry_representatives": indexed_equivalents,
        "parent_crystal_to_sample": gA,
        "q_parent_from_daughter": Q,
        "n_variants": len(family),
        "n_daughter_symmetry_representatives": len(m),
    }


def forward_inverse_roundtrip(
    parent_g: object, parent_from_daughter_q: object,
    parent_proper_symmetry: tuple[np.ndarray, ...],
    daughter_proper_symmetry: tuple[np.ndarray, ...],
    *,
    observed_daughter_g: object | None = None,
    weights: object | None = None,
    accepted_deg: float = 3.0,
) -> dict:
    """Compare observed daughter means to forward models and invert matches.

    A returned inverse residual equals the forward fit because inverse
    rotations preserve SO(3) distance. This is an identity/cross-tool contract
    test, NOT independent physical validation of the OR.
    """
    if not 0 < accepted_deg <= 20:
        raise ValueError("acceptance threshold must be in (0,20] degrees")
    predicted = forward_predict_daughter(
        parent_g, parent_from_daughter_q,
        parent_proper_symmetry, daughter_proper_symmetry,
    )
    gA = predicted["parent_crystal_to_sample"]
    family = predicted["variant_rotations_parent_from_daughter"]
    m = np.asarray(daughter_proper_symmetry, float)
    if observed_daughter_g is None:
        observed = _stack_orientations(
            predicted["daughter_symmetry_representatives"].reshape(-1, 3, 3)
        )
    else:
        observed = _stack_orientations(observed_daughter_g)
    if weights is None:
        w = np.ones(len(observed))
    else:
        w = np.asarray(weights, float).reshape(-1)
        if w.shape != (len(observed),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("Weights must be finite and nonnegative, total > 0")
    q_observed = Rotation.from_matrix(observed).as_quat()
    q_predictions = Rotation.from_matrix(
        predicted["daughter_symmetry_representatives"].reshape(-1, 3, 3)
    ).as_quat()
    products = np.abs(q_observed @ q_predictions.T)
    choices = np.argmax(products, axis=1)
    best_dot = products[np.arange(len(observed)), choices]
    forward_errors = np.rad2deg(2 * np.arccos(np.clip(best_dot, 0, 1)))
    n_sym = len(m)
    variant_idx = choices // n_sym
    child_idx = choices % n_sym
    # The observed gM has indexer representative S_M. Reconstruct beta by
    # gA' = gM @ S_M.T @ (S_parent @ Q).T.
    parent_candidates = np.einsum(
        "nij,njk,nkl->nil", observed,
        np.transpose(m[child_idx], (0, 2, 1)),
        np.transpose(family[variant_idx], (0, 2, 1)),
    )
    q_parent = Rotation.from_matrix(gA).as_quat()
    q_back = Rotation.from_matrix(parent_candidates).as_quat()
    inverse_errors = np.rad2deg(
        2 * np.arccos(np.clip(np.abs(q_back @ q_parent), 0, 1))
    )
    if not np.allclose(forward_errors, inverse_errors, atol=1e-4, rtol=0):
        raise AssertionError("Forward and inverse orientation residuals disagree")
    passing = forward_errors <= accepted_deg
    return {
        "schema": "cualni.parent-forward-inverse/1.0",
        "scientific_status": "conditional fitted OR, no physical parent or diffraction certification",
        "physical_or_certified": False,
        "absolute_parent_frame_certified": False,
        "n_observations": int(len(observed)),
        "n_predicted_variants": predicted["n_variants"],
        "n_proper_child_symmetries": int(n_sym),
        "median_forward_residual_deg": float(np.median(forward_errors)),
        "p90_forward_residual_deg": float(np.percentile(forward_errors, 90)),
        "max_forward_inverse_disagreement_deg": float(np.max(np.abs(forward_errors - inverse_errors))),
        "weighted_fraction_within_threshold": float(np.sum(w[passing]) / np.sum(w)),
        "acceptance_deg": float(accepted_deg),
        "variant_id_1based": (variant_idx + 1).astype(int).tolist(),
        "indexed_child_symmetry_id_1based": (child_idx + 1).astype(int).tolist(),
        "forward_residual_deg": forward_errors.tolist(),
        "inverse_parent_residual_deg": inverse_errors.tolist(),
        "recovered_parent_quaternions_xyzw": q_back.tolist(),
        "inversion_contract": "gA = gM @ S_child.T @ (S_parent @ Q).T",
        "raw_or_rebased_frame": "must match parent and daughter original Cartesian crystal axes",
    }


def _matlab_replay_script() -> str:
    """MATLAB/MTEX entry point; not executed or certified in Python."""
    return r"""% REPRODUCIBILITY CONTRACT -- MATLAB + installed MTEX required
% This is an EXTERNAL MTEX WORKFLOW, not a Python implementation of MCL.
% Place the original Oxford H5OINA beside this file and set the filename in
% or_contract.json. Do NOT silently change OR, phase frame, or source file.
clearvars;
contract = jsondecode(fileread('or_contract.json'));
Q = contract.or.Q_parent_from_daughter;
assert(norm(Q'*Q-eye(3),'fro')<1e-7 && abs(det(Q)-1)<1e-7, ...
       'Bad Q input; convention beta<-18R crystal Cartesian');
h = contract.h5oina_filename;
assert(exist(h,'file')==2, 'Source H5OINA absent. Copy locally (not in ZIP).');
ebsd = EBSD.load(h); % Oxford MTEX specimen-frame correction may differ from RAW CS1
ebsdM = ebsd('indexed');
assert(numel(unique(ebsdM.phaseId)) == 1, 'Expected one indexed daughter phase');
csM = ebsdM.CS;
% Cubic symmetry ONLY: parent lattice parameter not measured, no physical CIF inferred.
csA = crystalSymmetry('m-3m',[1 1 1],[90 90 90]*degree,'mineral','beta1 hypothetical');
p2c0 = orientation.byMatrix(Q',csA,csM);
[grains,ebsd] = calcGrains(ebsd,'angle',3*degree,'minPixel',2);
job = parentGrainReconstructor(ebsd,grains,p2c0);
job.calcVariantGraph('threshold',2.5*degree,'tolerance',1.5*degree);
job.clusterVariantGraph('numIter',10,'inflationPower',1.05);
job.calcParentFromVote('minProb',0.6,'minDelta',0.15);
job.mergeSimilar('threshold',3*degree);
parentEBSD = job.calcParentEBSD;
save('MTEX_frozen_OR_unverified.mat','contract','Q','p2c0',...
     'ebsd','grains','job','parentEBSD','-v7.3');
% Separate OR refinement: preserve the frozen model and original result.
job_refit = parentGrainReconstructor(ebsd,grains,p2c0);
job_refit.calcParent2Child('quantile',0.9);
p2c_refined = job_refit.p2c;
save('MTEX_refitted_OR_separate_unverified.mat','p2c0','p2c_refined',...
     'job_refit','-v7.3');
fprintf('MTEX finished. Inspect phase setting, frame, votes, and fit. NOT physical certification.\n');
"""


def _matrix_csv(matrices: np.ndarray) -> str:
    stream = StringIO()
    writer = csv.writer(stream)
    writer.writerow(["id", "qx", "qy", "qz", "qw", "phi1_deg", "Phi_deg", "phi2_deg"])
    rotation = Rotation.from_matrix(matrices)
    quats = rotation.as_quat()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        eulers = rotation.as_euler("ZXZ", degrees=True)
    for i, (q, e) in enumerate(zip(quats, eulers), 1):
        writer.writerow([i, *(float(v) for v in q), *(float(v) for v in e)])
    return stream.getvalue()


def export_replay_zip(
    parent_g: object, parent_from_daughter_q: object,
    parent_proper_symmetry: tuple[np.ndarray, ...],
    daughter_proper_symmetry: tuple[np.ndarray, ...],
    *,
    h5oina_filename: str = "source.h5oina",
    h5oina_sha256: str | None = None,
    parent_cell: dict | None = None,
    daughter_cell: dict | None = None,
    observed_daughter_g: object | None = None,
    weights: object | None = None,
    accepted_deg: float = 3.0,
    indexed_daughter_cif: bytes | None = None,
    parent_cif: bytes | None = None,
) -> bytes:
    """Portable ZIP: OR, predictions, replay report, and actual MTEX script.

    No proprietary EBSD bytes; the user copies the source file locally.
    """
    # Pass original CIF files through unchanged; do not fabricate structures.
    def checked_cif(raw: bytes | None, label: str) -> bytes | None:
        if raw is None:
            return None
        if not isinstance(raw, bytes) or len(raw) > 2 * 1024 * 1024:
            raise ValueError(f"{label} CIF must be bytes <= 2 MiB")
        try:
            content = raw.decode("utf-8")
        except UnicodeError as exc:
            raise ValueError(f"{label} CIF is not UTF-8") from exc
        if "data_" not in content or "_cell_length_a" not in content:
            raise ValueError(f"{label} CIF lacks a crystal-cell declaration")
        return raw
    child_cif = checked_cif(indexed_daughter_cif, "Indexed daughter")
    beta_cif = checked_cif(parent_cif, "Measured parent")
    source = Path(h5oina_filename).name
    if not source.lower().endswith(".h5oina"):
        raise ValueError("Only H5OINA source names accepted in the MTEX replay kit")
    if h5oina_sha256 is not None and (len(h5oina_sha256) != 64 or any(
        x not in "0123456789abcdef" for x in h5oina_sha256.lower()
    )):
        raise ValueError("Source SHA256 must contain exactly 64 hex characters")
    predicted = forward_predict_daughter(
        parent_g, parent_from_daughter_q,
        parent_proper_symmetry, daughter_proper_symmetry,
    )
    report = forward_inverse_roundtrip(
        parent_g, parent_from_daughter_q,
        parent_proper_symmetry, daughter_proper_symmetry,
        observed_daughter_g=observed_daughter_g, weights=weights,
        accepted_deg=accepted_deg,
    )
    contract = {
        "schema": "cualni.parent-interoperability/1.0",
        "h5oina_filename": source,
        "h5oina_sha256": h5oina_sha256,
        "raw_ebsd_embedded": False,
        "indexed_daughter_cif_supplied": child_cif is not None,
        "parent_cif_supplied": beta_cif is not None,
        "cif_atomic_validation": "externally required; source CIFs copied unchanged",
        "parent_crystal_to_sample_matrix": predicted["parent_crystal_to_sample"].tolist(),
        "parent_crystal_to_sample_quaternion_xyzw": Rotation.from_matrix(
            predicted["parent_crystal_to_sample"]).as_quat().tolist(),
        "or": mtex_or_contract(parent_from_daughter_q),
        "parent_cell": parent_cell,
        "daughter_cell": daughter_cell,
        "euler_export": "SciPy intrinsic ZXZ; verify sample axes against Oxford/MTEX before import",
        "daughter_variant_count": predicted["n_variants"],
        "scientific_status": "frozen numerical candidate; not independently anchored or physically certified",
    }
    memory = BytesIO()
    with ZipFile(memory, "w", compression=ZIP_DEFLATED) as z:
        z.writestr("or_contract.json", json.dumps(contract,indent=2,allow_nan=False)+"\n")
        z.writestr("forward_inverse_results.json", json.dumps(report,indent=2,allow_nan=False)+"\n")
        z.writestr("predicted_18r_variants_xyzw_bunge.csv", _matrix_csv(
            predicted["daughter_crystal_to_sample"]))
        z.writestr("conditional_beta_orientation_xyzw_bunge.csv", _matrix_csv(
            predicted["parent_crystal_to_sample"][None]))
        z.writestr("run_external_mtex_replay.m", _matlab_replay_script())
        if child_cif is not None:
            z.writestr("USER_SUPPLIED_indexed_18R_structure.cif", child_cif)
        if beta_cif is not None:
            z.writestr("USER_SUPPLIED_beta_parent_structure.cif", beta_cif)
        z.writestr("README.txt",
            "Standalone reproducibility kit: numerical candidate, not physical OR proof.\n"
            "Copy your ORIGINAL H5OINA alongside this kit; its bytes are not included.\n"
            "Run MATLAB + MTEX: run_external_mtex_replay.m; inspect output manually.\n"
            "Q = parent<-daughter crystal Cartesian, MTEX p2c = Q.T.\n"
            "The two phase CIFs, crystal setting and Oxford frame must be verified.\n"
            "The Euler/CSV table is an exchange reference, NOT a verified AZtec importer.\n"
            "MATLAB script is generated but not executed or parity-tested here.\n"
        )
    return memory.getvalue()
