from __future__ import annotations

"""Recompute an OR's operator residual from portable daughter-grain CSV tables.

The CSVs describe independently segmented daughter regions and their
adjacent-region pairs. This is MTEX-style *boundary evidence* and NOT MTEX MCL.
No instrument source pixels or fitted OR are shipped in this module.
"""

from io import BytesIO

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from .orientation_kernel import require_so3
from .parent_or_interop import frozen_or_boundary_audit


def replay_grain_boundary_tables(
    grain_csv: bytes, boundary_csv: bytes, q_parent_from_child: object,
    parent_symmetry: tuple[np.ndarray, ...],
    child_symmetry: tuple[np.ndarray, ...],
    *, threshold_deg: float = 2.5, transition_width_deg: float = 1.5,
) -> dict:
    if not 0 < len(grain_csv) <= 5*1024*1024 or not 0 < len(boundary_csv) <= 5*1024*1024:
        raise ValueError("CSV payloads must be nonempty and <=5 MiB each")
    q = require_so3(np.asarray(q_parent_from_child,float),
                    tolerance=2e-8,name="parent<-daughter OR")
    grains=pd.read_csv(BytesIO(grain_csv))
    boundaries=pd.read_csv(BytesIO(boundary_csv))
    required_g={"qx","qy","qz","qw"}
    required_b={"row_index_A_zero_based","row_index_B_zero_based","boundary_contacts"}
    if not required_g.issubset(grains.columns) or not required_b.issubset(boundaries.columns):
        raise ValueError("Grain or boundary CSV columns do not match the portable schema")
    if not 2 <= len(grains) <= 5000 or not 1 <= len(boundaries) <= 20000:
        raise ValueError("Expected 2..5000 grain rows and 1..20000 boundary rows")
    try:
        quat=grains[["qx","qy","qz","qw"]].to_numpy(dtype=float)
        ij=boundaries[["row_index_A_zero_based","row_index_B_zero_based"]].to_numpy(dtype=float)
        contacts=boundaries["boundary_contacts"].to_numpy(dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError("CSV orientation or contact fields are nonnumeric") from exc
    if (not np.isfinite(quat).all() or
        np.any(np.abs(np.linalg.norm(quat,axis=1)-1)>1e-4) or
        not np.isfinite(ij).all() or
        np.any(ij!=np.rint(ij)) or
        not np.isfinite(contacts).all() or np.any(contacts < 1)):
        raise ValueError("Invalid normalized grain quaternions or boundary pair indices")
    ij=ij.astype(int)
    if np.any(ij<0) or np.any(ij>=len(quat)) or np.any(ij[:,0]==ij[:,1]):
        raise ValueError("Out-of-range or self-referential boundary pair")
    # Duplicate edges are not independent measurements; reject rather than
    # inflating fit percentages through repeated rows.
    pairs=np.sort(ij,axis=1)
    if len(np.unique(pairs,axis=0))!=len(pairs):
        raise ValueError("Duplicated boundary grain pairs")
    ori=Rotation.from_quat(quat).as_matrix()
    observed=ori[ij]
    report=frozen_or_boundary_audit(
        observed,q,parent_symmetry,child_symmetry,
        threshold_deg=threshold_deg,transition_width_deg=transition_width_deg,
    )
    residual=np.asarray(report["operator_misfit_deg"],float)
    reported_delta=None
    if "frozen_OR_operator_misfit_degrees" in boundaries:
        old=boundaries["frozen_OR_operator_misfit_degrees"].to_numpy(float)
        if not np.isfinite(old).all():
            raise ValueError("Historical residual CSV contains nonfinite data")
        reported_delta=float(np.max(np.abs(residual-old)))
    return {
        "schema":"cualni.grain-boundary-replay/1.0",
        "n_daughter_grain_regions":len(grains),
        "n_adjacency_edges":len(boundaries),
        "n_total_boundary_contacts":int(np.sum(contacts)),
        "median_operator_misfit_deg":report["median_operator_misfit_deg"],
        "p90_operator_misfit_deg":report["p90_operator_misfit_deg"],
        "fraction_under_3deg":report["fraction_under_3deg"],
        "historical_operator_residual_max_abs_difference_deg":reported_delta,
        "mtex_style_edge_probability":report["mtex_style_probability"],
        "operator_misfit_deg":report["operator_misfit_deg"],
        "physical_or_certified":False,
        "mtex_graph_mcl_executed":False,
        "caveat":"Full daughter grain segmentation was done externally; this replays its adjacency evidence, not MTEX segmentation or Markov clustering.",
    }
