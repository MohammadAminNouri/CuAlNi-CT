from __future__ import annotations

"""OR-INDEPENDENT indexed CTF cartography, using existing EBSD mathematics.

The crystallographic point groups and cells are sourced from the CTF's
DECLARED phase table, not independently reindexed or certified. No reference
orientation relationship, correspondence, or theoretical twin answer is used.
"""

from collections import Counter
import hashlib
import io
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from cualni_cryst.ebsd_analysis import (
    build_neighbor_graph, extract_boundaries, grain_statistics,
    kernel_average_misorientation, segment_grains,
)
from cualni_cryst.ebsd_io import load_ctf
from cualni_cryst.ebsd_map import EBSDPhase
from cualni_cryst.lattice import Lattice

MAX_CTF_BYTES = 35 * 1024 * 1024


def declared_ctf_phases(content: bytes) -> tuple[dict[str, Any], ...]:
    """Read all AZtec phase declarations (including zero-indexed phases).

    CTF encodes a space-group NUMBER, but this is an indexing ASSUMPTION:
    its appearance in a file does not independently demonstrate phase identity.
    The table is never substituted for experimental diffraction validation.
    """
    if not content or len(content) > MAX_CTF_BYTES:
        raise ValueError("CTF is empty or exceeds the 35 MB limit")
    lines = content.decode("utf-8-sig", errors="strict").splitlines()
    if not lines or lines[0].strip() != "Channel Text File":
        raise ValueError("Not an Oxford/AZtec Channel Text File")
    header = None
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 2 and parts[0].lower() == "phases":
            try:
                count = int(parts[1])
            except ValueError as exc:
                raise ValueError("CTF Phases count is not an integer") from exc
            if not 0 <= count <= 256:
                raise ValueError("CTF phase declaration count out of range")
            header = (idx + 1, count)
            break
        if len(parts) >= 6 and parts[0].lower() == "phase":
            break
    if header is None:
        return ()
    index, n = header
    records = []
    import gemmi

    for phase_id in range(1, n + 1):
        if index + phase_id - 1 >= len(lines):
            raise ValueError("CTF phase header is truncated")
        cells = lines[index + phase_id - 1].split("\t")
        # Some synthetic/minimal CTF exports have no phase descriptors.
        tokens = lines[index + phase_id - 1].split()
        if (len(tokens) >= 6 and tokens[0].lower() == "phase"
                and "euler1" in (x.lower() for x in tokens)):
            # Some synthetic/legacy CTFs omit atomic phase declarations.
            return ()
        if len(cells) < 5:
            raise ValueError("CTF declared phase requires cell and space-group fields")
        try:
            lengths = tuple(float(x) for x in cells[0].split(";"))
            angles = tuple(float(x) for x in cells[1].split(";"))
            number = int(cells[4])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid CTF phase {phase_id} declaration") from exc
        if len(lengths) != 3 or len(angles) != 3:
            raise ValueError(f"CTF phase {phase_id} has invalid cell dimensions")
        lattice = Lattice(
            *lengths, *angles, length_unit="angstrom",
        )
        sg = gemmi.find_spacegroup_by_number(number)
        if sg is None:
            raise ValueError(f"CTF phase {phase_id} has invalid space-group number {number}")
        records.append({
            "phase_id": phase_id,
            "name": cells[2].strip() or f"AZtec phase {phase_id}",
            "space_group_number": number,
            "space_group": sg.hm,
            "declared_point_group": sg.point_group_hm(),
            "lengths_angstrom": tuple(float(v) for v in lengths),
            "angles_deg": tuple(float(v) for v in angles),
            "cell_metric_angstrom2": lattice.metric().tolist(),
            "provenance": "AZtec indexing declaration; NOT independently verified",
        })
    return tuple(records)


def compare_ctf_model_settings(
    declared: tuple[dict[str, Any], ...],
    project_payload, transformation_id,
    parent_phase_id: int, martensite_phase_id: int,
    *,
    length_relative_tolerance: float = 0.02,
    angle_tolerance_deg: float = 1.0,
) -> dict[str, Any]:
    """Reject unsupported equivalence between AZtec and model crystal bases.

    This comparison assumes identical declared conventional axes. If a
    physical rebase is needed it must be entered/proven separately, not guessed.
    """
    from cualni_cryst.project_io import project_from_dict
    from twin_app.input_explainer import convert_cell_lengths

    by_id = {p["phase_id"]: p for p in declared}
    state = project_from_dict(dict(project_payload), source="ctf:phase-audit")
    tr = state.project.transformation(transformation_id)
    roles = (
        ("parent", int(parent_phase_id), state.project.phase(tr.parent_phase_id)),
        ("martensite", int(martensite_phase_id), state.project.phase(tr.product_phase_id)),
    )
    results = []
    for role, phase_id, model in roles:
        item = by_id.get(phase_id)
        if item is None:
            results.append({"role": role, "phase_id": phase_id, "pass": False,
                            "reason": "AZtec header has no definition for this phase ID"})
            continue
        cell = model.lattice
        model_len = np.asarray(convert_cell_lengths(
            (cell.a, cell.b, cell.c),
            from_unit=cell.length_unit, to_unit="angstrom",
        ), dtype=float)
        input_len = np.asarray(item["lengths_angstrom"], dtype=float)
        length_error = float(np.max(np.abs(model_len / input_len - 1.)))
        angle_error = float(np.max(np.abs(
            np.asarray((cell.alpha_deg, cell.beta_deg, cell.gamma_deg), dtype=float)
            - np.asarray(item["angles_deg"], dtype=float)
        )))
        declared_group = item["declared_point_group"].replace(" ", "")
        selected_group = model.point_group_symbol.replace(" ", "")
        same_group = declared_group == selected_group
        ok = bool(
            length_error <= length_relative_tolerance
            and angle_error <= angle_tolerance_deg and same_group
        )
        results.append({
            "role": role, "phase_id": phase_id,
            "pass": ok, "maximum_length_relative_difference": length_error,
            "maximum_cell_angle_difference_deg": angle_error,
            "CTF_point_group": declared_group,
            "model_point_group": selected_group,
            "reason": (
                "same axis setting appears consistent within stated tolerances"
                if ok else
                "CELL SETTING OR SYMMETRY MISMATCH: verify the CTF phase,"
                " conventional basis and transformation correspondence"
            ),
        })
    return {
        "pass": bool(all(row["pass"] for row in results)),
        "comparisons": results,
        "scope": (
            "Input-consistency comparison only. Matching declared CTF cells "
            "does NOT independently prove phase identity, indexing quality, "
            "orientation relationship or parent-martensite correspondence."
        ),
        "length_tolerance_fraction": length_relative_tolerance,
        "angle_tolerance_deg": angle_tolerance_deg,
    }


def indexed_neighbor_disorientation(eb, phases, neighbors, *,
                                   threshold_deg, computed_angles=None):
    """Exact full proper-symmetry quotient for ALL indexed neighbor edges.

    min_{S1,S2 in G} angle(S1^T g1^T g2 S2)
        = min_{S in G} angle(g1^T g2 S),
    since G is closed and trace is cyclic. This vectorized calculation uses
    all declared proper operators; it does NOT approximate by angle alone.
    """
    edges=np.asarray(neighbors.edges,dtype=int).reshape(-1,2)
    if not len(edges):
        return {"n_indexed_neighbors":0,"median_deg":None,
                "fraction_below_segmentation_threshold":None,
                "fraction_above_30_deg":None}
    a,b=edges[:,0],edges[:,1]
    valid=eb.indexed[a]&eb.indexed[b]&(eb.phase_id[a]==eb.phase_id[b])
    a,b=a[valid],b[valid]
    if computed_angles is not None:
        selected=np.asarray(computed_angles,dtype=float)
        if selected.shape!=(len(edges),):
            raise ValueError("Precomputed edge-angle shape mismatch")
        angles=selected[np.isfinite(selected)]
        if not len(angles):
            return {"n_indexed_neighbors":0,"median_deg":None,
                    "fraction_below_segmentation_threshold":None,
                    "fraction_above_30_deg":None}
        return {
            "n_indexed_neighbors":int(len(angles)),
            "median_deg":float(np.median(angles)),
            "fraction_below_segmentation_threshold":float(np.mean(angles<=threshold_deg)),
            "fraction_above_30_deg":float(np.mean(angles>30.)),
            "p90_deg":float(np.quantile(angles,0.9)),
            "scientific_scope":(
                "Full proper-symmetry quotient of adjacent indexed phase-matched "
                "pixels, not a physical twin/plane certificate."
            ),
        }
    pieces=[]
    for phase_id,phase in phases.items():
        take=(eb.phase_id[a]==int(phase_id))
        if not np.any(take):
            continue
        ra=np.asarray(eb.orientations[a[take]])
        rb=np.asarray(eb.orientations[b[take]])
        rel=np.einsum("nji,njk->nik",ra,rb,optimize=True)
        proper=np.asarray(phase.proper_symmetry_cartesian,dtype=float)
        traces=np.einsum("nij,kji->nk",rel,proper,optimize=True)
        angles=np.rad2deg(np.arccos(np.clip(
            (np.max(traces,axis=1)-1.)/2.,-1.,1.
        )))
        pieces.append(angles)
    if not pieces:
        return {"n_indexed_neighbors":0,"median_deg":None,
                "fraction_below_segmentation_threshold":None,
                "fraction_above_30_deg":None}
    angles=np.concatenate(pieces)
    return {
        "n_indexed_neighbors":int(len(angles)),
        "median_deg":float(np.median(angles)),
        "fraction_below_segmentation_threshold":
            float(np.mean(angles<=threshold_deg)),
        "fraction_above_30_deg":float(np.mean(angles>30.)),
        "p90_deg":float(np.quantile(angles,0.9)),
        "scientific_scope":(
            "Symmetry-reduced adjacent-pixel orientation jumps, not "
            "Kikuchi reindexing, a physical grain-boundary declaration, "
            "nor an experimental Type I/II classification."
        ),
    }


def independent_ctf_cartography(
    content: bytes, filename: str,
    *, misorientation_threshold_deg: float = 3.0,
    minimum_grain_pixels: int = 3,
    kam_neighbor_cutoff_deg: float = 5.0,
    three_dimensional_angle_unit: str | None = None,
) -> dict[str, Any]:
    """Fully analyze measured indexed orientations WITHOUT an OR/theory model.

    Uses the frozen crystallographic grain and KAM algorithms, with declared
    CTF phase metrics/symmetry ONLY. Outputs never claim CT variants/twin types.
    """
    if Path(filename).name != filename or Path(filename).suffix.lower() != ".ctf":
        raise ValueError("Provide a simple .ctf filename")
    if not (math.isfinite(misorientation_threshold_deg)
            and misorientation_threshold_deg > 0):
        raise ValueError("Grain threshold must be positive")
    if not (1 <= int(minimum_grain_pixels) <= 1000):
        raise ValueError("Min. grain pixel count must be 1 to 1000")
    if not (math.isfinite(kam_neighbor_cutoff_deg)
            and kam_neighbor_cutoff_deg > 0):
        raise ValueError("KAM cutoff must be positive")
    phases_declared = declared_ctf_phases(content)
    if not phases_declared:
        raise ValueError(
            "The CTF has no complete AZtec phase declarations. "
            "Independent symmetry-aware grain analysis requires a verified "
            "phase metric and point group; no symmetry will be invented."
        )
    with tempfile.TemporaryDirectory(prefix="ctf-or-free-") as directory:
        path = Path(directory) / filename
        path.write_bytes(content)
        from cualni_cryst.ebsd_map import AngleUnit
        if three_dimensional_angle_unit not in {None,"degree","radian"}:
            raise ValueError("Invalid explicit 3-D CTF angle unit")
        eb = load_ctf(
            path, three_dimensional_angle_unit=(
                AngleUnit(three_dimensional_angle_unit)
                if three_dimensional_angle_unit is not None else None
            ),
        )

    by_id = {p["phase_id"]: p for p in phases_declared}
    indexed_ids = sorted(set(map(int, eb.phase_id[eb.indexed])))
    unknown = set(indexed_ids).difference(by_id)
    if unknown:
        raise ValueError(f"CTF contains indexed phases not declared: {sorted(unknown)}")
    if not indexed_ids:
        raise ValueError("CTF has no indexed orientations")
    phases = {}
    for ident in indexed_ids:
        phase = by_id[ident]
        phases[ident] = EBSDPhase.from_point_group(
            ident, phase["name"],
            Lattice(*phase["lengths_angstrom"],
                    *phase["angles_deg"], length_unit="angstrom"),
            phase["declared_point_group"],
        )

    neighbors = build_neighbor_graph(eb, radius_factor=1.15)
    # The reference/native algorithm remains authoritative for small maps.
    # For large real AZtec scans use the EXACT same proper-symmetry quotient
    # with batched matrix multiplications and connected components, rather
    # than millions of interpreted Python symmetry loops.
    accelerated = None
    if eb.n_points >= 3000:
        from twin_app.ctf_fast_grains import accelerated_ebsd_maps
        accelerated = accelerated_ebsd_maps(
            eb, phases, neighbors, threshold_deg=misorientation_threshold_deg,
            minimum_points=int(minimum_grain_pixels),
            kam_max_deg=kam_neighbor_cutoff_deg,
        )
        seg, kam = accelerated["segmentation"], accelerated["kam_deg"]
    else:
        seg = segment_grains(
            eb, phases, threshold_deg=misorientation_threshold_deg,
            neighbor_graph=neighbors, minimum_points=int(minimum_grain_pixels),
        )
        kam = kernel_average_misorientation(
            eb, phases, neighbor_graph=neighbors,
            maximum_neighbor_misorientation_deg=kam_neighbor_cutoff_deg,
        )
    # Large-map means follow the identical native symmetry-alignment and
    # SO(3) polar-mean equations, vectorized over each region's pixels.
    # For small maps, retain the original native reference implementation.
    if accelerated is not None:
        from twin_app.ctf_fast_grains import accelerated_grain_statistics
        grains = accelerated_grain_statistics(eb, seg, phases)
    else:
        grains = grain_statistics(eb, seg, phases)
    edges = extract_boundaries(eb, seg, phases)
    neighbor_audit = indexed_neighbor_disorientation(
        eb, phases, neighbors, threshold_deg=misorientation_threshold_deg,
        computed_angles=(None if accelerated is None else accelerated["edge_angles_deg"]),
    )
    boundary_pairs = Counter(
        tuple(sorted((int(e.grain_a), int(e.grain_b)))) for e in edges
    )
    grain_rows = [
        {
            "grain_id": g.grain_id, "phase_id": g.phase_id, "pixels": g.size,
            "centroid_x": float(g.centroid[0]),
            "centroid_y": float(g.centroid[1]),
            "gos_deg": g.gos_deg,
            "maximum_spread_deg": g.maximum_spread_deg,
            "mean_crystal_to_sample_rotation": g.mean_orientation.tolist(),
        } for g in grains
    ]
    indexed = eb.indexed
    grain_gos = np.full(eb.n_points,np.nan,dtype=float)
    for grain in grains:
        grain_gos[grain.point_indices] = grain.gos_deg
    neighbor_max = (
        accelerated["neighbor_max_deg"] if accelerated is not None
        else np.full(eb.n_points,np.nan,dtype=float)
    )
    boundary_mask = (
        accelerated["grain_boundary_mask"] if accelerated is not None
        else np.zeros(eb.n_points,dtype=bool)
    )
    if accelerated is None:
        for edge in edges:
            boundary_mask[edge.point_a] = True
            boundary_mask[edge.point_b] = True
    result = {
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "status": "MEASURED_GRAIN_DIAGNOSTICS_NOT_TWIN_CERTIFICATION",
        "orientation_convention": eb.metadata.get("orientation_convention", ""),
        "declared_phases": phases_declared,
        "indexed_phase_ids": indexed_ids,
        "n_pixels": eb.n_points,
        "n_indexed": int(np.count_nonzero(indexed)),
        "n_retained_indexed": int(np.count_nonzero(seg.grain_id>=0)),
        "neighbor_disorientation": neighbor_audit,
        "n_grains": len(grains),
        "n_neighbor_edges": len(neighbors.edges),
        "n_intergrain_edges": len(edges),
        "n_distinct_grain_adjacencies": len(boundary_pairs),
        "threshold_deg": misorientation_threshold_deg,
        "minimum_grain_pixels": int(minimum_grain_pixels),
        "kam_cutoff_deg": kam_neighbor_cutoff_deg,
        "grains": grain_rows,
        "adjacent_grain_pairs": [
            {"grain_a": a, "grain_b": b, "pixel_neighbor_edges": count}
            for (a,b),count in sorted(boundary_pairs.items())
        ],
        "maps": {
            "x": eb.x, "y": eb.y, "z": eb.z,
            "phase": np.where(indexed, eb.phase_id, 0),
            "grain_id": seg.grain_id,
            "kam_deg": kam,
            "gos_deg": grain_gos,
            "neighbor_max_deg": neighbor_max,
            "grain_boundary_mask": boundary_mask.astype(float),
            "MAD": np.where(indexed, eb.quality.get("MAD", np.full(eb.n_points,np.nan)),np.nan),
            "BC": np.where(indexed, eb.quality.get("BC", np.full(eb.n_points,np.nan)),np.nan),
            "BS": np.where(indexed, eb.quality.get("BS", np.full(eb.n_points,np.nan)),np.nan),
        },
        "source_boundary": (
            "No parent orientation, twin plane or correspondence is inferred. "
            "Phase symmetries are copied from the AZtec phase table, not "
            "verified from measured diffraction patterns."
        ),
    }
    return result


def plot_ctf_field(maps: dict, field: str):
    """Build one physical-coordinate plot (no fake IPF coloring)."""
    import plotly.graph_objects as go

    supported = {
        "phase", "grain_id", "kam_deg", "gos_deg",
        "neighbor_max_deg", "grain_boundary_mask",
        "MAD", "BC", "BS",
    }
    if field not in supported:
        raise ValueError("Unsupported EBSD map field")
    x, y = np.asarray(maps["x"]), np.asarray(maps["y"])
    values = np.asarray(maps[field], dtype=float)
    if len(x) != len(values):
        raise ValueError("Coordinate/value length mismatch")
    # Most AZtec grid maps are rectangular. Only use pixel heatmaps when
    # *all* coordinates establish a unique complete Cartesian grid.
    xs, xi = np.unique(np.round(x, 4), return_inverse=True)
    ys, yi = np.unique(np.round(y, 4), return_inverse=True)
    if len(xs) * len(ys) == len(values):
        pair = yi.astype(np.int64) * len(xs) + xi.astype(np.int64)
        if np.unique(pair).size == len(values):
            image = np.full((len(ys), len(xs)), np.nan)
            image[yi, xi] = values
            fig = go.Figure(data=go.Heatmap(
                z=image, x=xs, y=ys, colorscale="Viridis",
                colorbar=dict(title=field), hoverongaps=False,
            ))
        else:
            fig = None
    else:
        fig = None
    if fig is None:
        fig = go.Figure(data=go.Scattergl(
            x=x, y=y, mode="markers",
            marker=dict(color=values, size=4, colorscale="Viridis",
                        showscale=True, colorbar=dict(title=field)),
        ))
    fig.update_layout(
        title=field, height=530,
        margin=dict(l=20,r=20,t=50,b=30),
        xaxis_title="X (CTF coordinate units)",
        yaxis_title="Y (CTF coordinate units)",
    )
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    return fig
