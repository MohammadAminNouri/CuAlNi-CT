from __future__ import annotations

"""Generic project/crystal loading for PTCLab-style workflows.

This module deliberately separates a *benchmark preset* from the scientific
engine. James-Hane is one preset, not a global default data model.

User projects are JSON and can contain arbitrary valid lattice metrics,
conventional point groups, explicit symmetry matrices, stored ORs, exact
correspondence matrices, composition/state metadata, and optional explicit
basis-change links between alternative representations of one physical phase.
"""

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

from .correspondence import Correspondence
from .crystal_objects import CrystalBasisRef
from .lattice import Lattice
from .numerics import NumericalPolicy
from .point_groups import (
    point_group_operations,
    point_group_table,
    resolve_point_group,
)
from .project_state import (
    Composition,
    CompositionBasis,
    CompositionComponent,
    OrientationDefinition,
    OrientationState,
    OrientationTheoryOrigin,
    PhaseState,
    ProjectState,
    StateProvenance,
    ThermomechanicalState,
    TransformationState,
    james_hane_6m_reference_project,
)
from .provenance import DataStatus, SourceRef
from .representation import CartesianConvention

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RepresentationLink:
    """Explicit coordinate change between two cells of one physical phase.

    Convention
    ----------
    u_from = P_from_to @ u_to

    Therefore
        M_to = P_from_to.T @ M_from @ P_from_to
        p_to = P_from_to.T @ p_from

    The matrix may describe a conventional-cell/supercell relation and need
    not be unimodular. It must be invertible.
    """

    link_id: str
    from_phase_id: str
    to_phase_id: str
    coordinates_from_from_to: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    source_key: str = ""
    notes: str = ""

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(self.coordinates_from_from_to, dtype=float)


@dataclass(frozen=True)
class RepresentationLinkAudit:
    link_id: str
    from_phase_id: str
    to_phase_id: str
    same_physical_phase_label: bool
    determinant: float
    metric_relative_residual: float
    volume_jacobian_residual: float
    passed: bool
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "link_id": self.link_id,
            "from_phase_id": self.from_phase_id,
            "to_phase_id": self.to_phase_id,
            "same_physical_phase_label": self.same_physical_phase_label,
            "determinant": self.determinant,
            "metric_relative_residual": self.metric_relative_residual,
            "volume_jacobian_residual": self.volume_jacobian_residual,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass(frozen=True)
class LoadedProject:
    project: ProjectState
    representation_links: tuple[RepresentationLink, ...] = ()
    source: str = ""

    def representation_audits(self) -> tuple[RepresentationLinkAudit, ...]:
        return tuple(
            audit_representation_link(self.project, link)
            for link in self.representation_links
        )

    def assert_representation_links(self) -> None:
        failed = [audit for audit in self.representation_audits() if not audit.passed]
        if failed:
            text = "; ".join(
                f"{item.link_id}: metric={item.metric_relative_residual:.3e}, "
                f"volume={item.volume_jacobian_residual:.3e}"
                for item in failed
            )
            raise ValueError(f"Representation-link validation failed: {text}")


def _matrix3(
    values: object, *, name: str
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    array = np.asarray(values, dtype=float)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must be a finite 3x3 matrix; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return tuple(tuple(float(x) for x in row) for row in array)  # type: ignore[return-value]


def _exact_scalar(value: object) -> sp.Expr:
    if isinstance(value, bool):
        raise TypeError("Boolean is not a valid matrix scalar.")
    if isinstance(value, int):
        return sp.Integer(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Non-finite exact-matrix scalar.")
        return sp.Rational(str(value))
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Empty exact-matrix scalar.")
        try:
            fraction = Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(
                f"Exact matrix scalar {value!r} must be integer, decimal, or p/q."
            ) from exc
        return sp.Rational(fraction.numerator, fraction.denominator)
    raise TypeError(f"Unsupported exact-matrix scalar type: {type(value).__name__}")


def _exact_matrix3(values: object, *, name: str) -> sp.Matrix:
    if not isinstance(values, list):
        raise TypeError(f"{name} must be a 3x3 nested list.")
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 rows.")
    rows: list[list[sp.Expr]] = []
    for row in values:
        if not isinstance(row, list):
            raise TypeError(f"{name} rows must be lists.")
        if len(row) != 3:
            raise ValueError(f"{name} rows must contain exactly 3 values.")
        rows.append([_exact_scalar(value) for value in row])
    matrix = sp.Matrix(rows)
    if sp.simplify(matrix.det()) == 0:
        raise ValueError(f"{name} must be invertible.")
    return matrix


def _status(value: object, default: DataStatus = DataStatus.UNVERIFIED) -> DataStatus:
    if value is None or str(value).strip() == "":
        return default
    return DataStatus(str(value))


def _provenance(
    payload: object,
    *,
    default_status: DataStatus = DataStatus.UNVERIFIED,
) -> StateProvenance:
    if payload is None:
        data: dict[str, object] = {}
    elif isinstance(payload, dict):
        data = payload
    else:
        raise TypeError("provenance must be an object.")
    return StateProvenance(
        status=_status(data.get("status"), default_status),
        source_key=str(data.get("source_key", "")).strip(),
        uncertainty=str(data.get("uncertainty", "")).strip(),
        notes=str(data.get("notes", "")).strip(),
    )


def _source(payload: dict[str, object]) -> SourceRef:
    key = str(payload.get("key", "")).strip()
    citation = str(payload.get("citation", "")).strip()
    if not key or not citation:
        raise ValueError("Every source requires non-empty 'key' and 'citation'.")
    equations = payload.get("equations", [])
    if not isinstance(equations, list):
        raise TypeError("source.equations must be a list.")
    return SourceRef(
        key=key,
        citation=citation,
        doi=str(payload.get("doi", "")).strip(),
        url=str(payload.get("url", "")).strip(),
        pages=str(payload.get("pages", "")).strip(),
        equations=tuple(str(item) for item in equations),
        notes=str(payload.get("notes", "")).strip(),
    )


def _phase(payload: dict[str, object]) -> PhaseState:
    phase_id = str(payload.get("phase_id", "")).strip()
    if not phase_id:
        raise ValueError("phase.phase_id is required.")

    label = str(payload.get("label", phase_id)).strip() or phase_id
    physical_phase = str(payload.get("physical_phase", label)).strip() or label
    representation = str(payload.get("cell_representation", "user_cell")).strip()
    basis_id = str(payload.get("basis_id", f"{phase_id}_basis")).strip()

    cell = payload.get("cell")
    if not isinstance(cell, dict):
        raise TypeError(f"phase {phase_id!r} requires a 'cell' object.")

    required = ("a", "b", "c")
    missing = [name for name in required if name not in cell]
    if missing:
        raise ValueError(f"phase {phase_id!r} cell missing {missing}.")

    lattice = Lattice(
        a=float(cell["a"]),
        b=float(cell["b"]),
        c=float(cell["c"]),
        alpha_deg=float(cell.get("alpha_deg", cell.get("alpha", 90.0))),
        beta_deg=float(cell.get("beta_deg", cell.get("beta", 90.0))),
        gamma_deg=float(cell.get("gamma_deg", cell.get("gamma", 90.0))),
        label=str(cell.get("label", representation)),
        length_unit=str(cell.get("length_unit", "angstrom")),
    )

    explicit_ops = payload.get("symmetry_matrices")
    point_group = str(payload.get("point_group", "")).strip()
    if explicit_ops is not None:
        if not isinstance(explicit_ops, list):
            raise TypeError(f"phase {phase_id!r}: symmetry_matrices must be a list.")
        if not explicit_ops:
            raise ValueError(
                f"phase {phase_id!r}: symmetry_matrices must not be empty."
            )
        operations = tuple(
            _matrix3(matrix, name=f"{phase_id}.symmetry_matrices[{index}]")
            for index, matrix in enumerate(explicit_ops)
        )
        symbol = point_group or "explicit"
    else:
        if not point_group:
            raise ValueError(
                f"phase {phase_id!r} requires point_group or explicit symmetry_matrices. "
                "Use point_group='1' when no symmetry beyond identity is intended."
            )
        definition = resolve_point_group(point_group)
        operations = tuple(
            _matrix3(
                np.asarray(matrix, dtype=float), name=f"{phase_id}.{definition.symbol}"
            )
            for matrix in definition.operations()
        )
        symbol = definition.symbol

    basis = CrystalBasisRef(
        phase_id,
        basis_id,
        representation,
    )
    return PhaseState(
        phase_id=phase_id,
        label=label,
        physical_phase=physical_phase,
        cell_representation=representation,
        lattice=lattice,
        basis=basis,
        point_group_symbol=symbol,
        symmetry_operators=operations,
        provenance=_provenance(payload.get("provenance")),
    )


def _transformation(payload: dict[str, object]) -> TransformationState:
    transformation_id = str(payload.get("transformation_id", "")).strip()
    parent = str(payload.get("parent_phase_id", "")).strip()
    product = str(payload.get("product_phase_id", "")).strip()
    if not transformation_id or not parent or not product:
        raise ValueError(
            "Transformation requires transformation_id, parent_phase_id, "
            "and product_phase_id."
        )
    matrix_payload = payload.get("correspondence_M_from_A")
    if matrix_payload is None:
        matrix_payload = payload.get("correspondence")
    if matrix_payload is None:
        raise ValueError(
            f"transformation {transformation_id!r} requires correspondence_M_from_A."
        )
    correspondence = Correspondence(
        _exact_matrix3(
            matrix_payload,
            name=f"{transformation_id}.correspondence_M_from_A",
        ),
        label=str(payload.get("correspondence_label", transformation_id)),
        source=str(payload.get("correspondence_source", "")),
        derivation=str(payload.get("correspondence_derivation", "")),
    )
    return TransformationState(
        transformation_id=transformation_id,
        label=str(payload.get("label", transformation_id)),
        parent_phase_id=parent,
        product_phase_id=product,
        correspondence=correspondence,
        parent_cartesian_convention=CartesianConvention(
            payload.get(
                "parent_cartesian_convention",
                CartesianConvention.SYMMETRIC_METRIC.value,
            )
        ),
        product_cartesian_convention=CartesianConvention(
            payload.get(
                "product_cartesian_convention",
                CartesianConvention.SYMMETRIC_METRIC.value,
            )
        ),
        provenance=_provenance(payload.get("provenance")),
    )


def _orientation(payload: dict[str, object]) -> OrientationState:
    orientation_id = str(payload.get("orientation_id", "")).strip()
    reference = str(payload.get("reference_phase_id", "")).strip()
    moving = str(payload.get("moving_phase_id", "")).strip()
    if not orientation_id or not reference or not moving:
        raise ValueError(
            "Stored orientation requires orientation_id, reference_phase_id, "
            "and moving_phase_id."
        )
    matrix = payload.get("R_reference_from_moving", payload.get("matrix"))
    if matrix is None:
        raise ValueError(
            f"orientation {orientation_id!r} requires R_reference_from_moving."
        )
    return OrientationState(
        orientation_id=orientation_id,
        label=str(payload.get("label", orientation_id)),
        reference_phase_id=reference,
        moving_phase_id=moving,
        R_reference_from_moving=_matrix3(
            matrix,
            name=f"{orientation_id}.R_reference_from_moving",
        ),
        reference_cartesian_convention=CartesianConvention(
            payload.get(
                "reference_cartesian_convention",
                CartesianConvention.PTCLAB_A_X_C_XZ.value,
            )
        ),
        moving_cartesian_convention=CartesianConvention(
            payload.get(
                "moving_cartesian_convention",
                CartesianConvention.PTCLAB_A_X_C_XZ.value,
            )
        ),
        definition_method=OrientationDefinition(
            payload.get("definition_method", OrientationDefinition.USER_MATRIX.value)
        ),
        theory_origin=OrientationTheoryOrigin(
            payload.get("theory_origin", OrientationTheoryOrigin.USER_DEFINED.value)
        ),
        provenance=_provenance(payload.get("provenance")),
        notes=str(payload.get("notes", "")),
        transformation_id=str(payload.get("transformation_id", "")).strip(),
    )


def _composition(payload: object) -> Composition | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("composition must be an object.")
    basis = CompositionBasis(
        payload.get("basis", CompositionBasis.WEIGHT_PERCENT.value)
    )
    components_payload = payload.get("components")
    components: list[CompositionComponent] = []
    if isinstance(components_payload, dict):
        components = [
            CompositionComponent(str(element), float(value))
            for element, value in components_payload.items()
        ]
    elif isinstance(components_payload, list):
        for item in components_payload:
            if not isinstance(item, dict):
                raise TypeError("composition.components list entries must be objects.")
            components.append(
                CompositionComponent(str(item["element"]), float(item["value"]))
            )
    else:
        raise TypeError("composition.components must be an object or list.")
    return Composition(
        tuple(components),
        basis,
        balance_element=str(payload.get("balance_element", "")),
        provenance=_provenance(payload.get("provenance")),
    )


def _thermomechanical(payload: object) -> ThermomechanicalState:
    if payload is None:
        return ThermomechanicalState()
    if not isinstance(payload, dict):
        raise TypeError("thermomechanical_state must be an object.")
    stress = payload.get("cauchy_stress_mpa")
    return ThermomechanicalState(
        temperature_k=(
            None
            if payload.get("temperature_k") is None
            else float(payload["temperature_k"])
        ),
        cauchy_stress_mpa=(
            None
            if stress is None
            else _matrix3(stress, name="thermomechanical_state.cauchy_stress_mpa")
        ),
        history=str(payload.get("history", "")),
        provenance=_provenance(payload.get("provenance")),
    )


def _numerical_policy(payload: object) -> NumericalPolicy:
    if payload is None:
        return NumericalPolicy()
    if not isinstance(payload, dict):
        raise TypeError("numerical_policy must be an object.")
    defaults = NumericalPolicy()
    return NumericalPolicy(
        algebraic=float(payload.get("algebraic", defaults.algebraic)),
        representation=float(payload.get("representation", defaults.representation)),
        exact_eigenvalue=float(
            payload.get("exact_eigenvalue", defaults.exact_eigenvalue)
        ),
        rank_one=float(payload.get("rank_one", defaults.rank_one)),
        projective_angle_deg=float(
            payload.get("projective_angle_deg", defaults.projective_angle_deg)
        ),
    )


def _representation_link(payload: dict[str, object]) -> RepresentationLink:
    link_id = str(payload.get("link_id", "")).strip()
    from_phase = str(payload.get("from_phase_id", "")).strip()
    to_phase = str(payload.get("to_phase_id", "")).strip()
    if not link_id or not from_phase or not to_phase:
        raise ValueError(
            "representation link requires link_id, from_phase_id, and to_phase_id."
        )
    matrix = payload.get("coordinates_from_from_to")
    if matrix is None:
        raise ValueError(
            f"representation link {link_id!r} requires coordinates_from_from_to."
        )
    frozen = _matrix3(matrix, name=f"{link_id}.coordinates_from_from_to")
    if abs(float(np.linalg.det(np.asarray(frozen)))) <= 1.0e-14:
        raise ValueError(f"representation link {link_id!r} matrix is singular.")
    return RepresentationLink(
        link_id=link_id,
        from_phase_id=from_phase,
        to_phase_id=to_phase,
        coordinates_from_from_to=frozen,
        source_key=str(payload.get("source_key", "")).strip(),
        notes=str(payload.get("notes", "")).strip(),
    )


def project_from_dict(payload: dict[str, object], *, source: str = "") -> LoadedProject:
    version = int(payload.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project schema_version={version}; expected {SCHEMA_VERSION}."
        )

    phase_payload = payload.get("phases")
    if not isinstance(phase_payload, list):
        raise TypeError("Project 'phases' must be a list.")
    if not phase_payload:
        raise ValueError("Project requires at least one phase.")
    if not all(isinstance(item, dict) for item in phase_payload):
        raise TypeError("Every phases entry must be an object.")
    phases = tuple(_phase(item) for item in phase_payload)

    transformation_payload = payload.get("transformations", [])
    orientation_payload = payload.get("orientations", [])
    source_payload = payload.get("sources", [])
    links_payload = payload.get("representation_links", [])
    for label, value in (
        ("transformations", transformation_payload),
        ("orientations", orientation_payload),
        ("sources", source_payload),
        ("representation_links", links_payload),
    ):
        if not isinstance(value, list):
            raise TypeError(f"{label} must be a list.")

    for label, values in (
        ("transformations", transformation_payload),
        ("orientations", orientation_payload),
        ("sources", source_payload),
        ("representation_links", links_payload),
    ):
        if not all(isinstance(item, dict) for item in values):
            raise TypeError(f"Every {label} entry must be an object.")

    transformations = tuple(_transformation(item) for item in transformation_payload)
    orientations = tuple(_orientation(item) for item in orientation_payload)
    sources = tuple(_source(item) for item in source_payload)
    links = tuple(_representation_link(item) for item in links_payload)

    project = ProjectState(
        project_id=str(payload.get("project_id", "user_project")).strip(),
        title=str(payload.get("title", "User crystallography project")).strip(),
        phases=phases,
        transformations=transformations,
        sources=sources,
        composition=_composition(payload.get("composition")),
        thermomechanical_state=_thermomechanical(payload.get("thermomechanical_state")),
        numerical_policy=_numerical_policy(payload.get("numerical_policy")),
        notes=str(payload.get("notes", "")),
        orientations=orientations,
    )
    project.validate().assert_passed()

    known = {phase.phase_id for phase in project.phases}
    for link in links:
        if link.from_phase_id not in known or link.to_phase_id not in known:
            raise ValueError(
                f"representation link {link.link_id!r} references unknown phase."
            )
        if link.from_phase_id == link.to_phase_id:
            raise ValueError(
                f"representation link {link.link_id!r} must connect distinct representations."
            )

    loaded = LoadedProject(project=project, representation_links=links, source=source)
    loaded.assert_representation_links()
    return loaded


def project_from_json(path: str | Path) -> LoadedProject:
    project_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(project_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Project JSON not found: {project_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in {project_path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError("Top-level project JSON must be an object.")
    return project_from_dict(payload, source=str(project_path))


def _perez_cerrato_2024_beta3() -> LoadedProject:
    source = SourceRef(
        key="perez_cerrato2024",
        citation=(
            "M. Perez-Cerrato et al., Powder Metallurgy Processing to Enhance "
            "Superelasticity and Shape Memory in Polycrystalline Cu-Al-Ni Alloys, "
            "Materials 17 (2024) 6165."
        ),
        doi="10.3390/ma17246165",
        notes=(
            "EBSD martensite indexing reports monoclinic beta'3 (C2/m): "
            "a=1.3817 nm, b=0.52856 nm, c=0.43987 nm, beta=113.6 deg."
        ),
    )
    phase = PhaseState(
        phase_id="beta3_perez2024",
        label="Cu-Al-Ni beta'3 martensite (Perez-Cerrato 2024)",
        physical_phase="Cu-Al-Ni beta'3 martensite",
        cell_representation="C2/m EBSD conventional cell, unique b",
        lattice=Lattice.monoclinic_unique_b(
            a=13.817,
            b=5.2856,
            c=4.3987,
            beta_deg=113.6,
            label="beta'3 C2/m",
            length_unit="angstrom",
        ),
        basis=CrystalBasisRef(
            "beta3_perez2024",
            "beta3_c2m_ebsd",
            "C2/m EBSD conventional cell, unique b",
        ),
        point_group_symbol="2/m",
        symmetry_operators=tuple(
            _matrix3(np.asarray(matrix, dtype=float), name="2/m")
            for matrix in point_group_operations("2/m")
        ),
        provenance=StateProvenance(
            DataStatus.SOURCE_MEASURED,
            "perez_cerrato2024",
            notes="Published EBSD indexing cell; converted from nm to angstrom exactly by x10.",
        ),
    )
    project = ProjectState(
        project_id="perez_cerrato_2024_beta3",
        title="Perez-Cerrato 2024 Cu-Al-Ni beta'3 EBSD cell benchmark",
        phases=(phase,),
        sources=(source,),
        notes=(
            "Single-phase literature preset. No austenite cell or OR is invented. "
            "Combine with user/literature austenite data in a custom project for "
            "two-phase calculations."
        ),
    )
    project.validate().assert_passed()
    return LoadedProject(project, source="preset:perez_cerrato_2024_beta3")


_PRESETS = {
    "james_hane_2000": lambda: LoadedProject(
        james_hane_6m_reference_project(),
        source="preset:james_hane_2000",
    ),
    "perez_cerrato_2024_beta3": _perez_cerrato_2024_beta3,
}


def preset_names() -> tuple[str, ...]:
    return tuple(sorted(_PRESETS))


def load_project(source: str | Path | None) -> LoadedProject:
    if source is None or str(source).strip() == "":
        raise ValueError(
            "Project source is required. Use preset:james_hane_2000, another "
            "preset, or a JSON project path."
        )
    text = str(source).strip()
    if text.startswith("preset:"):
        name = text.split(":", 1)[1].strip()
        factory = _PRESETS.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown preset {name!r}. Available: {', '.join(preset_names())}."
            )
        return factory()
    return project_from_json(text)


def audit_representation_link(
    project: ProjectState,
    link: RepresentationLink,
) -> RepresentationLinkAudit:
    phase_from = project.phase(link.from_phase_id)
    phase_to = project.phase(link.to_phase_id)
    P = link.matrix
    determinant = float(np.linalg.det(P))
    M_from = phase_from.lattice.metric()
    M_to = phase_to.lattice.metric()
    predicted = P.T @ M_from @ P
    scale = max(float(np.linalg.norm(M_to)), float(np.linalg.norm(predicted)), 1.0)
    metric_residual = float(np.linalg.norm(predicted - M_to) / scale)

    volume_from = float(np.sqrt(np.linalg.det(M_from)))
    volume_to = float(np.sqrt(np.linalg.det(M_to)))
    predicted_volume = abs(determinant) * volume_from
    volume_scale = max(abs(volume_to), abs(predicted_volume), 1.0)
    volume_residual = abs(predicted_volume - volume_to) / volume_scale

    same_phase = (
        phase_from.physical_phase.strip().lower()
        == phase_to.physical_phase.strip().lower()
    )
    tolerance = project.numerical_policy.representation
    passed = (
        metric_residual <= tolerance
        and volume_residual <= tolerance
        and abs(determinant) > 1.0e-14
    )
    note = (
        "Metric-equivalent cell representations."
        if passed
        else "Provided basis change does not reproduce the target metric."
    )
    if not same_phase:
        note += " Physical-phase labels differ; review provenance/identity."

    return RepresentationLinkAudit(
        link_id=link.link_id,
        from_phase_id=link.from_phase_id,
        to_phase_id=link.to_phase_id,
        same_physical_phase_label=same_phase,
        determinant=determinant,
        metric_relative_residual=metric_residual,
        volume_jacobian_residual=volume_residual,
        passed=passed,
        note=note,
    )


def _cell_payload(text: str, unit: str) -> dict[str, object]:
    tokens = text.replace(",", " ").split()
    if len(tokens) != 6:
        raise ValueError("Cell must contain six numbers: 'a b c alpha beta gamma'.")
    values = [float(token) for token in tokens]
    return {
        "a": values[0],
        "b": values[1],
        "c": values[2],
        "alpha_deg": values[3],
        "beta_deg": values[4],
        "gamma_deg": values[5],
        "length_unit": unit,
    }


def _phase_payload(
    phase_id: str,
    cell_text: str,
    point_group: str,
    unit: str,
    *,
    label: str = "",
    physical_phase: str = "",
    representation: str = "user_cell",
) -> dict[str, object]:
    resolve_point_group(point_group)
    return {
        "phase_id": phase_id,
        "label": label or phase_id,
        "physical_phase": physical_phase or label or phase_id,
        "cell_representation": representation,
        "basis_id": f"{phase_id}_basis",
        "cell": _cell_payload(cell_text, unit),
        "point_group": point_group,
        "provenance": {
            "status": DataStatus.USER_MEASURED.value,
            "notes": "Created by cualni-project create-pair; review provenance.",
        },
    }


def generic_template() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "project_id": "my_crystallography_project",
        "title": "Arbitrary two-phase crystallography project",
        "phases": [
            {
                "phase_id": "phase_A",
                "label": "Reference phase",
                "physical_phase": "reference phase",
                "cell_representation": "conventional",
                "basis_id": "phase_A_basis",
                "cell": {
                    "a": 5.8,
                    "b": 5.8,
                    "c": 5.8,
                    "alpha_deg": 90.0,
                    "beta_deg": 90.0,
                    "gamma_deg": 90.0,
                    "length_unit": "angstrom",
                },
                "point_group": "m-3m",
                "provenance": {"status": "USER_MEASURED"},
            },
            {
                "phase_id": "phase_B",
                "label": "Moving phase",
                "physical_phase": "moving phase",
                "cell_representation": "conventional",
                "basis_id": "phase_B_basis",
                "cell": {
                    "a": 4.4,
                    "b": 5.3,
                    "c": 13.8,
                    "alpha_deg": 90.0,
                    "beta_deg": 100.0,
                    "gamma_deg": 90.0,
                    "length_unit": "angstrom",
                },
                "point_group": "2/m",
                "provenance": {"status": "USER_MEASURED"},
            },
        ],
        "transformations": [],
        "orientations": [],
        "sources": [],
        "representation_links": [],
        "notes": (
            "No OR or correspondence is assumed. Supply an OR explicitly to "
            "cualni-two, or store one in orientations."
        ),
    }


def _project_summary(loaded: LoadedProject) -> str:
    project = loaded.project
    lines = [
        "=" * 88,
        "GENERIC CRYSTALLOGRAPHY PROJECT",
        "=" * 88,
        f"source        : {loaded.source or '-'}",
        f"project id    : {project.project_id}",
        f"title         : {project.title}",
        f"phases        : {len(project.phases)}",
        f"transformations: {len(project.transformations)}",
        f"orientations  : {len(project.orientations)}",
        f"representation links: {len(loaded.representation_links)}",
        "",
        "PHASES",
    ]
    for phase in project.phases:
        lattice = phase.lattice
        volume = float(np.sqrt(np.linalg.det(lattice.metric())))
        lines.extend(
            [
                f"  {phase.phase_id}",
                f"    {phase.label}",
                (
                    f"    cell: a={lattice.a:g}, b={lattice.b:g}, c={lattice.c:g}, "
                    f"alpha={lattice.alpha_deg:g}, beta={lattice.beta_deg:g}, "
                    f"gamma={lattice.gamma_deg:g} {lattice.length_unit}"
                ),
                f"    volume={volume:.12g} {lattice.length_unit}^3",
                f"    point group={phase.point_group_symbol}, order={len(phase.symmetry_operators)}",
                f"    representation={phase.cell_representation}",
            ]
        )
    if loaded.representation_links:
        lines.extend(["", "REPRESENTATION LINKS"])
        for audit in loaded.representation_audits():
            lines.append(
                f"  {audit.link_id}: {'PASS' if audit.passed else 'FAIL'}  "
                f"metric={audit.metric_relative_residual:.3e}  "
                f"volume={audit.volume_jacobian_residual:.3e}"
            )
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic crystal/project loader and validator for CuAlNi-CT."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("presets")
    sub.add_parser("point-groups")

    show = sub.add_parser("show")
    show.add_argument("source")
    show.add_argument("--json", action="store_true")

    validate = sub.add_parser("validate")
    validate.add_argument("source")
    validate.add_argument("--json", action="store_true")

    template = sub.add_parser("template")
    template.add_argument("--output", default="")

    create = sub.add_parser("create-pair")
    create.add_argument("output")
    create.add_argument("--reference-id", required=True)
    create.add_argument("--reference-cell", required=True)
    create.add_argument("--reference-point-group", required=True)
    create.add_argument("--reference-label", default="")
    create.add_argument("--reference-physical-phase", default="")
    create.add_argument("--reference-representation", default="user_cell")
    create.add_argument("--moving-id", required=True)
    create.add_argument("--moving-cell", required=True)
    create.add_argument("--moving-point-group", required=True)
    create.add_argument("--moving-label", default="")
    create.add_argument("--moving-physical-phase", default="")
    create.add_argument("--moving-representation", default="user_cell")
    create.add_argument("--unit", default="angstrom")
    create.add_argument(
        "--title", default="Arbitrary two-phase crystallography project"
    )

    links = sub.add_parser("representation-audit")
    links.add_argument("source")
    links.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "presets":
        for name in preset_names():
            loaded = load_project(f"preset:{name}")
            print(f"{name:34s}  {loaded.project.title}")
        return 0

    if args.command == "point-groups":
        print("symbol   family        order   setting")
        print("------   ------------  -----   --------------------------------")
        for row in point_group_table():
            print(
                f"{row['symbol']:<8s} {row['crystal_family']:<13s} "
                f"{row['order']:>5d}   {row['conventional_setting']}"
            )
        return 0

    if args.command in {"show", "validate", "representation-audit"}:
        loaded = load_project(args.source)
        validation = loaded.project.validate()
        audits = loaded.representation_audits()
        if args.command == "show":
            if args.json:
                payload = loaded.project.to_dict()
                payload["representation_link_audits"] = [
                    item.to_dict() for item in audits
                ]
                print(json.dumps(payload, indent=2))
            else:
                print(_project_summary(loaded))
            return 0
        if args.command == "representation-audit":
            payload = [item.to_dict() for item in audits]
            if args.json:
                print(json.dumps(payload, indent=2))
            elif not audits:
                print("No representation links declared.")
            else:
                for item in audits:
                    print(
                        f"{item.link_id}: {'PASS' if item.passed else 'FAIL'}  "
                        f"metric={item.metric_relative_residual:.3e}  "
                        f"volume={item.volume_jacobian_residual:.3e}  "
                        f"det(P)={item.determinant:.12g}"
                    )
            return 0 if all(item.passed for item in audits) else 1

        result = {
            "passed": validation.passed and all(item.passed for item in audits),
            "errors": [issue.message for issue in validation.errors],
            "warnings": [issue.message for issue in validation.warnings],
            "representation_links": [item.to_dict() for item in audits],
        }
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("PASS" if result["passed"] else "FAIL")
            for warning in result["warnings"]:
                print(f"WARNING: {warning}")
            for error in result["errors"]:
                print(f"ERROR: {error}")
        return 0 if result["passed"] else 1

    if args.command == "template":
        text = json.dumps(generic_template(), indent=2) + "\n"
        if args.output:
            path = Path(args.output)
            path.write_text(text, encoding="utf-8")
            print(path)
        else:
            print(text, end="")
        return 0

    if args.command == "create-pair":
        payload = {
            "schema_version": SCHEMA_VERSION,
            "project_id": Path(args.output).stem,
            "title": args.title,
            "phases": [
                _phase_payload(
                    args.reference_id,
                    args.reference_cell,
                    args.reference_point_group,
                    args.unit,
                    label=args.reference_label,
                    physical_phase=args.reference_physical_phase,
                    representation=args.reference_representation,
                ),
                _phase_payload(
                    args.moving_id,
                    args.moving_cell,
                    args.moving_point_group,
                    args.unit,
                    label=args.moving_label,
                    physical_phase=args.moving_physical_phase,
                    representation=args.moving_representation,
                ),
            ],
            "transformations": [],
            "orientations": [],
            "sources": [],
            "representation_links": [],
            "notes": "No OR or correspondence assumed by create-pair.",
        }
        # Validate before writing.
        project_from_dict(payload, source="create-pair-preflight")
        path = Path(args.output)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(path)
        return 0

    raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
