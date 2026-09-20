from __future__ import annotations

"""Exact orientation↔correspondence operator cross-lock.

The central rule is intentionally simple:

* an operator is an exact double coset;
* a displayed disorientation is only a representative summary;
* orientation topology is compared with correspondence topology only when the
  OrientationState is explicitly bound to a TransformationState.

This module does not infer a correspondence from phase names.
"""

import argparse
import json
from dataclasses import dataclass

import numpy as np
import sympy as sp

from .cualni_models import TransformationBranch
from .group_theory import correspondence_groupoid
from .orientation import OrientationService
from .project_state import (
    OrientationState,
    ProjectState,
    james_hane_6m_reference_project,
)
from .symmetry import matrix_key
from .twin_compare import build_twin_atlas
from .twinning_ct import CTTwin, twins_from_operator

ExactMatrixKey = tuple[sp.Expr, ...]


def _exact_matrix(matrix: object) -> sp.Matrix:
    arr = np.asarray(matrix, dtype=float).reshape(3, 3)
    rows: list[list[sp.Expr]] = []
    for row in arr:
        exact_row: list[sp.Expr] = []
        for value in row:
            nearest = round(float(value))
            if abs(float(value) - nearest) <= 1.0e-12:
                exact_row.append(sp.Integer(nearest))
            else:
                exact_row.append(sp.nsimplify(float(value)))
        rows.append(exact_row)
    return sp.Matrix(rows)


def _key(matrix: object) -> ExactMatrixKey:
    return matrix_key(_exact_matrix(matrix))


def _key_strings(key: ExactMatrixKey) -> tuple[str, ...]:
    return tuple(str(value) for value in key)


def _vector(values: np.ndarray) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float).reshape(3).copy()
    arr[np.abs(arr) < 1.0e-14] = 0.0
    return tuple(float(value) for value in arr)


@dataclass(frozen=True)
class TwinElement:
    """One CT Type-I or Type-II result with its exact parent generator."""

    kind: str
    parent_symmetry: tuple[str, ...]
    plane_m: tuple[float, float, float]
    direction_m: tuple[float, float, float]
    shear: float
    rational_element: str

    @classmethod
    def from_ct(cls, twin: CTTwin) -> TwinElement:
        return cls(
            kind=twin.kind,
            parent_symmetry=_key_strings(matrix_key(twin.parent_symmetry)),
            plane_m=_vector(twin.plane_m),
            direction_m=_vector(twin.direction_m),
            shear=float(twin.shear),
            rational_element=twin.rational_element,
        )

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind,
            "parent_symmetry": list(self.parent_symmetry),
            "shear": self.shear,
            "rational_element": self.rational_element,
        }
        if self.kind == "I":
            result["K1"] = list(self.plane_m)
            result["eta1"] = list(self.direction_m)
        elif self.kind == "II":
            result["K2"] = list(self.plane_m)
            result["eta2"] = list(self.direction_m)
        else:
            result["plane_m"] = list(self.plane_m)
            result["direction_m"] = list(self.direction_m)
        return result


@dataclass(frozen=True)
class IndependentTwinCheck:
    """Independent Mallard/Ball-James cross-check summary for one C operator."""

    status: str
    relation_count: int
    max_plane_or_direction_angle_deg: float | None
    max_relative_shear_residual: float | None
    max_rank_one_residual: float | None
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "relation_count": self.relation_count,
            "max_plane_or_direction_angle_deg": self.max_plane_or_direction_angle_deg,
            "max_relative_shear_residual": self.max_relative_shear_residual,
            "max_rank_one_residual": self.max_rank_one_residual,
            "note": self.note,
        }


@dataclass(frozen=True)
class OperatorCrosslockRow:
    """One exact operator identity plus optional CT/BJ physics."""

    orientation_operator: int
    correspondence_operator: int | None
    inverse_orientation_operator: int
    inverse_correspondence_operator: int | None
    cayron_class: str
    size: int
    exact_parent_members: tuple[tuple[str, ...], ...]
    representative_disorientation_deg: float
    contains_parent_reflection: bool
    contains_parent_180_rotation: bool
    ct_twins: tuple[TwinElement, ...]
    independent_check: IndependentTwinCheck

    @property
    def twin_modes(self) -> str:
        modes = sorted({twin.kind for twin in self.ct_twins})
        return "+".join(modes) if modes else "--"

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_operator": self.orientation_operator,
            "correspondence_operator": self.correspondence_operator,
            "inverse_orientation_operator": self.inverse_orientation_operator,
            "inverse_correspondence_operator": self.inverse_correspondence_operator,
            "cayron_class": self.cayron_class,
            "size": self.size,
            "exact_parent_members": [
                list(member) for member in self.exact_parent_members
            ],
            "representative_disorientation_deg": self.representative_disorientation_deg,
            "representative_disorientation_is_operator_definition": False,
            "contains_parent_reflection": self.contains_parent_reflection,
            "contains_parent_180_rotation": self.contains_parent_180_rotation,
            "ct_twin_modes": self.twin_modes,
            "ct_twins": [twin.to_dict() for twin in self.ct_twins],
            "independent_mallard_ball_james": self.independent_check.to_dict(),
        }


@dataclass(frozen=True)
class OperatorCrosslockReport:
    orientation_id: str
    transformation_id: str
    reference_phase_id: str
    moving_phase_id: str
    h_t_exact: tuple[tuple[str, ...], ...]
    h_c_exact: tuple[tuple[str, ...], ...]
    exact_subgroups_equal: bool
    one_to_one_operator_crosslock: bool
    orientation_operator_count: int
    correspondence_operator_count: int
    rows: tuple[OperatorCrosslockRow, ...]
    note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "orientation_id": self.orientation_id,
            "transformation_id": self.transformation_id,
            "reference_phase_id": self.reference_phase_id,
            "moving_phase_id": self.moving_phase_id,
            "H_T_exact": [list(item) for item in self.h_t_exact],
            "H_C_exact": [list(item) for item in self.h_c_exact],
            "exact_subgroups_equal": self.exact_subgroups_equal,
            "one_to_one_operator_crosslock": self.one_to_one_operator_crosslock,
            "orientation_operator_count": self.orientation_operator_count,
            "correspondence_operator_count": self.correspondence_operator_count,
            "rows": [row.to_dict() for row in self.rows],
            "note": self.note,
        }


class OperatorCrosslockService:
    """Cross-lock a physical OR to one explicitly selected transformation hypothesis."""

    def __init__(self, project: ProjectState):
        self.project = project
        self.orientations = OrientationService(project)

    def _bound_transformation(self, state: OrientationState):
        if not state.transformation_id:
            raise ValueError(
                "OrientationState is not bound to a transformation. "
                "Bind it explicitly; correspondence is never inferred from phase names."
            )
        transformation = self.project.transformation(state.transformation_id)
        if transformation.parent_phase_id != state.reference_phase_id:
            raise ValueError(
                "Bound transformation parent phase does not match the OR reference phase."
            )
        if transformation.product_phase_id != state.moving_phase_id:
            raise ValueError(
                "Bound transformation product phase does not match the OR moving phase."
            )
        return transformation

    def _branch(self, state: OrientationState) -> TransformationBranch:
        transformation = self._bound_transformation(state)
        parent = self.project.phase(transformation.parent_phase_id)
        product = self.project.phase(transformation.product_phase_id)
        return TransformationBranch(
            name=transformation.transformation_id,
            parent_point_group=tuple(
                _exact_matrix(g) for g in parent.symmetry_matrices()
            ),
            product_point_group=tuple(
                _exact_matrix(g) for g in product.symmetry_matrices()
            ),
            correspondence=transformation.correspondence,
            notes="Built from the explicitly bound ProjectState transformation.",
        )

    def build(self, state: OrientationState) -> OperatorCrosslockReport:
        transformation = self._bound_transformation(state)
        reference = self.project.phase(state.reference_phase_id)
        moving = self.project.phase(state.moving_phase_id)

        topology = self.orientations.topology(state)
        branch = self._branch(state)
        c_groupoid = correspondence_groupoid(
            list(branch.parent_point_group),
            list(branch.product_point_group),
            branch.correspondence,
        )

        h_t_keys = tuple(
            sorted(
                _key_strings(_key(item.reference_crystallographic_matrix))
                for item in topology.full_intersection
            )
        )
        h_c_keys = tuple(
            sorted(_key_strings(matrix_key(item)) for item in c_groupoid.subgroup)
        )
        h_equal = h_t_keys == h_c_keys

        parent_by_index = {
            index: _exact_matrix(matrix)
            for index, matrix in enumerate(reference.symmetry_matrices())
        }

        orientation_signatures: list[frozenset[ExactMatrixKey]] = []
        for operator in topology.operators:
            orientation_signatures.append(
                frozenset(
                    matrix_key(parent_by_index[index])
                    for index in operator.parent_double_coset_symmetry_indices
                )
            )
        c_signatures = [
            frozenset(matrix_key(matrix) for matrix in operator)
            for operator in c_groupoid.operators
        ]

        one_to_one = h_equal and len(orientation_signatures) == len(c_signatures)
        c_match: list[int | None] = []
        if one_to_one:
            used: set[int] = set()
            for signature in orientation_signatures:
                matches = [
                    i for i, current in enumerate(c_signatures) if current == signature
                ]
                if len(matches) != 1 or matches[0] in used:
                    one_to_one = False
                    break
                used.add(matches[0])
                c_match.append(matches[0])
            if len(used) != len(c_signatures):
                one_to_one = False

        if not one_to_one:
            c_match = [None] * len(orientation_signatures)

        atlas = None
        atlas_error = ""
        try:
            atlas = build_twin_atlas(
                branch,
                reference.lattice.metric(),
                moving.lattice.metric(),
                tol=max(float(self.project.numerical_policy.algebraic), 1.0e-10),
            )
        except (AssertionError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            atlas_error = str(exc)

        rows: list[OperatorCrosslockRow] = []
        for i, operator in enumerate(topology.operators):
            c_index = c_match[i]
            twins: tuple[TwinElement, ...] = ()
            check = IndependentTwinCheck(
                status="not_applicable",
                relation_count=0,
                max_plane_or_direction_angle_deg=None,
                max_relative_shear_residual=None,
                max_rank_one_residual=None,
                note="No exact C-operator match; no C-derived twin claim is made.",
            )
            inverse_c: int | None = None

            if c_index is not None:
                c_operator = c_groupoid.operators[c_index]
                twins = tuple(
                    TwinElement.from_ct(twin)
                    for twin in twins_from_operator(
                        c_operator,
                        reference.lattice.metric(),
                        moving.lattice.metric(),
                        transformation.correspondence,
                    )
                )
                inverse_c = c_groupoid.inverse_operators[c_index] + 1

                if atlas is None:
                    check = IndependentTwinCheck(
                        status="unavailable",
                        relation_count=0,
                        max_plane_or_direction_angle_deg=None,
                        max_relative_shear_residual=None,
                        max_rank_one_residual=None,
                        note=(
                            "Independent Mallard/Ball-James adapter was not applicable "
                            f"to this transformation: {atlas_error}"
                        ),
                    )
                else:
                    summary = atlas.operators[c_index]
                    if not summary.relations:
                        check = IndependentTwinCheck(
                            status="not_applicable",
                            relation_count=0,
                            max_plane_or_direction_angle_deg=None,
                            max_relative_shear_residual=None,
                            max_rank_one_residual=None,
                            note=summary.note,
                        )
                    else:
                        max_angle = max(
                            relation.max_geometry_angle_deg
                            for relation in summary.relations
                        )
                        max_shear = max(
                            relation.max_relative_shear_residual
                            for relation in summary.relations
                        )
                        max_rank = max(
                            relation.max_rank_one_residual
                            for relation in summary.relations
                        )
                        status = (
                            "verified"
                            if max_angle
                            <= self.project.numerical_policy.projective_angle_deg
                            and max_shear <= self.project.numerical_policy.algebraic
                            and max_rank <= self.project.numerical_policy.rank_one
                            else "residuals_reported"
                        )
                        check = IndependentTwinCheck(
                            status=status,
                            relation_count=len(summary.relations),
                            max_plane_or_direction_angle_deg=max_angle,
                            max_relative_shear_residual=max_shear,
                            max_rank_one_residual=max_rank,
                            note=(
                                "CT and Mallard/Ball-James are computed independently. "
                                "Angles compare physical plane/direction geometry, not raw "
                                "crystallographic coordinate triples."
                            ),
                        )

            members = tuple(
                sorted(_key_strings(key) for key in orientation_signatures[i])
            )
            rows.append(
                OperatorCrosslockRow(
                    orientation_operator=operator.index,
                    correspondence_operator=(
                        c_index + 1 if c_index is not None else None
                    ),
                    inverse_orientation_operator=operator.inverse_operator_index,
                    inverse_correspondence_operator=inverse_c,
                    cayron_class=operator.cayron_class,
                    size=operator.size,
                    exact_parent_members=members,
                    representative_disorientation_deg=(
                        operator.minimum_crystallographic_disorientation_deg
                    ),
                    contains_parent_reflection=operator.contains_parent_reflection,
                    contains_parent_180_rotation=operator.contains_parent_180_rotation,
                    ct_twins=twins,
                    independent_check=check,
                )
            )

        note = (
            "Operator identity is the exact parent-symmetry double-coset membership. "
            "Representative disorientation is display metadata only. "
            "C/T operator equality is asserted only when exact H_T and H_C memberships "
            "and every full double-coset member set match."
        )
        return OperatorCrosslockReport(
            orientation_id=state.orientation_id,
            transformation_id=transformation.transformation_id,
            reference_phase_id=state.reference_phase_id,
            moving_phase_id=state.moving_phase_id,
            h_t_exact=h_t_keys,
            h_c_exact=h_c_keys,
            exact_subgroups_equal=h_equal,
            one_to_one_operator_crosslock=one_to_one,
            orientation_operator_count=len(topology.operators),
            correspondence_operator_count=len(c_groupoid.operators),
            rows=tuple(rows),
            note=note,
        )


def _format_vector(values: tuple[float, float, float]) -> str:
    return "[" + " ".join(f"{value:.8g}" for value in values) + "]"


def render_report(
    report: OperatorCrosslockReport,
    *,
    details: bool = False,
    members: bool = False,
) -> str:
    lines = [
        "=" * 92,
        "ORIENTATION ↔ CORRESPONDENCE OPERATOR CROSS-LOCK",
        "=" * 92,
        f"orientation       : {report.orientation_id}",
        f"transformation    : {report.transformation_id}",
        f"phases            : {report.reference_phase_id} <- {report.moving_phase_id}",
        f"exact H_T == H_C  : {'YES' if report.exact_subgroups_equal else 'NO'}",
        (
            "one-to-one O_T↔O_C: "
            f"{'YES' if report.one_to_one_operator_crosslock else 'NO'}"
        ),
        "",
        "OPERATOR TABLE",
        "  T   C   inv   class        size   CT twins   BJ/Mallard   repr. disorientation",
        "  --  --  ----  -----------  -----  ---------  ------------  --------------------",
    ]

    for row in report.rows:
        c_label = (
            str(row.correspondence_operator) if row.correspondence_operator else "--"
        )
        check = row.independent_check.status
        lines.append(
            f"  {row.orientation_operator:>2d}  {c_label:>2s}  "
            f"{row.inverse_orientation_operator:>4d}  "
            f"{row.cayron_class:<11s}  {row.size:>5d}  "
            f"{row.twin_modes:<9s}  {check:<12s}  "
            f"{row.representative_disorientation_deg:>12.6g} deg"
        )

    lines.extend(
        [
            "",
            "NOTE",
            "  Operator = exact double coset. The displayed disorientation is only",
            "  a representative summary and is never used to identify the operator.",
        ]
    )

    if details:
        lines.extend(["", "CT TWIN DETAILS"])
        for row in report.rows:
            if not row.ct_twins:
                continue
            lines.append(
                f"  O{row.orientation_operator} / C{row.correspondence_operator}:"
            )
            for twin in row.ct_twins:
                if twin.kind == "I":
                    lines.append(
                        f"    Type I   K1={_format_vector(twin.plane_m)}  "
                        f"eta1={_format_vector(twin.direction_m)}  s={twin.shear:.10g}"
                    )
                else:
                    lines.append(
                        f"    Type II  eta2={_format_vector(twin.direction_m)}  "
                        f"K2={_format_vector(twin.plane_m)}  s={twin.shear:.10g}"
                    )
            check = row.independent_check
            if check.relation_count:
                lines.append(
                    "    independent BJ/Mallard: "
                    f"max angle={check.max_plane_or_direction_angle_deg:.3e} deg, "
                    f"max rel shear={check.max_relative_shear_residual:.3e}, "
                    f"max rank-one={check.max_rank_one_residual:.3e}"
                )

    if members:
        lines.extend(["", "EXACT SUBGROUP MEMBERS"])
        lines.append("  H_T:")
        lines.extend(f"    {member}" for member in report.h_t_exact)
        lines.append("  H_C:")
        lines.extend(f"    {member}" for member in report.h_c_exact)
        lines.extend(["", "EXACT DOUBLE-COSET MEMBERS"])
        for row in report.rows:
            lines.append(f"  O_T{row.orientation_operator}:")
            lines.extend(f"    {member}" for member in row.exact_parent_members)

    return "\n".join(lines)


def _parse_matrix(text: str) -> np.ndarray:
    rows = [row.strip() for row in text.split(";") if row.strip()]
    if len(rows) != 3:
        raise ValueError("Matrix input must contain 3 semicolon-separated rows.")
    matrix = np.array([[float(value) for value in row.split()] for row in rows])
    if matrix.shape != (3, 3):
        raise ValueError("Matrix input must contain exactly 9 numbers.")
    return matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cualni-crosslock",
        description=(
            "Cross-lock a physical OR against one explicit correspondence hypothesis. "
            "No correspondence is inferred from phase names."
        ),
    )
    parser.add_argument(
        "--transformation",
        default="do3_to_6m_reference",
        help="Explicit transformation/correspondence id.",
    )
    parser.add_argument(
        "--matrix",
        help=(
            "Optional OR matrix R_reference<-moving, e.g. "
            "'1 0 0; 0 1 0; 0 0 1'. If omitted, use the polar candidate."
        ),
    )
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--members", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project = james_hane_6m_reference_project()
    orientations = OrientationService(project)
    transformation = project.transformation(args.transformation)

    if args.matrix:
        state = orientations.state_from_matrix(
            transformation.parent_phase_id,
            transformation.product_phase_id,
            _parse_matrix(args.matrix),
            orientation_id="user_bound_or",
            label="User OR explicitly bound to selected transformation",
            reference_convention=transformation.parent_cartesian_convention,
            moving_convention=transformation.product_cartesian_convention,
            transformation_id=transformation.transformation_id,
            repair=args.repair,
        )
    else:
        state = orientations.polar_orientation(transformation.transformation_id)

    report = OperatorCrosslockService(project).build(state)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_report(report, details=args.details, members=args.members))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
