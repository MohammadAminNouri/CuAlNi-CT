"""Examples for the Cayron-compatible orientation-topology layer."""

import numpy as np

from cualni_cryst.orientation import OrientationRenderer, OrientationService
from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.representation import CartesianConvention


def print_audit(service: OrientationService, state, title: str) -> None:
    topology = service.topology(state)
    audit = topology.audit
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print(
        "H_T full/proper =",
        audit.full_orientation_intersection_order,
        "/",
        audit.proper_orientation_intersection_order,
    )
    print(
        "N_T full/proper =",
        audit.full_orientation_variant_count,
        "/",
        audit.proper_orientation_variant_count,
    )
    print(
        "O_T full/proper =",
        audit.full_orientation_operator_count,
        "/",
        audit.proper_orientation_operator_count,
    )
    print(
        "H_C / N_C / O_C =",
        audit.correspondence_intersection_order,
        "/",
        audit.correspondence_variant_count,
        "/",
        audit.correspondence_operator_count,
    )
    print(
        "H_T == H_C =",
        audit.orientation_correspondence_intersections_equal,
    )
    print(
        "one-to-one C/T topology =",
        audit.one_to_one_correspondence_orientation_topology,
    )


def main() -> None:
    service = OrientationService(james_hane_6m_reference_project())
    renderer = OrientationRenderer()

    # Example 1: the current DO3 -> 6M correspondence-derived polar candidate.
    polar = service.polar_orientation("do3_to_6m_reference")
    print_audit(service, polar, "EXAMPLE 1 — DO3 -> 6M POLAR CANDIDATE")
    print(
        renderer.orientation(
            service.report(polar),
            variants=service.variants(polar),
            operators=service.operators(polar),
        )
    )

    # Example 2: a deliberately generic OR for the same phase pair.
    # This demonstrates why H_T must be calculated and must never be assumed
    # equal to H_C merely because that equality happens in the reference case.
    generic = service.state_from_axis_angle(
        "do3",
        "6m",
        np.array([1.0, 2.0, 3.0]),
        17.0,
        orientation_id="generic_demo_or",
    )
    print_audit(service, generic, "EXAMPLE 2 — GENERIC OR, SAME C")

    # Example 3: coordinate re-expression.  The matrix/axis-angle parameters
    # can change, but H_T, cosets, operators and physical mapping do not.
    ptclab = service.reexpress(
        polar,
        CartesianConvention.PTCLAB_A_X_C_XZ,
        CartesianConvention.PTCLAB_A_X_C_XZ,
    )
    print("\n" + "=" * 88)
    print("EXAMPLE 3 — SAME PHYSICAL OR, DIFFERENT CARTESIAN REPRESENTATION")
    print("=" * 88)
    print("symmetric-metric axis-angle:", service.report(polar).axis_angle)
    print("PTCLab axis-angle:          ", service.report(ptclab).axis_angle)
    print(
        "topology count in both:     ",
        len(service.variants(polar)),
        len(service.variants(ptclab)),
    )


if __name__ == "__main__":
    main()
