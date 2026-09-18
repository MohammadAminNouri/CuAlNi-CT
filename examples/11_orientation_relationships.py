"""Demonstrate the orientation-relationship engine."""

import numpy as np

from cualni_cryst.orientation import OrientationRenderer, OrientationService
from cualni_cryst.project_state import james_hane_6m_reference_project


def main() -> None:
    service = OrientationService(james_hane_6m_reference_project())
    renderer = OrientationRenderer()

    polar = service.polar_orientation("do3_to_6m_reference")
    print(
        renderer.orientation(
            service.report(polar),
            variants=service.variants(polar),
        )
    )

    print()
    parallel = service.state_from_parallelisms(
        "do3",
        "6m",
        "(0 1 0)",
        "(0 1 0)",
        "[1 0 0]",
        "[1 0 0]",
    )
    print(renderer.parallelisms(parallel))

    print()
    identity = service.state_from_matrix(
        "do3",
        "6m",
        np.eye(3),
        orientation_id="identity_test_or",
    )
    print(renderer.comparison(service.compare_orientations(identity, polar)))


if __name__ == "__main__":
    main()
