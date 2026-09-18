"""Build and inspect a reproducible DO3 -> 6M scientific project state."""

from pprint import pprint

from cualni_cryst.crystal_objects import Direction, Plane
from cualni_cryst.project_state import james_hane_6m_reference_project


def main() -> None:
    project = james_hane_6m_reference_project()
    report = project.validate()
    report.assert_passed()

    parent = project.phase("austenite_do3")
    direction_a = Direction((1, 0, 0), parent.basis)
    plane_a = Plane((0, 1, 0), parent.basis)

    direction_m = project.map_direction("do3_to_6m_reference", direction_a)
    plane_m = project.map_plane("do3_to_6m_reference", plane_a)

    print("PROJECT VALID:", report.passed)
    print("PHASES:", [phase.phase_id for phase in project.phases])
    print("COMPOSITION:")
    pprint(project.composition.to_dict() if project.composition else None)

    print("\nPARENT OBJECTS")
    pprint(direction_a.to_dict())
    pprint(plane_a.to_dict())

    print("\nMAPPED PRODUCT OBJECTS")
    pprint(direction_m.to_dict())
    pprint(plane_m.to_dict())

    print("\nREPRESENTATION PARITY")
    parity = project.bridge("do3_to_6m_reference").audit()
    pprint(
        {
            "maximum_residual": parity.maximum_residual,
            "parent_convention": project.transformation(
                "do3_to_6m_reference"
            ).parent_cartesian_convention.value,
            "product_convention": project.transformation(
                "do3_to_6m_reference"
            ).product_cartesian_convention.value,
        }
    )


if __name__ == "__main__":
    main()
