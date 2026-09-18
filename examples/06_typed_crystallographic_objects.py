"""Typed direct/reciprocal objects: metric and PTCLab-Cartesian views."""

from cualni_cryst.crystal_objects import (
    CrystalBasisRef,
    Direction,
    Plane,
    direction_plane_angle_deg,
)
from cualni_cryst.cualni_models import james_hane_6m_example_lattices
from cualni_cryst.representation import CartesianConvention


def main() -> None:
    _, martensite = james_hane_6m_example_lattices()
    basis = CrystalBasisRef(
        "martensite_6M",
        "reference_6M_unique_b",
        "6M",
    )

    direction = Direction((1, 0, 1), basis)
    plane = Plane((0, 1, 1), basis)

    print(direction.to_dict())
    print(plane.to_dict())
    print(f"|u| = {direction.length(martensite):.8f}")
    print(f"d(hkl) = {plane.spacing(martensite):.8f}")
    print(
        "direction-plane angle = "
        f"{direction_plane_angle_deg(direction, plane, martensite):.8f} deg"
    )

    for convention in CartesianConvention:
        print(
            convention.value,
            "u_cart=",
            direction.cartesian(martensite, convention),
            "n_cart=",
            plane.cartesian_normal(martensite, convention, normalize=True),
        )


if __name__ == "__main__":
    main()
