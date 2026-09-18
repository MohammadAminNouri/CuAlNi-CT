"""Compare the same DO3 -> 6M transformation in all Cartesian conventions."""

from cualni_cryst.cualni_models import (
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.representation import (
    CartesianConvention,
    RepresentationBridge,
)


def main() -> None:
    austenite, martensite = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    for parent_convention in CartesianConvention:
        for product_convention in CartesianConvention:
            bridge = RepresentationBridge(
                austenite,
                martensite,
                branch.correspondence,
                parent_convention,
                product_convention,
            )
            audit = bridge.audit()
            print(
                f"{parent_convention.value:22s} -> "
                f"{product_convention.value:22s} "
                f"lambda={bridge.principal_stretches()} "
                f"max parity residual={audit.maximum_residual:.3e}"
            )


if __name__ == "__main__":
    main()
