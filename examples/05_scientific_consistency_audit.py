"""Run the cross-theory mathematical consistency audit."""

from pprint import pprint

from cualni_cryst.compatibility_atlas import james_hane_benchmark_projection
from cualni_cryst.cualni_models import (
    do3_to_6m_branch,
    james_hane_6m_example_lattices,
)
from cualni_cryst.lattice import Lattice
from cualni_cryst.scientific_contracts import audit_core_theory_consistency


def main() -> None:
    A, M = james_hane_6m_example_lattices()
    branch = do3_to_6m_branch()

    print("SOURCE-ROUNDED JAMES-HANE BENCHMARK")
    pprint(audit_core_theory_consistency(A, M, branch.correspondence).to_dict())

    projection = james_hane_benchmark_projection()
    M_exact = Lattice.monoclinic_unique_b(
        M.a,
        M.b,
        M.c,
        projection.projected_beta_deg,
        label="HYPOTHETICAL_TEST exact-compatible beta projection",
    )

    print("\nHYPOTHETICAL EXACT-COMPATIBLE CONTROL")
    pprint(
        audit_core_theory_consistency(
            A,
            M_exact,
            branch.correspondence,
        ).to_dict()
    )


if __name__ == "__main__":
    main()
