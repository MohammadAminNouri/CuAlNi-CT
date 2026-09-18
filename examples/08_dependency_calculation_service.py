"""Demonstrate dependency-aware calculations and selective invalidation."""

from dataclasses import replace
from pprint import pprint

from cualni_cryst.calculation_service import CalculationKind, CalculationService
from cualni_cryst.lattice import Lattice
from cualni_cryst.project_state import james_hane_6m_reference_project

TRANSFORMATION_ID = "do3_to_6m_reference"


def main() -> None:
    project = james_hane_6m_reference_project()
    service = CalculationService(project)

    bundle = service.compute(
        CalculationKind.TRANSFORMATION_BUNDLE,
        TRANSFORMATION_ID,
    )

    print("REFERENCE TRANSFORMATION SUMMARY")
    pprint(
        {
            "variants": bundle.topology.n_variants,
            "operators": bundle.topology.n_operators,
            "lambda": bundle.metric.principal_stretches.tolist(),
            "lambda2_residual": bundle.metric.lambda2_residual,
            "ct_exact": bundle.am_compatibility.ct_exact,
            "contracts_passed": bundle.contracts.passed,
            "selected_frame_parity": (bundle.representation.maximum_parity_residual),
        }
    )

    print("\nCACHE AFTER FIRST BUNDLE")
    pprint(service.cache_stats)

    parent = project.phase("austenite_do3")
    product = project.phase("martensite_long_period")
    changed_lattice = Lattice.monoclinic_unique_b(
        product.lattice.a,
        product.lattice.b,
        product.lattice.c,
        96.0,
        label=product.lattice.label,
    )
    changed_project = replace(
        project,
        phases=(parent, replace(product, lattice=changed_lattice)),
    )

    print("\nIF ONLY MARTENSITE BETA CHANGES")
    pprint(service.impact(changed_project, TRANSFORMATION_ID).to_dict())

    service.set_project(changed_project)
    service.compute(CalculationKind.DISCRETE_TOPOLOGY, TRANSFORMATION_ID)

    print("\nCACHE AFTER REQUESTING TOPOLOGY ON CHANGED METRIC STATE")
    pprint(service.cache_stats)
    print(
        "The topology cache is reused because symmetry + correspondence did "
        "not change; metric-dependent nodes receive different fingerprints."
    )


if __name__ == "__main__":
    main()
