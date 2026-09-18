"""Quick PTCLab-style two-phase workbench example."""

from cualni_cryst.project_state import james_hane_6m_reference_project
from cualni_cryst.two_phase import TwoPhaseRenderer, TwoPhaseWorkbench


def main() -> None:
    workbench = TwoPhaseWorkbench(james_hane_6m_reference_project())
    renderer = TwoPhaseRenderer()
    state = workbench.resolve_orientation("polar:do3_to_6m_reference")

    print(renderer.summary(workbench.summary(state)))
    print()
    print(
        renderer.pair(
            workbench.pair(
                state,
                "[1 1 0]",
                "(1 0 1)",
                max_index=8,
            )
        )
    )
    print()
    print(
        renderer.variants(
            workbench.variant_angles(
                state,
                "[1 0 0]",
                "[1 0 0]",
                max_index=6,
            )
        )
    )


if __name__ == "__main__":
    main()
