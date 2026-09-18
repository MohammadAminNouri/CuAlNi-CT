"""First user-facing CuAlNi-CT crystallography-console workflow."""

from cualni_cryst.crystallography_console import (
    ConsoleRenderer,
    CrystallographyConsole,
)
from cualni_cryst.project_state import james_hane_6m_reference_project


def main() -> None:
    console = CrystallographyConsole(james_hane_6m_reference_project())
    renderer = ConsoleRenderer()

    print("AVAILABLE PHASES")
    for phase in console.phase_choices():
        print(
            f"  {phase['phase_id']:24s} "
            f"{phase['cell_representation']:6s} "
            f"{phase['label']}"
        )

    print()
    report = console.inspect("6m", "[1 0 1]")
    print(renderer.object_report(report, show_derivation=True))

    print()
    comparison = console.compare("6m", "[1 0 1]", "(0 1 1)")
    print(renderer.comparison_report(comparison, show_derivation=True))

    print()
    mapping = console.map(
        "do3_to_6m_reference",
        "do3",
        "[1 0 0]",
    )
    print(renderer.mapping_report(mapping, show_derivation=True))


if __name__ == "__main__":
    main()
