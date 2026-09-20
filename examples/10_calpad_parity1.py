"""Demonstrate the PTCLab-style Calpad parity-I workflow."""

from cualni_cryst.calpad import CalPadRenderer, CalPadService
from cualni_cryst.project_state import james_hane_6m_reference_project


def main() -> None:
    service = CalPadService(james_hane_6m_reference_project())
    renderer = CalPadRenderer()

    print(renderer.phase_cell(service.phase_cell("6m")))
    print()

    conversion = service.normal_conversion(
        "6m",
        "(1 0 1)",
        max_index=12,
    )
    print(renderer.normal_conversion(conversion, show_derivation=True))
    print()

    table = service.low_index_table(
        "6m",
        "[1 0 1]",
        candidate_kind="plane",
        max_index=2,
        limit=12,
    )
    print(renderer.low_index_table(table, show_derivation=True))


if __name__ == "__main__":
    main()
