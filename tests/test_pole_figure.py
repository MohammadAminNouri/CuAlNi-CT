
import numpy as np

from cualni_cryst.pole_figure import pole_family, stereographic_project
from cualni_cryst.project_state import james_hane_6m_reference_project


def test_cubic_100_plane_family_has_three_projective_poles():
    project = james_hane_6m_reference_project()
    transformation = project.transformation("do3_to_6m_reference")
    family = pole_family(
        project,
        transformation.parent_phase_id,
        kind="plane",
        coefficients=(1, 0, 0),
        reference_phase_id=transformation.parent_phase_id,
        projective=True,
    )
    assert len(family.points) == 3
    assert all(point.x * point.x + point.y * point.y <= 1.0 + 1e-12 for point in family.points)


def test_stereographic_project_identifies_projective_opposites():
    a = stereographic_project(np.array([1.0, 2.0, 3.0]), projective=True)
    b = stereographic_project(np.array([-1.0, -2.0, -3.0]), projective=True)
    assert np.allclose(a, b, atol=1e-14)
