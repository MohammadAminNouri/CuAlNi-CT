import numpy as np
import sympy as sp
from cualni_cryst.cualni_models import DO3_TO_6M, DO3_TO_2H, james_hane_6m_example_lattices
from cualni_cryst.james_hane import cube_edge_6m_reference_U, cube_edge_6m_variants
from cualni_cryst.stretch import stretch_from_metrics, generate_stretch_variants, match_matrix_family
from cualni_cryst.symmetry import cubic_proper_rotations


def parallel(a,b):
    return sp.Matrix(a).cross(sp.Matrix(b)) == sp.zeros(3,1)


def test_2h_source_orientation_relationships():
    # One reference variant consistent with literature: [100]_A -> [010]_2H;
    # a parent {110} plane maps to the 2H basal plane (001)_M.
    assert parallel(DO3_TO_2H.map_direction_a_to_m([1,0,0]), [0,1,0])
    assert parallel(DO3_TO_2H.map_plane_a_to_m([0,1,-1]), [0,0,1])


def test_6m_correspondence_reproduces_james_hane_cube_edge_reference():
    A,M=james_hane_6m_example_lattices()
    U=stretch_from_metrics(A.metric(),M.metric(),DO3_TO_6M)
    Uref=cube_edge_6m_reference_U(A.a,M.a,M.b,M.c,M.beta_deg)
    assert np.allclose(U,Uref,atol=1e-12)


def test_6m_correspondence_reproduces_all_twelve_stretches():
    A,M=james_hane_6m_example_lattices()
    U=stretch_from_metrics(A.metric(),M.metric(),DO3_TO_6M)
    derived=generate_stretch_variants(U,[np.array(q,float) for q in cubic_proper_rotations()])
    ref=cube_edge_6m_variants(A.a,M.a,M.b,M.c,M.beta_deg)
    ok,R=match_matrix_family(derived,ref,tol=1e-10)
    assert len(derived)==12 and ok
