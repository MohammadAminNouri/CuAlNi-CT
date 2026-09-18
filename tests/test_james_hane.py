import numpy as np

from cualni_cryst.james_hane import m18r_to_6m, six_m_exact_compatibility_angles


def test_m18r_to_6m_returns_physical_cell():
    c,beta=m18r_to_6m(4.4,38.0,89.0)
    assert c>0 and 0<beta<180


def test_exact_compatibility_angles_are_supplementary():
    a1,a2=six_m_exact_compatibility_angles(5.836,4.430,12.79)
    assert np.isclose(a1+a2,180.0)
