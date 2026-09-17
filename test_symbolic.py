import sympy as sp
from cualni_cryst.symbolic import do3_to_6m_pulled_metric, do3_to_2h_pulled_metric


def test_symbolic_6m_pulled_metric_structure():
    (a0,a,b,c,beta),G,D=do3_to_6m_pulled_metric()
    assert sp.simplify(G[0,0]-b**2)==0
    assert sp.simplify(G[1,2]-(a**2-c**2/9))==0
    assert sp.simplify(G[1,1]-(a**2+2*a*c*sp.cos(beta)/3+c**2/9))==0
    assert sp.simplify(G[2,2]-(a**2-2*a*c*sp.cos(beta)/3+c**2/9))==0


def test_symbolic_2h_pulled_metric_structure():
    (a0,a,b,c),G,D=do3_to_2h_pulled_metric()
    assert G == sp.Matrix([[b**2,0,0],[0,a**2+c**2,a**2-c**2],[0,a**2-c**2,a**2+c**2]])
