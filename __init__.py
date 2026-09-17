"""Cu-Al-Ni martensitic crystallography research toolkit.

Design principle: one crystallographic input set, several independent theory
branches (Cayron CT, PTMC, Ball-James/cofactor), then comparison to experiment.
"""
from .lattice import Lattice
from .correspondence import Correspondence
from .cualni_models import do3_to_6m_branch, do3_to_2h_branch

__all__=["Lattice","Correspondence","do3_to_6m_branch","do3_to_2h_branch"]
__version__="0.2.0"
