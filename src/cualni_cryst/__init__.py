"""Cu-Al-Ni martensitic crystallography research toolkit.

Design principle: one crystallographic input set, several independent theory
branches (Cayron CT, PTMC, Ball-James/cofactor), then comparison to experiment.
"""
from .correspondence import Correspondence
from .cualni_models import do3_to_2h_branch, do3_to_6m_branch
from .lattice import Lattice

__all__=["Correspondence", "Lattice", "do3_to_2h_branch", "do3_to_6m_branch"]
__version__="0.2.0"
