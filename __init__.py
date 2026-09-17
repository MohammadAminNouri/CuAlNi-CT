"""Cu-Al-Ni crystallography research toolkit.

The package intentionally separates:
- crystallographic metrics,
- correspondence/symmetry (Cayron CT),
- stretch/rank-one mechanics (Ball-James),
- PTMC-style laminate/habit-plane calculations,
- cofactor-condition checks,
- EBSD comparison utilities.

No Cu-Al-Ni correspondence matrix is silently hard-coded. A correspondence must
be supplied from a verified source or the user's experiment.
"""

from .lattice import Lattice
from .correspondence import Correspondence

__all__ = ["Lattice", "Correspondence"]
__version__ = "0.1.0"
