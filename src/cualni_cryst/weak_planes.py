from __future__ import annotations

"""Status and scaffolding for Cayron axial weak-plane analysis.

Cayron 2022 gives the concept and says GenOVa enumerates rational directions
using distance/angle tolerances, but its detailed Ref. 74 is listed as "in
preparation" in that paper.  Therefore this repository deliberately does NOT
present a guessed implementation as the published algorithm.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from .lattice import metric_dot, metric_norm


@dataclass(frozen=True)
class RationalDirectionCandidate:
    uvw: tuple[int,int,int]
    length: float
    angle_to_axis_deg: float


def enumerate_rational_directions(M:np.ndarray,axis:np.ndarray,max_index:int=4)->list[RationalDirectionCandidate]:
    """Transparent helper for future weak-plane screening; NOT Cayron/GenOVa itself."""
    axis=np.asarray(axis,float); axis=axis/metric_norm(axis,M); out=[]; seen=set()
    for v in itertools.product(range(-max_index,max_index+1),repeat=3):
        if v==(0,0,0): continue
        # canonical primitive integer direction and sign
        g=np.gcd.reduce(np.abs(np.array(v,dtype=int)))
        w=tuple(int(x//g) for x in v)
        nz=next((x for x in w if x),1)
        if nz<0: w=tuple(-x for x in w)
        if w in seen: continue
        seen.add(w); x=np.array(w,float); L=metric_norm(x,M)
        c=np.clip(metric_dot(x,axis,M)/L,-1,1)
        out.append(RationalDirectionCandidate(w,L,float(np.degrees(np.arccos(c)))))
    return sorted(out,key=lambda z:(sum(abs(i) for i in z.uvw),z.uvw))


def exact_genova_weak_plane_search(*args,**kwargs):
    raise NotImplementedError(
        "The detailed weak-plane algorithm cited by Cayron 2022 as Ref. 74 was 'in preparation'. "
        "This package will not fabricate that unpublished algorithm. Use enumerate_rational_directions "
        "only as an explicitly project-defined screening helper, or implement a later published source."
    )
