from __future__ import annotations

"""One-input / many-theory comparison harness.

The point of this module is methodological: CT, Ball-James/cofactor, and PTMC
must receive the same lattice and correspondence data.  Experimental data are
kept outside the predictors and are used only for validation residuals.
"""

from dataclasses import dataclass

import numpy as np

from .cofactor import evaluate_cofactor_conditions
from .correspondence import Correspondence
from .ct import analyze_austenite_martensite
from .group_theory import correspondence_groupoid
from .stretch import analyze_stretch, generate_stretch_variants
from .symmetry import cubic_proper_rotations
from .twinning_ct import twins_from_operator


@dataclass(frozen=True)
class TheoryInputs:
    M_a: np.ndarray
    M_m: np.ndarray
    correspondence: Correspondence
    parent_group: tuple
    product_group: tuple
    label: str = ""


@dataclass(frozen=True)
class TheoryComparison:
    inputs: TheoryInputs
    groupoid: object
    ct_am: object
    stretch: object
    stretch_variants: tuple[np.ndarray, ...]
    ct_twins_by_operator: tuple[tuple[object,...], ...]


def evaluate_same_inputs(inputs: TheoryInputs, tol: float=1e-8) -> TheoryComparison:
    groupoid = correspondence_groupoid(
        list(inputs.parent_group), list(inputs.product_group), inputs.correspondence
    )
    ct_am = analyze_austenite_martensite(inputs.M_a, inputs.M_m, inputs.correspondence, tol=tol)
    stretch = analyze_stretch(inputs.M_a, inputs.M_m, inputs.correspondence)
    proper = [np.array(q,float) for q in cubic_proper_rotations()]
    variants = tuple(generate_stretch_variants(stretch.U, proper))
    twins=[]
    for op in groupoid.operators:
        try:
            twins.append(tuple(twins_from_operator(op, inputs.M_a, inputs.M_m, inputs.correspondence)))
        except ValueError:
            twins.append(())
    return TheoryComparison(inputs, groupoid, ct_am, stretch, variants, tuple(twins))


def cofactor_for_twin(U:np.ndarray,a:np.ndarray,n:np.ndarray,tol:float=1e-7):
    """Small explicit adapter; intentionally does not infer a,n from CT."""
    return evaluate_cofactor_conditions(U,a,n,tol=tol)
