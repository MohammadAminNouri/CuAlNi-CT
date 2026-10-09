"""One consistent, non-diagnostic language and colour vocabulary for the workbench.

Colours are cues alongside written labels, never an alternative to them.
Bhattacharya's conventional symbols are retained in technical detail views.
"""

# Muted, deliberately limited palette. Gray: structure; blue: selected couple;
# teal: verified habit; amber: unverified classification.
COLOR_FAMILY = "#647583"
COLOR_SELECTED = "#5C85AD"
COLOR_HABIT = "#458C81"
COLOR_UNRESOLVED = "#B68C4F"
COLOR_EDGE = "#788491"

QUANTITY_HELP = {
    "s": "Twin shear magnitude: how much one martensite variant shears relative to another.",
    "K₁": "Twin plane (hkl), in the product crystal's reciprocal coordinates.",
    "η₁": "Shear direction [uvw], in the product crystal's direct coordinates.",
    "a": "Shape/shear vector in parent Cartesian coordinates for the rank-one twin relation.",
    "n̂": "Unit normal to the M/M twin interface in parent Cartesian coordinates.",
    "λ": "Fraction of the other martensite variant inside this twinned laminate (0 to 1).",
    "b": "Austenite–martensite shape-strain vector. Its length and direction both matter.",
    "m": "Unit normal of the austenite–martensite habit plane; (hkl) requires a frame conversion.",
}


def twin_name(classification: str) -> str:
    """Do not invent a crystallographic Type I/II when cross-lock is unresolved."""
    if classification == "Type I":
        return "Type I twin · mirror-related"
    if classification == "Type II":
        return "Type II twin · twofold-related"
    if classification == "Compound":
        return "Compound twin · two valid descriptions"
    if "unresolved" in classification.lower():
        return "Twin geometry found · type not verified"
    return f"Twin geometry · {classification}"


def habit_status(habit_solutions: int, continuum: bool = False) -> str:
    if continuum:
        return "Continuous compatibility — no single habit plane"
    if habit_solutions:
        return "Exact habit plane available"
    return "No exact habit plane for this twin"


def representative_habit_solutions(solutions):
    """Display one exact +/− representative, preserve every remaining solution.

    If both λ and its complement are available, prefer λ <= 1/2 for a compact
    book-like view. This never changes a calculated normal, vector or fraction.
    """
    if not solutions:
        return (), ()
    preferred = [s for s in solutions if s.other_variant_volume_fraction <= .5 + 1e-8]
    pool = preferred if preferred else list(solutions)
    ordered = sorted(pool, key=lambda s: (-s.habit_branch, s.other_variant_volume_fraction))
    by_branch = {}
    for sol in ordered:
        if sol.habit_branch not in by_branch:
            by_branch[sol.habit_branch] = sol
    shown = tuple(by_branch.values())
    selected_ids = {id(sol) for sol in shown}
    hidden = tuple(sol for sol in solutions if id(sol) not in selected_ids)
    return shown, hidden
