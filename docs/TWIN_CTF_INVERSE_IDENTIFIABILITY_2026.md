# CTF-driven crystallography: evidence levels and inverse identifiability

**Scope:** the existing feature-branch Streamlit Twin Family Explorer. No
Cayron, BJ, PTMC or cofactor solver equation is modified by this extension.

## What one already-indexed CTF determines

- CTF declared parent/product conventional cells, phase IDs and space-group
  number. These can prefill the crystal editor after user selection.
- Indexed pixel Euler angles and sample-plane positions, vendor confidence and
  phase assignments, which permit *conditional* symmetry-aware orientation,
  IPF-color and misorientation/region analyses.
- Martensite orientation boundaries and candidate variant operators. These are
  measurements of orientation relationships between labeled domains, not
  automatically type-I, type-II or Compound twin certification.

The imported phase and space-group information is **AZtec's assumed indexing
model**. It cannot alone validate 2H/18R stacking, sample orientation or atomic
symmetry. Before comparison with theory, verify the conventional setting.

## Correspondence is NOT orientation relationship

Use Cayron (2019), *Acta Crystallographica A* 75, 411–437,
DOI 10.1107/S205327331900038X, and Cayron (2026), *Acta Materialia* 316,
122399, DOI 10.1016/j.actamat.2026.122399.

For direct crystal coordinates, C maps a parent direction to the
corresponding martensite direction. The actual orientation relationship
between the two crystal lattices, the crystal lattice distortion and the
correspondence are DISTINCT objects (Cayron 2019). A matched pair of cell
metrics does not specify the atom mapping or a unique exact rational C.

The upload route *never* supplies a guessed matrix C. If the existing NiTi
editable example remains in the editor, the UI explicitly warns and blocks
the new CTF input route from calculating before a deliberate scientific
confirmation. The code preserves all existing editor input values.

## Blind OR inverse problem: an observable class, not a unique R

Given only child-child orientations, the parent symmetry group G_A enters
the observed operator family as:

    H(R) = R.T @ G_A @ R.

A rotation N from the normalizer of G_A gives the same observable
subgroup for N @ R. Therefore even exact, noise-free child-only
boundaries cannot in general uniquely determine the physical R.
This also cannot identify the lattice correspondence C.

Our existing independently tested blind solver is reused as an
OPTIONAL exploratory operation. It reports
`identified_observable_class`, `ambiguous`, or
`insufficient_evidence`, never a measured unique physical OR. Strong
high-angle neighbor jumps and pseudo-symmetry make its practical
identifiability still weaker. Parent orientations reconstructed from
a **separately validated OR** are likewise hypotheses until sufficient
domain evidence supports them.

## IPF

Native Bunge ZXZ EBSD rotations are crystal-to-sample Cartesian g.
For sample direction s, the crystal direction is g.T @ s.
The external orix TSL IPFColorKey (Laue symmetry sector) assigns
symmetry-equivalent crystal directions a well-defined standard color.
Users select sample X, Y or Z; phase selection is explicit.
The key itself is shown with the map and its convention is exported.

A valid standard IPF color map is NOT independent proof that the original
Kikuchi patterns were correctly indexed or that the specimen axes are
registered to the assumed sample frame.

## What Cayron/CT, BJ and PTMC can calculate **after C is supplied**

With declared/validated metrics M_A, M_M, groups and an independently
justified C, the existing theoretical solvers calculate:

- M/M transformation twins using correspondence variants and parent
  symmetry (Type I / Type II / compound where the equations admit one);
- A/M invariant-plane compatibility by Cayron's CMC degeneracy and
  SMC-derived shear vector;
- A/M/M shear/shear compatibility, separately compared with the
  Ball–James rank-one and PTMC laminate frameworks;
- theoretical K1, eta1, shear, habit-plane candidates and residuals.

These outputs are **predictions for the input transformation model**.
They are not automatically observations in a 2D indexed EBSD map.

## Why measured 3D planes are not supplied by a single section

A 2D EBSD boundary trace constrains a plane to a pencil of 3D normals.
One section alone does not determine the unique 3D normal, nor does an
image of a boundary establish a direct/reciprocal twinning element.
To measure K1 and shear experimentally requires independent specimen
geometry, a second nonparallel trace/serial section or suitable
diffraction/TEM evidence and variant cross-identification. A selected
CT-predicted K1 or eta1 is NOT a measured one.

## Scientific test contract

- Autofill preserves exact correspondence editor entries, works with
  cubic-to-2H and cubic-to-monoclinic 18R source cells, and fails closed
  for absent phase declarations.
- orix IPF uses its documented Laue key and the inverse of the native
  crystal-to-sample Bunge matrix; no ad-hoc Euler RGB is used.
- Blind OR explicitly reports the observable inverse-problem gauge,
  alternatives and insufficiency without silent physical identification.
- Existing CI and P0/P1 solver audit remain the regression gate.

Cu–Al–Ni 2H experimental structural constraint context:
X Chen et al. (2000), *Micron* 31, 17–25,
DOI 10.1016/S0968-4328(99)00060-8. The 2H basal plane and
[010] axis parent relationships constrain *directions*, but do
not uniquely determine all three independent direct
correspondence vectors from a CTF header.

Distinct phase comparisons should be made for beta3→2H and beta3→18R.

## Optional registered two-section interface plane reconstruction

The Step-01 EBSD workspace accepts TWO experimentally registered traces
of the same physical interface in a common sample Cartesian frame,
each paired with its actual sample-section normal. The code checks:
- both traces are orthogonal to their corresponding section normals;
- the sections are nonparallel and the traces independently intersect;
- a valid crystal-to-sample SO(3) EBSD grain orientation is supplied.

For two measured trace direction vectors t1 and t2, the unoriented
sample-frame plane normal is

    n_sample = +/- (t1 x t2) / |t1 x t2|,

and the (unrounded) reciprocal-index covector in the indexed product
conventional cell is

    h = B_M.T @ g_M.T @ n_sample,

where g_M is the indexed crystal-to-sample orientation and
B_M.T @ B_M = M_M. The result includes an approximate first-order
angular error amplification proportional to 1/sin(angle(t1,t2)),
explicitly NOT a full confidence interval.

One single-section CTF offers neither independent second trace nor
section registration. Thus this separate user-supplied method cannot
magically infer a 3D habit plane from one EBSD map. Nor does it
independently certify Type I/II, a twinning shear direction or
interfacial atomic structure; those require a valid crystallographic
crosswalk and additional physical evidence.
