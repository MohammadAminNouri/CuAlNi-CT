# Exact correspondence input — Cayron (2019)

Scientific basis: Cyril Cayron (2019), *The transformation matrices
(distortion, orientation, correspondence), their continuous forms and their variants*,
Acta Crystallographica A 75, 411–437; DOI 10.1107/S205327331900038X.
Consult §§2.3 and 3 (especially Eqs. 5–6, 9, 11, 13). The 2019 published
corrigendum DOI 10.1107/S2053273319009276 corrects distinct point-group
versus metric-invariance claims in Appendices B4/B5; those equalities must not
be reused as a point-group-membership rule.

## Exactly what C means

The paper uses gamma for parent and alpha for product. This workbench uses A
for parent and M for daughter/product.

  [u]_M = C_(M<-A) [u]_A

The columns of C contain the daughter-crystal coordinates of the mapped parent
basis directions. All components are rational numbers, not estimated rotation
angles. C is not an orientation relationship T and not a distortion F.

## Two interchangeable exact entry routes

1. **Three pairs of source/destination direct lattice vectors**. Form
   P = [u_source v_source w_source] and
   Q = [u_destination v_destination w_destination], writing corresponding
   column vectors in their *own* crystallographic bases. Calculate
   C_(destination<-source) = Q P^(-1) exactly (Cayron Eq. 9).
   The three columns of each matrix must be linearly independent.
2. **Direct 3 × 3 rational C matrix**. Fractions such as -1/2 are exact.
   A typed decimal is interpreted as its *exact terminating rational*;
   the UI does not pretend 0.3333 is exactly 1/3.

Selecting M -> A inputs produces the exact inverse automatically before
submitting canonical C_(M<-A) to the unchanged scientific solver
(Cayron Eq. 13). Direct-space column actions are previewed as [u]_M = C[u]_A.

**Reciprocal-space Miller planes are covectors**: [h]_M = C^(-T)[h]_A
(Cayron Eq. 11), *not* C[h]. Coordinate triplets in different conventional
basis settings are not a shared Cartesian frame. Neither the symmetry group
nor lattice metrics alone determine the physical correspondence.

An identity-column supercell is just a mathematically exact reversible
representative of C; it is NOT necessarily a minimal/primitive or physically
observed supercell. Verified chemistry, centering, specimens, measured
orientation relationships and atom shuffle remain outside the automatic
correspondence conversion.

## UI and numerical invariants

Changing entry format or direction attempts a lossless copy of the exact
canonical C; it does not replace a valid user input with the NiTi example.
Invalid or singular inputs fail before running the scientific backend.
The existing interactive direct-space vector action and report export remain
source-preserving. Input previews show the actual C sent to the solver.

## Bidirectional display and evidence conservation

In both entry formats the SOURCE is on the LEFT, the DESTINATION is on the RIGHT,
and the UI presents an explicit rightward arrow. A → M previews the canonical C;
M → A previews C⁻¹ with rightward source-to-destination semantics. The backend
retains the exact canonical forward C. Reciprocal planes follow D⁻ᵀ for the
selected input-direction D. Switching display format preserves any originally
entered paired supercell representatives when the canonical mapping is unchanged.
