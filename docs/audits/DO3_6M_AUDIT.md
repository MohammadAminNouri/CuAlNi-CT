# DO3 -> 6M audit

## Scope

This audit covers the current implementation of the DO3 -> 6M Cu-Al-Ni reference branch, as it stands on the active research-hardening branch, before further scientific extension.

## 1. What the current code assumes

- The direct-space lattice correspondence is stored as a 3x3 exact matrix and used as the transform from parent crystallographic coordinates to martensite coordinates:

  u_M = C_M_from_A u_A

- The code uses the same matrix for both the parent-to-daughter action and the daughter-to-parent inverse, but it keeps the inverse explicit via C_A_from_M = C_M_from_A^{-1}.
- Parent cubic symmetry is generated exactly as signed permutation matrices, giving 48 operations in m-3m.
- Daughter 6M symmetry is represented as the unique-b monoclinic 2/m group with four matrices.
- CT discrete subgroup analysis is implemented via exact SymPy intersection and exact coset/double-coset enumeration.
- The reference 6M choice is benchmarked against the James-Hane cube-edge stretch family, rather than being treated as a default material law.
- All calculations keep a conceptual separation between dimensional quantities (with metric M) and normalized comparison quantities (e.g. G_hat, CMC_hat), which is scientifically appropriate.

## 2. Exact correspondence convention

The package uses the explicit matrix names:

- C_M_from_A
- C_A_from_M

with the direct-space convention

u_M = C_M_from_A u_A,

and the reciprocal-space covector convention

p_M = C_M_from_A^{-T} p_A.

This is the correct pair of coordinate actions for the chosen basis. The lowercase alias names C_m_from_a and C_a_from_m remain as compatibility aliases but are not the authoritative names in the scientific API.

The current exact reference matrix is:

C_DO3_TO_6M =
[[0, 1, 1],
 [1, 0, 0],
 [0, 1/3, -1/3]]

This matrix is source-derived from the revised 6M basis relation and is validated by the James-Hane family reconstruction tests.

## 3. Exact metric convention

The direct metric is defined by lattice lengths and angles as:

M =
[[a^2, ab cos(gamma), ac cos(beta)],
 [ab cos(gamma), b^2, bc cos(alpha)],
 [ac cos(beta), bc cos(alpha), c^2]]

For the monoclinic 6M basis, the code stores the crystal cell as a unique-b monoclinic lattice with explicitly declared alpha = 90°, gamma = 90°, and beta = reported monoclinic angle.

This convention is valid only when the cell is genuinely in the unique-b monoclinic setting. The implementation does not silently rename a general angle as beta unless the basis ordering is explicitly checked. The data model is therefore structurally stronger than a generic 3x3 metric convention but must remain precise about the basis basis in any future source comparison.

## 4. Source-derived vs computation-derived results

### Source-derived / source-supported

- The reference 6M lattice correspondence matrix C_DO3_TO_6M.
- The cubic parent group as exact signed permutation matrices.
- The parent and daughter metric definitions.
- The James-Hane example lattice parameters and their provenance status.
- The direct correspondence benchmark against the published James-Hane stretch set.

### Computation-derived

- H_C via exact subgroup intersection in parent coordinates.
- Exact left coset partition of G_A into correspondence variants.
- Exact double-coset partition and the number of operator classes.
- Burnside cross-check for the operator count.
- The pulled-back metric G_C = C^T M_M C.
- The dimensional CMC and normalized CMC_hat.
- The generalized eigenvalue spectrum and reproducible U = positive_sqrt(G_C / a0^2).

### Hypotheses or not yet fully source-grounded

- Any weak-plane or nonconventional junction classification that requires a physical OR or fit parameter not yet source-supported.
- Any CT twin classification that is not independently cross-checked against a Ball-James / Mallard law solution.
- Any experimental EBSD or sample-orientation comparison outside the exact crystallographic branch.

## 5. Current ambiguities and weaknesses

- The code has multiple naming conventions in parallel: C_m_from_a, C_a_from_m, C_M_from_A, C_A_from_M. The scientific API should prefer the uppercase names everywhere for clarity.
- The direct-space and reciprocal-space transforms are implemented but not yet consistently enforced across every module through a single explicit API contract.
- The 6M monoclinic metric is structurally correct only for unique-b convention; a non-standard basis would require a clear basis transformation before the same formulas are reused.
- Some modules still use lowercase names in docstrings and comments; this is not a numerical bug, but it weakens auditability.
- The package contains several mathematically complete mathematical objects but only a partial set of high-level report-generation and provenance endpoints for the full research-grade chain of source -> derivation -> implementation -> verification.

## 6. Insufficient tests

The current test set covers the strong benchmark family reconstruction, but there are still gaps relative to the requested scientific fortification:

- explicit first-principles direct/reciprocal convention checks, including p^T u = 0 before and after mapping;
- exact symbolic checks for the inverse-transpose plane transform, not only numeric convenience tests;
- explicit monoclinic metric symmetry/positive-definiteness tests with the chosen unique-b basis specification;
- stronger provenance and source status checks for every report result;
- more explicit audit files for unresolved hypotheses and unsupported assumptions;
- tests that check the exact distinction between parent direct vectors and reciprocal covectors in 6M, instead of relying on integer index conventions.

## 7. Mathematical strengths of the current implementation

- Exact discrete finite-group logic in SymPy.
- Deterministic coset and double-coset generation.
- Good separation between metric and normalization branches.
- Strong benchmark reproduction for the published James-Hane 12-stretch family.
- Good provenance structure for source-origin tracking of the example data.

## 8. Scientific caution

This branch is not a full general-purpose crystallographic solver. It is a reference branch for one exact DO3 -> 6M correspondence on one source benchmark, and its main strength is reproducibility and auditability.

Any claim that a physical twin or habit plane is experimentally confirmed would be premature without source-backed data, independent geometry, and explicit tolerance classifications.
