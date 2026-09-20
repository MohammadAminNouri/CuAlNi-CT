# Cu–Al–Ni martensitic crystallography — CT vs PTMC vs Ball–James

**Version 0.2 — transparent research code, not a black box.**

The central question is not whether Cayron's Correspondence Theory *can be applied* to Cu–Al–Ni. It is whether CT **predicts the real transformation crystallography correctly**, and how those predictions compare with classical PTMC and Ball–James/cofactor theory when all branches receive the same independently sourced inputs.

## What this version does substantially better

- carries two physically distinct Cu–Al–Ni branches: DO3→6M monoclinic long-period martensite and DO3→2H orthorhombic martensite;
- stores explicit source-derived reference correspondences with their convention and derivation status;
- derives exact symbolic pulled-back metrics \(C^TM_MC\) for both branches;
- proves the 6M correspondence by recovering the independently transcribed 12-matrix James–Hane cube-edge stretch family;
- proves the 2H branch against published orientation anchors and the cubic→orthorhombic stretch family;
- generates the full 48-element cubic group exactly, correspondence subgroup, left cosets, double cosets, inverse operators, adjacency, multivalued groupoid composition and Burnside cross-check;
- performs Cayron CMC/SMC, Type-I/II twin and shear/shear calculations in crystallographic metrics;
- builds Ball–James stretches and independent rank-one/Mallard-law checks;
- implements Chen–Srivastava–Dabade–James CC1–CC3;
- contains an independent PTMC laminate compatibility branch;
- supports EBSD variant/operator/trace residuals without letting experiment silently tune the prediction;
- generates professor-readable derivation and verification reports.

## The key mathematical object

For a verified correspondence `u_M = C u_A`,

\[
G_C=C^T M_M C.
\]

Everything branches from this same object:

- **Cayron CT:** \(CMC=G_C-M_A\);
- **normalized comparison:** \(\hat G=M_A^{-1/2}G_CM_A^{-1/2}\), \(\widehat{CMC}=\hat G-I\);
- **Ball–James, cubic parent:** \(U^2=G_C/a_0^2\).

Thus CT and Ball–James do not receive different crystallography.

## Exact discrete results of the current reference branches

These are computation-derived consequences of the stated source-derived correspondences:

| branch | |G_A| | |G_M| | |H_C| | correspondence variants | operators | double-coset sizes |
|---|---:|---:|---:|---:|---:|---|
| DO3→6M | 48 | 4 | 4 | 12 | 8 | 4,4,4,4,8,8,8,8 |
| DO3→2H | 48 | 8 | 8 | 6 | 3 | 8,8,32 |

Operator counts are independently checked by Burnside's lemma.

## Install and verify

```bash
python -m pip install -e ".[dev]"
pytest -q
python -m cualni_cryst symbolic
python -m cualni_cryst report --out reports
```

## Read in this order

1. `docs/CONVENTIONS.md`
2. `docs/derivations/DO3_TO_6M.md`
3. `docs/derivations/DO3_TO_2H.md`
4. `docs/THEORY_COMPARISON.md`
5. `docs/VERIFICATION_MATRIX.md`
6. `docs/EXPLAIN_TO_PROFESSOR.md`

## Scientific restraint

- Literature benchmark lattice parameters are never defaults for an unknown sample.
- A source-derived correspondence is explicitly labelled as such and is checked against independent published consequences.
- A group-theory-allowed boundary is not claimed to be physically preferred until experiment confirms it.
- Near-zero compatibility residuals are not called exact if source lattice parameters are rounded.
- Hysteresis/reversibility are secondary observables, not pure crystallographic truth, because defects, precipitates, kinetics and grain constraints contribute.
