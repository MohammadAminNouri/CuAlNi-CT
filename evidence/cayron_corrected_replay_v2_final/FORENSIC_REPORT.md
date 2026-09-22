# Forensic result — why the first reveal failed

The first reveal did **not** expose a need to loosen tolerances. It exposed a
correspondence-direction error in the validation input for the two NiTi cases.

Cayron prints

`C = [[0,1,-1],[0,1,1],[1,0,0]]`.

Under Cayron's crystallographic convention, its columns are daughter basis
vectors expressed in the parent basis. The decisive exact identity is:

`C * [010]_B19' = [110]_B2`.

Therefore, under this repository's explicit coordinate-action convention
(`u_M = C_M_from_A u_A`), Cayron's printed matrix is **A_from_M**, not
**M_from_A**.

The old case_003 and case_004 files incorrectly declared the same raw matrix as
`"mode": "M_from_A"`. This reverses the physical correspondence while leaving
some group-theory counts invariant. That is why topology passed and O2 could
pass accidentally while O4/O5/O6 failed.

The correct internal matrix is:

`C_M_from_A = inverse(C_paper)
             = [[0,0,1],[1/2,1/2,0],[-1/2,1/2,0]]`.

It immediately gives the published rational CT elements:

- `(011bar)_B2 -> (1bar11)_B19'`
- `[011bar]_B2 -> [2bar11]_B19'`
- `(011)_B2 -> (111)_B19'`
- `[011]_B2 -> [211]_B19'`
- `(010)_B2 -> (011)_B19'`
- `[010]_B2 -> [011]_B19'`

It also fixes the metric diagnostic. For Kudoh's nominal values,

`lambda2 - 1 = 4.108 / (sqrt(2)*3.01) - 1
             = -0.03495194115802758`.

Using the wrongly directed raw matrix in `C^T M_M C` instead gives a completely
different middle stretch (about `+0.58319` relative to 1), which is the
fingerprint of the convention error.

## Integrity rule

Do not delete, rewrite, or relabel the failed v1 campaign. It is evidence of the
defect. Run this v2 as a **corrected-input replay**, not as a retroactively
successful blind campaign.

The solver phase and literature reveal remain separate. The solver replay does
not load the oracle or expected literature values.
