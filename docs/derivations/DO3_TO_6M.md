# DO3 → 6M reference branch: derivation and verification

## 1. Why 6M is not just a renamed 18R cell

James & Hane explain that the long-period Cu-based martensites historically described with 18R/M18R cells can also be indexed by a 6M cell. The revised 6M choice matters because changing the cell changes the lattice correspondence, and the 6M description is consistent with the accepted monoclinic symmetry. The code therefore stores **physical phase**, **cell representation**, and **correspondence** as separate concepts.

## 2. Internal correspondence convention

Every module uses

\[
\mathbf u_M=C_{M\leftarrow A}\mathbf u_A.
\]

Plane covectors transform as

\[
\mathbf p_M=C_{M\leftarrow A}^{-T}\mathbf p_A.
\]

The chosen reference correspondence is source-derived from the revised 6M geometry and is *not* claimed to be a verbatim matrix printed in the paper:

\[
C_{M\leftarrow A}=
\begin{pmatrix}
0&1&1\\
1&0&0\\
0&1/3&-1/3
\end{pmatrix}.
\]

The decisive validation is not aesthetic: this correspondence must reproduce the independently transcribed James–Hane cube-edge stretch family.

## 3. Exact metric pullback

With the monoclinic unique-b metric

\[
M_M=\begin{pmatrix}
a^2&0&ac\cos\beta\\
0&b^2&0\\
ac\cos\beta&0&c^2
\end{pmatrix},
\]

the exact pulled-back daughter metric is

\[
G_C=C^T M_M C=
\begin{pmatrix}
b^2&0&0\\
0&a^2+\frac{2ac\cos\beta}{3}+\frac{c^2}{9}&a^2-\frac{c^2}{9}\\
0&a^2-\frac{c^2}{9}&a^2-\frac{2ac\cos\beta}{3}+\frac{c^2}{9}
\end{pmatrix}.
\]

For cubic parent metric \(M_A=a_0^2I\),

\[
U^2=G_C/a_0^2,
\qquad
\mathrm{CMC}=G_C-a_0^2I.
\]

Thus Cayron's CMC and Ball–James stretch theory are generated from the **same exact object**.

## 4. Verification criterion

The repository constructs \(U=\sqrt{G_C/a_0^2}\), generates \(Q UQ^T\) using the 24 proper cubic rotations, and compares the resulting family to an independent transcription of James–Hane Eq. (10). For the literature Cu–14 wt% Al–4 wt% Ni benchmark, all twelve matrices match to floating-point roundoff.

This is a much stronger check than merely obtaining “12 variants.”

## 5. Discrete CT result

Using full cubic \(m\bar3m\) (48 operations), monoclinic \(2/m\) (4 operations), and this correspondence, exact SymPy group arithmetic gives:

- \(|H_C|=4\),
- 12 correspondence variants,
- 8 double-coset operator classes,
- double-coset sizes 4,4,4,4,8,8,8,8.

The double-coset count is independently checked with Burnside's lemma. These are **computation-derived consequences of the stated correspondence**, not numbers copied from a source.

## 6. What is still experimental

The existence of a mathematically allowed operator does not imply that the interface is frequently observed. EBSD and boundary traces must test variant occurrence, operator occurrence, and predicted plane geometry. Composition-dependent lattice parameters then change the metric-dependent twin elements, CMC/SMC and supercompatibility residuals while the discrete groupoid remains fixed as long as the phase/correspondence branch remains fixed.
