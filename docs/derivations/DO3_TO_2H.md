# DO3 → 2H orthorhombic reference branch

## Experimental crystallographic anchors

Published Cu–Al–Ni EBSD reports that the basal planes of 2H correspondence variants originate from parent \(\{110\}\) planes and that \([010]_{2H}\) originates from parent \(\langle001\rangle\). A later Cu–Al–Ni paper states the lattice correspondence in one convention as

\[
[100]_{2H}\parallel[110]_A,\quad
[010]_{2H}\parallel[\bar110]_A,\quad
[001]_{2H}\parallel[001]_A,
\]

up to choice of symmetry-equivalent reference variant and basis labeling.

The repository's reference matrix chooses an equivalent cubic variant with

\[
[100]_A\mapsto[010]_{2H}
\]

and one parent \(\{110\}\) plane mapping to \((001)_{2H}\). Tests verify these relations explicitly.

## Exact metric pullback

For

\[
C=\begin{pmatrix}0&1&1\\1&0&0\\0&1&-1\end{pmatrix},
\qquad
M_M=\mathrm{diag}(a^2,b^2,c^2),
\]

\[
C^TM_MC=
\begin{pmatrix}
b^2&0&0\\
0&a^2+c^2&a^2-c^2\\
0&a^2-c^2&a^2+c^2
\end{pmatrix}.
\]

Therefore the reference stretch belongs to the standard cubic-to-orthorhombic face-diagonal family. The code verifies this against an independent James–Hane Eq. (9) implementation.

## Discrete groupoid result

Exact group arithmetic gives \(|H_C|=8\), 6 correspondence variants and 3 double-coset operator classes for this reference branch. As with 6M, these counts are derived from the explicit correspondence and point groups rather than inserted as assumptions.
