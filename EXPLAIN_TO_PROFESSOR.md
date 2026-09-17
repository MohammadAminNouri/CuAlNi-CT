# Five-minute explanation of the code

1. **I first freeze the crystallography.** A material state is not just “CuAlNi”; it is a parent phase, a particular martensite branch (6M or 2H), a cell convention, a correspondence and measured lattice parameters.

2. **I separate discrete and metric information.** Symmetry + correspondence gives an exact finite group problem. Lattice parameters then deform the metric continuously.

3. **The 48 cubic operations are not 48 variants.** I compute the correspondence subgroup \(H_C\), then variants are left cosets \(G_A/H_C\), while intervariant relationship classes are double cosets \(H_C\backslash G_A/H_C\).

4. **I keep real and normalized metrics.** The physical calculations use Å-based metrics. A dimensionless whitened metric is also calculated for comparison across compositions. No physical data are changed by normalization.

5. **The same metric/correspondence produces both CT and Ball–James quantities.** \(G_C=C^TM_MC\). Cayron uses \(G_C-M_A\) as CMC; for cubic parent, Ball–James uses \(U^2=G_C/a_0^2\).

6. **Then the theories separate.** CT derives twins directly from parent symmetry/correspondence and computes CMC/SMC/shear–shear compatibility. Ball–James solves rank-one connections and cofactor conditions. PTMC independently computes laminate habit-plane compatibility.

7. **EBSD is the judge, not an input used to tune the theory.** The code assigns measured domains to predicted variants, measured neighboring relations to theoretical operators, and compares predicted interface traces with measured ones.

8. **Across composition**, as long as the phase and correspondence stay the same, the groupoid topology stays fixed while the metric-dependent quantities move continuously. If the martensite switches from 6M to 2H, the discrete topology changes too.
