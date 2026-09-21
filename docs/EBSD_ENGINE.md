# EBSD map engine

## Scientific contract

The EBSD layer has one internal orientation convention:

\[
v_{\rm sample} = g\,v_{\rm crystal,cart}.
\]

Every vendor convention is converted to this form exactly once at the I/O
boundary.  Downstream reconstruction never guesses whether a matrix is active,
passive, crystal-to-sample, sample-to-crystal, degrees, radians, or which sample
frame a vendor used.

This matters because EBSD file formats are not a universal orientation
standard.  EDAX ANG, Oxford CTF, H5EBSD/H5OINA and laboratory HDF5 archives may
store different metadata and reference-frame assumptions.

The engine therefore separates:

1. **file parsing**;
2. **orientation convention conversion**;
3. **phase crystallography**;
4. **map geometry and neighbors**;
5. **grain reconstruction**;
6. **parent reconstruction**;
7. **variant/operator classification**.

## Native and universal input paths

Native text readers:

- EDAX/TSL `.ang`
- Oxford/HKL `.ctf`

Universal explicit readers:

- CSV/TSV with a declared column schema;
- HDF5/H5EBSD/H5OINA/EDAX-H5/research archives with explicit dataset paths
  through the optional `h5py` adapter;
- direct in-memory matrices, quaternions or Euler arrays.

This is the safe meaning of "any EBSD map": any dataset can enter the internal
engine once its orientation representation and phase/coordinate fields are
declared.  Proprietary binary layouts are not guessed.

## ANG

The standard ANG row is interpreted as

```text
phi1 Phi phi2 x y IQ CI phase SEM fit
```

with Euler angles in radians.  Phase `<= 0` is unindexed.

The parser preserves IQ, CI, SEM and Fit but does not turn them into angular
uncertainty.  Those quality metrics are vendor/algorithm dependent.

## CTF

The parser expects the declared CTF columns including

```text
Phase X Y Euler1 Euler2 Euler3
```

plus optional Bands, Error, MAD, BC, BS and GrainIndex.

Standard 2-D CTF is treated as Bunge ZXZ degrees.  Historical 3-D CTF exports
have used different angle units; a 3-D file therefore requires an explicit
unit rather than a magnitude-based guess.

## Reference-frame corrections

Reference-frame corrections are explicit proper rotations:

\[
g_{\rm internal}
=
Q_{\rm sample}\,
g_{\rm raw,c\to s}\,
Q_{\rm crystal}^{T}.
\]

A raw sample-to-crystal matrix is transposed before these corrections.

No EDAX/Oxford frame correction is silently inserted.  This is essential for
hexagonal materials, where vendor crystal-frame choices can produce apparent
30-degree differences if mixed.

## Map quality audit

The map audit reports:

- total/indexed points;
- indexed fraction;
- phase populations;
- duplicate coordinates;
- maximum SO(3) residual;
- map dimensionality;
- nearest-neighbor spacing.

Indexed orientations must be proper rotations.  Only roundoff-scale departures
from SO(3) may be projected back to SO(3); larger errors fail loudly.

## Grain segmentation

Neighbors are built from physical coordinates, using declared X/Y/Z step sizes
when present or a nearest-neighbor estimate otherwise.

Two points merge only when:

- both are indexed;
- both belong to the same phase;
- their fully symmetry-reduced disorientation is below the threshold.

Different phases are never merged merely because their Euler angles are close.

## Orientation mean, KAM and GOS

Grain means use an iterative symmetry-aligned chordal mean:

1. move each orientation to the symmetry-equivalent representative nearest the
   current mean;
2. take the weighted matrix mean;
3. project that matrix to SO(3);
4. iterate to convergence.

This avoids the common error of averaging unrelated symmetry copies.

KAM uses phase-aware nearest-neighbor disorientations.  Grain orientation spread
(GOS) is the mean symmetry-reduced distance to the reconstructed grain mean.

## Parent reconstruction

For a parent/product OR variant

\[
x_A = R_{A\leftarrow M}^{(k)}x_M,
\]

the measured product orientation obeys

\[
g_M
=
g_A R_{A\leftarrow M}^{(k)}.
\]

Therefore every product grain provides parent candidates

\[
g_A^{(k)}
=
g_M
\left(R_{A\leftarrow M}^{(k)}\right)^T.
\]

The reconstruction algorithm performs a deterministic weighted consensus over
all candidates and then alternates:

- variant selection;
- parent-symmetry alignment;
- SO(3) averaging.

It returns:

- reconstructed parent orientation;
- best and second-best variant residual for every product grain;
- ambiguity margin;
- inlier fraction;
- mean/max reconstruction residual.

Quality fields such as CI or MAD are never converted into weights unless the
user explicitly supplies a weighting rule.

## Variant assignment

For a reconstructed parent,

\[
g_M^{(k)} = g_A R_{A\leftarrow M}^{(k)}
\]

is compared with the measurement under the complete proper product symmetry.
The engine reports both the best residual and the gap to the second-best
variant.  A small gap is flagged as crystallographically ambiguous.

## Martensite-martensite / product-product operators

Operator classification is matrix-based, not angle-only and not an
axis-angle Euclidean score.

For two measured product orientations,

\[
\Delta = g_1^T g_2.
\]

The full product-symmetry quotient is explored,

\[
S_1^T\Delta S_2,
\]

and compared geodesically on SO(3) against the complete theoretical operator
orbit.  This is robust near zero and 180 degrees, where an axis itself becomes
poorly conditioned.

## Uncertainty policy

The engine never manufactures uncertainty from EBSD quality fields.

If a microscope/software pipeline provides a calibrated angular uncertainty,
it can be carried as a separate field and used downstream.  CI, IQ, MAD, BC,
BS and pattern fit are preserved as observables, not silently converted to
degrees or probabilities.

## File-format background

The implementation follows the documented standard ANG column ordering and
the standard CTF Bunge ZXZ orientation model.  DREAM3D-NX documentation is used
as an external file-format reference, including its warning that reference
frames must be treated explicitly and its description of vendor differences in
H5EBSD/H5OINA workflows.

Relevant external references:

- DREAM3D-NX `ReadAngDataFilter`
- DREAM3D-NX `ReadCtfDataFilter`
- DREAM3D-NX `ReadH5EbsdFilter`
- DREAM3D-NX `ReadH5OinaDataFilter`
- MTEX ANG import/export documentation

These references define I/O behavior only.  They are not used as hidden
crystallographic fitting inputs.
