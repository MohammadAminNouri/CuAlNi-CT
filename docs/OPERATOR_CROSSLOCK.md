# Operator cross-lock: simple UI, exact crystallography

This layer freezes the discrete orientation/correspondence topology before the
project moves to CT-vs-Ball-James-vs-PTMC theory adapters.

## The user-facing rule

Run:

```bash
cualni-crosslock --transformation do3_to_6m_reference
```

The default table is intentionally compact. It tells you:

- orientation operator `T`;
- exactly matched correspondence operator `C`;
- inverse operator;
- Cayron class;
- operator size;
- available CT twin modes;
- independent Mallard/Ball-James check;
- one representative disorientation.

For the actual \(K_1,\eta_1,K_2,\eta_2,s\):

```bash
cualni-crosslock --transformation do3_to_6m_reference --details
```

For exact subgroup and double-coset matrices:

```bash
cualni-crosslock --transformation do3_to_6m_reference --members
```

For machine-readable output:

```bash
cualni-crosslock --transformation do3_to_6m_reference --json
```

Any physical OR can be tested against an explicit correspondence hypothesis by
supplying a proper rotation matrix:

```bash
cualni-crosslock \
  --transformation do3_to_6m_reference \
  --matrix "1 0 0; 0 1 0; 0 0 1"
```

The program never guesses a correspondence from phase names.

## Scientific contracts

### 1. Explicit binding

`OrientationState.transformation_id` is optional for a free experimental/user
OR, but mandatory for any C/T cross-comparison. A bound OR must have the same
reference/product phase endpoints as the selected `TransformationState`.

### 2. Operator identity

The operator is the exact double coset

\[
O_k=H g_k H.
\]

The minimum disorientation angle is only a useful representative summary. It is
not used to identify, merge or compare operators.

### 3. Ambivalent versus polar

The general groupoid definition is used:

\[
O_k\ \text{ambivalent}\iff O_k^{-1}=O_k.
\]

Otherwise the operator is polar and its inverse belongs to a complementary
operator.

This is kept separate from standard twin-generator content:

- parent mirror present -> Type-I CT candidate;
- parent proper two-fold rotation present -> Type-II CT candidate.

An operator can therefore be self-inverse without containing a standard
Type-I/Type-II generator. The software does not force a twin label in that case.

### 4. Exact C/T cross-lock

A one-to-one `O_T <-> O_C` result is reported only if:

1. exact parent subgroup memberships satisfy `H_T == H_C`;
2. the number of double cosets agrees;
3. every orientation double-coset member set has exactly one equal
   correspondence double-coset member set.

No angle-based matching is used.

### 5. CT twin elements

For every matched correspondence operator, all eligible parent mirrors and
proper 180-degree rotations are sent through the existing Cayron CT twin
backend.

Type I reports:

\[
K_1,\quad \eta_1,\quad s.
\]

Type II reports:

\[
K_2,\quad \eta_2,\quad s.
\]

The exact parent generator is retained as provenance.

### 6. Independent Mallard/Ball-James check

Where the transformation has the required stretch/twin structure, the existing
independent Mallard/Ball-James solver is run and the maximum:

- physical plane/direction angular mismatch;
- relative shear mismatch;
- rank-one residual

are reported.

If that independent adapter is not applicable to a future transformation, the
result is `unavailable` rather than a fabricated comparison.

## DO3 -> 6M truth-lock

For the present reference branch the exact common subgroup is

\[
H_T=H_C=
\left\{
I,\;
\operatorname{diag}(1,-1,-1),\;
-I,\;
\operatorname{diag}(-1,1,1)
\right\}.
\]

The regression suite freezes those four matrices and the complete memberships
of all eight full double cosets, not merely the counts.

The exact inverse-operator signature is:

```text
O1 -> O1
O2 -> O2
O3 -> O3
O4 -> O4
O5 -> O5
O6 -> O7
O7 -> O6
O8 -> O8
```

Thus `O6/O7` are a complementary polar pair. `O4` is an important regression
case: it is self-inverse/ambivalent even though it contains no parent mirror or
proper two-fold generator. This is why ambivalence must not be defined merely
by testing for a Type-I/II generator.

## Why this matters for the research question

The final comparison must keep independent objects independent:

```text
C / correspondence
    -> H_C -> C variants -> C operators
                     |
                     +-> Cayron CT twin elements

T or R / orientation candidate
    -> H_T -> OR variants -> OR operators

then, only when mathematically justified:
    O_T <-> O_C

and independently:
    CT twins <-> Mallard/Ball-James
    CT A/M compatibility <-> Ball-James / cofactor / PTMC
    theory predictions <-> EBSD / experiment
```

That makes the program useful both as a PTCLab-like crystallographic tool and
as a falsifiable test platform for Correspondence Theory in Cu-Al-Ni.
