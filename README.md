# Cu–Al–Ni crystallography: CT vs PTMC vs Ball–James

A deliberately transparent Python package for testing martensitic crystallography in Cu–Al–Ni.

## Central scientific question

This repository does **not** assume Cayron's Correspondence Theory (CT) is correct. It is designed to test:

> How accurately does CT predict real Cu–Al–Ni martensite variants, intervariant operators, twin/junction geometry, A/M habit planes and supercompatibility, and how do those predictions compare with classical PTMC and Ball–James/cofactor theory when all theories receive the same crystallographic input?

## What is already implemented

- dimensional crystallographic metric tensors;
- an optional dimensionless normalized metric layer;
- full 48-operation cubic `m-3m` group and 24 proper rotations;
- monoclinic `2/m` and orthorhombic `mmm` point groups;
- exact correspondence subgroups, left cosets, double cosets and operator adjacency using SymPy;
- Cayron CMC and normalized CMC;
- CMC degeneracy analysis and compatible habit-plane extraction;
- Cayron SMC and A→M shear from a habit plane;
- Cayron Type-I and Type-II transformation-twin formulas from parent reflection / twofold symmetries;
- Cayron shear/shear supercompatibility residual `epsilon_CT`;
- stretch tensor `U` from correspondence + metrics;
- symmetry-generated stretch variants;
- numerical rank-one compatibility solver for independent checking;
- a numerical PTMC laminate solver based on the classical middle-singular-value condition;
- Chen–Srivastava–Dabade–James cofactor conditions;
- Cayron crystallographic cross tensor / quaternion product;
- basic EBSD variant-assignment and boundary-trace tools;
- strict provenance model and a sourced James–Hane Cu–Al–Ni lattice example.

## What is intentionally NOT hard-coded

The repository currently does **not** hard-code a Cu–Al–Ni `DO3 -> 6M` or `DO3 -> 2H` correspondence matrix.

That is deliberate. The exact correspondence is convention-sensitive and must be reconstructed from a primary crystallographic source, with the basis convention documented and independently checked against published stretch variants. Until that verification is done, this package refuses to present an exploratory matrix as a fact.

This is a reliability feature, not a missing numerical shortcut.

## Install

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e '.[dev]'
pytest
```

## First examples

```bash
python examples/01_metrics_and_normalization.py
python examples/02_cubic_symmetry.py
python examples/03_run_ct_when_correspondence_is_verified.py
```

## The calculation chain

```text
composition / temperature / state
        ↓
identify martensite branch: 6M long-period / 2H / other
        ↓
measured lattice parameters
        ↓
dimensional M_A, M_M  +  normalized metric layer
        ↓
verified correspondence C + G_A + G_M
        ↓
Cayron: H_C -> variants -> double-coset operators -> twins
        ↓
Cayron: CMC -> A/M habit planes -> SMC -> shear
        ↓
Cayron: shear/shear residual epsilon_CT

same C, M_A, M_M
        ↓
Ball-James: U -> stretch variants -> rank-one twins
        ↓
cofactor conditions

same inputs
        ↓
PTMC laminate -> habit plane / shape strain

all predictions
        ↓
EBSD / measured interfaces / mechanical data
        ↓
quantitative residuals and theory comparison
```

Read `docs/THEORY.md`, `docs/CONVENTIONS.md` and `docs/METHODOLOGY.md` before using the package for real data.
