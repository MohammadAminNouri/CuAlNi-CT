# PTCLab baseline and CuAlNi-CT workstation scope

## Purpose

The CuAlNi-CT workstation uses the interaction principles that made PTCLab useful — phase-first project setup, crystallographic object tools, orientation relationships, transformation models, variants, compact calculator functions, graphical/tabular output — while keeping the CuAlNi-CT scientific architecture explicit and testable.

Primary comparison source: **Xinfu Gu, PTCLab User Manual v1.18.8 (2014)**, especially Chapters 2–6.

This file is a scope contract, not a marketing parity claim. A control is exposed only when this repository has an actual backend calculation for it. Unsupported theories are never represented by placeholder tabs that look operational.

## PTCLab basics retained

PTCLab's manual organizes the workflow around:

- crystal structures and conventional coordinate systems;
- pole/stereographic representation and symmetry-equivalent objects;
- phase-transformation crystallography;
- variants generated from symmetry and orientation relationships;
- a calculation pad for single-/two-phase crystallographic geometry and orientation conversion;
- diffraction / EBSD-oriented visualization and indexing utilities.

The CuAlNi-CT public workstation keeps the same scientist-facing logic where the corresponding mathematics exists, but presents it as a persistent web project rather than a collection of modal desktop dialogs.

## Workstation improvements in this release

### 1. Crystal-family-aware phase editor

The point group is selected from the exact 32-group registry. The UI exposes only the independent conventional-cell parameters:

| family | editable conventional parameters |
| --- | --- |
| cubic | a |
| tetragonal | a, c |
| orthorhombic | a, b, c |
| hexagonal | a, c (hexagonal axes) |
| trigonal built-in setting | a, c (hexagonal axes) |
| monoclinic unique-b | a, b, c, beta |
| triclinic | a, b, c, alpha, beta, gamma |

The project loader still independently verifies metric positivity and point-group/metric consistency. UI constraints do not replace scientific validation.

### 2. Correspondence and orientation are distinct

The workstation never treats lattice correspondence `C` and physical orientation relationship `R` as synonyms.

- `C(product <- parent)` maps direct crystallographic coordinates and its inverse transpose maps plane covectors.
- `R(parent <- product)` is a proper physical Cartesian rotation.
- Both forward and reverse object mappings are exposed and labelled.

### 3. Orientation relationship analysis and variants

The OR can be supplied as:

- an explicit proper rotation matrix;
- ZXZ Euler angles with active/passive convention stated;
- two crystallographic parallelisms;
- the polar rotation of the correspondence deformation, explicitly labelled as a *candidate* rather than an experimental/theoretical OR identity.

For a chosen OR, the backend reports Cayron-style orientation intersection topology, symmetry-distinct variants, double-coset operators, disorientation information, axis/angle, Euler representations and representation-parity diagnostics.

### 4. Bidirectional parent/product reconstruction

The workstation treats reconstruction as a first-class workflow in both directions. With

`x_sample = g(sample <- crystal) x_crystal`

and `R(parent <- product)`, it enumerates symmetry-distinct candidates using the backend-generated OR variants:

- observed parent -> candidate product orientations;
- observed product -> candidate parent orientations.

No EBSD vendor Euler convention is guessed. Users must provide/convert to the stated matrix convention.

This goes beyond the 2014 PTCLab manual's documented state, where reconstruction appears in the EBSD/Variant interface but is also listed among future features in Chapter 7.

### 5. Martensite workflow

For selected stretch variants, the workstation composes existing scientific modules:

1. registered proper parent symmetries;
2. Mallard Type-I / Type-II twin solutions where the selected pair satisfies the backend relation;
3. cofactor CC1–CC3 evaluation;
4. classical single-shear PTMC volume-fraction solutions;
5. optional independent numerical rank-one residual cross-check.

No missing twin system is invented. A manual `a tensor n` route is available for expert-specified lattice-invariant shear data.

**Double-shear PTMC is not claimed in this release** because no validated double-shear solver is currently exposed by the frozen backend.

### 6. Metric-correct CalPad

The public CalPad exposes backend calculations for:

- direct and reciprocal cell information;
- direct/reciprocal metrics;
- physical plane-normal <-> crystallographic-index conversion using the actual non-cubic metric;
- nearest low-index representations with their angular mismatch;
- low-index direction/plane ranking with projective or oriented angle sense.

An approximate low-index representative is never presented as an exact Miller-index identity.

## Current scientific method scope

| Method / workflow | Public workstation status |
| --- | --- |
| CMC / normalized CMC / SMC | available |
| stretch tensor / principal stretches | available |
| CT exact compatibility / habit planes | available |
| nearest-degeneracy diagnostic | available, explicitly non-exact |
| Ball–James single-variant rank-one | available |
| correspondence variants / operators | available |
| physical OR variants / operators | available |
| parent <-> product reconstruction | available |
| Mallard twins | available for qualifying selected stretch pairs |
| classical single-shear PTMC | available |
| cofactor CC1–CC3 | available |
| metric CalPad | available |
| double-shear PTMC | not claimed |
| invariant-line / O-line / E2E | not claimed in public workstation |
| 3-D NCS / Moire-delta-g / O-lattice / NCRL | not claimed in public workstation |
| TEM / Kikuchi / EBSD pattern simulation and indexing | not claimed in public workstation |
| X-ray profile simulation | not claimed in public workstation |

The last rows are deliberate omissions rather than hidden approximations. They may be added only through explicit scientific implementations and tests.

## UI acceptance rules

1. No crystallographic equation is independently reimplemented in Streamlit.
2. Every scientific button reaches an existing Python backend solver/service.
3. Exact and approximate classifications remain visibly distinct.
4. Direct directions and reciprocal plane covectors remain distinct types/conventions.
5. Parent/product direction conventions are shown near matrices.
6. Project metadata and internal IDs are secondary, not the main normal-user workflow.
7. Save/reload/export must preserve the project calculation state.
8. Invalid domains fail with explanatory messages instead of fabricated results.
9. The frozen scientific package must remain unchanged during workstation installation.
