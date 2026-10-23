# Boundary classification investigation — read-only

This package does not modify the production repository. It uses the published orthorhombic Cu–Al–Ni benchmark's *input*, changes one lattice parameter **synthetically**, and captures the full direct Type-I/Type-II symmetry witnesses at `epsilon=1e-9`.

From the existing `feature/twin-family-tree-app-v1` Codespace, upload the script to the repository root, then run:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u diagnose_symmetry_conflict.py
```

The script writes `audit_findings/boundary_classification_witness_2026_10_21.json` containing the exact offending target/source construction IDs, symmetry-generator index, both ordered-stretch residuals, dyad residual, and whether two distinct variants accidentally fit within the same classification tolerance. The alternative tolerances are **diagnostics**, not a physics modification or validation.

It uses the existing `numerical_boundary_falsification.py` and `upstream_twin_falsification.py` scripts already installed in the Codespace. It neither commits nor pushes. Inspect the output evidence before any source edits, and do not relabel contradictions as Compound without direct independent crystallographic evidence.
