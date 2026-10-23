# Scientific repair worklist

These are TEST FINDINGS until independently reproduced. Never modify solver tolerances or physics just to silence a test.

## Numerical boundary audit — commit 7dc7a4427a12 (variants)

- Report: `audit_findings/numerical_boundary_2026_10_21/audit.json`
- Repair ledger: `audit_findings/numerical_boundary_2026_10_21/repair_ledger.md`
- New findings: 7. Status OPEN_UNREVIEWED pending independent reproduction.


## Reproduced variant-orbit identity faults: 2026-10-21
- [P1_VARIANT_LOSS_PTMC] epsilon=3e-9, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-9, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=3e-10, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=3e-10, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-10, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-10, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-12, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-12, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.

## Reproduced variant-orbit identity faults: 2026-10-21
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-9, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-9, scale=1.0, order=reverse, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-9, scale=1.0, order=sorted, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-10, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-10, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-10, scale=1.0, order=reverse, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-10, scale=1.0, order=reverse, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-10, scale=1.0, order=sorted, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-10, scale=1.0, order=sorted, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=1.0, order=reverse, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-11, scale=1.0, order=reverse, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=1.0, order=sorted, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-11, scale=1.0, order=sorted, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-12, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-12, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-12, scale=1.0, order=reverse, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-12, scale=1.0, order=reverse, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-12, scale=1.0, order=sorted, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-12, scale=1.0, order=sorted, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-9, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-9, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-9, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-9, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-9, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-9, scale=0.01, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-9, scale=100.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=1.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=0.01, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-11, scale=0.01, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_PTMC] epsilon=1e-11, scale=100.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ptmc_adapter.py ::_build_variants; DO NOT fix with another arbitrary identity threshold.
- [P1_VARIANT_LOSS_BJ] epsilon=1e-11, scale=100.0, order=forward, reference=6, production=3. Investigate src/cualni_cryst/ball_james_adapter.py ::generate_stretch_variants; DO NOT fix with another arbitrary identity threshold.
## Numerical boundary audit — commit 7dc7a4427a12 (tangency)

- Report: `audit_findings/numerical_boundary_2026_10_21/audit.json`
- Repair ledger: `audit_findings/numerical_boundary_2026_10_21/repair_ledger.md`
- New findings: 3. Status OPEN_UNREVIEWED pending independent reproduction.

