# Open questions / deliberately unresolved items

1. **Exact DO3 -> 6M correspondence matrix for the chosen Cu–Al–Ni cell convention**
   - must be re-derived from the primary cell construction;
   - must pass the James–Hane `U_i` reconstruction unit test before being promoted to a reference constant.

2. **Exact DO3 -> 2H correspondence for Cu–Al–Ni**
   - needs primary-source extraction, not analogy to another alloy.

3. **18R/M18R/6M representation policy**
   - the code supports explicit change-of-cell logic, but the exact transformation matrix between a user's reported cell and the canonical 6M cell must be recorded with source/convention.

4. **Weak junctions**
   - current version provides the group-theory foundation and Type-I/II exact formulas but does not yet implement Cayron's tolerance-ranked weak-plane search.

5. **General non-cubic-parent CT habit-plane extraction**
   - CMC is implemented generally through metric whitening; experimental validation should begin with cubic DO3 parent before extending further.

6. **EBSD vendor adapters**
   - current utilities use orientation matrices with one explicit convention. ANG/CTF/H5 importers should be added only after the actual lab export format is known.
