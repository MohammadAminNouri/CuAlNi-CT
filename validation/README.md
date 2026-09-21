# Validation workspace

This directory is for **evidence against the frozen backend**, not for changing
the backend until an answer appears.

Start every case from a compact manifest containing at least:

```text
case_id
input_sha256
input_format
adapter
backend_tag = ct-ebsd-reliability-gate-v1
backend_commit = f7c616379108c0ec06d804e4692cd437c35ec4e0
configuration
solver-visible metadata
withheld information
outputs
output hashes
comparison status
```

Recommended layout:

```text
validation/
  cases/
    <case-id>/
      manifest.json
      config.json
      notes.md
  protocols/
  README.md
```

Do not commit raw `.ctf`, `.ang`, HDF5, `.npz`, image stacks, or ZIP archives.
