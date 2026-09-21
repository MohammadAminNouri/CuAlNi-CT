# Validation cases

One directory per real or synthetic validation case.

A case should be immutable once its prediction is frozen. If analysis protocol
changes materially, create a new case/version instead of overwriting the old
one.

The solver-visible input must not contain the expected crystallographic answer
when the case is intended to be blind.
