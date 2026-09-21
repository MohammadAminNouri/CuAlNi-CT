# EBSD phase-survival forensics

## Why this layer exists

A valid indexed phase can disappear before crystallographic inference if every
orientation/spatial component is smaller than a configured minimum grain size.

That is not the same as "phase absent" and not the same as "no
crystallographic relation".

The phase-forensics layer is therefore a mandatory diagnostic before any
parent reconstruction or blind transformation inference.

## Evidence hierarchy

For every phase the layer reports:

1. indexed-pixel count;
2. raw same-phase spatial connected components;
3. raw component-size distribution;
4. same-phase neighbor disorientation distribution;
5. generic quality-field quantiles;
6. orientation-connected components at every configured threshold;
7. survival under minimum sizes 1, 2, 3, 5, 10, 20.

A phase can no longer disappear silently between steps 1 and 7.

## Direct interface route

When a minority phase does not form conventional grains but has genuine
cross-phase spatial interfaces, the code can still test whether independent
orientation-component interfaces share a common relative-orientation class.

For an anonymous unordered phase pair `a < b`:

\[
R_{a\leftarrow b} = g_a^T g_b.
\]

This is coordinate bookkeeping only.  It does **not** assert that `a` is a
parent or `b` is a product.

Equivalent observations obey

\[
R' = S_a^T R S_b,
\]

so every residual uses the complete proper left/right symmetry quotient.

Interface evidence is built from unique orientation-component pairs, not every
pixel edge.  This avoids pseudo-replication from a long interface.

## Fit / holdout / null design

Interface units are deterministically divided into fit and holdout subsets.

The candidate relation is discovered on fit units only.

The frozen relation is then tested on untouched holdout units.

Finally, a pairing-permutation null test shuffles holdout side-B component
orientations while preserving side-A orientations and marginal orientation
sets.  The fitted candidate is never re-optimized on the null permutations.

Thus the null test asks whether the true spatial pairing supports the discovered
relation better than randomized phase pairing.

## What the layer never does

It never:

- lowers a minimum grain size automatically;
- converts isolated pixels into "grains";
- inserts a literature OR;
- assigns parent/product roles;
- names a famous OR;
- interprets a quality field as angular uncertainty;
- forces a relationship when interface evidence is insufficient.

## Required future pipeline behavior

Any production EBSD transformation workflow should run this preflight before
grain-level crystallographic inference.

If a phase has indexed pixels but no retained grains, the orchestrator should
report the survival audit and choose only among predeclared evidence routes:

- grain-level inference when retained grain topology is sufficient;
- direct interphase-interface relation analysis when enough independent
  component interfaces exist;
- `insufficient_evidence` when neither route is supported.

Silent phase extinction is a pipeline error, not a valid scientific conclusion.
