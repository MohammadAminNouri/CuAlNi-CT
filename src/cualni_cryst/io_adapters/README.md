# I/O adapters

Only format/convention adapters belong here.

An adapter may:

- parse an external format;
- validate its schema;
- convert documented orientation conventions;
- map vendor metadata into the existing internal data model;
- preserve provenance and warnings.

An adapter may **not**:

- alter CT/twin/orientation mathematics;
- silently guess missing crystallographic metadata;
- tune scientific thresholds to a dataset;
- inject an expected OR, variant, twin, or literature answer.
