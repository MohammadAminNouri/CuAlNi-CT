# Cu–Al–Ni 2H / modified 18R blind experimental geometry protocol

Status: SOFTWARE GEOMETRY ASSESSMENT ONLY. NOT EXPERIMENTALLY VALIDATED.

## Three user-provided JSON files

The Streamlit Research Audit panel accepts predictions, independent observations, and crystal/frame geometry. No literature values or measured answers are passed to the solver. A prediction-set hash is a content fingerprint, not proof that the file was frozen before measurements were examined.

Predictions JSON must contain: specimen_id, explicit phase (gamma1_prime_2H or beta1_prime_M18R), solver_commit (40 lowercase hex characters), crystal_inputs_sha256 (64 lowercase hex characters), and a nonempty predictions array. Each predicted candidate requires unique candidate_id, interface_kind (twin/habit), independently labelled variant_pair, and normal_parent_cartesian (3 finite components). Freeze the entire solver input state including lattice parameters, C and setting outside the comparison engine; preserve a third-party timestamp.

Observations JSON must contain exactly matching specimen_id and sample_identity, phase, DOI, source_location (specific page/figure/pattern), measurement_method, mass-percent Cu/Al/Ni, measurement_temperature_K, lattice_setting and measured_from_prediction=false. Every observation requires observation_id, interface_kind, frame, coordinates and uncertainty_deg. A source_independently_reviewed flag is a declaration only, not authenticated evidence.

Allowed frames: parent_cartesian; parent_reciprocal_miller; product_cartesian; product_reciprocal_miller; sample_plane_normal_cartesian; sample_surface_trace_cartesian. The last requires surface_normal_sample_cartesian and independent parent-to-sample orientation; the surface trace constrains only azimuth, not full 3D plane inclination. Product reciprocal Miller indices require specimen product metric and independently measured product-to-parent OR. All rotations are proper Cartesian, NOT correspondence matrices.

Geometry JSON must contain parent_metric (symmetric positive-definite 3x3). Optional, but required according to frames: product_metric; product_to_parent_rotation; parent_to_sample_rotation. The Cartesian metric embedding and OR conventions must be independently and consistently established. C is NOT derivable solely from these two metrics.

Output includes every eligible candidate and residual, best and runner-up, number of candidates inside uncertainty, unmatched observations, and unobservable traces. Twin and habit planes never cross-match. Isotropic Monte Carlo null fraction quantifies multiple-candidate chance for a hypothetical uniformly random undirected 3D plane or 2D tangent trace; it is not an experimentally calibrated p-value. A near-zero angle, unique nearest branch or low null fraction still does not establish physical validity.

## Real experimental sources and what they support

- Chen et al., Micron 31 (2000) 17–25, DOI 10.1016/S0968-4328(99)00060-8: 2H EBSD variant OR and {121}/{101} mirror twin families. Missing verified one-to-one link from archived numerical fixture input C/cell to that observed specimen and complete quantitative 3D interface normals.
- Sari and Aksoy, Journal of Alloys and Compounds 417 (2006) 138–142, DOI 10.1016/j.jallcom.2005.09.049: heat-treatment dependence and diffraction identification of CuAlNi 2H and modified 18R. Not a complete specimen-specific twin/habit geometry dataset.
- Nó, Caillard and San Juan, Acta Materialia 57 (2009) 1004–1014, DOI 10.1016/j.actamat.2008.10.017: nine experimentally studied CuAlNi habit planes and ORs; abstract alone does not provide independent numerical per-interface metrics and uncertainty. Full primary figures and measurements must be reviewed.
- Zárubová et al., ESOMAT (2009), DOI 10.1051/esomat/200902036: in-situ TEM 2H/18R and interface observations. Additional indexed specimen-specific geometry is required for blind acceptance.

Historical 6M reduced-cell representation must not be silently substituted for modified 18R conventional indexing. The labels phase, cell, stacking, and crystallographic correspondence require independent confirmation.

## Acceptance criteria

Freeze predictions and complete solver inputs before revealing observations. Independently source and cross-index the full specimen composition, temperature, phase, C, OR and metric setting. Use measured 3D normals or explicitly labelled 2D trace constraints with physical uncertainty. Evaluate every branch, unmatched observation, angular residual, multiplicity, orientation/frame adversaries, wrong-phase and wrong-specimen negative controls. Do not mark experimental validation PASS until a scientist has verified original data and reproduction independently.

Current experimental status for quantitative twin/habit predictions of both phases: NOT VALIDATED.
