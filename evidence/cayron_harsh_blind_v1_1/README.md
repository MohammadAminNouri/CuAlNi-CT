# Cayron harsh blind campaign v1.1

This supersedes v1.  v1 contained a bad first orientation input
(BCC-HCP/Burgers-like) and correctly failed before producing a prediction.

v1.1 uses the actual 2006 Cayron FCC->BCC orientation relationships:

- case_001: Nishiyama-Wassermann, input only as one plane parallelism and one
  direction parallelism;
- case_002: Kurdjumov-Sachs, same blind-input principle;
- case_003: 3D B2 -> B19' metric/correspondence case;
- case_004: second B2 -> B19' metric/correspondence state;
- case_005: Cu-Al-Ni beta1(D03) -> gamma' metric/correspondence case.

No literature variant count, operator count, expected misorientation, expected
habit plane, twin plane, shear, compatibility result or observed answer is
stored in the case files.

New safeguard:
- every orientation case is subjected to an exact crystallographic-incidence
  preflight before the solver runs.  For the plane/direction definitions used
  here, h*u+k*v+l*w must be exactly zero on both parent and product sides.
  This would have rejected the bad v1 case immediately.

Run outside the repository:

```bash
python /tmp/cayron_harsh_blind_campaign_v1_1/run_harsh_blind_campaign.py \
  --repo /workspaces/CuAlNi-CT
```
