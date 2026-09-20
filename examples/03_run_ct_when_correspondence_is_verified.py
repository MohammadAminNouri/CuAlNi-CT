"""Template showing the full CT pipeline once C has been verified from a primary source."""
import sympy as sp

from cualni_cryst.correspondence import Correspondence
from cualni_cryst.group_theory import correspondence_groupoid
from cualni_cryst.symmetry import cubic_full_m3m, monoclinic_2_over_m_unique_b

# Replace ONLY after source verification. This identity is intentionally a toy example.
C = Correspondence(sp.eye(3), label="TOY ONLY", source="HYPOTHETICAL_TEST")
res = correspondence_groupoid(cubic_full_m3m(), monoclinic_2_over_m_unique_b(), C)
print("TOY subgroup order:", len(res.subgroup))
print("TOY number of correspondence variants:", len(res.variants))
print("TOY number of operators:", len(res.operators))
print("Do not interpret these numbers as Cu-Al-Ni results.")
