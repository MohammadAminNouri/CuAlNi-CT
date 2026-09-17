from cualni_cryst.symmetry import cubic_full_m3m, cubic_proper_rotations, classify_cubic_operation

G = cubic_full_m3m()
Gp = cubic_proper_rotations()
print("Full cubic m-3m order:", len(G))
print("Proper rotational subgroup order:", len(Gp))
counts = {}
for g in G:
    kind = classify_cubic_operation(g)["kind"]
    counts[kind] = counts.get(kind, 0) + 1
print("Operation classes:", counts)
