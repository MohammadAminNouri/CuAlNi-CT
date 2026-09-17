"""Small professor-friendly sanity check: no installation details hidden."""
from cualni_cryst.symmetry import cubic_full_m3m, cubic_proper_rotations

print("Full cubic m-3m operations:", len(cubic_full_m3m()))
print("Proper cubic rotations:", len(cubic_proper_rotations()))
print("If these are not 48 and 24, stop: the symmetry layer is wrong.")
