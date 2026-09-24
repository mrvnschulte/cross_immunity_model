"""
Shows that the invasion ability is not transitive, even under independence
assumption.
"""

from cross_immunity_model import invasion_reproduction_number


sigma_13 = 1
r0_1 = 2.5
r0_2 = 2.25
r0_3 = 2
sigma_12 = 0.8
sigma_23 = 0.8

r0_inv21 = invasion_reproduction_number(r0_1, r0_2, sigma_12)
r0_inv31 = invasion_reproduction_number(r0_1, r0_3, sigma_13)
r0_inv32 = invasion_reproduction_number(r0_2, r0_3, sigma_23)

print(
    "Assume that variant 1 has two descendants 2 and 3 and setting the parameters:"
)
print(rf"R_{{0,1}}={r0_1}")
print(rf"R_{{0,2}}={r0_2}")
print(rf"R_{{0,3}}={r0_3}")
print(rf"sigma_{{12}}={sigma_12}")
print(rf"sigma_{{13}}={sigma_13}")
print(rf"sigma_{{23}}={sigma_23}")

print("\nWe then have:")
print(rf"R_{{0,3to2}}={r0_inv32}")
print(rf"R_{{0,2to1}}={r0_inv21}")
print(rf"R_{{0,3to1}}={round(r0_inv31, 1)}")
