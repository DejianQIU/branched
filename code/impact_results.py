import matplotlib.pyplot as plt
import numpy as np

# Define pillar width, height and gap between pillars and height
w = 2
h = 2
g = np.linspace(2, 8, 4)

# Define roughness factor and A_p/w values
new_roughness_factor = 4 * (w * h) / (w + g) ** 2
r = np.concatenate(([0], new_roughness_factor[::-1]))   # 0, 0.16, 0.25, 0.44, 1
Ap_w = [10, 30, 60, 90, 120, 150, 200]

"""
# linear: xp=0.02, Np=40, BpP=50
# Define deposition points as (r, A_p/w) pairs
deposition_points = [
    (r[1], Ap_w[3]), (r[1], Ap_w[2]), (r[1], Ap_w[1]),
    (r[2], Ap_w[2]), (r[2], Ap_w[3]),
    (r[3], Ap_w[2]), (r[3], Ap_w[3]),
    (r[4], Ap_w[2]),
    (r[0], Ap_w[1]), (r[0], Ap_w[2]), (r[0], Ap_w[3]), (r[0], Ap_w[4])
]

# Define rebound with satellite droplet points
satellite_points = [
    (r[1], Ap_w[6]),
    (r[2], Ap_w[1])
]
"""

# branched: xp=0.02, bb=50, Np=20, BpP=50
# Define deposition points as (r, A_p/w) pairs
deposition_points = [
    (r[1], Ap_w[3]), (r[1], Ap_w[2]), (r[1], Ap_w[1]),
    (r[2], Ap_w[2]), (r[2], Ap_w[3]),
    (r[3], Ap_w[2]), (r[3], Ap_w[3]),
    (r[4], Ap_w[2]),
    (r[0], Ap_w[1]), (r[0], Ap_w[2]), (r[0], Ap_w[3]), (r[0], Ap_w[4])
]

# Define rebound with satellite droplet points
satellite_points = [
    (r[1], Ap_w[6]),
    (r[2], Ap_w[1])
]

# Create all possible data points (CORRECTED)
all_points = []
for r_values in r:
    for ap_w in Ap_w:
        all_points.append((r_values, ap_w)) # Use the scalar value r_values

# Separate data into deposition, rebound, and satellite points
deposition_r = [point[0] for point in deposition_points]
deposition_ap_w = [point[1] for point in deposition_points]

satellite_r = [point[0] for point in satellite_points]
satellite_ap_w = [point[1] for point in satellite_points]

# Points that are neither deposition nor satellite are regular rebound
rebound_points = [point for point in all_points if point not in deposition_points and point not in satellite_points]
rebound_r = [point[0] for point in rebound_points]
rebound_ap_w = [point[1] for point in rebound_points]

# Rest of the plotting code remains the same...
# Create the plot
plt.figure(figsize=(10, 8))

# Plot deposition points (red squares)
plt.scatter(deposition_r, deposition_ap_w, color='red', marker='s', s=100,
            label='Deposition', zorder=2)

# Plot rebound points (blue circles)
plt.scatter(rebound_r, rebound_ap_w, color='blue', marker='o', s=100,
            label='Rebound', zorder=1)

# Plot rebound with satellite droplet points (green triangles)
plt.scatter(satellite_r, satellite_ap_w, color='deepskyblue', marker='^', s=120,
            label='Rebound with satellite residual', zorder=3)

# Set up the plot
plt.xlabel('roughness factor r', fontsize=16)
plt.ylabel(r'$\mathrm{A_{p/w}}$', fontsize=16)

plt.legend(frameon=False, fontsize=14, loc='best', bbox_to_anchor=(1.1, 195), bbox_transform=plt.gca().transData)

# Adjust axis
plt.xlim(-0.1, 1.2)  # Adjusted to include r=0
plt.ylim(0, 210)
plt.xticks(r, [f'{x:.2f}' for x in r], fontsize=16)  # 2 decimal points for x axis
plt.yticks(Ap_w, fontsize=16)

# Show plot
plt.tight_layout()
plt.savefig('impact_result.png', dpi=300, bbox_inches='tight')
plt.show()