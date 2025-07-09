import matplotlib.pyplot as plt
import numpy as np



h = 2
w = np.concatenate(([0.5], np.arange(1, 11)))
g = np.linspace(0, 100, 101)

fig, ax = plt.subplots(figsize=(10,8))
for width_values in w:
    #roughness_factor = (4 * (width_values * h) + (width_values + g)**2) / (width_values + g) ** 2
    roughness_factor = 4*(width_values * h) / (g + width_values) ** 2
    ax.plot(g, roughness_factor, marker='o', markersize=3, linewidth=2, label=f'{width_values: .1f}')
plt.axhline(y=0, linestyle='--', linewidth=2, color='r')

ax.set_xlim(0, 100)
ax.set_yticks(range(0, 19, 1))
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_xlabel("Gap", fontsize=16)
ax.set_ylabel("Roughness Factor", fontsize=16)
ax.set_title("Roughness Factor vs. Gap", fontsize=16)
ax.legend(frameon=False, title='Width value', fontsize=16, title_fontsize=16)
plt.savefig('./roughness_factor/r_gap.png', dpi=300, bbox_inches='tight')
plt.show()