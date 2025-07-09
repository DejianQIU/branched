import matplotlib.pyplot as plt
import numpy as np



h = 2
g = np.concatenate(([0.5], np.arange(1, 11)))
w = np.linspace(0, 100, 101)

fig, ax = plt.subplots(figsize=(10,8))
for gap_values in g:
    #roughness_factor = (4 * (w * h) + (w + gap_values)**2) / (w + gap_values) ** 2
    roughness_factor = 4*(w * h) / (w + gap_values) ** 2
    ax.plot(w, roughness_factor, marker='o', markersize=4, linewidth=2, label=f'{gap_values: .1f}')
plt.axhline(y=0, linestyle='--', linewidth=2, color='r')

ax.set_xlim(0, 100)
ax.set_yticks(np.arange(0, 4.5, 0.5))
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.set_xlabel("Width", fontsize=16)
ax.set_ylabel("Roughness Factor", fontsize=16)
ax.set_title("Roughness Factor vs. Width", fontsize=16)
ax.legend(frameon=False, title='Gap value', fontsize=16, title_fontsize=16)
plt.savefig('./roughness_factor/r.png', dpi=300, bbox_inches='tight')
plt.show()