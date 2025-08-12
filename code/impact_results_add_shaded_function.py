import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button

# Define pillar width, height and gap between pillars and height
w = 2
h = 2
g = np.linspace(2, 8, 4)

# color
normal_red = '#ef233c'
blue1 = '#0e518d'
blue2= '#1976D2'
navyblue = '#0D47A1'
cyan = '#b8f2e6'

# Define roughness factor and A_p/w values
new_roughness_factor = 4 * (w * h) / (w + g) ** 2
r = np.concatenate(([0], new_roughness_factor[::-1]))  # 0, 0.16, 0.25, 0.44, 1
Ap_w = [10, 30, 60, 90, 120, 150, 200]

# Define deposition points as (r, A_p/w) pairs
deposition_points = [
    (r[0], Ap_w[1]), (r[0], Ap_w[2]), (r[0], Ap_w[3]), (r[0], Ap_w[5]), (r[0], Ap_w[6]),
    (r[1], Ap_w[1]), (r[1], Ap_w[2]), (r[1], Ap_w[3]), (r[1], Ap_w[4]),
    (r[2], Ap_w[2]), (r[2], Ap_w[3]),
    (r[3], Ap_w[2]), (r[3], Ap_w[3]),
    (r[4], Ap_w[2])
]

# Define rebound with satellite droplet points
satellite_points = [
    #(r[2], Ap_w[5]), (r[2], Ap_w[6]),
    #(r[3], Ap_w[4]), (r[3], Ap_w[5]), (r[3], Ap_w[6])
]

# Create all possible data points
all_points = [(rv, aw) for rv in r for aw in Ap_w]

# Separate into categories
deposition_r = [p[0] for p in deposition_points]
deposition_ap_w = [p[1] for p in deposition_points]
satellite_r = [p[0] for p in satellite_points]
satellite_ap_w = [p[1] for p in satellite_points]
rebound_points = [pt for pt in all_points if pt not in deposition_points and pt not in satellite_points]
rebound_r = [pt[0] for pt in rebound_points]
rebound_ap_w = [pt[1] for pt in rebound_points]

# Setup figure and axis
fig, ax = plt.subplots(figsize=(12, 9))
plt.subplots_adjust(bottom=0.42)

# Initial boundary values
initial_upper_y = [133, 108, 102, 98, 76]
initial_lower_y = [17, 21, 39, 51, 49.5]
control_x = [0, 0.16, 0.25, 0.44, 1.0]
initial_left_edge = -0.017
initial_right_edge = 1.02

# Initial smoothed curve
smooth_x = np.linspace(initial_left_edge, initial_right_edge, 100)
upper_smooth = np.interp(smooth_x, control_x, initial_upper_y)
lower_smooth = np.interp(smooth_x, control_x, initial_lower_y)

fill_between = ax.fill_between(smooth_x, lower_smooth, upper_smooth,
                               color='green', alpha=0.2, zorder=0)
upper_line, = ax.plot(smooth_x, upper_smooth, 'g--', linewidth=3, zorder=0)
lower_line, = ax.plot(smooth_x, lower_smooth, 'g--', linewidth=3, zorder=0)

# Plot categorized points
ax.scatter(deposition_r, deposition_ap_w, color=normal_red, marker='s', s=100,
           label='Deposition', zorder=2)
ax.scatter(rebound_r, rebound_ap_w, color=navyblue, marker='o', s=100,
           label='Rebound', zorder=1)
ax.scatter(satellite_r, satellite_ap_w, color=cyan, marker='^', s=120,
           label='Rebound with satellite residual', zorder=3)

ax.set_xlabel('roughness factor r', fontsize=16)
ax.set_ylabel(r'$\mathrm{A_{p/w}}$', fontsize=16)
ax.legend(fontsize=14, loc='upper right', frameon=False)
ax.set_xlim(-0.1, 1.2)
ax.set_ylim(0, 210)
ax.set_xticks(r)
ax.set_xticklabels([f'{x:.2f}' for x in r], fontsize=16)
ax.set_yticks(Ap_w)
ax.set_yticklabels([str(y) for y in Ap_w], fontsize=16)

# Slider setup
axcolor = 'lightgoldenrodyellow'
upper_sliders = []
lower_sliders = []

for i, x_val in enumerate(control_x):
    ax_upper = plt.axes([0.15, 0.35 - 0.04 * i, 0.3, 0.03], facecolor=axcolor)
    ax_lower = plt.axes([0.6, 0.35 - 0.04 * i, 0.3, 0.03], facecolor=axcolor)

    slider_upper = Slider(ax_upper, f'Upper r={x_val:.2f}', 0, 210, valinit=initial_upper_y[i])
    slider_lower = Slider(ax_lower, f'Lower r={x_val:.2f}', 0, 210, valinit=initial_lower_y[i])

    upper_sliders.append(slider_upper)
    lower_sliders.append(slider_lower)

# Edge sliders with extended range
ax_left = plt.axes([0.15, 0.05, 0.3, 0.03], facecolor=axcolor)
slider_left = Slider(ax_left, 'Left Edge X', -0.2, 0.9, valinit=initial_left_edge)

ax_right = plt.axes([0.6, 0.05, 0.3, 0.03], facecolor=axcolor)
slider_right = Slider(ax_right, 'Right Edge X', 0.1, 1.2, valinit=initial_right_edge)

# Update function
def update(val):
    global fill_between
    upper_y_values = [s.val for s in upper_sliders]
    lower_y_values = [s.val for s in lower_sliders]
    left_edge = slider_left.val
    right_edge = slider_right.val

    smooth_x = np.linspace(left_edge, right_edge, 100)
    upper_smooth = np.interp(smooth_x, control_x, upper_y_values)
    lower_smooth = np.interp(smooth_x, control_x, lower_y_values)

    upper_line.set_xdata(smooth_x)
    lower_line.set_xdata(smooth_x)
    upper_line.set_ydata(upper_smooth)
    lower_line.set_ydata(lower_smooth)

    fill_between.remove()
    fill_between = ax.fill_between(smooth_x, lower_smooth, upper_smooth,
                                   color='green', alpha=0.2, zorder=0)
    fig.canvas.draw_idle()

# Attach update callbacks
for s in upper_sliders + lower_sliders:
    s.on_changed(update)

slider_left.on_changed(update)
slider_right.on_changed(update)

# Save button
save_ax = plt.axes([0.8, 0.01, 0.1, 0.04])
save_button = Button(save_ax, 'Save')

def save(event):
    # Collect values
    upper_y_values = [s.val for s in upper_sliders]
    lower_y_values = [s.val for s in lower_sliders]
    left_edge = slider_left.val
    right_edge = slider_right.val
    smooth_x = np.linspace(left_edge, right_edge, 100)
    upper_smooth = np.interp(smooth_x, control_x, upper_y_values)
    lower_smooth = np.interp(smooth_x, control_x, lower_y_values)

    # New clean figure
    export_fig, export_ax = plt.subplots(figsize=(12, 9))

    export_ax.fill_between(smooth_x, lower_smooth, upper_smooth, color='green', alpha=0.2, zorder=0)
    export_ax.plot(smooth_x, upper_smooth, 'g--', linewidth=3, zorder=0)
    export_ax.plot(smooth_x, lower_smooth, 'g--', linewidth=3, zorder=0)

    export_ax.scatter(deposition_r, deposition_ap_w, color=normal_red, marker='s', s=100,
                      label='Deposition', zorder=2)
    export_ax.scatter(rebound_r, rebound_ap_w, color=navyblue, marker='o', s=100,
                      label='Rebound', zorder=1)
    export_ax.scatter(satellite_r, satellite_ap_w, color=cyan, marker='^', s=120,
                      label='Rebound with satellite residual', zorder=3)

    export_ax.set_xlabel('roughness factor r', fontsize=16)
    export_ax.set_ylabel(r'$\mathrm{A_{p/w}}$', fontsize=16)
    export_ax.legend(fontsize=14, loc='upper right', frameon=False)
    export_ax.set_xlim(-0.1, 1.2)
    export_ax.set_ylim(0, 210)
    export_ax.set_xticks(r)
    export_ax.set_xticklabels([f'{x:.2f}' for x in r], fontsize=16)
    export_ax.set_yticks(Ap_w)
    export_ax.set_yticklabels([str(y) for y in Ap_w], fontsize=16)

    export_fig.savefig('../impact_result_custom_trend.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'impact_result_custom_trend.png'")
    plt.close(export_fig)

save_button.on_clicked(save)

# Print button
print_ax = plt.axes([0.6, 0.01, 0.1, 0.04])
print_button = Button(print_ax, 'Print Values')

def print_values(event):
    upper_y_values = [s.val for s in upper_sliders]
    lower_y_values = [s.val for s in lower_sliders]
    left_edge = slider_left.val
    right_edge = slider_right.val

    print("\nCurrent Values for Copy-Paste:")
    print("upper_x =", control_x)
    print("upper_y =", upper_y_values)
    print("lower_x =", control_x)
    print("lower_y =", lower_y_values)
    print("left_edge_x =", left_edge)
    print("right_edge_x =", right_edge)

print_button.on_clicked(print_values)

plt.show()
