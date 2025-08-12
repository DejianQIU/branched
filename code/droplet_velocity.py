import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

def load_velocity_data(directory: str = '.') -> dict:
    data = {}
    pattern = re.compile(r'droplet_velocity_xp([\d.]+)_Np(\d+)_BpP(\d+)_mcl(\d+)_bl(\d+)_nb(\d+)_v(\d+)_wgh(\d+)_pw(\d+)\.csv')

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            match = pattern.match(filename)
            if match:
                xp, Np, BpP, mcl, bl, nb, v, wgh, pw = match.groups()
                filepath = os.path.join(directory, filename)

                # Nested dictionary structure
                if xp not in data:
                    data[xp] = {}
                if Np not in data[xp]:
                    data[xp][Np] = {}
                if mcl not in data[xp][Np]:
                    data[xp][Np][mcl] = {}
                bl_nb = f"bl{bl}_nb{nb}"
                if bl_nb not in data[xp][Np][mcl]:
                    data[xp][Np][mcl][bl_nb] = {}
                if v not in data[xp][Np][mcl][bl_nb]:
                    data[xp][Np][mcl][bl_nb][v] = {}
                if wgh not in data[xp][Np][mcl][bl_nb][v]:
                    data[xp][Np][mcl][bl_nb][v][wgh] = {}

                # Read the header manually to ensure correct column names
                with open(filepath, 'r') as f:
                    f.readline()  # Skip title
                    header_line = f.readline().strip()
                    column_names = header_line.replace('#', '').strip().split()

                # Load the data with sep=r'\s+' to handle multiple spaces
                df = pd.read_csv(filepath, skiprows=2, header=None, names=column_names, sep=r'\s+', skipinitialspace=True, index_col=False)
                data[xp][Np][mcl][bl_nb][v][wgh][pw] = df

    if not data:
        print(f"Warning: No valid velocity data files found in {directory}")
    return data

def get_data_for_params(data, params):
    """Safely get data for given parameters."""
    try:
        current_data = data
        param_order = ['xp', 'Np', 'mcl', 'bl_nb', 'v', 'wgh', 'pw']
        for param_name in param_order:
            if param_name in params:
                if params[param_name] not in current_data:
                    print(f"Warning: {param_name}={params[param_name]} not found. Available: {list(current_data.keys())}")
                    return None
                current_data = current_data[params[param_name]]
        return current_data
    except Exception as e:
        print(f"Error accessing data with params: {params}. Details: {str(e)}")
        return None

def plot_velocity_comparison(data, comparison_type, fixed_params, varying_params,
                             velocity_types: list = ['v_axi', 'v_rad'],
                             save_dir="../plot/droplet_velocity", max_time=600,
                             time_ranges: dict = None):
    for vel_type in velocity_types:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Shaded regions for stages (same as your polymer script)
        spreading = plt.axvspan(0, 46, color='lightblue', alpha=0.3, label='Spreading')
        retracting = plt.axvspan(46, 100, color='gold', alpha=0.3, label='Retraction')
        hopping = plt.axvspan(100, 300, color='lightcoral', alpha=0.3, label='Hopping')
        shaded_regions = [spreading, retracting, hopping]

        data_lines = []

        # Comparison logic
        if comparison_type == 'bl_nb':
            for bl_nb in varying_params:
                params = fixed_params.copy()
                params['bl_nb'] = bl_nb
                dataset = get_data_for_params(data, params)
                if dataset is not None and vel_type in dataset:
                    # Apply time range filter if specified
                    t_min = 0
                    t_max = max_time
                    if time_ranges and bl_nb in time_ranges:
                        t_min, t_max = time_ranges[bl_nb]
                    mask = (dataset['Time'] >= t_min) & (dataset['Time'] <= t_max)
                    line, = plt.plot(dataset['Time'][mask], dataset[vel_type][mask], label=f'bl_nb={bl_nb}',
                                     linewidth=2)
                    data_lines.append(line)
                title = f"xp={fixed_params['xp']}, Np={fixed_params['Np']}, mcl={fixed_params['mcl']}, v={fixed_params['v']}, wgh={fixed_params['wgh']}, pw={fixed_params['pw']}"

        elif comparison_type == 'pw':
            for pw in varying_params:
                params = fixed_params.copy()
                params['pw'] = pw
                dataset = get_data_for_params(data, params)
                if dataset is not None and vel_type in dataset:
                    # Apply time range filter if specified
                    t_min = 0
                    t_max = max_time
                    if time_ranges and pw in time_ranges:
                        t_min, t_max = time_ranges[pw]
                    mask = (dataset['Time'] >= t_min) & (dataset['Time'] <= t_max)
                    line, = plt.plot(dataset['Time'][mask], dataset[vel_type][mask], label=f'pw={pw}', linewidth=2)
                    data_lines.append(line)
                title = f"xp={fixed_params['xp']}, Np={fixed_params['Np']}, mcl={fixed_params['mcl']}, bl_nb={fixed_params['bl_nb']}, v={fixed_params['v']}, wgh={fixed_params['wgh']}"

        elif comparison_type == 'wgh':
            for wgh in varying_params:
                params = fixed_params.copy()
                params['wgh'] = wgh
                dataset = get_data_for_params(data, params)
                if dataset is not None and vel_type in dataset:
                    # Apply time range filter if specified
                    t_min = 0
                    t_max = max_time
                    if time_ranges and wgh in time_ranges:
                        t_min, t_max = time_ranges[wgh]
                    mask = (dataset['Time'] >= t_min) & (dataset['Time'] <= t_max)
                    line, = plt.plot(dataset['Time'][mask], dataset[vel_type][mask], label=f'wgh={wgh}', linewidth=2)
                    data_lines.append(line)
                title = f"xp={fixed_params['xp']}, Np={fixed_params['Np']}, mcl={fixed_params['mcl']}, bl_nb={fixed_params['bl_nb']}, v={fixed_params['v']}, pw={fixed_params['pw']}"

                # Inset for v_axi
                if vel_type == 'v_axi':
                    inset_ax = inset_axes(ax, width=3, height=2, loc='center', bbox_to_anchor=(0.33, 0.72),
                                          bbox_transform=ax.transAxes)
                    for line in data_lines:
                        inset_ax.plot(line.get_xdata(), line.get_ydata(),
                                      color=line.get_color(), linewidth=2)  # Match colors
                    inset_ax.set_xlim(25, 50)  # Updated x range
                    inset_ax.set_ylim(0, 100)  # Updated y range
                    inset_ax.set_xticks(range(25, 51, 5))  # Ticks every 5 from 25 to 50
                    inset_ax.set_yticks(range(0, 101, 20))  # Ticks every 20 from 0 to 100
                    inset_ax.tick_params(axis='both', labelsize=10)

                    # Background shading for inset
                    inset_ax.axvspan(25, 46, color='lightblue', alpha=0.3)  # Spreading (adjusted to fit 25-50)
                    inset_ax.axvspan(46, 50, color='gold', alpha=0.3)  # Retraction (adjusted to fit 25-50)

                    # Add zoom indicator: rectangle and connection lines
                    rect = plt.Rectangle((25, 0), 25, 100, fill=False, edgecolor='black',
                                         linestyle='--')  # Width = 50-25, height = 100-0
                    ax.add_patch(rect)
                    con1 = ConnectionPatch(xyA=(50, 100), xyB=(50, 0), coordsA="data", coordsB="data",
                                           axesA=ax, axesB=inset_ax, color="black", linestyle='--')
                    con2 = ConnectionPatch(xyA=(25, 100), xyB=(25, 0), coordsA="data", coordsB="data",
                                           axesA=ax, axesB=inset_ax, color="black", linestyle='--')
                    ax.add_artist(con1)
                    ax.add_artist(con2)

        # Labels and formatting
        plt.xlabel("Time", fontsize=16)
        plt.ylabel(f"Velocity ({vel_type})", fontsize=16)
        plt.xticks(range(0, max_time + 1, 50), fontsize=14)
        plt.yticks(fontsize=14)
        plt.tick_params(axis='both', which='both', top=True, right=True, direction='in')
        plt.xlim(0, max_time)
        plt.title(f"{title}\nVelocity Type: {vel_type}")

        # Legends
        leg1 = ax.legend(data_lines, [l.get_label() for l in data_lines], loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False, title='Surface Type')
        ax.add_artist(leg1)
        ax.legend(shaded_regions, ['Spreading', 'Retracting', 'Hopping'], loc='upper right', bbox_to_anchor=(0.85, 1.0), frameon=False, title='Time Regions')

        # Save plot
        plt.tight_layout()
        filename_parts = [f"vel_{comparison_type}_{vel_type}"]
        for key, value in fixed_params.items():
            if key != comparison_type:
                filename_parts.append(f"{key}{value}")
        filename = "_".join(filename_parts) + ".png"

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")
        plt.close()

# Example usage
if __name__ == "__main__":
    data_dir = os.path.join("..", "data", "droplet_velocity", "branched")
    velocity_data = load_velocity_data(data_dir)

    fixed_params = {
        'xp': '0.02',
        'Np': '20',
        'mcl': '50',
        'bl_nb': 'bl5_nb10',
        'v': '2',
        #'pw': '90',
        'wgh': '222',
    }
    #varying_wgh = ['000', '222', '242', '262', '282']
    varying_pw = ['10', '30', '60', '90', '120', '150', '200']
    new_range1= {
        '10': (0, 150),  # pw=10, plot t=0 to t=100
        '30': (0, 150),  # pw=30, plot t=0 to t=100
        '120': (0, 150),  # pw=30, plot t=0 to t=100
        '150': (0, 150),  # pw=30, plot t=0 to t=100
        '200': (0, 150),  # pw=30, plot t=0 to t=100
        # Others use default max_time (600)
    }

    plot_velocity_comparison(velocity_data, 'pw', fixed_params, varying_pw, time_ranges=new_range1)