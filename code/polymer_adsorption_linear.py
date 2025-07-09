import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import re
from mpl_toolkits.axes_grid1.inset_locator import inset_axes    # plot in plot
from matplotlib.patches import ConnectionPatch  # connection line for plot in plot

def load_linear_polymer_adsorption_data(directory: str = '.') -> dict:
    """
    Load linear polymer adsorption CSV files into a nested dictionary.
    Structure: data[xp][cl][v][wgh][pw]
    """
    data = {}
    pattern = re.compile(r'polymer_adsorption_xp([\d.]+)_cl(\d+)_v(\d+)_wgh(\d+)_pw(\d+)\.csv')

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            match = pattern.match(filename)
            if match:
                xp, cl, v, wgh, pw = match.groups()
                filepath = os.path.join(directory, filename)

                # Build nested dictionary
                if xp not in data:
                    data[xp] = {}
                if cl not in data[xp]:
                    data[xp][cl] = {}
                if v not in data[xp][cl]:
                    data[xp][cl][v] = {}
                if wgh not in data[xp][cl][v]:
                    data[xp][cl][v][wgh] = {}

                with open(filepath, 'r') as f:
                    f.readline()  # Skip title
                    header_line = f.readline().strip()
                    column_names = header_line.replace('#', '').strip().split()

                df = pd.read_csv(filepath, skiprows=2, header=None, names=column_names, sep=r'\s+', skipinitialspace=True)
                data[xp][cl][v][wgh][pw] = df

    if not data:
        print(f"Warning: No valid data files found in {directory}")
    return data

def get_data_for_params(data, params):
    try:
        current_data = data
        for param in ['xp', 'cl', 'v', 'wgh', 'pw']:
            if param in params and params[param] in current_data:
                current_data = current_data[params[param]]
            else:
                print(f"Warning: {param}={params.get(param)} not found. Available: {list(current_data.keys())}")
                return None
        return current_data
    except Exception as e:
        print(f"Error accessing data: {str(e)}")
        return None

def plot_adsorption_comparison(data, comparison_type, fixed_params, varying_params,
                              adsorption_types: list = ['n_ads_surface', 'n_ads_top', 'n_ads_sides', 'n_ads_edges'],
                              save_dir="./polymer_adsorption", max_time=600):

    for ads_type in adsorption_types:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Background shading for stages
        spreading = plt.axvspan(0, 46, color='lightblue', alpha=0.3, label='Spreading')
        retracting = plt.axvspan(46, 100, color='gold', alpha=0.3, label='Retracting')
        hopping = plt.axvspan(100, 300, color='lightcoral', alpha=0.3, label='Hopping')
        shaded_regions = [spreading, retracting, hopping]

        data_lines = []

        if comparison_type == 'cl':
            for cl in varying_params:
                params = fixed_params.copy()
                params['cl'] = cl
                dataset = get_data_for_params(data, params)
                if dataset is not None and ads_type in dataset:
                    line, = plt.plot(dataset['Time'], dataset[ads_type], label=f'cl={cl}', linewidth=2)
                    data_lines.append(line)
            title = f"xp={fixed_params['xp']}, v={fixed_params['v']}, A_pw={fixed_params['pw']}, wgh={fixed_params['wgh']}"

        elif comparison_type == 'pw':
            for pw in varying_params:
                params = fixed_params.copy()
                params['pw'] = pw
                dataset = get_data_for_params(data, params)
                if dataset is not None and ads_type in dataset:
                    line, = plt.plot(dataset['Time'], dataset[ads_type], label=f'pw={pw}', linewidth=2)
                    data_lines.append(line)
            title = f"xp={fixed_params['xp']}, cl={fixed_params['cl']}, v={fixed_params['v']}, wgh={fixed_params['wgh']}"

        elif comparison_type == 'wgh':
            for wgh in varying_params:
                params = fixed_params.copy()
                params['wgh'] = wgh
                dataset = get_data_for_params(data, params)
                if dataset is not None and ads_type in dataset:
                    line, = plt.plot(dataset['Time'], dataset[ads_type], label=f'wgh={wgh}', linewidth=2)
                    data_lines.append(line)
            title = f"xp={fixed_params['xp']}, cl={fixed_params['cl']}, v={fixed_params['v']}, A_pw={fixed_params['pw']}"

        # Labels and ticks
        plt.xlabel("Time", fontsize=16)
        plt.ylabel(f"Number of Adsorbed Beads ({ads_type})", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tick_params(axis='both', which='both', top=True, right=True, direction='in')
        # Custom y-ticks based on adsorption type
        if ads_type == 'n_ads_edges':
            plt.yticks(range(0, 1001, 100))
        elif ads_type == 'n_ads_surface':
            plt.yticks(range(0, 2001, 200))
        plt.xlim(0, max_time)
        plt.title(f"{title}\nAdsorption Type: {ads_type}")

        # Inset for n_ads_surface, zoom on all curves
        if ads_type == 'n_ads_surface':
            inset_ax = inset_axes(ax, width=3, height=1.5, loc='center', bbox_to_anchor=(0.33, 0.72), bbox_transform=ax.transAxes)
            for line in data_lines:
                inset_ax.plot(line.get_xdata(), line.get_ydata(),
                              color=line.get_color(), linewidth=2)  # Match colors
            inset_ax.set_xlim(0, 100)
            inset_ax.set_ylim(0, 120)
            inset_ax.set_xticks(range(0, 101, 25))
            inset_ax.set_yticks(range(0, 121, 20))
            inset_ax.tick_params(axis='both', labelsize=10)

            # Background shading for inset
            inset_ax.axvspan(0, 46, color='lightblue', alpha=0.3)  # Spreading
            inset_ax.axvspan(46, 100, color='gold', alpha=0.3)  # Retraction
            # Hopping (100–300) is outside inset range, so omitted

            # Add zoom indicator: rectangle and connection lines
            rect = plt.Rectangle((0, 0), 100, 100, fill=False, edgecolor='black', linestyle='--')
            ax.add_patch(rect)
            con1 = ConnectionPatch(xyA=(100, 100), xyB=(100, 0), coordsA="data", coordsB="data",
                                   axesA=ax, axesB=inset_ax, color="black", linestyle='--')
            con2 = ConnectionPatch(xyA=(0, 100), xyB=(0, 0), coordsA="data", coordsB="data",
                                   axesA=ax, axesB=inset_ax, color="black", linestyle='--')
            ax.add_artist(con1)
            ax.add_artist(con2)

        # Legends
        leg1 = ax.legend(data_lines, [l.get_label() for l in data_lines], loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False, title='Surface Type')
        ax.add_artist(leg1)
        ax.legend(shaded_regions, ['Spreading', 'Retracting', 'Hopping'], loc='upper right', bbox_to_anchor=(0.85, 1.0), frameon=False, title='Time Regions')

        # Save figure
        plt.tight_layout()
        filename_parts = [f"ads_{comparison_type}_{ads_type}"]
        for key, value in fixed_params.items():
            if key != comparison_type:
                filename_parts.append(f"{key}{value}")
        filename = "_".join(filename_parts) + ".png"

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Figure saved as {filename}")
        plt.close()

        # Save figure
        #plt.tight_layout()
        #filename = f"linear_ads_{comparison_type}_{ads_type}_xp{fixed_params['xp']}_cl{fixed_params['cl']}_v{fixed_params['v']}_pw{fixed_params['pw']}.png"
        #os.makedirs(save_dir, exist_ok=True)
        #plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        #print(f"Figure saved as {filename}")
        #plt.close()

if __name__ == "__main__":
    data_dir = os.path.join(".", "polymer_adsorption", "linear", "sim5")
    adsorption_data = load_linear_polymer_adsorption_data(data_dir)

    fixed_params = {
        'xp': '0.02',
        'cl': '50',
        'v': '2',
        'pw': '30',
        #'wgh': '222',
    }
    varying_wgh = ['000', '222', '242', '262', '282']
    #varying_pw = ['10', '30', '60', '90', '120', '150', '200']

    # Plot for different adsorption types
    plot_adsorption_comparison(adsorption_data, 'wgh', fixed_params, varying_wgh)