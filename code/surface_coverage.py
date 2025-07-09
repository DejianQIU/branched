import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import re
import numpy as np
from typing import Dict, List, Tuple, Union

def load_surface_coverage_data(directory: str = '.') -> Dict:
    """
    Load all surface coverage CSV files from the specified directory and return a nested dictionary.
    Structure: data[xp][Np][mcl][bl_nb][v][wgh][pw]
    """
    data = {}
    pattern = re.compile(
        r'surface_coverage_xp([\d.]+)_Np(\d+)_BpP(\d+)_hawk_mcl(\d+)_bl(\d+)_nb(\d+)_v(\d+)_wgh(\d+)_pw(\d+)\.csv')

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            match = pattern.match(filename)
            if match:
                xp, Np, BpP, mcl, bl, nb, v, wgh, pw = match.groups()
                filepath = os.path.join(directory, filename)

                # Create nested dictionary structure
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

                # Read and process headers
                with open(filepath, 'r') as f:
                    f.readline()  # Skip first line (title)
                    header_line = f.readline().strip()
                    # Remove '#' and split on whitespace
                    column_names = header_line.replace('#', '').strip().split()

                # Read data using pandas with fixed-width format
                df = pd.read_csv(filepath,
                               skiprows=2,  # Skip first two lines
                               header=None,  # No header in data
                               names=column_names,  # Use our processed column names
                               sep=r'\s+',  # Split on whitespace
                               skipinitialspace=True)  # Skip extra spaces

                data[xp][Np][mcl][bl_nb][v][wgh][pw] = df

    if len(data) == 0:
        print(f"Warning: No valid data files found in {directory}")

    return data

def get_data_for_params(data, params):
    """Helper function to safely navigate the nested dictionary"""
    try:
        current_data = data
        # Update the parameter order to match the actual data structure
        param_order = ['xp', 'Np', 'mcl', 'bl_nb', 'v', 'wgh', 'pw']

        for param_name in param_order:
            if param_name in params:
                if params[param_name] not in current_data:
                    print(f"Warning: {param_name}={params[param_name]} not found in data")
                    print(f"Available {param_name} values: {list(current_data.keys())}")
                    return None
                current_data = current_data[params[param_name]]
        return current_data
    except Exception as e:
        print(f"Error accessing data with params: {params}")
        print(f"Error details: {str(e)}")
        return None

def plot_surface_coverage_comparison(data, comparison_type, fixed_params, varying_params,
                                   metric='ads_Pol', save_dir="./surface_coverage_plots"):
    """
    Plot selected metric vs time for multiple datasets, varying one parameter while keeping others fixed.
    
    Parameters:
    -----------
    metric : str
        One of 'coverage', 'Diameter', 'ads_Solv', 'ads_Pol'
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Add background shading for different stages
    region1 = plt.axvspan(0, 46, color='lightblue', alpha=0.3, label='Spreading')
    region2 = plt.axvspan(46, 100, color='gold', alpha=0.3, label='Retraction')
    region3 = plt.axvspan(100, 300, color='lightcoral', alpha=0.3, label='Hopping')

    # Store shaded regions for separate legend
    shaded_regions = [region1, region2, region3]
    region_labels = ['Spreading', 'Retraction', 'Hopping']

    data_lines = []  # Store data lines for separate legend

    if comparison_type == 'bl_nb':
        # Plot different bl_nb combinations
        for bl_nb in varying_params:
            params = fixed_params.copy()
            params['bl_nb'] = bl_nb
            dataset = get_data_for_params(data, params)
            if dataset is not None:
                line, = plt.plot(dataset['Time'], dataset[metric],
                        label=f'bl_nb={bl_nb}', linewidth=2)
                data_lines.append(line)
        title = f"XP={fixed_params['xp']}, Np={fixed_params['Np']}, MCL={fixed_params['mcl']}"

    elif comparison_type == 'pw':
        # Plot different pw values
        for pw in varying_params:
            params = fixed_params.copy()
            params['pw'] = pw
            dataset = get_data_for_params(data, params)
            if dataset is not None:
                line, = plt.plot(dataset['Time'], dataset[metric],
                        label=f'pw={pw}', linewidth=2)
                data_lines.append(line)
        title = f"XP={fixed_params['xp']}, Np={fixed_params['Np']}, MCL={fixed_params['mcl']}, bl_nb={fixed_params['bl_nb']}"

    elif comparison_type == 'wgh':
        # Plot different wgh values
        for wgh in varying_params:
            params = fixed_params.copy()
            params['wgh'] = wgh
            dataset = get_data_for_params(data, params)
            if dataset is not None:
                line, = plt.plot(dataset['Time'], dataset[metric],
                        label=f'wgh={wgh}', linewidth=2)
                data_lines.append(line)
        title = f"XP={fixed_params['xp']}, Np={fixed_params['Np']}, MCL={fixed_params['mcl']}, bl_nb={fixed_params['bl_nb']}, pw={fixed_params['pw']}"

    plt.xlabel("Time", fontsize=16)
    plt.ylabel(f"{metric.replace('_', ' ').title()}", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tick_params(axis='both', which='both', top=True, right=True, direction='in')

    # Create two legends
    # First legend for data curves (on the right)
    first_legend = plt.legend(data_lines, [l.get_label() for l in data_lines],
                              loc='upper right', frameon=False)

    # Add the first legend back to the plot
    ax.add_artist(first_legend)

    # Second legend for shaded regions (at the top middle)
    plt.legend(shaded_regions, region_labels,
               loc='upper center', frameon=False, title='Time Regions',
               bbox_to_anchor=(0.8, 0.98))

    plt.title(title)
    plt.tight_layout()

    # Create filename based on comparison type
    if comparison_type == 'pw':
        filename = f"{metric}_{comparison_type}_xp{fixed_params['xp']}_Np{fixed_params['Np']}_mcl{fixed_params['mcl']}_blnb{fixed_params['bl_nb']}_v{fixed_params['v']}_wgh{fixed_params['wgh']}.png"
    elif comparison_type == 'wgh':
        filename = f"{metric}_{comparison_type}_xp{fixed_params['xp']}_Np{fixed_params['Np']}_mcl{fixed_params['mcl']}_blnb{fixed_params['bl_nb']}_v{fixed_params['v']}_pw{fixed_params['pw']}.png"
    else:
        filename = f"{metric}_{comparison_type}_xp{fixed_params['xp']}_Np{fixed_params['Np']}_mcl{fixed_params['mcl']}_v{fixed_params['v']}_wgh{fixed_params['wgh']}_pw{fixed_params['pw']}.png"

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Figure saved as {filename}")

    plt.close()

"""
def plot_3d_surface_coverage(data, fixed_params, varying_params, metric='ads_Pol', emphasize_pw=None,
                           save_dir="./surface_coverage_plots"):

    #Create a 3D line plot of selected metric vs time with optional emphasis on a specific pw value.
    
    Parameters:
    -----------
    metric : str
        One of 'coverage', 'Diameter', 'ads_Solv', 'ads_Pol'

    from mpl_toolkits.mplot3d import Axes3D

    # First, get a sample data to determine time points
    sample_params = fixed_params.copy()
    sample_params.update({'pw': varying_params['pw'][0], 'wgh': varying_params['wgh'][0]})
    sample_data = get_data_for_params(data, sample_params)
    if sample_data is None:
        print("Error: Could not get sample data")
        return

    # Convert time points to numpy array
    time_points = sample_data['Time'].to_numpy()
    pw_values = np.array([float(pw) for pw in varying_params['pw']])

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors
    main_colors = ['royalblue', 'red', 'chocolate', 'magenta', 'lime']
    background_colors = ['#ADD8E6', '#FFB6C1', '#DEB887', '#FFE4E1', '#90EE90']

    if emphasize_pw is not None:
        # Plot background curves
        for wgh_idx, wgh in enumerate(varying_params['wgh']):
            for pw in pw_values:
                if str(int(pw)) != emphasize_pw:
                    params = fixed_params.copy()
                    params['pw'] = str(int(pw))
                    params['wgh'] = wgh
                    dataset = get_data_for_params(data, params)
                    if dataset is not None:
                        ax.plot(time_points, np.full_like(time_points, pw),
                               dataset[metric].to_numpy(),
                               color=background_colors[wgh_idx], alpha=0.15, linewidth=0.5)

        # Plot emphasized curves
        for wgh_idx, wgh in enumerate(varying_params['wgh']):
            params = fixed_params.copy()
            params['pw'] = emphasize_pw
            params['wgh'] = wgh
            dataset = get_data_for_params(data, params)
            if dataset is not None:
                ax.plot(time_points, np.full_like(time_points, float(emphasize_pw)),
                        dataset[metric].to_numpy(),
                        color=main_colors[wgh_idx], alpha=1.0, linewidth=2.5,
                        label=f'Surface: wgh={wgh} (pw={emphasize_pw})')
    else:
        # Plot all curves with same emphasis
        for wgh_idx, wgh in enumerate(varying_params['wgh']):
            for pw in pw_values:
                params = fixed_params.copy()
                params['pw'] = str(int(pw))
                params['wgh'] = wgh
                dataset = get_data_for_params(data, params)
                if dataset is not None:
                    if pw == float(varying_params['pw'][0]):
                        ax.plot(time_points, np.full_like(time_points, pw),
                               dataset[metric].to_numpy(),
                               color=main_colors[wgh_idx], alpha=0.7, linewidth=1.5,
                               label=f'Surface: wgh={wgh}')
                    else:
                        ax.plot(time_points, np.full_like(time_points, pw),
                               dataset[metric].to_numpy(),
                               color=main_colors[wgh_idx], alpha=0.7, linewidth=1.5)

    ax.set_xlabel('Time', labelpad=20)
    ax.set_ylabel('Polymer-Wall Interaction (pw)', labelpad=20)
    ax.set_zlabel(f'{metric.replace("_", " ").title()}', labelpad=20)

    y_ticks = np.arange(0, 220, 20)
    ax.set_yticks(y_ticks)

    ax.invert_xaxis()
    ax.invert_yaxis()

    title = f"XP={fixed_params['xp']}, Np={fixed_params['Np']}, MCL={fixed_params['mcl']}\nbl_nb={fixed_params['bl_nb']}"
    plt.title(title)

    ax.legend()
    ax.view_init(elev=30, azim=120)

    filename_suffix = f"_emphasis_pw{emphasize_pw}" if emphasize_pw is not None else ""
    filename = f"3D_{metric}_xp{fixed_params['xp']}_Np{fixed_params['Np']}_mcl{fixed_params['mcl']}_blnb{fixed_params['bl_nb']}{filename_suffix}.png"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', pad_inches=1.0)
    plt.close()
"""


if __name__ == "__main__":
    data_dir = os.path.join(".", "surface_coverage", "branched")
    coverage_data = load_surface_coverage_data(data_dir)

    # Example: Plot varying pw while keeping other parameters fixed
    fixed_params = {
        'xp': '0.02',
        'Np': '20',
        'mcl': '50',
        'bl_nb': 'bl5_nb10',
        'v': '2',
        #'pw': '120',
        'wgh': '222',
    }
    varying_pw = ['10', '30', '60', '90', '120', '150', '200']
    varying_wgh = ['000', '222', '242', '262', '282']
    
    # Plot ads_Pol vs time
    plot_surface_coverage_comparison(coverage_data, 'pw', fixed_params, varying_pw, metric='ads_Pol')


"""
    # Example: 3D plot
    fixed_params = {
        'xp': '0.02',
        'Np': '20',
        'mcl': '50',
        'bl_nb': 'bl5_nb10',
        'v': '2'
    }
    varying_params = {
        'pw': ['10', '30', '60', '90', '120', '150', '200'],
        'wgh': ['000', '222', '242', '262', '282']
    }
    #plot_3d_surface_coverage(coverage_data, fixed_params, varying_params, metric='ads_Pol')
"""