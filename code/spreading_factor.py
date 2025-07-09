import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import re
import numpy as np
from typing import Dict, List, Tuple, Union


class DataValidator:
    """
    Validates the relationships between parameters in the branched polymer system.
    """
    TOTAL_PARTICLES = 100000

    @staticmethod
    def validate_polymer_particles(xp: str, Np: str, BpP: str) -> bool:
        """
        Validates if xp, Np, and BpP follow the relationship:
        BpP = (xp * TOTAL_PARTICLES) / Np
        """
        xp_float = float(xp)
        Np_int = int(Np)
        BpP_int = int(BpP)
        expected_BpP = (xp_float * DataValidator.TOTAL_PARTICLES) / Np_int
        return abs(BpP_int - expected_BpP) < 1  # Allow for small rounding differences

    @staticmethod
    def validate_branch_parameters(BpP: str, mcl: str, bl: str, nb: str) -> bool:
        """
        Validates if BpP = mcl + bl*nb
        """
        return int(BpP) == int(mcl) + int(bl) * int(nb)


def load_data_files(directory: str = '.') -> Dict:
    """
    Load all CSV files from the specified directory and return a nested dictionary.
    Structure: data[xp][Np][mcl][bl_nb][v][wgh][pw]
    """
    data = {}
    pattern = re.compile(
        r'dropsize_xp([\d.]+)_Np(\d+)_BpP(\d+)_hawk_mcl(\d+)_bl(\d+)_nb(\d+)_v(\d+)_wgh(\d+)_pw(\d+)\.csv')

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


def calculate_spreading_factor(data: Dict) -> Dict:
    """
    Calculate spreading factor for each dataset.
    Maintains the same nested dictionary structure as input.
    """
    spreading_factors = {}

    for xp in data:
        spreading_factors[xp] = {}
        for Np in data[xp]:
            spreading_factors[xp][Np] = {}
            for mcl in data[xp][Np]:
                spreading_factors[xp][Np][mcl] = {}
                for bl_nb in data[xp][Np][mcl]:
                    spreading_factors[xp][Np][mcl][bl_nb] = {}
                    for v in data[xp][Np][mcl][bl_nb]:
                        spreading_factors[xp][Np][mcl][bl_nb][v] = {}
                        for wgh in data[xp][Np][mcl][bl_nb][v]:
                            spreading_factors[xp][Np][mcl][bl_nb][v][wgh] = {}
                            for pw in data[xp][Np][mcl][bl_nb][v][wgh]:
                                df = data[xp][Np][mcl][bl_nb][v][wgh][pw]

                                initial_diameter = df['Radius'].iloc[:5].mean() * 2
                                spreading_factor = 2 * df['Base_radius'] / initial_diameter

                                spreading_factors[xp][Np][mcl][bl_nb][v][wgh][pw] = pd.DataFrame({
                                    'Time': df['Time'],
                                    'Spreading_Factor': spreading_factor
                                })

    return spreading_factors
### end of function calculate_spreading_factor

def get_data_for_params(spreading_factors, params):
    """Helper function to safely navigate the nested dictionary"""
    try:
        current_data = spreading_factors
        # Update the parameter order to match the actual data structure
        param_order = ['xp', 'Np', 'mcl', 'bl_nb', 'v', 'wgh', 'pw']

        for param_name in param_order:
            if param_name in params:
                if params[param_name] not in current_data:
                    print(f"Warning: {param_name}={params[param_name]} not found in data")
                    #print(f"Available {param_name} values: {list(current_data.keys())}")
                    return None
                current_data = current_data[params[param_name]]
        return current_data
    except Exception as e:
        print(f"Error accessing data with params: {params}")
        print(f"Error details: {str(e)}")
        return None


def plot_comparison(spreading_factors, comparison_type, fixed_params, varying_params,
                    save_dir="./spreading_factor"):
    """
    Plot spreading factor for multiple datasets, varying one parameter while keeping others fixed.
    Comparison types: 'pw', 'wgh', or 'bl_nb'
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # First add the background shading
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
            data = get_data_for_params(spreading_factors, params)
            if data is not None:
                line, = plt.plot(data['Time'], data['Spreading_Factor'],
                                 label=f'bl_nb={bl_nb}', linewidth=2)
                data_lines.append(line)
        title = f"XP={fixed_params['xp']}, Np={fixed_params['Np']}, MCL={fixed_params['mcl']}"

    elif comparison_type == 'pw':
        # Plot different pw values
        for pw in varying_params:
            params = fixed_params.copy()
            params['pw'] = pw
            data = get_data_for_params(spreading_factors, params)
            if data is not None:
                line, = plt.plot(data['Time'], data['Spreading_Factor'],
                                 label=f'pw={pw}', linewidth=2)
                data_lines.append(line)
        title = f"XP={fixed_params['xp']}, Np={fixed_params['Np']}, MCL={fixed_params['mcl']}, bl_nb={fixed_params['bl_nb']}"

    elif comparison_type == 'wgh':
        # Plot different wgh values
        for wgh in varying_params:
            params = fixed_params.copy()
            params['wgh'] = wgh
            data = get_data_for_params(spreading_factors, params)
            if data is not None:
                line, = plt.plot(data['Time'], data['Spreading_Factor'],
                                 label=f'wgh={wgh}', linewidth=2)
                data_lines.append(line)
        title = f"XP={fixed_params['xp']}, Np={fixed_params['Np']}, MCL={fixed_params['mcl']}, bl_nb={fixed_params['bl_nb']}, pw={fixed_params['pw']}"

    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Spreading Factor", fontsize=16)
    plt.xticks(np.arange(0, 650, 50), fontsize=14)
    plt.yticks(np.arange(0, 3, 0.5), fontsize=14)
    plt.tick_params(axis='both', which='both', top=True, right=True, direction='in')

    # Set x-axis limits exactly to data range
    plt.xlim(0, 600)

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
        filename = f"beta_{comparison_type}_xp{fixed_params['xp']}_Np{fixed_params['Np']}_mcl{fixed_params['mcl']}_blnb{fixed_params['bl_nb']}_v{fixed_params['v']}_wgh{fixed_params['wgh']}.png"
    elif comparison_type == 'wgh':
        filename = f"beta_{comparison_type}_xp{fixed_params['xp']}_Np{fixed_params['Np']}_mcl{fixed_params['mcl']}_blnb{fixed_params['bl_nb']}_v{fixed_params['v']}_pw{fixed_params['pw']}.png"
    else:
        filename = f"beta_{comparison_type}_xp{fixed_params['xp']}_Np{fixed_params['Np']}_mcl{fixed_params['mcl']}_v{fixed_params['v']}_wgh{fixed_params['wgh']}_pw{fixed_params['pw']}.png"

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Figure saved as {filename} in {save_dir}")

    plt.close()
### end of plot comparison spreading factor function

"""
def plot_3d_spreading_factor(spreading_factors, fixed_params, varying_params, emphasize_pw=None, emphasize_wgh=None,
                             save_dir="./spreading_factor_plot"):
    #Create a 3D line plot with optional emphasis on specific pw or wgh values.
    #Simplified legends to show only one entry per emphasized value.
    
    from mpl_toolkits.mplot3d import Axes3D

    if emphasize_pw is not None and emphasize_wgh is not None:
        print("Warning: Both emphasize_pw and emphasize_wgh are specified. Using emphasize_wgh.")
        emphasize_pw = None

    # Get sample data and setup
    sample_params = fixed_params.copy()
    sample_params.update({'pw': varying_params['pw'][0], 'wgh': varying_params['wgh'][0]})
    sample_data = get_data_for_params(spreading_factors, sample_params)
    if sample_data is None:
        print("Error: Could not get sample data")
        return

    time_points = sample_data['Time'].to_numpy()
    pw_values = np.array([float(pw) for pw in varying_params['pw']])

    # Create figure
    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax = fig.add_subplot(111, projection='3d')

    # Define colors
    main_colors = ['royalblue', 'red', 'chocolate', 'magenta', 'lime']
    background_colors = ['#ADD8E6', '#FFB6C1', '#DEB887', '#FFE4E1', '#90EE90']

    if emphasize_pw is not None:
        # Plot non-emphasized curves
        for wgh_idx, wgh in enumerate(varying_params['wgh']):
            for pw in pw_values:
                if str(int(pw)) != emphasize_pw:
                    params = fixed_params.copy()
                    params['pw'] = str(int(pw))
                    params['wgh'] = wgh
                    data = get_data_for_params(spreading_factors, params)
                    if data is not None:
                        ax.plot(time_points, np.full_like(time_points, pw),
                                data['Spreading_Factor'].to_numpy(),
                                color=background_colors[wgh_idx], alpha=0.15, linewidth=0.5)

        # Plot emphasized curves - one label per wgh
        for wgh_idx, wgh in enumerate(varying_params['wgh']):
            params = fixed_params.copy()
            params['pw'] = emphasize_pw
            params['wgh'] = wgh
            data = get_data_for_params(spreading_factors, params)
            if data is not None:
                # Simplified label without pw value
                ax.plot(time_points, np.full_like(time_points, float(emphasize_pw)),
                        data['Spreading_Factor'].to_numpy(),
                        color=main_colors[wgh_idx], alpha=1.0, linewidth=2.5,
                        label=f'Surface: wgh={wgh}')

    elif emphasize_wgh is not None:
        # Plot non-emphasized curves
        for wgh_idx, wgh in enumerate(varying_params['wgh']):
            if wgh != emphasize_wgh:
                for pw in pw_values:
                    params = fixed_params.copy()
                    params['pw'] = str(int(pw))
                    params['wgh'] = wgh
                    data = get_data_for_params(spreading_factors, params)
                    if data is not None:
                        ax.plot(time_points, np.full_like(time_points, pw),
                                data['Spreading_Factor'].to_numpy(),
                                color=background_colors[wgh_idx], alpha=0.15, linewidth=0.5)

        # Plot emphasized curves - only one label for the emphasized wgh
        wgh_idx = varying_params['wgh'].index(emphasize_wgh)
        label_added = False
        for pw in pw_values:
            params = fixed_params.copy()
            params['pw'] = str(int(pw))
            params['wgh'] = emphasize_wgh
            data = get_data_for_params(spreading_factors, params)
            if data is not None:
                # Only add label for the first curve
                if not label_added:
                    ax.plot(time_points, np.full_like(time_points, pw),
                            data['Spreading_Factor'].to_numpy(),
                            color=main_colors[wgh_idx], alpha=1.0, linewidth=2.5,
                            label=f'Surface: wgh={emphasize_wgh}')
                    label_added = True
                else:
                    ax.plot(time_points, np.full_like(time_points, pw),
                            data['Spreading_Factor'].to_numpy(),
                            color=main_colors[wgh_idx], alpha=1.0, linewidth=2.5)

    else:
        # Plot all curves - one label per wgh
        for wgh_idx, wgh in enumerate(varying_params['wgh']):
            label_added = False
            for pw in pw_values:
                params = fixed_params.copy()
                params['pw'] = str(int(pw))
                params['wgh'] = wgh
                data = get_data_for_params(spreading_factors, params)
                if data is not None:
                    if not label_added:
                        ax.plot(time_points, np.full_like(time_points, pw),
                                data['Spreading_Factor'].to_numpy(),
                                color=main_colors[wgh_idx], alpha=0.7, linewidth=1.5,
                                label=f'Surface: wgh={wgh}')
                        label_added = True
                    else:
                        ax.plot(time_points, np.full_like(time_points, pw),
                                data['Spreading_Factor'].to_numpy(),
                                color=main_colors[wgh_idx], alpha=0.7, linewidth=1.5)

    # Customize the plot
    ax.set_xlabel('Time', labelpad=20)
    ax.set_ylabel('Polymer-Wall Interaction (pw)', labelpad=20)
    ax.set_zlabel('Spreading Factor', labelpad=20)

    y_ticks = np.arange(0, 220, 20)
    ax.set_yticks(y_ticks)

    ax.invert_xaxis()
    ax.invert_yaxis()

    title = f"XP={fixed_params['xp']}, Np={fixed_params['Np']}, MCL={fixed_params['mcl']}\nbl_nb={fixed_params['bl_nb']}"
    plt.title(title)

    ax.legend(frameon=False)
    ax.view_init(elev=30, azim=120)

    # Save figure
    emphasis_suffix = ""
    if emphasize_pw is not None:
        emphasis_suffix = f"_emphasis_pw{emphasize_pw}"
    elif emphasize_wgh is not None:
        emphasis_suffix = f"_emphasis_wgh{emphasize_wgh}"

    filename = f"3D_spreading_factor_xp{fixed_params['xp']}_Np{fixed_params['Np']}_mcl{fixed_params['mcl']}_blnb{fixed_params['bl_nb']}{emphasis_suffix}.png"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight', pad_inches=1.0)
    plt.close()
### end of plot 3d spreading factor function
"""

# Example usage
if __name__ == "__main__":
    data_dir = os.path.join(".", "dropsize", "branched")
    dropsize_data = load_data_files(data_dir)
    spreading_factors = calculate_spreading_factor(dropsize_data)

    # Plot comparison: varying one parameter while keeping other parameters fixed
    fixed_params = {
        'xp': '0.02',
        'Np': '20',
        'mcl': '50',
        'bl_nb': 'bl10_nb5',
        #'bl_nb': 'bl5_nb10',
        'v': '2',
        'pw': '90',
        #'wgh': '262'
    }

    # Use actual bl_nb combinations from your data
    #varying_pw = ['10', '30', '60', '90', '120', '150', '200']
    varying_wgh = ['000', '222', '242', '262', '282']

    plot_comparison(spreading_factors, 'wgh', fixed_params, varying_wgh)  # here change comparison type and varying parameter




    """
    # Set up parameters for 3D plot
    fixed_params = {
        'xp': '0.06',
        'Np': '20',
        'mcl': '50',
        'bl_nb': 'bl10_nb25',
        'v': '2'
    }

    varying_params = {
        'pw': ['10', '30', '60', '90', '120', '150', '200'],
        'wgh': ['000', '222', '242', '262', '282']
    }

    # Print the structure of the first few levels of data
    print("\nData structure check:")
    for xp in list(spreading_factors.keys())[:1]:
        for Np in list(spreading_factors[xp].keys())[:1]:
            for mcl in list(spreading_factors[xp][Np].keys())[:1]:
                print(f"Sample data structure for xp={xp}, Np={Np}, mcl={mcl}:")
                sample_data = spreading_factors[xp][Np][mcl]
                print(f"Branch config at this level: {list(sample_data.keys())}")

    #plot_3d_spreading_factor(spreading_factors, fixed_params, varying_params, emphasize_wgh='282')
"""


