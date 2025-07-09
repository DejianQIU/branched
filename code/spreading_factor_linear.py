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
    Validates the relationships between parameters in the linear polymer system.
    """
    TOTAL_PARTICLES = 100000

    @staticmethod
    def validate_polymer_particles(xp: str, cl: str) -> bool:
        """
        Validates if xp and cl follow the expected relationship
        for linear polymers
        """
        # Add validation logic specific to linear polymers if needed
        return True


def load_data_files(directory: str = '.') -> Dict:
    """
    Load all CSV files from the specified directory and return a nested dictionary.
    Structure: data[xp][cl][v][wgh][pw]
    """
    data = {}
    pattern = re.compile(
        r'dropsize_xp([\d.]+)_cl(\d+)_v(\d+)_wgh(\d+)_pw(\d+)\.csv')

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            match = pattern.match(filename)
            if match:
                xp, cl, v, wgh, pw = match.groups()
                filepath = os.path.join(directory, filename)

                # Create nested dictionary structure
                if xp not in data:
                    data[xp] = {}
                if cl not in data[xp]:
                    data[xp][cl] = {}
                if v not in data[xp][cl]:
                    data[xp][cl][v] = {}
                if wgh not in data[xp][cl][v]:
                    data[xp][cl][v][wgh] = {}

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

                data[xp][cl][v][wgh][pw] = df

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
        for cl in data[xp]:
            spreading_factors[xp][cl] = {}
            for v in data[xp][cl]:
                spreading_factors[xp][cl][v] = {}
                for wgh in data[xp][cl][v]:
                    spreading_factors[xp][cl][v][wgh] = {}
                    for pw in data[xp][cl][v][wgh]:
                        df = data[xp][cl][v][wgh][pw]

                        initial_diameter = df['Radius'].iloc[:5].mean() * 2
                        spreading_factor = 2 * df['Base_radius'] / initial_diameter

                        spreading_factors[xp][cl][v][wgh][pw] = pd.DataFrame({
                            'Time': df['Time'],
                            'Spreading_Factor': spreading_factor
                        })

    return spreading_factors


def get_data_for_params(spreading_factors, params):
    """Helper function to safely navigate the nested dictionary"""
    try:
        current_data = spreading_factors
        # Update the parameter order to match the linear polymer data structure
        param_order = ['xp', 'cl', 'v', 'wgh', 'pw']

        for param_name in param_order:
            if param_name in params:
                if params[param_name] not in current_data:
                    print(f"Warning: {param_name}={params[param_name]} not found in data")
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
    Comparison types: 'pw' or 'wgh'
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

    if comparison_type == 'pw':
        # Plot different pw values
        for pw in varying_params:
            params = fixed_params.copy()
            params['pw'] = pw
            data = get_data_for_params(spreading_factors, params)
            if data is not None:
                line, = plt.plot(data['Time'], data['Spreading_Factor'],
                                 label=f'pw={pw}', linewidth=2)
                data_lines.append(line)
        title = f"XP={fixed_params['xp']}, CL={fixed_params['cl']}"

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
        title = f"XP={fixed_params['xp']}, CL={fixed_params['cl']}, pw={fixed_params['pw']}"

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
        filename = f"linear_{comparison_type}_xp{fixed_params['xp']}_cl{fixed_params['cl']}_v{fixed_params['v']}_wgh{fixed_params['wgh']}.png"
    elif comparison_type == 'wgh':
        filename = f"linear_{comparison_type}_xp{fixed_params['xp']}_cl{fixed_params['cl']}_v{fixed_params['v']}_pw{fixed_params['pw']}.png"

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Figure saved as {filename} in {save_dir}")

    plt.close()


# Example usage
if __name__ == "__main__":
    data_dir = os.path.join(".", "dropsize", "linear", "sim5")
    dropsize_data = load_data_files(data_dir)
    spreading_factors = calculate_spreading_factor(dropsize_data)

    # Plot comparison: varying one parameter while keeping other parameters fixed
    fixed_params = {
        'xp': '0.02',
        'cl': '50',
        'v': '2',
        'pw': '90',
    }

    # Example values - update with your actual values
    varying_wgh = ['000', '222', '242', '262', '282']

    plot_comparison(spreading_factors, 'wgh', fixed_params, varying_wgh)