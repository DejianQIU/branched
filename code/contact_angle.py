import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from typing import Dict

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
                    column_names = header_line.replace('#', '').strip().split()

                # Read data using pandas
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

def get_data_for_params(data: Dict, params: Dict) -> pd.DataFrame:
    """Helper function to safely navigate the nested dictionary"""
    try:
        current_data = data
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

def plot_contact_angle(data: Dict, comparison_type: str, fixed_params: Dict, varying_params: list,
                      smoothing_method: str = None, smoothing_params: Dict = None,
                      save_dir: str = "./contact_angle") -> None:
    """
    Plot contact angle for multiple datasets with time regions and optional smoothing.
    
    Args:
        data: Nested dictionary containing all the data
        comparison_type: Type of comparison ('pw', 'wgh', or 'bl_nb')
        fixed_params: Dictionary of fixed parametersl

        varying_params: List of values to vary for comparison
        smoothing_method: One of 'moving_avg', 'savgol', 'gaussian', or None
        smoothing_params: Dictionary of parameters for the chosen smoothing method
        save_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add time region shading
    region1 = plt.axvspan(0, 46, color='lightblue', alpha=0.3, label='Spreading')
    region2 = plt.axvspan(46, 100, color='gold', alpha=0.3, label='Retraction')
    region3 = plt.axvspan(100, 300, color='lightcoral', alpha=0.3, label='Hopping')

    # Store shaded regions for legend
    shaded_regions = [region1, region2, region3]
    region_labels = ['Spreading', 'Retraction', 'Hopping']
    
    data_lines = []  # Store data lines for legend
    
    if smoothing_params is None:
        smoothing_params = {}

    for param_value in varying_params:
        params = fixed_params.copy()
        params[comparison_type] = param_value
        df = get_data_for_params(data, params)
        
        if df is not None:
            contact_angle_data = df['contact_angle']
            
            # Apply smoothing if specified
            if smoothing_method == 'moving_avg':
                window = smoothing_params.get('window', 15)
                smoothed_data = contact_angle_data.rolling(window=window, center=True).mean()
            elif smoothing_method == 'savgol':
                window = smoothing_params.get('window', 11)
                poly_order = smoothing_params.get('poly_order', 2)
                smoothed_data = savgol_filter(contact_angle_data, window_length=window, polyorder=poly_order)
            elif smoothing_method == 'gaussian':
                sigma = smoothing_params.get('sigma', 2)
                smoothed_data = gaussian_filter1d(contact_angle_data, sigma=sigma)
            else:
                smoothed_data = contact_angle_data

            line, = plt.plot(df['Time'], smoothed_data, 
                           label=f'{comparison_type}={param_value}', linewidth=2)
            data_lines.append(line)

    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Contact Angle (degrees)", fontsize=16)
    plt.xticks(np.arange(0, 650, 50), fontsize=14)
    plt.yticks(np.arange(0, 180, 20), fontsize=14)
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

    # Set title based on fixed parameters
    title_parts = [f"{k.upper()}={v}" for k, v in fixed_params.items() if k != comparison_type]
    plt.title(", ".join(title_parts))
    
    plt.tight_layout()
    
    # Create filename
    smooth_suffix = f"_smooth_{smoothing_method}" if smoothing_method else ""
    filename = f"contact_angle_{comparison_type}"
    for k, v in fixed_params.items():
        if k != comparison_type:
            filename += f"_{k}{v}"
    filename += f"{smooth_suffix}.png"
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Figure saved as {filename}")
    plt.close()

# Example usage
if __name__ == "__main__":
    # Load data
    data_dir = "./dropsize"
    dropsize_data = load_data_files(data_dir)
    
    # Example: Plot varying pw while keeping other parameters fixed
    fixed_params = {
        'xp': '0.02',
        'Np': '20',
        'mcl': '50',
        'bl_nb': 'bl10_nb5',
        'v': '2',
        'wgh': '000'
    }
    
    varying_pw = ['10', '30', '60', '90', '120', '150', '200']
    
    # Example plots with different smoothing methods
    smoothing_methods = ['gaussian']  #'moving_avg', 'savgol',
    for method in smoothing_methods:
        smoothing_params = {
            #'moving_avg': {'window': 15},
            #'savgol': {'window': 11, 'poly_order': 2},
            'gaussian': {'sigma': 2}
        }
        
        plot_contact_angle(
            dropsize_data, 
            'pw',
            fixed_params, 
            varying_pw,
            smoothing_method=method,
            smoothing_params=smoothing_params.get(method, {})
        )
