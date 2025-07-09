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


def calculate_contact_line_velocity(spreading_factors: Dict) -> Dict:
    """
    Calculate the contact line velocity (derivative of spreading factor with respect to time)
    for each dataset. Maintains the same nested dictionary structure as input.

    Also calculate maximum and average retraction velocities during the retraction phase (time 46-80).
    """
    contact_velocities = {}
    retraction_velocities = {}  # Store retraction velocity metrics

    # Define time regions based on your description
    spreading_time = (0, 46)
    retraction_time = (46, 80)
    hopping_time = (80, 600)  # Assuming it goes to the end of your data

    for xp in spreading_factors:
        contact_velocities[xp] = {}
        retraction_velocities[xp] = {}

        for cl in spreading_factors[xp]:
            contact_velocities[xp][cl] = {}
            retraction_velocities[xp][cl] = {}

            for v in spreading_factors[xp][cl]:
                contact_velocities[xp][cl][v] = {}
                retraction_velocities[xp][cl][v] = {}

                for wgh in spreading_factors[xp][cl][v]:
                    contact_velocities[xp][cl][v][wgh] = {}
                    retraction_velocities[xp][cl][v][wgh] = {}

                    for pw in spreading_factors[xp][cl][v][wgh]:
                        sf_df = spreading_factors[xp][cl][v][wgh][pw]

                        # Calculate velocity using gradient function (central differences)
                        # This gives the rate of change of spreading factor with respect to time
                        time = sf_df['Time'].values
                        spreading_factor = sf_df['Spreading_Factor'].values

                        # Calculate velocity using numpy's gradient function
                        # This uses central differences for interior points and one-sided
                        # differences for the endpoints
                        velocity = np.gradient(spreading_factor, time)

                        # Create DataFrame with contact line velocity
                        contact_velocities[xp][cl][v][wgh][pw] = pd.DataFrame({
                            'Time': time,
                            'Spreading_Factor': spreading_factor,
                            'Contact_Line_Velocity': velocity
                        })

                        # Extract velocities during retraction phase
                        retraction_mask = (time >= retraction_time[0]) & (time <= retraction_time[1])
                        retraction_time_points = time[retraction_mask]
                        retraction_velocities_values = velocity[retraction_mask]

                        # Filter to only include negative velocities (actual retraction)
                        negative_velocity_mask = retraction_velocities_values < 0
                        negative_velocities = retraction_velocities_values[negative_velocity_mask]

                        # Calculate metrics for retraction phase
                        if len(negative_velocities) > 0:
                            # Find most negative value (maximum retraction velocity)
                            max_retraction_velocity = np.min(negative_velocities)
                            # Calculate mean of negative velocities (average retraction velocity)
                            avg_retraction_velocity = np.mean(negative_velocities)

                            retraction_velocities[xp][cl][v][wgh][pw] = {
                                'max_retraction_velocity': max_retraction_velocity,
                                'avg_retraction_velocity': avg_retraction_velocity
                            }
                        else:
                            print(
                                f"Warning: No negative velocities in retraction phase for xp={xp}, cl={cl}, v={v}, wgh={wgh}, pw={pw}")
                            retraction_velocities[xp][cl][v][wgh][pw] = {
                                'max_retraction_velocity': 0,
                                'avg_retraction_velocity': 0
                            }

    return contact_velocities, retraction_velocities


def get_data_for_params(data_dict, params):
    """Helper function to safely navigate the nested dictionary"""
    try:
        current_data = data_dict
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


def plot_contact_line_velocity(contact_velocities, comparison_type, fixed_params, varying_params,
                               save_dir="./contact_line_velocity"):
    """
    Plot contact line velocity for multiple datasets, varying one parameter while keeping others fixed.
    Comparison types: 'pw' or 'wgh'
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # First add the background shading for different regions
    region1 = plt.axvspan(0, 46, color='lightblue', alpha=0.3, label='Spreading')
    region2 = plt.axvspan(46, 80, color='gold', alpha=0.3, label='Retraction')
    region3 = plt.axvspan(80, 300, color='lightcoral', alpha=0.3, label='Hopping')

    # Store shaded regions for separate legend
    shaded_regions = [region1, region2, region3]
    region_labels = ['Spreading', 'Retraction', 'Hopping']

    data_lines = []  # Store data lines for separate legend

    if comparison_type == 'pw':
        # Plot different pw values
        for pw in varying_params:
            params = fixed_params.copy()
            params['pw'] = pw
            data = get_data_for_params(contact_velocities, params)
            if data is not None:
                line, = plt.plot(data['Time'], data['Contact_Line_Velocity'],
                                 label=f'pw={pw}', linewidth=2)
                data_lines.append(line)
        title = f"Contact Line Velocity: XP={fixed_params['xp']}, CL={fixed_params['cl']}, WGH={fixed_params['wgh']}"

    elif comparison_type == 'wgh':
        # Plot different wgh values
        for wgh in varying_params:
            params = fixed_params.copy()
            params['wgh'] = wgh
            data = get_data_for_params(contact_velocities, params)
            if data is not None:
                line, = plt.plot(data['Time'], data['Contact_Line_Velocity'],
                                 label=f'wgh={wgh}', linewidth=2)
                data_lines.append(line)
        title = f"Contact Line Velocity: XP={fixed_params['xp']}, CL={fixed_params['cl']}, PW={fixed_params['pw']}"

    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Contact Line Velocity", fontsize=16)
    plt.xticks(np.arange(0, 650, 50), fontsize=14)
    plt.tick_params(axis='both', which='both', top=True, right=True, direction='in')

    # Set x-axis limits exactly to data range
    plt.xlim(0, 600)

    # Add a horizontal line at y=0 to distinguish positive and negative velocities
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)

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
        filename = f"velocity_{comparison_type}_xp{fixed_params['xp']}_cl{fixed_params['cl']}_v{fixed_params['v']}_wgh{fixed_params['wgh']}.png"
    elif comparison_type == 'wgh':
        filename = f"velocity_{comparison_type}_xp{fixed_params['xp']}_cl{fixed_params['cl']}_v{fixed_params['v']}_pw{fixed_params['pw']}.png"

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Figure saved as {filename} in {save_dir}")

    plt.close()


def plot_retraction_velocity_comparison(retraction_velocities, metric_type, comparison_type,
                                        fixed_params, varying_params, save_dir="./retraction_velocity"):
    """
    Plot either maximum or average retraction velocity against the varying parameter.

    Parameters:
    - metric_type: 'max' or 'avg' to select which retraction velocity metric to plot
    - comparison_type: 'pw' or 'wgh' to determine which parameter is being varied
    - fixed_params: dictionary of fixed parameters
    - varying_params: list of values for the parameter being varied
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine which metric to use
    if metric_type == 'max':
        metric_key = 'max_retraction_velocity'
        y_label = 'Maximum Retraction Velocity (magnitude)'
        title_prefix = 'Maximum'
    else:  # 'avg'
        metric_key = 'avg_retraction_velocity'
        y_label = 'Average Retraction Velocity'
        title_prefix = 'Average'

    # Collect data for plotting
    x_values = []
    y_values = []

    for param_value in varying_params:
        params = fixed_params.copy()
        params[comparison_type] = param_value

        # Navigate to the data
        data = get_data_for_params(retraction_velocities, params)

        if data is not None:
            x_values.append(param_value)
            # Take the absolute value of the retraction velocity (convert negative to positive)
            y_values.append(abs(data[metric_key]))

    # Convert to numeric for proper plotting if they're strings
    if comparison_type == 'wgh' or comparison_type == 'pw':
        x_values_numeric = [int(x) for x in x_values]
        plt.plot(x_values_numeric, y_values, 'o-', linewidth=2)
        plt.xticks(x_values_numeric)
    else:
        plt.plot(x_values, y_values, 'o-', linewidth=2)

    # Add labels and title
    plt.xlabel(f"{'Ap/w (pw)' if comparison_type == 'pw' else 'Surface Type (wgh)'}", fontsize=16)
    plt.ylabel(y_label, fontsize=16)

    # Create title based on what's fixed and what's varying
    if comparison_type == 'pw':
        title = f"{title_prefix} Retraction Velocity vs Ap/w: XP={fixed_params['xp']}, CL={fixed_params['cl']}, WGH={fixed_params['wgh']}"
    elif comparison_type == 'wgh':
        title = f"{title_prefix} Retraction Velocity vs Surface Type: XP={fixed_params['xp']}, CL={fixed_params['cl']}, PW={fixed_params['pw']}"

    plt.title(title)
    #plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Create filename based on comparison and metric type
    filename = f"{metric_type}_retraction_velocity_vs_{comparison_type}_xp{fixed_params['xp']}_cl{fixed_params['cl']}_v{fixed_params['v']}"

    if comparison_type == 'pw':
        filename += f"_wgh{fixed_params['wgh']}.png"
    elif comparison_type == 'wgh':
        filename += f"_pw{fixed_params['pw']}.png"

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Figure saved as {filename} in {save_dir}")

    plt.close()


def plot_retraction_velocity_vs_pw_multiple_surfaces(retraction_velocities, metric_type,
                                                     fixed_params, surface_types, pw_values,
                                                     save_dir="./retraction_velocity"):
    """
    Plot either maximum or average retraction velocity against polymer-wall interaction (pw)
    for multiple surface types on the same graph.

    Parameters:
    - metric_type: 'max' or 'avg' to select which retraction velocity metric to plot
    - fixed_params: dictionary of fixed parameters (xp, cl, v)
    - surface_types: list of surface types (wgh values)
    - pw_values: list of polymer-wall interaction values to plot on x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set the style
    #plt.style.use('ggplot')

    # Define colors for different surface types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']

    # Determine which metric to use
    if metric_type == 'max':
        metric_key = 'max_retraction_velocity'
        y_label = 'Maximum Retraction Velocity'
        title_prefix = 'Maximum'
    else:  # 'avg'
        metric_key = 'avg_retraction_velocity'
        y_label = 'Average Retraction Velocity'
        title_prefix = 'Average'

    # Plot each surface type
    for i, wgh in enumerate(surface_types):
        x_values = []
        y_values = []

        for pw in pw_values:
            params = fixed_params.copy()
            params['wgh'] = wgh
            params['pw'] = pw

            # Navigate to the data
            data = get_data_for_params(retraction_velocities, params)

            if data is not None:
                x_values.append(int(pw))
                # Take the absolute value of retraction velocity (convert negative to positive)
                y_values.append(abs(data[metric_key]))

        if x_values:
            line, = plt.plot(x_values, y_values, '-',
                             color=colors[i % len(colors)],
                             marker=markers[i % len(markers)],
                             markersize=8,
                             linewidth=2,
                             label=f'Surface: wgh={wgh}')

            # Add value labels to each point
            for x, y in zip(x_values, y_values):
                plt.text(x, y + 0.005, f'{y:.3f}',
                         ha='center', va='bottom',
                         color=colors[i % len(colors)],
                         fontsize=9)

    # Set up the plot
    plt.xlabel('Polymer-Wall Interaction (pw)', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(f'{title_prefix} Retraction Velocity vs Polymer-Wall Interaction\nfor Different Surface Types',
              fontsize=16)
    #plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', frameon=True)

    # Add details box
    details = f"Details: xp={fixed_params['xp']}, Np=20, mcl={fixed_params['cl']}, bl_nb=bl5_nb10, v={fixed_params['v']}"
    plt.figtext(0.5, 0.01, details, ha='center', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Create and save the figure
    filename = f"{metric_type}_retraction_velocity_vs_pw_multiple_surfaces_xp{fixed_params['xp']}_cl{fixed_params['cl']}_v{fixed_params['v']}.png"

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Figure saved as {filename} in {save_dir}")

    plt.close()


# Example usage
if __name__ == "__main__":
    data_dir = os.path.join(".", "dropsize", "linear", "sim5")
    dropsize_data = load_data_files(data_dir)
    spreading_factors = calculate_spreading_factor(dropsize_data)

    # Calculate contact line velocities and retraction velocities
    contact_velocities, retraction_velocities = calculate_contact_line_velocity(spreading_factors)

    # Plot contact line velocity: varying one parameter while keeping other parameters fixed
    fixed_params = {
        'xp': '0.02',
        'cl': '50',
        'v': '2',
        'pw': '90',
    }

    # Example values - update with your actual values
    varying_wgh = ['000', '222', '242', '262', '282']

    # Plot contact line velocity for different surface types
    plot_contact_line_velocity(contact_velocities, 'wgh', fixed_params, varying_wgh)

    # Plot maximum retraction velocity vs surface type
    plot_retraction_velocity_comparison(retraction_velocities, 'max', 'wgh', fixed_params, varying_wgh)

    # Plot average retraction velocity vs surface type
    plot_retraction_velocity_comparison(retraction_velocities, 'avg', 'wgh', fixed_params, varying_wgh)

    # Now plot with fixed surface type and varying pw
    fixed_params_pw = {
        'xp': '0.02',
        'cl': '50',
        'v': '2',
        'wgh': '242',
    }

    # Example values for pw - update with your actual values
    varying_pw = ['10', '30', '50', '70', '90']

    # Plot contact line velocity for different pw values
    plot_contact_line_velocity(contact_velocities, 'pw', fixed_params_pw, varying_pw)

    # Plot maximum retraction velocity vs pw
    plot_retraction_velocity_comparison(retraction_velocities, 'max', 'pw', fixed_params_pw, varying_pw)

    # Plot average retraction velocity vs pw
    plot_retraction_velocity_comparison(retraction_velocities, 'avg', 'pw', fixed_params_pw, varying_pw)

    # Create combined plots with multiple surface types on one graph
    # Fixed parameters common for all
    fixed_params_base = {
        'xp': '0.02',
        'cl': '50',
        'v': '2',
    }

    # Define surface types and pw values
    surface_types = ['000', '222', '242', '262', '282']  # wgh values
    pw_values = ['10', '30', '60', '90', '120', '150', '200']  # pw values

    # Plot maximum retraction velocity vs pw for all surface types
    plot_retraction_velocity_vs_pw_multiple_surfaces(retraction_velocities, 'max',
                                                     fixed_params_base, surface_types, pw_values)

    # Plot average retraction velocity vs pw for all surface types
    plot_retraction_velocity_vs_pw_multiple_surfaces(retraction_velocities, 'avg',
                                                     fixed_params_base, surface_types, pw_values)