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


from scipy.signal import savgol_filter


def calculate_contact_line_velocity(spreading_factors: Dict) -> Dict:
    """
    Calculate the contact line velocity (derivative of spreading factor with respect to time)
    for each dataset. Uses adaptive time range detection for retraction phase and
    applies Savitzky-Golay smoothing to handle oscillations.
    """
    contact_velocities = {}
    retraction_velocities = {}  # Store retraction velocity metrics

    # Approximate time regions based on your previous description (used as initial guidance)
    spreading_time_approx = (0, 46)
    retraction_time_approx = (46, 80)
    hopping_time_approx = (80, 600)

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

                        # Extract data
                        time = sf_df['Time'].values
                        spreading_factor = sf_df['Spreading_Factor'].values

                        # 1. Apply Savitzky-Golay filter to smooth the data
                        # Window length must be odd and greater than polyorder
                        # Increasing window_length increases smoothing
                        window_length = min(21, len(spreading_factor) - 1)
                        if window_length % 2 == 0:  # Must be odd
                            window_length -= 1

                        if window_length > 3:  # Need at least 3 points for smoothing
                            spreading_factor_smoothed = savgol_filter(spreading_factor,
                                                                      window_length=window_length,
                                                                      polyorder=3)
                        else:
                            spreading_factor_smoothed = spreading_factor

                        # 2. Calculate velocity using gradient on smoothed data
                        velocity_raw = np.gradient(spreading_factor, time)
                        velocity_smoothed = np.gradient(spreading_factor_smoothed, time)

                        # 3. Apply another smoothing pass to velocity itself to further reduce oscillations
                        if window_length > 3:
                            velocity = savgol_filter(velocity_smoothed,
                                                     window_length=window_length,
                                                     polyorder=3)
                        else:
                            velocity = velocity_smoothed

                        # Store data in DataFrame
                        contact_velocities[xp][cl][v][wgh][pw] = pd.DataFrame({
                            'Time': time,
                            'Spreading_Factor': spreading_factor,
                            'Spreading_Factor_Smoothed': spreading_factor_smoothed,
                            'Contact_Line_Velocity_Raw': velocity_raw,
                            'Contact_Line_Velocity': velocity
                        })

                        # 4. Adaptive detection of retraction phase
                        # First, find when spreading transitions to retraction (should be after peak spreading)
                        # Look for the first significant negative velocity after the approximate spreading phase

                        # Define threshold for "significant" negative velocity
                        velocity_threshold = -0.01  # Adjust based on your data sensitivity
                        stability_threshold = 0.005  # Threshold for defining "stabilizes"
                        window_size = 5  # Number of consecutive points to check for stability

                        # Find max spreading factor (should occur near end of spreading phase)
                        max_sf_idx = np.argmax(spreading_factor_smoothed)

                        # Start looking for retraction after max spreading
                        potential_start_indices = np.where((time > time[max_sf_idx]) &
                                                           (velocity < velocity_threshold))[0]

                        # Default to approximate values if detection fails
                        retraction_start_idx = np.where(time >= retraction_time_approx[0])[0][0]
                        retraction_end_idx = np.where(time >= retraction_time_approx[1])[0][0]

                        # Find the first significant negative velocity after max spreading
                        if len(potential_start_indices) > 0:
                            retraction_start_idx = potential_start_indices[0]

                            # Find where velocity stabilizes (close to zero consistently)
                            stable_indices = []
                            for i in range(retraction_start_idx + 1, len(velocity) - window_size):
                                velocity_window = velocity[i:i + window_size]
                                # Check if all values in window are within stability threshold of zero
                                if np.all(np.abs(velocity_window) < stability_threshold):
                                    stable_indices.append(i)
                                    break

                            # If we found stable points, use the first one as the end of retraction
                            if stable_indices:
                                retraction_end_idx = stable_indices[0]
                            else:
                                # If no stability is found, use the approximate end time
                                retraction_end_idx = np.where(time >= retraction_time_approx[1])[0][0]

                        # Get the time range for detected retraction phase
                        retraction_start_time = time[retraction_start_idx]
                        retraction_end_time = time[retraction_end_idx]

                        # Extract velocities during detected retraction phase
                        retraction_mask = (time >= retraction_start_time) & (time <= retraction_end_time)
                        retraction_time_points = time[retraction_mask]
                        retraction_velocities_values = velocity[retraction_mask]

                        # Filter to only include negative velocities (actual retraction)
                        negative_velocity_mask = retraction_velocities_values < 0
                        negative_velocities = retraction_velocities_values[negative_velocity_mask]

                        # Calculate comprehensive metrics for retraction phase
                        if len(negative_velocities) > 0:
                            # Basic metrics (now with smoothed data)
                            max_retraction_velocity = np.min(negative_velocities)  # Most negative value
                            avg_retraction_velocity = np.mean(negative_velocities)  # Average of all negative values
                            median_retraction_velocity = np.median(
                                negative_velocities)  # Median (more robust to outliers)

                            # Percentile-based metrics (more robust for oscillating data)
                            p5_retraction_velocity = np.percentile(negative_velocities,
                                                                   5)  # 5th percentile (close to max)
                            p25_retraction_velocity = np.percentile(negative_velocities, 25)  # 25th percentile

                            retraction_velocities[xp][cl][v][wgh][pw] = {
                                'max_retraction_velocity': max_retraction_velocity,
                                'avg_retraction_velocity': avg_retraction_velocity,
                                'median_retraction_velocity': median_retraction_velocity,
                                'p5_retraction_velocity': p5_retraction_velocity,
                                'p25_retraction_velocity': p25_retraction_velocity,
                                'retraction_start_time': retraction_start_time,
                                'retraction_end_time': retraction_end_time,
                                'num_negative_points': len(negative_velocities)
                            }
                        else:
                            print(
                                f"Warning: No negative velocities in detected retraction phase for xp={xp}, cl={cl}, v={v}, wgh={wgh}, pw={pw}")
                            retraction_velocities[xp][cl][v][wgh][pw] = {
                                'max_retraction_velocity': 0,
                                'avg_retraction_velocity': 0,
                                'median_retraction_velocity': 0,
                                'p5_retraction_velocity': 0,
                                'p25_retraction_velocity': 0,
                                'retraction_start_time': retraction_start_time,
                                'retraction_end_time': retraction_end_time,
                                'num_negative_points': 0
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
    plt.grid(True, linestyle='--', alpha=0.7)
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
    for multiple surface types on the same graph. Includes information about adaptive time ranges.

    Parameters:
    - metric_type: 'max', 'avg', 'median', 'p5', or 'p25' to select which retraction velocity metric to plot
    - fixed_params: dictionary of fixed parameters (xp, cl, v)
    - surface_types: list of surface types (wgh values)
    - pw_values: list of polymer-wall interaction values to plot on x-axis
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Set the style
    plt.style.use('ggplot')

    # Define colors for different surface types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']

    # Determine which metric to use
    if metric_type == 'max':
        metric_key = 'max_retraction_velocity'
        y_label = 'Maximum Retraction Velocity'
        title_prefix = 'Maximum'
    elif metric_type == 'avg':
        metric_key = 'avg_retraction_velocity'
        y_label = 'Average Retraction Velocity'
        title_prefix = 'Average'
    elif metric_type == 'median':
        metric_key = 'median_retraction_velocity'
        y_label = 'Median Retraction Velocity'
        title_prefix = 'Median'
    elif metric_type == 'p5':
        metric_key = 'p5_retraction_velocity'
        y_label = '5th Percentile Retraction Velocity'
        title_prefix = '5th Percentile'
    elif metric_type == 'p25':
        metric_key = 'p25_retraction_velocity'
        y_label = '25th Percentile Retraction Velocity'
        title_prefix = '25th Percentile'
    else:
        raise ValueError(f"Unknown metric_type: {metric_type}. Expected 'max', 'avg', 'median', 'p5', or 'p25'.")

    # Create a second plot for time ranges
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    time_ranges = {}

    # Plot each surface type
    for i, wgh in enumerate(surface_types):
        x_values = []
        y_values = []
        start_times = []
        end_times = []
        point_counts = []

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

                # Also collect time range data for the second plot
                start_times.append(data['retraction_start_time'])
                end_times.append(data['retraction_end_time'])
                point_counts.append(data['num_negative_points'])

        if x_values:
            # Plot velocity data
            line, = ax.plot(x_values, y_values, '-',
                            color=colors[i % len(colors)],
                            marker=markers[i % len(markers)],
                            markersize=8,
                            linewidth=2,
                            label=f'Surface: wgh={wgh}')

            # Add value labels to each point
            for x, y in zip(x_values, y_values):
                ax.text(x, y + 0.005, f'{y:.3f}',
                        ha='center', va='bottom',
                        color=colors[i % len(colors)],
                        fontsize=9)

            # Plot time range data
            ax2.plot(x_values, start_times, '--',
                     color=colors[i % len(colors)],
                     marker='o',
                     linewidth=1.5,
                     label=f'Start: wgh={wgh}')

            ax2.plot(x_values, end_times, '-',
                     color=colors[i % len(colors)],
                     marker='s',
                     linewidth=1.5,
                     label=f'End: wgh={wgh}')

            # Store time ranges for reference
            time_ranges[wgh] = {
                'pw_values': x_values,
                'start_times': start_times,
                'end_times': end_times,
                'point_counts': point_counts
            }

    # Set up the velocity plot
    ax.set_xlabel('Polymer-Wall Interaction (pw)', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(f'{title_prefix} Retraction Velocity vs Polymer-Wall Interaction\nfor Different Surface Types',
                 fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', frameon=True)

    # Add details box
    details = f"Details: xp={fixed_params['xp']}, Np=20, mcl={fixed_params['cl']}, bl_nb=bl5_nb10, v={fixed_params['v']}"
    fig.figtext(0.5, 0.01, details, ha='center', fontsize=10,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Create and save the velocity figure
    filename = f"{metric_type}_retraction_velocity_vs_pw_multiple_surfaces_xp{fixed_params['xp']}_cl{fixed_params['cl']}_v{fixed_params['v']}.png"

    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Figure saved as {filename} in {save_dir}")

    # Set up the time range plot
    ax2.set_xlabel('Polymer-Wall Interaction (pw)', fontsize=14)
    ax2.set_ylabel('Time', fontsize=14)
    ax2.set_title(f'Adaptive Retraction Phase Time Ranges\nfor Different Surface Types', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='best', frameon=True)

    # Add the same details box
    fig2.figtext(0.5, 0.01, details, ha='center', fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    fig2.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Create and save the time range figure
    time_range_filename = f"retraction_time_ranges_xp{fixed_params['xp']}_cl{fixed_params['cl']}_v{fixed_params['v']}.png"
    fig2.savefig(os.path.join(save_dir, time_range_filename), dpi=300, bbox_inches='tight')
    print(f"Time range figure saved as {time_range_filename} in {save_dir}")

    plt.close(fig)
    plt.close(fig2)

    return time_ranges


# Example usage
if __name__ == "__main__":
    data_dir = os.path.join(".", "dropsize", "linear")
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


"""
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
"""