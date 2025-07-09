import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re


class PolymerStructureFactorAnalyzer:
    def __init__(self):
        # Set up paths
        self.script_dir = Path.cwd()
        self.data_dir = self.script_dir / "polymersmsf"
        self.branched_dir = self.data_dir / "branched"
        self.linear_dir = self.data_dir / "linear"
        self.output_dir = self.data_dir  # Save plots to polymersmsf directory

        # Make sure output directory exists, if not, make it
        self.output_dir.mkdir(exist_ok=True)

        # Dictionary to store all data
        self.data = {}

        # Load all data files
        self.load_all_data()

    def parse_filename(self, filename):
        """Parse polymer system parameters from filename for both formats."""
        # Try to match branched polymer format first
        branched_pattern = r'polymer_smsf_xp([\d.]+)_Np(\d+)_BpP(\d+)_mcl(\d+)_bl(\d+)_nb(\d+)_(\w+)'       # \d (0-9), \d+ (\d + more digits), [\d.]+ (\d+ + decimal points), \w+ one or more word characters
        branched_match = re.match(branched_pattern, filename)   # re.match checks if the pattern matches the start of the string

        if branched_match:
            return {
                'xp': float(branched_match.group(1)),
                'Np': int(branched_match.group(2)),
                'BpP': int(branched_match.group(3)),
                'mcl': int(branched_match.group(4)),
                'bl': int(branched_match.group(5)),
                'nb': int(branched_match.group(6)),
                'simulation_type': branched_match.group(7),
                'polymer_type': 'branched'
            }

        # Try to match linear polymer format
        linear_pattern = r'polymer_smsf_xp([\d.]+)_cl(\d+)_(\w+)'
        linear_match = re.match(linear_pattern, filename)

        if linear_match:
            return {
                'xp': float(linear_match.group(1)),
                'cl': int(linear_match.group(2)),  # Chain length for linear polymers
                'simulation_type': linear_match.group(3),
                'polymer_type': 'linear',
                # Set default values for parameters that don't apply to linear polymers
                'Np': None,
                'BpP': None,
                'mcl': None,
                'bl': None,
                'nb': None
            }
        # if no match then:
        return None

    def load_data_file(self, filepath):
        """Load individual data file."""
        try:
            data = pd.read_csv(filepath,
                               comment='#',
                               sep=r'\s+',
                               names=['q', 'S(q)'])
            return data
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return None

    def load_all_data(self):
        """Load all data files from both branched and linear directories."""
        # Process branched polymer data
        if self.branched_dir.exists():
            for file in self.branched_dir.glob('polymer_smsf_*.csv'):
                params = self.parse_filename(file.stem)
                if params:
                    data = self.load_data_file(file)
                    if data is not None:
                        key = self._create_key(params)
                        self.data[key] = {'params': params, 'data': data}

        # Process linear polymer data
        if self.linear_dir.exists():
            for file in self.linear_dir.glob('polymer_smsf_*.csv'):
                params = self.parse_filename(file.stem) # extract file name
                if params:  # if successfully get file name specs
                    data = self.load_data_file(file)    # load the actual data
                    if data is not None:
                        key = self._create_key(params)
                        self.data[key] = {'params': params, 'data': data}

    def _create_key(self, params):
        """Create a unique key for a dataset based on its parameters."""
        if params['polymer_type'] == 'branched':
            return (
                params['polymer_type'],
                params['xp'],
                params['Np'],
                params['BpP'],
                params['mcl'],
                params['bl'],
                params['nb'],
                params['simulation_type']
            )
        else:  # linear polymer
            return (
                params['polymer_type'],
                params['xp'],
                params['cl'],
                params['simulation_type']
            )

    def find_matching_data(self, **kwargs):
        """Find data matching the specified parameters."""
        matching_data = []

        for _, value in self.data.items():
            params = value['params']

            # Check if all provided parameters match
            matches = True
            for param, val in kwargs.items():
                if param in params:
                    # If parameter is not specified for this dataset (None) and user is looking for a specific value
                    if params[param] is None and val is not None:
                        matches = False
                        break
                    # If parameter value doesn't match user's request
                    elif params[param] is not None and params[param] != val:
                        matches = False
                        break
                # If user is asking for a parameter that doesn't exist in this dataset
                else:
                    matches = False
                    break

            if matches:
                matching_data.append(value)

        return matching_data

    def list_available_data(self):
        """Print information about all available datasets."""
        print(f"Total number of datasets: {len(self.data)}")

        print("\nBranched polymer datasets:")
        branched_count = 0
        for _, value in self.data.items():
            if value['params']['polymer_type'] == 'branched':
                branched_count += 1
                self._print_dataset_info(value['params'])
        print(f"Total branched datasets: {branched_count}")

        print("\nLinear polymer datasets:")
        linear_count = 0
        for _, value in self.data.items():
            if value['params']['polymer_type'] == 'linear':
                linear_count += 1
                self._print_dataset_info(value['params'])
        print(f"Total linear datasets: {linear_count}")

    def _print_dataset_info(self, params):
        """Print formatted information about a dataset."""
        if params['polymer_type'] == 'branched':
            print(f"xp={params['xp']}, Np={params['Np']}, BpP={params['BpP']}, " +
                  f"mcl={params['mcl']}, bl={params['bl']}, nb={params['nb']}, " +
                  f"type={params['simulation_type']}")
        else:
            print(f"xp={params['xp']}, cl={params['cl']}, " +
                  f"type={params['simulation_type']}")

    def plot_structure_factor(self, output_filename=None, **kwargs):
        """
        Plot structure factor for systems matching the specified parameters.

        Parameters:
        - output_filename: Custom filename for the plot (optional)
        - **kwargs: Parameters to match datasets (can include):
            - polymer_type: 'branched' or 'linear'
            - xp: polymer number density
            - Np: number of polymers (branched only)
            - BpP: beads per polymer (branched only)
            - mcl: main chain length (branched only)
            - bl: branch length (branched only)
            - nb: number of branches (branched only)
            - cl: chain length (linear only)
            - simulation_type: simulation type (e.g., 'equil')
        """
        matching_data = self.find_matching_data(**kwargs)

        if not matching_data:
            print("No matching data found for the specified parameters.")
            return

        plt.figure(figsize=(10, 7))

        # Store fitting results for all datasets
        fitting_results = []

        # Plot for each matching dataset
        for i, data_dict in enumerate(matching_data):
            data = data_dict['data']
            params = data_dict['params']

            # Use different line styles and colors for different datasets
            line_color = f"C{i}"
            line_style = '-'

            # Create label based on polymer type
            if params['polymer_type'] == 'branched':
                label = f"Branched: xp={params['xp']}, Np={params['Np']}, BpP={params['BpP']}, " + \
                        f"mcl={params['mcl']}, bl={params['bl']}, nb={params['nb']}"
            else:
                label = f"Linear: xp={params['xp']}, cl={params['cl']}"

            # Plot data
            plt.loglog(data['q'], data['S(q)'], linestyle=line_style, color=line_color, label=label)

            # Perform fitting in the range 0.5 to 1
            mask = (data['q'] >= 0.5) & (data['q'] <= 1.0)
            if sum(mask) > 2:  # Make sure we have enough points for fitting
                fit_data_x = data['q'][mask]
                fit_data_y = data['S(q)'][mask]

                # Fit in log space
                log_x = np.log(fit_data_x)
                log_y = np.log(fit_data_y)

                # Linear fit in log space
                slope, intercept = np.polyfit(log_x, log_y, 1)
                mu = -1 / slope  # Since S(q) ~ q^(-1/mu)

                # Generate fitting line
                fit_x = np.linspace(0.5, 1.0, 100)
                fit_y = np.exp(intercept + slope * np.log(fit_x))
                plt.loglog(fit_x, fit_y, '--', color=line_color, alpha=0.7)

                # Store fitting results
                fitting_results.append((label, mu))

        # Set axis labels and title
        plt.xlabel('q')
        plt.ylabel('S(q)')
        plt.title('Polymer Structure Factor')

        # Set axis limits if there's data
        plt.xlim(0.1, 2.5)
        plt.ylim(data['S(q)'].min() * 0.5, data['S(q)'].max() * 2)

        # Add legend
        plt.legend(loc='best', frameon=False)

        # Add Flory exponent values in a textbox
        if fitting_results:
            textbox = "\n".join([f"{label.split(':')[0]}: μ = {mu:.3f}" for label, mu in fitting_results])
            plt.figtext(0.02, 0.02, textbox, fontsize=9, verticalalignment='bottom')

        # Generate filename for the plot
        if output_filename:
            plot_path = self.output_dir / f"{output_filename}.png"
        else:
            # Create default filename based on filter parameters
            plot_filename = '_'.join(f"{k}{v}" for k, v in kwargs.items() if v is not None)
            if not plot_filename:
                plot_filename = "all_data"
            plot_path = self.output_dir / f"structure_factor_{plot_filename}.png"

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved as: {plot_path}")

        # Print fitting parameters
        for label, mu in fitting_results:
            print(f"Flory exponent (μ) for {label}: {mu:.3f}")

    def plot_comparison(self, branched_params, linear_params, output_filename=None):
        """
        Plot comparison between branched and linear polymers with specified parameters.

        Parameters:
        - branched_params: Dictionary of parameters to match branched polymer data
        - linear_params: Dictionary of parameters to match linear polymer data
        - output_filename: Custom filename for the plot (optional)
        """
        # Add polymer_type to the parameter dictionaries
        branched_params['polymer_type'] = 'branched'
        linear_params['polymer_type'] = 'linear'

        # Find matching data
        branched_data = self.find_matching_data(**branched_params)
        linear_data = self.find_matching_data(**linear_params)

        if not branched_data:
            print("No matching branched polymer data found.")
            return

        if not linear_data:
            print("No matching linear polymer data found.")
            return

        plt.figure(figsize=(10, 7))

        # Plot branched polymer data
        branched = branched_data[0]
        b_data = branched['data']
        b_params = branched['params']

        plt.loglog(b_data['q'], b_data['S(q)'], 'r-', label='Branched')

        # Perform fitting for branched polymer
        b_mask = (b_data['q'] >= 0.5) & (b_data['q'] <= 1.0)
        if sum(b_mask) > 2:
            b_fit_x = b_data['q'][b_mask]
            b_fit_y = b_data['S(q)'][b_mask]
            b_log_x = np.log(b_fit_x)
            b_log_y = np.log(b_fit_y)
            b_slope, b_intercept = np.polyfit(b_log_x, b_log_y, 1)
            b_mu = -1 / b_slope

            # Generate fitting line
            b_line_x = np.linspace(0.5, 1.0, 100)
            b_line_y = np.exp(b_intercept + b_slope * np.log(b_line_x))
            plt.loglog(b_line_x, b_line_y, 'r--', alpha=0.7)
        else:
            b_mu = None

        # Plot linear polymer data
        linear = linear_data[0]
        l_data = linear['data']
        l_params = linear['params']

        plt.loglog(l_data['q'], l_data['S(q)'], 'b-', label='Linear')

        # Perform fitting for linear polymer
        l_mask = (l_data['q'] >= 0.5) & (l_data['q'] <= 1.0)
        if sum(l_mask) > 2:
            l_fit_x = l_data['q'][l_mask]
            l_fit_y = l_data['S(q)'][l_mask]
            l_log_x = np.log(l_fit_x)
            l_log_y = np.log(l_fit_y)
            l_slope, l_intercept = np.polyfit(l_log_x, l_log_y, 1)
            l_mu = -1 / l_slope

            # Generate fitting line
            l_line_x = np.linspace(0.5, 1.0, 100)
            l_line_y = np.exp(l_intercept + l_slope * np.log(l_line_x))
            plt.loglog(l_line_x, l_line_y, 'b--', alpha=0.7)
        else:
            l_mu = None

        # Set axis labels and title
        plt.xlabel('q')
        plt.ylabel('S(q)')
        plt.title('Comparison: Branched vs Linear Polymer Structure Factor')

        # Add parameter information
        branched_label = f"Branched: xp={b_params['xp']}, Np={b_params['Np']}, BpP={b_params['BpP']}, "
        branched_label += f"mcl={b_params['mcl']}, bl={b_params['bl']}, nb={b_params['nb']}"

        linear_label = f"Linear: xp={l_params['xp']}, cl={l_params['cl']}"

        plt.figtext(0.02, 0.98, branched_label, fontsize=9, verticalalignment='top')
        plt.figtext(0.02, 0.94, linear_label, fontsize=9, verticalalignment='top')

        # Add Flory exponent information
        if b_mu is not None and l_mu is not None:
            plt.figtext(0.02, 0.02, f"Branched: μ = {b_mu:.3f}\nLinear: μ = {l_mu:.3f}",
                        fontsize=9, verticalalignment='bottom')

        # Add legend
        plt.legend(loc='best', frameon=False)

        # Set axis limits
        plt.xlim(0.1, 2.5)
        y_min = min(b_data['S(q)'].min(), l_data['S(q)'].min()) * 0.5
        y_max = max(b_data['S(q)'].max(), l_data['S(q)'].max()) * 2
        plt.ylim(y_min, y_max)

        # Generate filename for the plot
        if output_filename:
            plot_path = self.output_dir / f"{output_filename}.png"
        else:
            b_key = f"xp{b_params['xp']}_mcl{b_params['mcl']}_bl{b_params['bl']}_nb{b_params['nb']}"
            l_key = f"xp{l_params['xp']}_cl{l_params['cl']}"
            plot_path = self.output_dir / f"comparison_branched_{b_key}_linear_{l_key}.png"

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved as: {plot_path}")


# Example usage
if __name__ == "__main__":
    analyzer = PolymerStructureFactorAnalyzer()

    # List all available data
    analyzer.list_available_data()

    # Example 1: Plot for a specific branched polymer system
    analyzer.plot_structure_factor(
        polymer_type='branched',
        xp=0.02,
        Np=20,
        BpP=100,
        mcl=50,
        bl=5,
        nb=10,
        simulation_type='equil'
    )

    # Example 2: Plot for a specific linear polymer system
    analyzer.plot_structure_factor(
        polymer_type='linear',
        xp=0.02,
        cl=50,
        simulation_type='equil'
    )

    # Example 3: Plot all systems with xp=0.02 (both linear and branched)
    analyzer.plot_structure_factor(xp=0.02)

    # Example 4: Compare a branched and a linear polymer system
    analyzer.plot_comparison(
        branched_params={
            'xp': 0.02,
            'Np': 20,
            'BpP': 100,
            'mcl': 50,
            'bl': 5,
            'nb': 10,
            'simulation_type': 'equil'
        },
        linear_params={
            'xp': 0.02,
            'cl': 50,
            'simulation_type': 'equil'
        },
        output_filename='branched_linear_comparison'
    )