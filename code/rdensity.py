import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path


class RadialDensityAnalyzer:
    def __init__(self):
        # Create paths for data and plot directories
        self.script_dir = Path.cwd()
        self.rdensity_dir = self.script_dir / "rdensity"
        self.branched_dir = self.rdensity_dir / "branched"
        self.linear_dir = self.rdensity_dir / "linear"
        self.plot_dir = self.rdensity_dir

        # Create plots directory if it doesn't exist
        self.plot_dir.mkdir(exist_ok=True)

        # Dictionary to store all data
        self.data = {}

        # Load all data files
        self.load_all_data()

    def parse_filename(self, filename):
        """Parse parameters from filename for both linear and branched polymer formats."""
        # Try to match branched polymer format first
        branched_pattern = r'rdensity_xp([\d.]+)_Np(\d+)_BpP(\d+)_mcl(\d+)_bl(\d+)_nb(\d+)_(\w+)'
        branched_match = re.match(branched_pattern, filename)

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
        linear_pattern = r'rdensity_xp([\d.]+)_Np(\d+)_BpP(\d+)_(\w+)'
        linear_match = re.match(linear_pattern, filename)

        if linear_match:
            return {
                'xp': float(linear_match.group(1)),
                'Np': int(linear_match.group(2)),
                'BpP': int(linear_match.group(3)),
                'simulation_type': linear_match.group(4),
                'mcl': None,
                'bl': None,
                'nb': None,
                'polymer_type': 'linear'
            }

        return None

    def load_data_file(self, filepath):
        """Load individual data file."""
        try:
            data = pd.read_csv(filepath,
                               sep=r'\s+',
                               skiprows=2,
                               names=['r', 'rho_total', 'rho_S', 'rho_P'])
            return data
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return None

    def load_all_data(self):
        """Load all data files from both branched and linear directories."""
        # Process branched polymer data
        if self.branched_dir.exists():
            for file in self.branched_dir.glob('rdensity_*'):
                params = self.parse_filename(file.stem)
                if params:
                    data = self.load_data_file(file)
                    if data is not None:
                        key = self._create_key(params)
                        self.data[key] = {'params': params, 'data': data}

        # Process linear polymer data
        if self.linear_dir.exists():
            for file in self.linear_dir.glob('rdensity_*'):
                params = self.parse_filename(file.stem)
                if params:
                    data = self.load_data_file(file)
                    if data is not None:
                        key = self._create_key(params)
                        self.data[key] = {'params': params, 'data': data}

    def _create_key(self, params):
        """Create a unique key for a dataset based on its parameters."""
        key_parts = [
            ('xp', params['xp']),
            ('Np', params['Np']),
            ('BpP', params['BpP']),
            ('simulation_type', params['simulation_type']),
            ('polymer_type', params['polymer_type'])
        ]

        # Add branched polymer specific parameters if available
        if params['polymer_type'] == 'branched':
            key_parts.extend([
                ('mcl', params['mcl']),
                ('bl', params['bl']),
                ('nb', params['nb'])
            ])

        return tuple(v for _, v in key_parts)

    def find_matching_data(self, **kwargs):
        """Find data matching the specified parameters."""
        matching_data = []

        for _, value in self.data.items():
            params = value['params']

            # Check if all provided parameters match
            matches = True
            for param, val in kwargs.items():
                # If parameter is not specified for this dataset (None) and user is looking for a specific value
                if param in params and params[param] is None and val is not None:
                    matches = False
                    break
                # If parameter value doesn't match user's request
                elif param in params and params[param] is not None and params[param] != val:
                    matches = False
                    break
                # If user is asking for a parameter that doesn't exist in this dataset
                elif param not in params:
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
            print(f"xp={params['xp']}, Np={params['Np']}, BpP={params['BpP']}, " +
                  f"type={params['simulation_type']}")

    def plot_density_profiles(self, output_filename=None, **kwargs):
        """
        Plot density profiles for systems matching the specified parameters.

        Parameters:
        - output_filename: Custom filename for the plot (optional)
        - **kwargs: Parameters to match datasets (can include):
            - xp: polymer number density
            - Np: number of polymers
            - BpP: beads per polymer
            - mcl: main chain length (branched only)
            - bl: branch length (branched only)
            - nb: number of branches (branched only)
            - simulation_type: simulation type (e.g., 'equil')
            - polymer_type: 'branched' or 'linear'
        """
        matching_data = self.find_matching_data(**kwargs)

        if not matching_data:
            print("No matching data found for the specified parameters.")
            return

        plt.figure(figsize=(10, 6))

        # Plot for each matching dataset
        for i, data_dict in enumerate(matching_data):
            data = data_dict['data']
            params = data_dict['params']

            # Use different line styles for different datasets if multiple
            linestyle_total = '-' if i == 0 else '-.'
            linestyle_S = '--' if i == 0 else ':'
            linestyle_P = '--' if i == 0 else ':'

            # Assign labels only for the first dataset to avoid duplicate legend entries
            if i == 0:
                plt.plot(data['r'], data['rho_total'], 'k' + linestyle_total, label='Total density')
                plt.plot(data['r'], data['rho_S'], 'b' + linestyle_S, label='Solvent density')
                plt.plot(data['r'], data['rho_P'], 'r' + linestyle_P, label='Polymer density')
            else:
                plt.plot(data['r'], data['rho_total'], 'k' + linestyle_total)
                plt.plot(data['r'], data['rho_S'], 'b' + linestyle_S)
                plt.plot(data['r'], data['rho_P'], 'r' + linestyle_P)

            # Add dataset specific label
            if params['polymer_type'] == 'branched':
                dataset_label = f"Dataset {i + 1}: xp={params['xp']}, Np={params['Np']}, BpP={params['BpP']}, "
                dataset_label += f"mcl={params['mcl']}, bl={params['bl']}, nb={params['nb']}, "
                dataset_label += f"type={params['polymer_type']}"
            else:
                dataset_label = f"Dataset {i + 1}: xp={params['xp']}, Np={params['Np']}, BpP={params['BpP']}, "
                dataset_label += f"type={params['polymer_type']}"

            plt.text(0.02, 0.98 - i * 0.05, dataset_label, transform=plt.gca().transAxes,
                     fontsize=8, verticalalignment='top')

        plt.xlim(0, 20)
        plt.ylim(0, 10)
        plt.xlabel('Distance from COM (r)')
        plt.ylabel('Density ρ(r)')
        plt.title(f"Radial Density Profiles ({len(matching_data)} datasets)")

        plt.legend(frameon=False)
        plt.grid(False)

        # Generate filename for the plot
        if output_filename:
            plot_path = self.plot_dir / f"{output_filename}.png"
        else:
            # Create default filename based on filter parameters
            plot_filename = '_'.join(f"{k}{v}" for k, v in kwargs.items() if v is not None)
            plot_path = self.plot_dir / f"density_profile_{plot_filename}.png"

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved as: {plot_path}")

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

        plt.figure(figsize=(12, 8))

        # Plot branched polymer data
        branched = branched_data[0]
        plt.plot(branched['data']['r'], branched['data']['rho_total'], 'k-', label='Branched: Total')
        plt.plot(branched['data']['r'], branched['data']['rho_S'], 'b--', label='Branched: Solvent')
        plt.plot(branched['data']['r'], branched['data']['rho_P'], 'r--', label='Branched: Polymer')


        # Plot linear polymer data
        linear = linear_data[0]
        plt.plot(linear['data']['r'], linear['data']['rho_total'], 'k-.', label='Linear: Total')
        plt.plot(linear['data']['r'], linear['data']['rho_S'], 'b:', label='Linear: Solvent')
        plt.plot(linear['data']['r'], linear['data']['rho_P'], 'r:', label='Linear: Polymer')


        plt.xlabel('Distance from COM (r)')
        plt.ylabel('Density ρ(r)')
        plt.title('Comparison: Branched vs Linear Polymer Density Profiles')

        # Create descriptive labels
        b_params = branched['params']
        l_params = linear['params']

        branched_label = f"Branched: xp={b_params['xp']}, Np={b_params['Np']}, BpP={b_params['BpP']}, "
        branched_label += f"mcl={b_params['mcl']}, bl={b_params['bl']}, nb={b_params['nb']}"

        linear_label = f"Linear: xp={l_params['xp']}, Np={l_params['Np']}, BpP={l_params['BpP']}"

        plt.text(0.02, 0.98, branched_label, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='top')
        plt.text(0.02, 0.94, linear_label, transform=plt.gca().transAxes,
                 fontsize=9, verticalalignment='top')

        plt.legend(frameon=False)
        plt.grid(False)

        # Generate filename for the plot
        if output_filename:
            plot_path = self.plot_dir / f"{output_filename}.png"
        else:
            # Create default filename
            plot_path = self.plot_dir / "branched_vs_linear_comparison.png"

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved as: {plot_path}")


# Example usage
if __name__ == "__main__":
    analyzer = RadialDensityAnalyzer()

    # List all available data
    analyzer.list_available_data()

    # Example 1: Plot for a specific branched polymer system
    analyzer.plot_density_profiles(
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
    analyzer.plot_density_profiles(
        polymer_type='linear',
        xp=0.02,
        Np=40,
        BpP=50,
        simulation_type='equil'
    )

"""
    # Example 3: Compare a branched and a linear polymer system
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
            'Np': 40,
            'BpP': 50,
            'simulation_type': 'equil'
        },
        output_filename='branched_linear_comparison'
    )

    # Example 4: Plot all systems with xp=0.02
    analyzer.plot_density_profiles(xp=0.02)
"""