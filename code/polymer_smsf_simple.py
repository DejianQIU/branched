import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def load_polymer_data(data_dir):
    """
    Load all polymer structure factor data from directory with simplified naming.
    Returns a dictionary with keys as file indices (e.g., '1', '2') and values as dataframes.
    """
    data_dict = {}

    for filename in os.listdir(data_dir):
        if filename.startswith('polymer_smsf') and filename.endswith('.csv'):
            # Extract the index number from the filename (e.g., '1' from 'polymer_smsf1.csv')
            try:
                # Remove 'polymer_smsf' prefix and '.csv' suffix to get the index
                index = filename.replace('polymer_smsf', '').replace('.csv', '')

                # Load data
                data = pd.read_csv(os.path.join(data_dir, filename),
                                   comment='#',
                                   sep='\s+',
                                   names=['q', 'S(q)'])

                data_dict[index] = data
                print(f"Loaded dataset {index}: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return data_dict


def plot_structure_factor(data_dict, indices=None, output_dir=None, colors=None):
    """
    Plot structure factor for specified indices or all datasets.

    Parameters:
    - data_dict: Dictionary containing loaded data
    - indices: List of indices to plot (e.g., ['1', '2']). If None, plot all datasets.
    - output_dir: Directory to save plots. If None, plots will only be displayed.
    - colors: List of colors for each dataset. If None, default color cycle is used.
    """
    if indices is None:
        indices = list(data_dict.keys())

    if colors is None:
        colors = plt.cm.tab10.colors  # Default color cycle

    # Create figure with log-log axes
    plt.figure(figsize=(10, 8))

    flory_exponents = {}

    for i, idx in enumerate(indices):
        if idx not in data_dict:
            print(f"Warning: Dataset {idx} not found in loaded data.")
            continue

        data = data_dict[idx]
        color = colors[i % len(colors)]

        # Plot data
        plt.loglog(data['q'], data['S(q)'], '-', color=color, label=f'Data {idx}')

        # Perform fitting in the range 0.5 to 1
        mask = (data['q'] >= 0.5) & (data['q'] <= 1.0)
        fit_data_x = data['q'][mask]
        fit_data_y = data['S(q)'][mask]

        # Check if we have enough data points for fitting
        if sum(mask) < 2:
            print(f"Warning: Not enough data points in the fitting range for dataset {idx}")
            continue

        # Fit in log space
        log_x = np.log(fit_data_x)
        log_y = np.log(fit_data_y)

        # Linear fit in log space
        slope, intercept = np.polyfit(log_x, log_y, 1)
        mu = -1 / slope  # Since S(q) ~ q^(-1/mu)
        flory_exponents[idx] = mu

        # Generate fitting line
        fit_x = np.linspace(0.5, 1.0, 100)
        fit_y = np.exp(intercept + slope * np.log(fit_x))
        plt.loglog(fit_x, fit_y, '--', color=color, label=f'Fit {idx}: μ = {mu:.3f}')

    # Set axis limits
    plt.xlim(0.1, 2.5)
    plt.ylim(0.01, 1)

    # Set specific ticks and labels
    plt.xticks([0.1, 1.0], ['0.1', '1.0'])
    plt.yticks([0.01, 0.1, 1.0], ['0.01', '0.1', '1.0'])

    # Labels
    plt.xlabel('q')
    plt.ylabel('S(q)')

    # Add legend
    plt.legend()

    # Add title
    datasets_str = ', '.join(indices)
    plt.title(f'Structure Factor for Datasets: {datasets_str}')

    # Save or show plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filename = f'structure_factor_{"_".join(indices)}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")

    plt.show()

    # Print fitting parameters
    print("\nFlory Exponents (μ):")
    for idx, mu in flory_exponents.items():
        print(f"Dataset {idx}: μ = {mu:.3f}")

    return flory_exponents


def plot_all_separate(data_dict, output_dir=None):
    """
    Create individual plots for each dataset.
    """
    for idx, data in data_dict.items():
        plt.figure(figsize=(8, 6))

        # Plot data
        plt.loglog(data['q'], data['S(q)'], 'k-', label='Data')

        # Perform fitting in the range 0.5 to 1
        mask = (data['q'] >= 0.5) & (data['q'] <= 1.0)
        fit_data_x = data['q'][mask]
        fit_data_y = data['S(q)'][mask]

        # Check if we have enough data points for fitting
        if sum(mask) < 2:
            print(f"Warning: Not enough data points in the fitting range for dataset {idx}")
            continue

        # Fit in log space
        log_x = np.log(fit_data_x)
        log_y = np.log(fit_data_y)

        # Linear fit in log space
        slope, intercept = np.polyfit(log_x, log_y, 1)
        mu = -1 / slope  # Since S(q) ~ q^(-1/mu)

        # Generate fitting line
        fit_x = np.linspace(0.5, 1.0, 100)
        fit_y = np.exp(intercept + slope * np.log(fit_x))
        plt.loglog(fit_x, fit_y, 'r--', label=f'Fit: μ = {mu:.3f}')

        # Set axis limits
        plt.xlim(0.1, 2.5)
        plt.ylim(0.01, 1)

        # Set specific ticks and labels
        plt.xticks([0.1, 1.0], ['0.1', '1.0'])
        plt.yticks([0.01, 0.1, 1.0], ['0.01', '0.1', '1.0'])

        # Labels
        plt.xlabel('q')
        plt.ylabel('S(q)')

        # Add legend
        plt.legend()

        # Add title
        plt.title(f'Structure Factor for Dataset {idx}')

        # Save or show plot
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = f'structure_factor_{idx}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')

        plt.close()
        print(f"Created plot for dataset {idx}, Flory exponent (μ): {mu:.3f}")


def main():
    # Define directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir)  # Adjust this to your data directory
    output_dir = os.path.join(script_dir)

    # Load all data
    data_dict = load_polymer_data(data_dir)

    if not data_dict:
        print("No data files found! Please check the data directory.")
        return

    print(f"\nFound {len(data_dict)} datasets: {', '.join(data_dict.keys())}")

    # Example 1: Plot a specific dataset
    # plot_structure_factor(data_dict, ['1'], output_dir)

    # Example 2: Plot multiple datasets together
    # plot_structure_factor(data_dict, ['1', '2', '3'], output_dir)

    # Example 3: Plot all datasets together
    # plot_structure_factor(data_dict, output_dir=output_dir)

    # Example 4: Create individual plots for each dataset
    # plot_all_separate(data_dict, output_dir)

    # Interactive menu for user selection
    while True:
        print("\nOptions:")
        print("1. Plot specific dataset(s)")
        print("2. Plot all datasets together")
        print("3. Create individual plots for all datasets")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            datasets = input("Enter dataset indices to plot (comma-separated, e.g., 1,2,3): ")
            indices = [idx.strip() for idx in datasets.split(',')]
            plot_structure_factor(data_dict, indices, output_dir)

        elif choice == '2':
            plot_structure_factor(data_dict, output_dir=output_dir)

        elif choice == '3':
            plot_all_separate(data_dict, output_dir)

        elif choice == '4':
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()