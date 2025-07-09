import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def parse_filename(filename):
    """Parse polymer system parameters from filename."""
    parts = filename.replace('.csv', '').split('_')
    params = {
        'xp': float(parts[2].replace('xp', '')),
        'Np': int(parts[3].replace('Np', '')),
        'BpP': int(parts[4].replace('BpP', '')),
        'mcl': int(parts[5].replace('mcl', '')),
        'bl': int(parts[6].replace('bl', '')),
        'nb': int(parts[7].replace('nb', ''))
    }
    return params


def load_polymer_data(data_dir):
    """Load all polymer structure factor data from directory."""
    data_dict = {}
    for filename in os.listdir(data_dir):
        if filename.startswith('polymer_smsf') and filename.endswith('equil.csv'):
            params = parse_filename(filename)

            # Create a unique key for the dataset
            key = f"xp{params['xp']}_Np{params['Np']}_BpP{params['BpP']}_mcl{params['mcl']}_bl{params['bl']}_nb{params['nb']}"

            # Load data
            data = pd.read_csv(os.path.join(data_dir, filename),
                               comment='#',
                               sep='\s+',
                               names=['q', 'S(q)'])
            data_dict[key] = {'data': data, 'params': params}

    return data_dict


def plot_structure_factor(data_dict, xp, Np, BpP, mcl, bl, nb, output_dir):
    """Plot structure factor for specified parameters."""
    # Create key for the specified parameters
    key = f"xp{xp}_Np{Np}_BpP{BpP}_mcl{mcl}_bl{bl}_nb{nb}"

    if key not in data_dict:
        raise ValueError(f"No data found for parameters: {key}")

    data = data_dict[key]['data']
    params = data_dict[key]['params']

    # Create figure with log-log axes
    plt.figure(figsize=(8, 6))

    # Plot data
    plt.loglog(data['q'], data['S(q)'], 'k-', label='Data')

    # Perform fitting in the range 0.5 to 1
    mask = (data['q'] >= 0.5) & (data['q'] <= 1.0)
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

    # Add title with system parameters
    title = f'Structure Factor (xp={xp}, Np={Np}, BpP={BpP}\nmcl={mcl}, bl={bl}, nb={nb})'
    plt.title(title)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'structure_factor_{key}.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print fitting parameter
    print(f"Flory exponent (μ) for {key}: {mu:.3f}")


def main():
    # Define directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'polymersmsf')
    output_dir = os.path.join(script_dir, 'polymersmsf_plot')

    # Load all data
    data_dict = load_polymer_data(data_dir)

    # Example: Plot for specific parameters
    plot_structure_factor(
        data_dict,
        xp=0.02,
        Np=20,
        BpP=100,
        mcl=50,
        bl=5,
        nb=10,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()