import numpy as np
import matplotlib.pyplot as plt

def vector_autocorrelation(data, max_lag=None):
    """
    Calculate the autocorrelation function of 3D vector data.

    Parameters:
        data (array-like): The input signal (Nx3 array, where each row is a 3D vector).
        max_lag (int, optional): Maximum lag for which the correlation is computed.
                                 If None, max_lag = len(data) - 1.

    Returns:
        lags (numpy.ndarray): Array of lag values.
        correlation (numpy.ndarray): Autocorrelation values.
    """
    # Convert input to a NumPy array
    data = np.asarray(data)
    
    # Ensure the input has the correct shape
    #if data.ndim != 2 or data.shape[1] != 3:
    #    raise ValueError("Input data must be an Nx3 array representing 3D vectors.")
    
    n = len(data)
    
    # Compute mean vector
    mean_vector = np.mean(data, axis=0)
    print('mean = ', mean_vector)
    # Center the data by subtracting the mean
    centered_data = data - mean_vector
    
    # Compute variance using the dot product of centered vectors
    variance = np.mean([np.dot(v, v) for v in centered_data])
    if variance == 0:
        raise ValueError("Variance of the data is zero; correlation function is undefined.")
    print('variance = ', variance)
    # Default max_lag
    if max_lag is None or max_lag >= n:
        max_lag = n - 1
    
    # Compute autocorrelation
    correlation = np.zeros(max_lag + 1)
    with open('out-acf-avg.txt', 'w') as a:
            
     for lag in range(max_lag + 1):
        dot_products = [
            np.dot(centered_data[i], centered_data[i + lag])
            for i in range(n - lag)
        ]
        correlation[lag] = np.mean(dot_products) / variance
        a.write(f"{correlation[lag]}\n")
    lags = np.arange(max_lag + 1)
    return lags, correlation

def radius_of_gyration(positions):
    """
    Calculate the radius of gyration (R_g) for a single polymer chain.
    
    Parameters:
        positions (numpy.ndarray): Array of shape (N, 3) representing the positions (x, y, z) of N atoms in the polymer chain.
        
    Returns:
        float: Radius of gyration (R_g).
    """
    # Calculate the center of mass
    center_of_mass = np.mean(positions, axis=0)
    
    # Compute squared distances from the center of mass
    squared_distances = np.sum((positions - center_of_mass)**2, axis=1)
    
    # Radius of gyration
    R_g = np.sqrt(np.mean(squared_distances))
    
    return R_g

# Specify the file name
#file_name = "pol.trj"
file_name = "traj.gro"
num_pol = 40
# Read the file and find lines containing "1 1 1"
with open(file_name, "r") as file:
    count = 0.
    matrix = []
    lines = file.readlines()
    #boxx = float(lines[5].split()[1])
    #boxy = float(lines[6].split()[1])
    #boxz = float(lines[7].split()[1])
    boxx = 100
    boxy = 100
    boxz = 100
    box_size = np.array([boxx, boxy, boxz])
    half = box_size/2
    neg = -box_size/2
    numatom = float(lines[2].split()[0])
    pl = int(numatom/num_pol)
    pl = 50
    print('num pol = ', num_pol)
    print('pol length = ', pl)
    h = 2             # headline
    hh = int(numatom + h+1)
    #hh = 100000 + h+1
    max_step = int(len(lines)/hh)
    #for line_number, line in enumerate(lines, start=1):
    with open('end-to-end.txt', 'w') as f:
        f.write('end-to-end distance            radius of gyration\n')
        for i in range (0,max_step):
            sum_delta = []
            sum_dist = 0.
            positions = []
            rg_values = []
            for j in range (num_pol):
                x = np.zeros(pl+1)
                y = np.zeros(pl+1)
                z = np.zeros(pl+1)
                d = np.zeros([pl, 3])
                pos = np.zeros([pl+1,3])
                count1 = np.zeros(3)
                count2 = np.zeros(3)
                for k in range (1, pl+1, 1):
                    x[k] = float(lines[h+k+(j*pl)+i*hh].split()[3])
                    y[k] = float(lines[h+k+(j*pl)+i*hh].split()[4])
                    z[k] = float(lines[h+k+(j*pl)+i*hh].split()[5])
                    pos[k] = np.array([x[k], y[k], z[k]])
                    if k > 1:
                        dd = np.array([x[k]-x[k-1], y[k]-y[k-1], z[k]-z[k-1]])
                        for n in range(3):  # Loop through x, y, z dimensions
                            if dd[n] >= half[n]:
                                count1[n] += 1.
                            if dd[n] < neg[n]:
                                count2[n] += 1.
                        for n in range(3):  # Loop through x, y, z dimensions
                            if count1[n] > 0:
                                pos[k,n] -= count1[n] * box_size[n]
                            if count2[n] > 0:
                                pos[k,n] += count2[n] * box_size[n]

                    positions.append(pos[k])
                R_g = radius_of_gyration(positions)
                rg_values.append(R_g)
                delta = np.array(pos[pl]-pos[1])   
                #print(pos[pl],'   ', pos[1])           
                dist = np.sqrt((delta[0]**2)+(delta[1]**2)+(delta[2]**2))
                sum_dist = sum_dist + dist
                sum_delta += [delta]
            mean_dist = sum_dist/num_pol
            mean_delta = np.mean(sum_delta, axis=0)
            matrix = matrix + [mean_dist]
            rg_avg = np.mean(rg_values)
            f.write(str(mean_dist)+'\t\t\t'+str(rg_avg)+'\n')

data = np.array(matrix)
step = int(len(data))
print(len(data))
print('step num = ',step)
print(boxx, boxy, boxz)

lags, corr = vector_autocorrelation(data, max_lag=step)
    
print(lags,corr)
# Plot the autocorrelation function
plt.figure(figsize=(8, 4))
plt.plot(lags, corr, marker='o')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.title("3D Vector Autocorrelation Function")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid(True)
plt.show()

