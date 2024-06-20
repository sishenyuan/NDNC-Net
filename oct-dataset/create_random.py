import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
# from tqdm import tqdm
import argparse

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Create random waveform data')
parser.add_argument('--n_terms', '-n', type=int, default=10,
                    help='Number of terms in the Fourier series')
parser.add_argument('--num_datasets', '-d', type=int, default=10,
                    help='Number of datasets to generate')
args = parser.parse_args()

# Create a folder to save the CSV files
save_dir = 'dynamics_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Set the number of terms in the Fourier series
n_terms = args.n_terms
num_datasets = args.num_datasets    

x = np.linspace(0, 4*np.pi, 6224)

for dataset in range(num_datasets):
    y = np.zeros_like(x)
    for i in range(1, n_terms+1):
        amplitude = np.random.rand()  # Random amplitude
        phase = np.random.rand() * 4 * np.pi  # Random phase
        y += amplitude * np.sin(i * x + phase)

    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    y = y * 2 - 1

    data_df = pd.DataFrame({'X': x, 'Y': y})

    filename = os.path.join(save_dir, f'{dataset + 1}.csv')
    data_df.to_csv(filename, index=False, header=False)

    # Plot the waveform
    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    # plt.set
    plt.axhline(y=0, color='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Dataset {dataset + 1}')
    plt.savefig(filename.replace('.csv', '.png'))
    plt.close()

print(f"All files have been saved to the '{save_dir}' folder.")
