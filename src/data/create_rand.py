import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os

from scipy.interpolate import interp1d

parser = argparse.ArgumentParser(description='Create random waveform data')
parser.add_argument('--num_datasets', '-d', type=int, default=10,
                    help='Number of datasets to generate')
args = parser.parse_args()

save_dir = 'dynamics_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def random_curve(start=0):
    rand_amp = np.random.uniform(0.8, 1.1)
    rand_freq = np.random.randint(30, 80)
    x = np.linspace(0, 1/(2*rand_freq), 1000)
    y = np.sin(2 * np.pi * rand_freq * x) * rand_amp
    x_end = x[-1] + start
    x += start
    return x, y, x_end

if __name__ == '__main__':
    num_datasets = args.num_datasets

    for dataset in range(num_datasets):
        x = np.array([])
        y = np.array([])
        x_end = 0

        for i in range(32):
            x_temp, y_temp, x_end = random_curve(x_end)
            x = np.append(x, x_temp)
            if i % 2 == 0:
                y_temp = y_temp * -1
            y = np.append(y, y_temp)

        # Define the new_x range
        new_x = np.linspace(0, x_end, 6224)
        
        # Interpolate y values for new_x
        interpolation_function = interp1d(x, y, kind='linear')
        new_y = interpolation_function(new_x)

        data_df = pd.DataFrame({'X': new_x, 'Y': new_y})
        filename = os.path.join(save_dir, f'{dataset + 1}.csv')
        data_df.to_csv(filename, index=False, header=False)

        plt.figure(figsize=(10, 5))
        plt.plot(new_x, new_y)
        # plt.set
        plt.axhline(y=0, color='k')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Dynamics {dataset + 1}')
        plt.savefig(filename.replace('.csv', '.png'))
        plt.close()
