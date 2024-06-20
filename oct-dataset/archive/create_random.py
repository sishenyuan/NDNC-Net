import os
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

# 指定要保存CSV文件的文件夹名称
folder_name = 'waveform_data'

# 使用os模块检查文件夹是否存在，如果不存在则创建
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for dataset in range(70):
    print(f"Generating dataset {dataset + 1}...")
    # 创建时间序列
    x = np.linspace(0, 4 * np.pi, 1000)
    y = np.zeros_like(x)
    n_peaks = np.random.randint(16, 24)

    for i in range(n_peaks):
        amplitude = np.random.uniform(0, 10)
        start_index = i * len(x) // n_peaks
        end_index = (i + 1) * len(x) // n_peaks if i < n_peaks - 1 else len(x)
        x_segment = x[start_index:end_index]
        
        if i % 2 == 0:
            y[start_index:end_index] = amplitude * np.sin(2.5 * (x_segment - x[start_index])) + 10
        else:
            y[start_index:end_index] = -amplitude * np.sin(2.5 * (x_segment - x[start_index])) + 10

    cs = CubicSpline(x, y)
    x_new = np.linspace(0, 4 * np.pi, 6224)
    y_new = cs(x_new)

    data_df = pd.DataFrame({'X': x_new, 'Y': y_new})

    # 构建文件路径，包括文件夹名和文件名
    filename = os.path.join(folder_name, f'{dataset + 1}.csv')
    # 保存CSV文件，不包括索引和列名
    data_df.to_csv(filename, index=False, header=False)

print(f"All files have been saved to the '{folder_name}' folder.")
