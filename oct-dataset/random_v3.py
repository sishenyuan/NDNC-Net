import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
from scipy.interpolate import CubicSpline

parser = argparse.ArgumentParser(description='Create random waveform data')
parser.add_argument('--num_datasets', '-d', type=int, default=10,
                    help='Number of datasets to generate')
args = parser.parse_args()

save_dir = 'dynamics_data'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

if __name__ == '__main__':
    num_datasets = args.num_datasets

    for dataset in range(num_datasets):
        # 生成两个周期的正弦函数
        x = np.linspace(0, 6224, 6224)
        y = 0.0002 * np.sin(4 * np.pi * x / 6224)

        # 添加高频噪声
        noise = 0.000002 * np.random.randn(len(x))
        y_noisy = y + noise

        # 生成100个均匀分布的散点索引
        total_indices = np.linspace(0, len(x) - 1, 100, dtype=int)
        np.random.shuffle(total_indices)  # 随机化选择的索引

        # 拆分为正和负区域的索引
        pos_indices = total_indices[y[total_indices] > 0][:50]
        neg_indices = total_indices[y[total_indices] <= 0][:50]

        # 确保两类散点交替分布
        def generate_alternating_indices(length, num_type1, num_type2):
            alternating_labels = np.zeros(length, dtype=int)
            type1_count, type2_count = 0, 0
            for i in range(length):
                if type1_count < num_type1 and (type2_count >= num_type2 or np.random.rand() < 0.5):
                    alternating_labels[i] = 1
                    type1_count += 1
                else:
                    alternating_labels[i] = 2
                    type2_count += 1
            return alternating_labels

        # 随机选择负坐标区域的第一类和第二类散点数
        num_neg_type1 = np.random.randint(20, 26)
        num_neg_type2 = 50 - num_neg_type1

        # 生成负坐标区域的交替散点标签
        neg_alternating_labels = generate_alternating_indices(len(neg_indices), num_neg_type1, num_neg_type2)

        # 将交替散点标签与负坐标区域的索引对应
        neg_type1_indices = neg_indices[neg_alternating_labels == 1]
        neg_type2_indices = neg_indices[neg_alternating_labels == 2]

        # 确保正坐标区域的散点有2-5个在0.0002到0.00025范围内
        num_pos_high = np.random.randint(2, 6)
        high_pos_indices = np.random.choice(pos_indices, num_pos_high, replace=False)
        low_pos_indices = np.setdiff1d(pos_indices, high_pos_indices)

        # 配置散点
        # 正坐标区域的散点
        y_noisy[high_pos_indices] = np.random.uniform(0.0002, 0.00025, len(high_pos_indices))
        y_noisy[low_pos_indices] = np.random.uniform(0, 0.0002, len(low_pos_indices))

        # 负坐标区域的散点
        y_noisy[neg_type1_indices] = np.random.uniform(-0.0001, 0.0001, len(neg_type1_indices))
        y_noisy[neg_type2_indices] = np.random.uniform(-0.00025, -0.0002, len(neg_type2_indices))

        # 将所有纵坐标限制在 -0.0003 和 0.0003 范围内
        y_noisy = np.clip(y_noisy, -0.0003, 0.0003)

        # 选择所有的散点进行拟合
        selected_indices = np.concatenate((pos_indices, neg_indices))
        x_selected = x[selected_indices]
        y_selected = y_noisy[selected_indices]

        # 对选择的点进行排序，并确保 x_selected 是严格递增的
        sorted_indices = np.argsort(x_selected)
        x_selected = x_selected[sorted_indices]
        y_selected = y_selected[sorted_indices]

        # 使用样条插值进行拟合
        cs = CubicSpline(x_selected, y_selected)

        # 生成更密集的 x 坐标用于平滑曲线
        new_x = np.linspace(x.min(), x.max(), 6224)
        y_smooth = cs(new_x)

        # 将平滑曲线的所有纵坐标限制在 -0.0003 和 0.0003 范围内
        new_y = np.clip(y_smooth, -0.0003, 0.0003)


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
