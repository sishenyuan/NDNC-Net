# modified from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd
import os
import argparse

from grid_sample import grid_sample
from single_dataset import CustomDataset
from scipy.interpolate import interp1d
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from matplotlib import pyplot as plt

# Define the transformation for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create an instance of the custom dataset
custom_dataset = CustomDataset(root_dir='.', transform=transform)

# Create a data loader for the dataset
data_loader = torch.utils.data.DataLoader(
    custom_dataset, batch_size=1, shuffle=False, num_workers=4) 


def grid_sample(input, grid, canvas=None):
    output = F.grid_sample(input, grid, align_corners=False)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid, align_corners=False)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


def gen_dynamics(width, dynamics_data_path):
    data = pd.read_csv(dynamics_data_path)
    y = data.iloc[:, 1]
    feature_series = 0.0003
    scaled_y = y - np.mean(y)
    scaled_y = feature_series * scaled_y / abs(np.min(scaled_y))
    return scaled_y


def random_flow_gen(original_tensor, y_data, batch_idx):
    size = original_tensor[:, :1, :, 1:].size()
    y_data = torch.FloatTensor(y_data)
    y_data = y_data.unsqueeze(0).unsqueeze(-1)
    y_data = y_data.unsqueeze(0)
    rand_flow = y_data.expand(size[0], 1024, size[2], size[3])
    y_flow = torch.zeros_like(rand_flow)
    rand_flow = torch.cat((rand_flow, y_flow), -1)
    return rand_flow.to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create distorted images')
    parser.add_argument('--load_dir', '-l', type=str, default='dynamics_data',
                        help='Directory to load the dynamics files')
    parser.add_argument('--img_dir', '-s', type=str, default='distorted_segment_images',
                        help='Directory to save the distorted images')
    parser.add_argument('--csv_dir', '-c', type=str,
                        default='vector', help='Directory to save the vector data')
    parser.add_argument('--img_width', '-w', type=int,
                        default=6224, help='Width of the image')
    parser.add_argument('--num_samples', '-n', type=int,
                        default=20, help='Number of samples to generate')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the directories
    img_dir = os.path.join("results", args.img_dir)
    csv_dir = os.path.join("results", args.csv_dir)
    labels_dir = os.path.join("results", "labels")
    comp_dis_img_dir = os.path.join("results", "complete_distorted_images")
    dynamics_dir = os.path.join("results", "dynamic_curves")
    ori_dir = os.path.join("results", "original_segments")

    for directory in [img_dir, csv_dir, comp_dis_img_dir, dynamics_dir, ori_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    if os.path.exists(labels_dir):
        for file in os.listdir(labels_dir):
            os.remove(os.path.join(labels_dir, file))
    else:
        os.makedirs(labels_dir)

    num_img = len(data_loader)

    all_csv = [f for f in sorted(os.listdir(args.load_dir), key=lambda x: int(
        x.split('.')[0])) if f.endswith('.csv')]
    num_csv = len(all_csv)
    save_idx = 0
    save_complete_idx = 0

    for csv in tqdm(all_csv):
        csv_file = os.path.join(args.load_dir, csv)
        df = pd.read_csv(csv_file, usecols=[1])
        dynamics_data = gen_dynamics(args.img_width, csv_file)

        below_zero_segments = []
        segment_start = None
        min_segment_length = 0

        for i, value in enumerate(dynamics_data):
            if value < -0.0001 and segment_start is None:
                segment_start = i
            elif value >= -0.0001 and segment_start is not None:
                if i - segment_start > min_segment_length:
                    below_zero_segments.append((segment_start, i))
                    segment_start = None
        if segment_start and len(dynamics_data) - segment_start > min_segment_length:
            below_zero_segments.append((segment_start, len(dynamics_data)))

        if below_zero_segments:
            print(f"Found {len(below_zero_segments)} segments")
        else:
            print("No segments found")
            print("Skipping...")
            continue

        for batch_idx, (data) in enumerate(data_loader):
            target = data.to(device)
            h = target.size()[2]
            w = target.size()[3]
            theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            theta = theta.view(-1, 2, 3)
            grid = F.affine_grid(
                theta, [1, 1, h, w], align_corners=False).to(device)
            grid_start = grid[:, :, :1, :]
            grid = torch.diff(grid, dim=2)
            flow = random_flow_gen(grid, dynamics_data, batch_idx).to(device)
            grid = grid + flow

            grid = torch.cat((grid_start, grid), dim=2)
            grid = torch.cumsum(grid, dim=2)

            data = grid[:, :, :, 0]
            normal_data = (data - data.min()) / (data.max() - data.min())
            normal_data = normal_data * 2 - 1
            grid[:, :, :, 0] = normal_data

            transformed_image = grid_sample(target, grid).cpu()

            target = target.cpu()*0.5 + 0.5
            target = target[0].permute(1, 2, 0).detach().numpy() * 255
            transformed_image = transformed_image*0.5+0.5
            test = transformed_image[0].permute(1, 2, 0).detach().numpy() * 255

            # save the transformed image
            total_samples = args.num_samples
            if save_complete_idx >= total_samples:
                print(f"  ed generating {total_samples} images")
                exit()

            save_complete_idx += 1
            cv2.imwrite(
                f'{comp_dis_img_dir}/{save_complete_idx}.png', test[:, 0:3000])
            save_complete_idx += 1
            cv2.imwrite(
                f'{comp_dis_img_dir}/{save_complete_idx}.png', test[:, 3224:6224])

            # plot the dynamics data
            fig, ax = plt.subplots(3, 1, dpi = 300, figsize=(10, 6))
            ax[0].imshow(target[:, :, 0], cmap='gray')
            ax[0].axis('off')

            ax[1].imshow(test[:, :, 0], cmap='gray')
            ax[1].axis('off')
            
            # set ax[1] figsize to be the same as ax[0]
            ax[2].plot(dynamics_data)
            # make the font size smaller
            ax[2].tick_params(axis='both', which='major', labelsize=8)
            ax[2].axhline(y=0, color='r')
            # find zero crossing in the dynamics data and plot vertical lines
            zero_crossings = np.where(np.diff(np.sign(dynamics_data)))[0]
            # plot the band below zero
            for start, end in below_zero_segments:
                if end - start > min_segment_length:
                    ax[2].axvspan(start, end, color='red', alpha=0.1)
                    ax[1].axvspan(start, end, color='red', alpha=0.1)
                
            ax[2].set_xbound(0, len(dynamics_data))

            plt.tight_layout()
            plt.savefig(f'{dynamics_dir}/{save_complete_idx}.png')
            plt.close()

            for idx, (start, end) in enumerate(below_zero_segments):
                segment_data = dynamics_data[start:end]
                segment_image = test[:, start:end, 0]
                ori_segment_image = target[:, start:end, 0]
                segment_len = 1000

                # Save the segment image
                original_image = np.zeros((segment_image.shape[0], segment_len))
                padded_image = np.zeros((segment_image.shape[0], segment_len))
                width = min(segment_len, segment_image.shape[1])
                length = min(segment_len, len(segment_data))
                original_image[:, :width] = ori_segment_image[:, :width]
                padded_image[:, :width] = segment_image[:, :width]
                
                save_idx += 1

                cv2.imwrite(f'{img_dir}/{save_idx}.png', padded_image)
                cv2.imwrite(f'{ori_dir}/{save_idx}.png', original_image)

                # Save the segment data to a CSV file
                padded_data = np.zeros(segment_len)
                padded_data[:length] = segment_data[:length]
                segment_df = pd.DataFrame(
                    {'DynamicsData': padded_data})
                segment_df.to_csv(
                    f'{csv_dir}/{save_idx}.csv', index=False)

                # save YOLO format label
                center_x = (start + end) / 2
                center_y = test.shape[0] / 2
                label_width = end - start
                label_height = test.shape[0]
                img_width = 3000
                img_height = test.shape[0]

                if center_x < 3000:
                    label_width = min(
                        center_x + label_width / 2, label_width, 3000 - center_x + label_width / 2)
                    center_x = center_x / img_width
                    center_y = center_y / img_height
                    label_width = label_width / img_width
                    label_height = label_height / img_height
                    with open(f'{labels_dir}/{save_complete_idx - 1}.txt', 'a') as f:
                        f.write(f'0 {center_x} {center_y} {label_width} {label_height}\n')

                elif center_x > 3224:
                    center_x -= 3224
                    label_width = min(
                        center_x + label_width / 2, label_width, 3000 - center_x + label_width / 2)
                    
                    center_x = center_x / img_width
                    center_y = center_y / img_height
                    label_width = label_width / img_width
                    label_height = label_height / img_height
                    with open(f'{labels_dir}/{save_complete_idx}.txt', 'a') as f:
                        f.write(f'0 {center_x} {center_y} {label_width} {label_height}\n')

        print(f"Already generated {save_complete_idx} samples")
