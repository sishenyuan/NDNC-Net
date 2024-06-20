import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d

import torch
import torch.nn.functional as F

from PIL import Image, ImageOps
from torchvision import transforms

from yolo.detect import detect_nurd
from ncnet import ImageVectorDataset, VisionTransformer
from torch.utils.data import DataLoader
import time
import argparse

def gen_dynamics(dynamics_data_path, width, width_full = 6224):
    data = pd.read_csv(dynamics_data_path)
    original_y = np.array(data)[:, 0]
    original_y = original_y[original_y != 0]
    original_y = original_y[:width-1]

    original_x = np.linspace(0,width-2,width-1)

    scale_factor = width_full / width
    scaled_x = (width - 1) * (original_x - np.min(original_x)) / \
        (np.max(original_x))
    interpolation_func = interp1d(scaled_x, original_y, kind='linear')

    new_x = np.arange(width - 1)
    interpolated_y = interpolation_func(new_x)

    scaled_y = 0.0003 * scale_factor * interpolated_y / abs(np.min(interpolated_y))

    return scaled_y

def random_flow_gen(original_tensor, y_data, height = 1000):
    size = original_tensor[:, :1, :, 1:].size()
    y_data = torch.FloatTensor(y_data)
    y_data = y_data.unsqueeze(0).unsqueeze(-1)
    y_data = y_data.unsqueeze(0)
    rand_flow = y_data.expand(size[0], height, size[2], size[3])
    y_flow = torch.zeros_like(rand_flow)
    rand_flow = torch.cat((rand_flow, y_flow), -1)
    return rand_flow.to(device)

def read_csv_pandas(file_path):
    data = pd.read_csv(file_path)
    return data

def fill_zero_columns(image):
    image = image.astype(float)
    channels, height, width = image.shape
    zero_count = 0
    for c in range(channels):
        col = 0
        while col < width:
            # find the start of zero rwo
            if np.all(image[c, :, col] == 0):
                start_col = col
                
                # find the end of zero rwo
                while col < width and np.all(image[c, :, col] == 0):
                    col += 1
                
                # find the nearlest left non-zero rwo
                left_col = start_col - 1
                while left_col >= 0 and np.all(image[c, :, left_col] == 0):
                    left_col -= 1
                
                # find the nearlest right non-zero rwo
                right_col = col
                while right_col < width and np.all(image[c, :, right_col] == 0):
                    right_col += 1
                
                zero_count += (col - start_col - 1)
                if left_col >= 0 and right_col < width:
                    x = [left_col, right_col]
                    y = np.arange(0, height)
                    z = np.array([image[c, :, left_col], image[c, :, right_col]]).T

                    f = interp2d(x, y, z, kind='linear')
                    image[c, :, start_col:col] = f(range(start_col, col), range(height))
                elif left_col >= 0:
                    image[c, :, start_col:col] = np.repeat(image[c, :, left_col][:, np.newaxis], col - start_col, axis=1)
                    image[c, :, start_col:col] = []
                elif right_col < width:
                    image[c, :, start_col:col] = np.repeat(image[c, :, right_col][:, np.newaxis], col - start_col, axis=1)
                    image[c, :, start_col:col] = []
            else:
                col += 1
    
    widthnew = int((width - zero_count) * (3000 / 6224))
    image = np.transpose(image, (1, 2, 0))
    image = cv2.resize(image, (widthnew, height))
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.uint8)
    return image

def restore_nurd(distort_image_path, dynamics_data_path, index):
    '''
    input
    distorted_img: H,W,C -> the distorted image section
    dynamics_data_path: the path to dynamic vector
    
    output
    restored_img: H,W,C -> the restorted image
    '''
    distorted_img = cv2.imread(distort_image_path)
    h, w, c = distorted_img.shape    
    dynamics_data = gen_dynamics(dynamics_data_path, w, 6224)  

    theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
    theta = theta.view(-1, 2, 3)
    grid = F.affine_grid(
        theta, [1, 1, h, w], align_corners=False).to(device)

    grid_start = grid[:, :, :1, :]
    grid = torch.diff(grid, dim=2)
    flow = random_flow_gen(grid, dynamics_data, height=1024).to(device)
    grid = grid + flow
    grid = torch.cat((grid_start, grid), dim=2)
    grid = torch.cumsum(grid, dim=2)

    # normalize the grid to fit the whole width
    data = grid[:,:,:,0]
    normal_data = (data - data.min()) / (data.max() - data.min())
    normal_data = normal_data * 2 - 1
    grid[:,:,:,0] = normal_data

    forward_flow = grid.squeeze().permute(2, 0, 1).cpu().numpy()
    forward_flow = (forward_flow + 1 ) * 0.5
    forward_flow[0] *= (w - 1)
    forward_flow = forward_flow.astype(int)

    restored_img = np.zeros((h,w,3))
    for i in range(w):
        restored_img[:,forward_flow[0,0,i]] = distorted_img[:,i]
    
    restored_img = np.transpose(restored_img, (2, 0, 1))
    restored_img = fill_zero_columns(restored_img)
    restored_img = np.transpose(restored_img, (1, 2, 0))

    # plt.figure()
    # xx = np.linspace(0,len(forward_flow[0,0,:]),len(forward_flow[0,0,:]))
    # plt.plot(xx, forward_flow[0,0,:])
    # plt.savefig(os.path.join("test", "forward_flow"+ str(index) + ".png"))   

    return restored_img

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', help='The path to input data.', type=str, default='datas')
    parser.add_argument('--output_path', help='The path to output.', type=str, default='outputs')

    parser.add_argument('--net2_path', help='The path to the weight of detection net.', type=str, default='yolo/train/weights/best.pt')
    parser.add_argument('--net3_path', help='The path to the weight of correction net.', type=str, default='./weights/epoch_102.pt')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_path, exist_ok=True)

    image_dir_list = [os.path.join(args.data_path, image) for image in os.listdir(args.data_path)]
    image_dir_list.sort()

    for image_dir in image_dir_list:
        stacked_filelist = [distort_img for distort_img in os.listdir(image_dir) if '.png' in distort_img and 'original' not in distort_img]
        stacked_filelist.sort()

        for distort_img in stacked_filelist:
            oct_image_whole_path = os.path.join(image_dir, distort_img)
            undistort_image_index = int(image_dir[6:])
            image_index = int(distort_img[:-4])

            oct_image_whole = cv2.imread(oct_image_whole_path)

            nurd_coord = detect_nurd(args.net2_path, oct_image_whole)
            nurd_coord = nurd_coord.astype(np.int64)

            ########  net3
            image = Image.open(oct_image_whole_path)
            image_width, image_height = image.size

            # cut and concatenate the images
            output_dir = os.path.join(args.output_path, "vector", str(undistort_image_index) + '_' + str(image_index))
            os.makedirs(output_dir, exist_ok=True)

            new_image_width = 1000
            black_strip = Image.new('L', (image_width, image_height), 0)

            full_images=[]
            cropped_images = []

            for i in range(len(nurd_coord)):
                left, right = int(nurd_coord[i,0]), int(nurd_coord[i,1])
                cropped = image.crop((left, 0, right, image_height))
                full_images.append(cropped)
                padded = ImageOps.pad(cropped, (new_image_width, image_height), color=0, centering=(0,0))
                cropped_images.append(padded)

            new_image_path1 = os.path.join(args.output_path, "distorted_cropped_image", str(undistort_image_index) + '_' + str(image_index))
            os.makedirs(new_image_path1, exist_ok=True)

            for i in range(0,len(full_images)):
                path = os.path.join(new_image_path1, str(i+1)+".png")
                new_image = full_images[i]
                new_image.save(path)

            new_image_path2 = os.path.join(args.output_path, "distorted_segment_images", str(undistort_image_index) + '_' + str(image_index))
            os.makedirs(new_image_path2, exist_ok=True)
            for i in range(0,len(cropped_images)):
                path = os.path.join(new_image_path2, str(i+1)+".png")
                new_image = cropped_images[i]
                new_image.save(path)

            batch_size = 1
            img_size = [125, 128]
            patch_size = [img_size[0] // 16, img_size[1] // 1]
            dim = 64
            num_classes = 1000

            # 定义数据转换
            transform = transforms.Compose([
                transforms.Resize((img_size[0], img_size[1])),
                transforms.ToTensor(),
            ])

            # 创建数据集和数据加载器
            dataset = ImageVectorDataset(new_image_path2, transform=transform)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # 初始化模型并加载预训练权重
            model = VisionTransformer(img_size, patch_size, num_classes, dim).to(device)
            model.load_state_dict(torch.load(args.net3_path))
            model.eval()

            # 预测
            with torch.no_grad():
                for images, img_names in data_loader:
                    images = images.to(device)
                    time1 = time.time()
                    outputs = model(images)
                    outputs = outputs.cpu().numpy()
                    print(time.time()-time1)

                    for img_name, output, in zip(img_names, outputs):
                        output_file = os.path.join(output_dir, img_name.replace('.png', '.csv'))
                        print(output)
                        result = pd.DataFrame({'Prediction':output/50000})
                        result.to_csv(output_file, index=False)


            ########  inverse process

            corrected_image_path = os.path.join(args.output_path, "corrected_image", str(undistort_image_index) + '_' + str(image_index))
            os.makedirs(corrected_image_path, exist_ok=True)

            #按顺序修复nurd图并保存
            for i in range(0,len(full_images)):
                distort_image_path = os.path.join(args.output_path, "distorted_cropped_image", str(undistort_image_index) + '_' + str(image_index), str(i+1) + ".png")
                dynamic_vector_path = os.path.join(args.output_path, "vector", str(undistort_image_index) + '_' + str(image_index), str(i+1) + ".csv")

                distort_image = cv2.imread(distort_image_path)
                restored_img = restore_nurd(distort_image_path, dynamic_vector_path, i)

                corrected_image_save_path = os.path.join(corrected_image_path, str(i+1) + ".png")
                cv2.imwrite(corrected_image_save_path, restored_img)

            #拼接为修复后的整张图并保存
            correct_whole = []
            for i in range(0,len(full_images)):
                corrected_image_save_path = os.path.join(corrected_image_path, str(i+1) + ".png")
                corrected_image = cv2.imread(corrected_image_save_path)

                if i == 0:
                    correct_whole.append(oct_image_whole[:,:nurd_coord[i,0]+1,:])

                if i == len(full_images) - 1:
                    correct_whole.append(corrected_image)
                    correct_whole.append(oct_image_whole[:,nurd_coord[i,1]:,:])
                else:
                    correct_whole.append(corrected_image)
                    correct_whole.append(oct_image_whole[:,nurd_coord[i,1]:nurd_coord[i+1,0],:])

            correct_whole = np.concatenate(correct_whole, 1)

            correct_whole_path = os.path.join(corrected_image_path, "whole.png")
            cv2.imwrite(correct_whole_path, correct_whole)

    print("Done.")


# csv_predpath = "results_new/vector/1/8.csv"
# csv_gtpath = "results_new/vector_gt/1/8.csv"

# data_gt = pd.read_csv(csv_gtpath)
# gt_y = np.array(data_gt)[:, 0]
# gt_y = gt_y[gt_y != 0]
# width = gt_y.shape[0]
# data_pred = pd.read_csv(csv_predpath)
# pred_y = np.array(data_pred)[:, 0]
# pred_y = pred_y[:width]

# x = np.linspace(0,width,width)
# plt.figure()
# plt.plot(x, gt_y)
# plt.plot(x, pred_y)
# plt.legend(('Ground Truth','Prediction'))
# plt.savefig(os.path.join("test", "dynamic.png"))
