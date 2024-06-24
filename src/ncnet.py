import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader

# 载入 (10, 2) 矩阵
# matrix = np.loadtxt('./original_distorted_data/nurd/matrix.txt', delimiter=' ')  # 修改文件路径为实际路径
# image_path = './original_distorted_data/distorted_image/1.png'  # 修改文件路径为实际路径

# # 从路径中载入图像
# image = Image.open(image_path)
# image_width, image_height = image.size

# # 切割并拼接图像
# new_image_width = 1000
# black_strip = Image.new('L', (image_width, image_height), 0)

# full_images=[]
# cropped_images = []

# for i, (left, right) in enumerate(matrix):
#     left, right = int(left), int(right)
#     cropped = image.crop((left, 0, right, image_height))
#     full_images.append(cropped)
#     padded = ImageOps.pad(cropped, (new_image_width, image_height), color=0)
#     cropped_images.append(padded)


# new_image_path1 = 'original_distorted_data/distorted_cropped_image/'
# for i in range(0,len(full_images)):
#     path=new_image_path1+str(i+1)+'.png'
#     new_image = full_images[i]
#     new_image.save(path)

# new_image_path2 = 'results/distorted_segment_images/'
# for i in range(0,len(cropped_images)):
#     path=new_image_path2+str(i+1)+'.png'
#     new_image = cropped_images[i]
#     new_image.save(path)

# Define dataset (Image-Vector with the same weight)
class ImageVectorDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, img_name

# Define Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim):
        super(VisionTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        self.dim = dim

        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=(patch_size[0], patch_size[1]),
                                         stride=(patch_size[0], patch_size[1]))
        self.bn1 = nn.BatchNorm2d(dim)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=4, dropout=0.1),
                                                 num_layers=12)
        self.fc = nn.Linear(dim * self.num_patches, num_classes * 2)
        self.bn2 = nn.BatchNorm1d(num_classes * 2)
        self.Relu = nn.LeakyReLU()
        self.f = nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.bn1(x)
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 0, 2).flatten(1)
        x = self.fc(x)
        x = self.bn2(x)
        x = self.Relu(x)
        x = self.f(x)
        return x

# 设置路径和参数
# image_dir = './results/distorted_segment_images/'
# weights_path = "./weights/epoch_99.pt"
# output_dir = "./results/vector"
# os.makedirs(output_dir, exist_ok=True)

# batch_size = 1
# img_size = [125, 128]
# patch_size = [img_size[0] // 16, img_size[1] // 1]
# dim = 64
# num_classes = 1000
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 定义数据转换
# transform = transforms.Compose([
#     transforms.Resize((img_size[0], img_size[1])),
#     transforms.ToTensor(),
# ])

# # 创建数据集和数据加载器
# dataset = ImageVectorDataset(image_dir, transform=transform)
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# # 初始化模型并加载预训练权重
# model = VisionTransformer(img_size, patch_size, num_classes, dim).to(device)
# model.load_state_dict(torch.load(weights_path))
# model.eval()

# # 预测
# with torch.no_grad():
#     for images, img_names in data_loader:
#         images = images.to(device)
#         time1 = time.time()
#         outputs = model(images)
#         outputs = outputs.cpu().numpy()
#         print(time.time()-time1)

#         for img_name, output, in zip(img_names, outputs):
#             output_file = os.path.join(output_dir, img_name.replace('.png', '.csv'))
#             print(output)
#             result = pd.DataFrame({'Prediction':output/50000})
#             result.to_csv(output_file, index=False)
