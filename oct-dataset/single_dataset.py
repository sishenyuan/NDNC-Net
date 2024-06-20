# modified from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html
import torch
import os
from PIL import Image
from torchvision import datasets, transforms

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        # path= root_dir + "/images/*.png"
        img_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        self.image_files = [os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith('.png')]
        self.image_files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)
        
        return image
    
# Define the transformation for the dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomCrop([1024,3200]),
    # transforms.Resize([1024,6400]),
    transforms.Normalize((0.5,), (0.5,)) # Assume that the data is normal distribution, with mean 0.5 and SD 0.5. 
    # Here the z score normalization is carried out such taht the mean becomes 0 and SD = 1. 
    # This will allow better convergence and training speed. 
])

# Create an instance of the custom dataset
custom_dataset = CustomDataset(root_dir='.', transform=transform)

# Create a data loader for the dataset
data_loader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=False, num_workers=4)
