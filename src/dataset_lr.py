import torch
import sys
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, image, label, transform=None, target_transform=None,diff=0):
        
        if (diff != 0):
            image_transformation_2 = [
                transforms.RandomVerticalFlip(p=diff),
                transforms.ColorJitter(brightness=diff, contrast= diff, hue= 0.1 if diff < 0.4 else 0.2),
                transforms.GaussianBlur(kernel_size=(7, 9), sigma=(0.1, diff*10+0.1))
            ]
        else: 
            image_transformation_2 = []

        image_transformation_2 = transforms.Compose(image_transformation_2)

        if transform:
            self.transform_2 = transform
        else:
            self.transform_2 = image_transformation_2


        self.img_data = image
        self.label_data = label
        self.target_transform = target_transform #TODO:

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        
        image = self.img_data[idx]
        label = self.label_data[idx]
        image = self.transform_2(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    

    
