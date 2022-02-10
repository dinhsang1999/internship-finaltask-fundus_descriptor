import torch
import sys
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CustomImageDataset(Dataset):
    '''Augmentation when training'''

    def __init__(self, image, label, transform=None, target_transform=None, diff=False):
        '''
        Args:
            image(list): list of all input image, components are weight of image
            label(list): list of labels
            transform(function): transform function on images
                                    Default: None
            target_transform(function): transform function on labels
                                    Default: None
            diff(bool): Turn on Augmention mode
                                    Default: False
        '''
        np.random.seed(1)
        torch.manual_seed(1)
        if (diff == True):
            image_transformation_2 = [
                transforms.RandomAutocontrast(p=0.2),
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.GaussianBlur(kernel_size=(7, 9), sigma=(0.1, 2))]), p=0.5),
                transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)]), p=0.5),
                transforms.RandomInvert(p=0.2),
                transforms.RandomPosterize(bits=2),
                transforms.RandomAdjustSharpness(sharpness_factor=2),
            ]
        else:
            image_transformation_2 = [
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]

        image_transformation_2 = transforms.Compose(image_transformation_2)

        if transform:
            self.transform_2 = transform
        else:
            self.transform_2 = image_transformation_2

        self.img_data = image
        self.label_data = label
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        image = self.img_data[idx]
        label = self.label_data[idx]
        label = (torch.FloatTensor(label))
        image = self.transform_2(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
