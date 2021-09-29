import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torchvision.transforms as transforms
import pandas as pd

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = [224, 224]


class CustomImageDataset(Dataset):
    def __init__(
            self,
            img_tensor,
            label_tensor,
            transform=None,
            target_transform=None,
            augment=True,
            rotation=0,
            translation=None,
            scaling=None):
        """Construct dataset object, prepare image for dataloader

        Args:
            img_tensor (torch.Tensor): Image tensor of shape [batch_size, 3, 224, 224]
            label_tensor (torch.Tensor): Label tensor of shape [batch_size, 6]
            transform (torchvision.transforms.transforms.Compose, optional): The combination of transformations applied on image. Defaults to None.
            target_transform (Callable, optional): The transformations applied on label tensor. Defaults to None.
            augment (bool, optional): Whether to apply the augmentation transformation or not. Defaults to True.
            rotation (int or list, optional): range of rotation degree of augmentation. Defaults to 0.
            translation (tupple or list, optional): range of translation degree of augmentation. Defaults to None.
            scaling (tupple or list, optional): range of scaling degree of augmentation. Defaults to None.
        """
        # print("init dataset")
        torch.manual_seed(17)

        image_transformation = [
            transforms.RandomAffine(
                degrees=rotation,
                translate=translation,
                scale=scaling)]
        image_transformation = transforms.Compose(image_transformation)

        self.img_tensor = img_tensor
        self.label_tensor = label_tensor

        if transform is None:
            self.transform = image_transformation
        else:
            self.transform = transform

        self.target_transform = target_transform
        self.augment = augment

    def __len__(self):
        """ Get the number of samples in the dataset
        """
        return len(self.img_tensor)

    def __getitem__(self, idx):
        """ Get the image and label of index "idx" in the dataset

        Args:
            idx (int): index number of a sample in the dataset

        Returns:
            image (torch.Tensor): image tensor of a sample with shape [3, 224, 224]
            label (torch.Tensor): label tensor of a sample with shape [6]
        """
        image = self.img_tensor[idx]
        label = self.label_tensor[idx]

        if self.augment:
            if self.transform:
                image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
