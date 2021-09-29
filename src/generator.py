import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torchvision.transforms as transforms
import pandas as pd
import json
from src.utils import preprocess
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Generator(Dataset):
    def __init__(self, df, data_dir, transform=None, target_transform=None):
        """ Construct generator object for converting image files and dataframes to preprocessed tensors of image and label

        Args:
            df (pandas.core.frame.DataFrame): dataframe of image filenames and labels
            data_dir (string): path to image folder
            transform (torchvision.transforms.transforms.Compose, optional): The combination of transformations applied on image. Defaults to None.
            target_transform (Callable, optional): The transformations applied on label tensor. Defaults to None.
        """
        image_size = [224, 224]
        image_transformation = [
            transforms.Resize(image_size),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        image_transformation = transforms.Compose(image_transformation)

        self.img_labels = df
        self.img_dir = data_dir
        self.transform = image_transformation
        self.target_transform = target_transform

        self.image_list = []
        self.label_list = []
        print("generator constructor success!")

    def __len__(self):
        """ Get the number of samples in the data generator
        """
        return len(self.img_labels)

    def __call__(self, train=False):  # , train=False
        """ Convert image files and dataframes to preprocessed tensors of image and label

        Args:
            train (bool, optional): Whether to save the resulting tensors as of training set or test set. Defaults to False.

        Returns:
            image_array (torch.Tensor): image tensor of the dataset with shape [len_dataset, 3, 224, 224]
            label_array (torch.Tensor): label tensor of a sample with shape [len_dataset, 6]
        """
        for idx in range(len(self.img_labels)):
            img_path = os.path.join(
                self.img_dir, str(self.img_labels.iloc[idx, 0]))
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)

            label = (torch.FloatTensor(label))

            self.image_list.append(image)
            self.label_list.append(label)

        image_array = torch.stack(self.image_list, dim=0)
        label_array = torch.stack(self.label_list, dim=0)

        if train:
            torch.save(image_array, "tensor/train_imgs_array.pt")
            torch.save(label_array, "tensor/train_labels_array.pt")
        else:
            torch.save(image_array, "tensor/test_imgs_array.pt")
            torch.save(label_array, "tensor/test_labels_array.pt")
        print(".pt file saved!")

        return image_array, label_array
