import torch
from torchvision.io import read_image
import os
import torchvision.transforms as transforms
import numpy as np
import skimage.io as io
import sys
import pandas as pd
import csv

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DataGenerator:
    def __init__(self, df, data_dir, train=True):
        image_size = [224, 224]

        image_transformation_1 = [
            transforms.Resize(image_size),
        ]
        image_transformation_2 = [
            transforms.RandomHorizontalFlip(p=1)
        ]
        image_transformation_3 = [
            transforms.RandomVerticalFlip(p=1)
        ]

        image_transforms_1 = transforms.Compose(image_transformation_1)

        image_transforms_2 = transforms.Compose(image_transformation_2)

        image_transforms_3 = transforms.Compose(image_transformation_3)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.img_labels = df
        self.img_dir = data_dir
        self.train_mode = train
        self.agu_label = torch.FloatTensor().to(self.device)
        self.agu_img = torch.FloatTensor().to(self.device)
        self.agu_img = []
        self.agu_label = []

        self.transform_1 = image_transforms_1
        self.transform_2 = image_transforms_2
        self.transform_3 = image_transforms_3

    def __len__(self):
        return len(self.img_labels)

    def __call__(self):
        image = torch.FloatTensor().to(self.device)
        image_2 = torch.FloatTensor().to(self.device)
        image_3 = torch.FloatTensor().to(self.device)
        image_4 = torch.FloatTensor().to(self.device)
        for idx in range(len(self.img_labels)):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            src = self.img_labels.iloc[idx, 2]
            #

            if src == "CTEH":
                image = check_size_crop_square(image)

            image = self.transform_1(image)
            self.agu_img.append(image)
            self.agu_label.append(label)

            if self.train_mode:
                image_3 = self.transform_3(image)
                self.agu_img.append(image_3)
                self.agu_label.append(label)

        return self.agu_img, self.agu_label


def check_size_crop_square(image):
    if image.shape[1] != image.shape[2]:

        if image.shape[1] < image.shape[2]:
            rateCrop = image.shape[1]
        else:
            rateCrop = image.shape[2]

        transforms_size = transforms.Compose([
            transforms.CenterCrop(
                size=[round(0.95*image.shape[1]), round(0.95*image.shape[2])]),
            transforms.CenterCrop(round(rateCrop*0.95))
        ])

        tranformation = transforms_size(image)

    else:
        tranformation = image

    return tranformation

#swap label if using HorizontalFlip
def swap_label(label):  # Exp: label = [1, 0, 0, 1, 1, 0, 0]
    if label[2] == 0:
        label[2] = 1
    else:
        label[2] = 0

    if label[3] == 0:
        label[3] = 1
    else:
        label[3] = 0

    return label
