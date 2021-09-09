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
    def __init__(self, df, data_dir):
        image_size = [224, 224]

        image_transformation_1 = [
            transforms.Resize(image_size)
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

        self.img_labels = df
        self.img_dir = data_dir
        self.agu_img = []
        self.agu_label = []
            
        self.transform_1 = image_transforms_1
        self.transform_2 = image_transforms_2
        self.transform_3 = image_transforms_3
    
    def __len__(self):
        return len(self.img_labels)

    def __call__(self):
        for idx in range(len(self.img_labels)):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            src = self.img_labels.iloc[idx, 2]

            #
            if src == 1:
                image = check_size_crop_square(image)
            
            image = self.transform_1(image)
            self.agu_img.append(image)
            self.agu_label.append(label)


            # image_2 = self.transform_2(image)
            # self.agu_img.append(image_2)
            # self.agu_label.append(label)
            
            # image_3 = self.transform_3(image)
            # self.agu_img.append(image_3)
            # self.agu_label.append(label)

            # image_4 = self.transform_3(image_2)
            # self.agu_img.append(image_4)
            # self.agu_label.append(label)
        return self.agu_img, self.agu_label

def check_size_crop_square(image):
    if image.shape[1] != image.shape[2]:

        if image.shape[1] < image.shape[2]:
            rateCrop=image.shape[1]
        else:
            rateCrop=image.shape[2]
        
        transforms_size = transforms.Compose([
        transforms.CenterCrop(size=[round(0.95*image.shape[1]),round(0.95*image.shape[2])]),
        transforms.CenterCrop(round(rateCrop*0.95))  
        ])

        tranformation = transforms_size(image)

    else: tranformation = image
        
    return tranformation 

def swap_label(label):
    if label == 0:
        label = 1
    else:
        label = 0
    return label