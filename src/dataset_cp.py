import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torchvision.transforms as transforms
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CustomImageDataset(Dataset):
    def __init__(self, df, data_dir, transform=None, type_transform=True, target_transform=None):
        image_size = [224, 224]

        image_transformation_1 = [
            transforms.Resize(image_size),
            transforms.ConvertImageDtype(torch.float)
        ]

        image_transformation_2 = [
            transforms.Resize(image_size),
            transforms.ConvertImageDtype(torch.float),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomVerticalFlip(p=0.3)
        ]

        image_transformation_1 = transforms.Compose(image_transformation_1)
        image_transformation_2 = transforms.Compose(image_transformation_2)

        self.img_labels = df
        self.img_dir = data_dir
        self.type_transform = transform
        self.transform_1 = image_transformation_1
        self.transform_2 = image_transformation_2
        self.first_ = type_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = check_size_crop_square(image)
        label = self.img_labels.iloc[idx, 1]
        if self.type_transform:
            if self.first_:
                image = self.transform_1(image)
            else:
                image = self.transform_2(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def check_size_crop_square(image):

    if image.shape[1] != image.shape[2]:
        if image.shape[1] < image.shape[2]:
            rateCrop = image.shape[1]
        else:
            rateCrop = image.shape[2]

        transforms_size = transforms.Compose([
            transforms.CenterCrop(
                [round(image.shape[1]*0.95), round(image.shape[2]*0.95)]),
            transforms.CenterCrop(round(rateCrop*0.95))
        ])

        tranformation = transforms_size(image)

    else:
        tranformation = image

    return tranformation

# def flip_transform(imgdir,label):
#     img = imread(imgdir)
#     x = np.array(img)
#     y = label
#     final_train_data = []
#     final_target_train = []
#     final_train_data.append(x)
#     final_train_data.append(np.fliplr(x))
#     final_target_train.append(y)
#     final_target_train.append(y)

#     final_train = np.array(final_train_data)
#     final_train.transpose(0,3,1,2)
#     out_data=torch.from_numpy(final_train)

#     final_target_train = np.array(final_target_train)
#     final_target_train = final_target_train.astype(int)
#     out_label=torch.from_numpy(final_target_train)

#     return out_data,out_label
