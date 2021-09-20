import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torchvision.transforms as transforms
import pandas as pd

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CustomImageDataset(Dataset):    
    def __init__(self, img_tensor, label_tensor, transform=None, target_transform=None, augment=True, rotation = 0, translation=None, scaling=None):
        # print("init dataset")
        torch.manual_seed(17)
        image_size = [224, 224]
        image_transformation = [
            #type of augmentation ???
            # transforms.Resize(image_size),
            # transforms.ConvertImageDtype(torch.float),
            # transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]

            # transforms.CenterCrop(size=180),

            transforms.RandomAffine(degrees=rotation, translate=translation, scale=scaling)
            # transforms.RandomAffine(degrees=0, translate=None, scale=None)
            # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            # transforms..GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        ]
        image_transformation = transforms.Compose(image_transformation)
        print(image_transformation)
        self.img_tensor = img_tensor
        self.label_tensor = label_tensor
        if transform is None:
            self.transform = image_transformation
        else:
            self.transform = transform

        self.target_transform = target_transform
        self.augment = augment
        # print(self.img_labels)

    def __len__(self):
        # print("len")
        return len(self.img_tensor)

    def __getitem__(self, idx):
        # print("'get item", idx)
        # print("idx: " + str(idx))
        # print(self.img_labels.iloc[idx, 0])
        # print(self.img_labels.iloc[idx, 1])
        # print()
        image = self.img_tensor[idx]
        label = self.label_tensor[idx]

        # print(self.transform)
        if self.augment == True:
            if self.transform:
                image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)        

        # label = (torch.FloatTensor(label))

        # print(label)
        # print(label.dtype)
        # print((label.long()).dtype)
        return image, label

# if __name__ == '__main__':
    # my_dataframe = pd.read_csv('../pseudolabel_done.csv')
    # my_dataset = CustomImageDataset(my_dataframe, '../data/data')
    # print(my_dataset.__len__())
    # print(my_dataset.__getitem__()

