import torch
from torch import nn
from torchvision import models


class NeuralNetwork(nn.Module):
    """
    Model architecture for training
    """

    def __init__(self, model_type):
        """
        Construct NeuralNetwork object and initialize member variables
        """
        super(NeuralNetwork, self).__init__()

        self.model_type = model_type.lower()

        if self.model_type == 'resnet18':
            # Get pretrained model
            self.pretrained_block = models.resnet18(
                pretrained=True, progress=True)

            # Finetune the last layer for binary classification
            num_features = self.pretrained_block.fc.in_features
            self.pretrained_block.fc = torch.nn.Linear(num_features, 2)
            self.softmax = torch.nn.LogSoftmax(dim=1)

        if self.model_type == 'vgg16':
            self.pretrained_block = models.vgg16(
                pretrained=True, progress=True)
            num_features = self.pretrained_block.classifier[6].in_features
            self.pretrained_block.classifier[6] = torch.nn.Linear(
                num_features, 2)
            self.softmax = torch.nn.LogSoftmax(dim=1)

        if self.model_type == 'alexnet':
            self.pretrained_block = models.alexnet(
                pretrained=True, progress=True)
            num_features = self.pretrained_block.classifier[6].in_features
            self.pretrained_block.classifier[6] = torch.nn.Linear(
                num_features, 2)
            self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Define forward pass in network training

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224]
                (a batch of input images)

        Returns:
            x (torch.Tensor): Output tensor of shape [batch_size, 2]
                (binary classification propability for each image in
                input batch)
        """
        x = self.pretrained_block(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    trial_model = NeuralNetwork('resnet18')  # resnet18 #vgg16
    print(trial_model)
