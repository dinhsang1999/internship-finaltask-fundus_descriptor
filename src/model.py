import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


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
            self.pretrained_block.fc = nn.Linear(
                self.pretrained_block.fc.in_features, 7)
            self.sigmoid = torch.nn.Sigmoid()

        if self.model_type == 'resnet50':
            # Get pretrained model
            self.pretrained_block = models.resnet50(
                pretrained=True, progress=True)
            # Finetune the last layer for binary classification

            num_features = self.pretrained_block.fc.in_features
            self.pretrained_block.fc = torch.nn.Linear(num_features, 7)
            self.softmax = torch.nn.Sigmoid()

        if self.model_type == 'resnet101':
            # Get pretrained model
            self.pretrained_block = models.resnet101(
                pretrained=True, progress=True)

            # Finetune the last layer for binary classification
            num_features = self.pretrained_block.fc.in_features
            self.pretrained_block.fc = nn.Linear(
                self.pretrained_block.fc.in_features, 7)
            self.sigmoid = torch.nn.Sigmoid()

        if self.model_type == 'vgg16':
            self.pretrained_block = models.vgg16(
                pretrained=True, progress=True)
            num_features = self.pretrained_block.classifier[6].in_features
            self.pretrained_block.classifier[6] = torch.nn.Linear(
                num_features, 7)
            self.sigmoid = torch.nn.Sigmoid()

        if self.model_type == 'alexnet':
            self.pretrained_block = models.alexnet(
                pretrained=True, progress=True)
            num_features = self.pretrained_block.classifier[6].in_features
            self.pretrained_block.classifier[6] = torch.nn.Linear(
                num_features, 3)
            self.softmax = torch.nn.LogSoftmax(dim=1)

        if self.model_type == 'densenet121':
            self.pretrained_block = models.densenet121(
                pretrained=True, progress=True)
            num_features = self.pretrained_block.fc.in_features
            self.pretrained_block.fc = torch.nn.Linear(num_features, 3)
            self.softmax = torch.nn.LogSoftmax(dim=1)

        if self.model_type == 'efficientnet_b0':
            self.pretrained_block = EfficientNet.from_name(
                'efficientnet-b0')  # EfficientNet.from_name('efficientnet-b0')
            num_features = self.pretrained_block._fc.in_features
            self.pretrained_block._dropout = torch.nn.Dropout(
                p=0.5, inplace=True)
            self.pretrained_block._fc = torch.nn.Linear(num_features, 7)
            self.sigmoid = torch.nn.Sigmoid()

        if self.model_type == 'efficientnet_b1':
            self.pretrained_block = EfficientNet.from_pretrained(
                'efficientnet-b1', num_classes=7)  # EfficientNet.from_name('efficientnet-b0')
            self.pretrained_block._dropout = torch.nn.Dropout(
                p=0.5, inplace=True)

            self.sigmoid = torch.nn.Sigmoid()

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
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    trial_model = NeuralNetwork('efficientnet_b0')  # resnet18 #vgg16
    print(trial_model)
