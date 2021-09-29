import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet


class NeuralNetwork(nn.Module):
    """
    Model architecture for training
    """

    def __init__(self, model_type, dropout_rate=0.2):
        """Construct NeuralNetwork object and initialize member variables

        Args:
            model_type (String): name of model architecture ("resnet34" or "efficientnet_b0")
            dropout_rate (float, optional): the dropout rate before the final dense layer. Defaults to 0.2.
        """
        super(NeuralNetwork, self).__init__()
        print("construct model")
        self.model_type = model_type.lower()
        self.dropout_rate = dropout_rate
        # print("self.dropout_rate", self.dropout_rate)

        if self.model_type == 'resnet34':
            # Get pretrained model
            self.pretrained_block = models.resnet34(
                pretrained=True, progress=True)

            # Finetune the last layer for binary classification with multilabel
            num_features = self.pretrained_block.fc.in_features
            self.pretrained_block.fc = nn.Sequential(
                nn.Dropout(
                    p=self.dropout_rate, inplace=False), nn.Linear(
                    num_features, 6))
            self.sigmoid = torch.nn.Sigmoid()

        if self.model_type == 'efficientnet_b0':
            # Get pretrained model
            self.pretrained_block = EfficientNet.from_pretrained(
                'efficientnet-b0')

            # Finetune the last layer for binary classification with multilabel
            num_features = self.pretrained_block._fc.in_features
            self.pretrained_block._dropout = nn.Dropout(
                p=self.dropout_rate, inplace=False)
            self.pretrained_block._fc = torch.nn.Linear(num_features, 6)
            self.sigmoid = torch.nn.Sigmoid()

        print("initialize model complete")

    def forward(self, x):
        """
        Define forward pass in network training

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 3, 224, 224]
                (a batch of input images)

        Returns:
            x (torch.Tensor): Output tensor of shape [batch_size, 6]
                (binary classification propability for each of 6 labels per samples in
                input batch)
        """
        x = self.pretrained_block(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    trial_model = NeuralNetwork('resnet34')  # 'efficientnet_b0'
    print(trial_model)
