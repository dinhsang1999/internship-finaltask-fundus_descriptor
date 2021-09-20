import torch
from torch import nn
from torchvision import models
from efficientnet_pytorch import EfficientNet



class NeuralNetwork(nn.Module):
    """
    Model architecture for training
    """

    def __init__(self, model_type, dropout_rate = 0.2):
        """
        Construct NeuralNetwork object and initialize member variables
        """
        super(NeuralNetwork, self).__init__()
        print("construct model")
        self.model_type = model_type.lower()
        self.dropout_rate = dropout_rate
        print("self.dropout_rate", self.dropout_rate)


        if self.model_type == 'resnet18':
            # Get pretrained model
            self.pretrained_block = models.resnet18(
                pretrained=True, progress=True)

            # Finetune the last layer for binary classification
            num_features = self.pretrained_block.fc.in_features
            self.pretrained_block.fc = torch.nn.Linear(num_features, 6)
            # self.softmax = torch.nn.LogSoftmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()

        if self.model_type == 'resnet34':
            # Get pretrained model
            self.pretrained_block = models.resnet34(
                pretrained=True, progress=True)

            # Finetune the last layer for binary classification
            num_features = self.pretrained_block.fc.in_features
            # self.pretrained_block.fc = torch.nn.Linear(num_features, 6)
            # self.softmax = torch.nn.LogSoftmax(dim=1)
            self.pretrained_block.fc = nn.Sequential(nn.Dropout(p=self.dropout_rate,inplace=False),
                                                        nn.Linear(num_features,6),
                                                    )
            self.sigmoid = torch.nn.Sigmoid()


        if self.model_type == 'vgg16':
            self.pretrained_block = models.vgg16(
                pretrained=True, progress=True)
            num_features = self.pretrained_block.classifier[6].in_features
            self.pretrained_block.classifier[6] = torch.nn.Linear(
                num_features, 6)
            # self.softmax = torch.nn.LogSoftmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()

        if self.model_type == 'googlenet':
            self.pretrained_block = models.googlenet(
                pretrained=True, progress=True)
            num_features = self.pretrained_block.fc.in_features
            self.pretrained_block.fc = torch.nn.Linear(
                num_features, 6)
            # # self.softmax = torch.nn.LogSoftmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()            

        if self.model_type == 'densenet121':
            self.pretrained_block = models.densenet121(
                pretrained = True, progress= True)
            num_features = self.pretrained_block.classifier.in_features
            # print(num_features)
            self.pretrained_block.classifier = torch.nn.Linear(num_features, 6)
            # self.softmax = torch.nn.LogSoftmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()




        if self.model_type == 'efficientnet_b0':
            self.pretrained_block = EfficientNet.from_pretrained('efficientnet-b0') #EfficientNet.from_name('efficientnet-b0')
            num_features = self.pretrained_block._fc.in_features
            self.pretrained_block._dropout = nn.Dropout(p=self.dropout_rate,inplace=False)
            self.pretrained_block._fc = torch.nn.Linear(num_features, 6)
            
            # self.softmax = torch.nn.LogSoftmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()

            # features = model.extract_features(img)
            # print(features.shape) # torch.Size([1, 1280, 7, 7])

        if self.model_type == 'efficientnet_b1':
            self.pretrained_block = EfficientNet.from_pretrained('efficientnet-b1') #EfficientNet.from_name('efficientnet-b0')
            num_features = self.pretrained_block._fc.in_features

            self.pretrained_block._fc = torch.nn.Linear(num_features, 6)
            
            # self.softmax = torch.nn.LogSoftmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()

        if self.model_type == 'efficientnet_b2':
            self.pretrained_block = EfficientNet.from_pretrained('efficientnet-b2') #EfficientNet.from_name('efficientnet-b0')
            num_features = self.pretrained_block._fc.in_features

            self.pretrained_block._fc = torch.nn.Linear(num_features, 6)
            
            # self.softmax = torch.nn.LogSoftmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()
        
        if self.model_type == 'efficientnet_b3':
            self.pretrained_block = EfficientNet.from_pretrained('efficientnet-b3') #EfficientNet.from_name('efficientnet-b0')
            num_features = self.pretrained_block._fc.in_features

            self.pretrained_block._fc = torch.nn.Linear(num_features, 6)
            
            # self.softmax = torch.nn.LogSoftmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()

        if self.model_type == 'efficientnet_b4':
            self.pretrained_block = EfficientNet.from_pretrained('efficientnet-b4') #EfficientNet.from_name('efficientnet-b0')
            num_features = self.pretrained_block._fc.in_features

            self.pretrained_block._fc = torch.nn.Linear(num_features, 6)
            
            # self.softmax = torch.nn.LogSoftmax(dim=1)
            self.sigmoid = torch.nn.Sigmoid()
        print("initialize model complete")


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
        # x = self.softmax(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    trial_model = NeuralNetwork('resnet34') #'resnet34')  # resnet18 #vgg16 #'densenet121'  #'efficientnet_b0' #'googlenet'
    print(trial_model)

