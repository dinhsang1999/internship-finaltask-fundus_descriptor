from src.model import NeuralNetwork
import torch
from torchvision import transforms
import json
from src.trainer_odm import CustomTrainer
from src.dataGenerator import check_size_crop_square
import os

from torchvision.io import read_image
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pprint import pprint


class Predicter():
    '''
    Predict the label of new image base on trained model
    '''

    def __init__(self,path_predict,src, model_type='resnet18', using_gpu=True):
        """
        Construct the predicter object.

        Args:
            model_type (str, optional): Type of model architecture.
                Defaults to 'resnet18'.
            using_gpu (bool, optional): GPU enable option. Defaults to True.
        """

        # Load training parameters
        params = json.load(open('config/config.json', 'r'))

        # Create CustomTrainer instance with loaded training parameters
        trainer = CustomTrainer(**params)

        # Check device
        self.device = 'cuda' if torch.cuda.is_available() and using_gpu else 'cpu'

        # Create model
        self.model = NeuralNetwork(trainer.ARCHITECTURE).to(self.device)

        # Load trained model
        self.model.load_state_dict(torch.load(os.path.join(
            trainer.MODEL_DIR, "trial-" + trainer.TRIAL + ".pth")))

        # Switch model to evaluation mode
        self.model.eval()

        # Image processing
        self.height = 224
        self.width = self.height * 1
        self.transform = transforms.Compose([
            transforms.Resize(
                (int(self.width),
                int(self.height)))])

        self.image_path = path_predict
        self.src = src

    def predict(self):
        """
        Predict image in image_path is peripheral or central.

        Args:
            image_path (str): Directory of image file.

        Returns:
            result (dict): Dictionary of propability of 2 classes,
                and predicted class of the image.
        """

        # Read image
        image = read_image(self.image_path)

        if self.src == "CTEH":
            image = check_size_crop_square(image)
        # Transform image
        image = self.transform(image)
        image = image.view(1, *image.size()).to(self.device)
        # Result
        labels = ['od', 'macula','null']
        result = {'prob_od': 0, 'prob_macula': 0,'prob_null': 0, 'label': labels[0]}

        # Predict image
        with torch.no_grad():
            output = self.model(image.float())
            ps = torch.exp(output)
            result['prob_od'] = float(ps[0][0].item())
            result['prob_macula'] = float(ps[0][1].item())
            result['prob_null'] = float(ps[0][2].item())
            result['label'] = labels[0] if (result['prob_od'] > result['prob_macula']) and (result['prob_od'] > result['prob_null']) else labels[1] if result['prob_macula'] > result['prob_null'] else labels[2]
        
        return result
