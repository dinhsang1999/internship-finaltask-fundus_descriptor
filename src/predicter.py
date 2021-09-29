from src.model import NeuralNetwork
import torch
from torchvision import transforms
import json
from src.trainer import CustomTrainer
import os

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Predicter():
    '''
    Predict the label of new image base on trained model
    '''

    def __init__(self, model_type='resnet34', using_gpu=True):
        """
        Construct the predicter object.

        Args:
            model_type (str, optional): Type of model architecture.
                Defaults to 'resnet34'.
            using_gpu (bool, optional): GPU enable option. Defaults to True.
        """

        # Load training parameters
        params = json.load(open('config/config_predict.json', 'r'))

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
                 int(self.height))),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])])

        self.image_path = trainer.PREDICT_IMAGE_PATH

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
        image = Image.open(self.image_path).convert('RGB')

        # Transform image
        image = self.transform(image)
        image = image.view(1, *image.size()).to(self.device)
        # Result
        labels = ['central', 'peripheral', 'left', 'right', 'od', 'macula']
        result = {
            'prob_central': 0,
            'prob_peripheral': 0,
            'prob_left': 0,
            'prob_right': 0,
            'prob_od': 0,
            'prob_macula': 0,
            'label': []}

        # Predict image
        with torch.no_grad():
            # Forward pass
            output = self.model(image)

            # Decode output
            result['prob_central'] = float(output[0][0].item())
            result['prob_peripheral'] = float(output[0][1].item())
            result['prob_left'] = float(output[0][2].item())
            result['prob_right'] = float(output[0][3].item())
            result['prob_od'] = float(output[0][4].item())
            result['prob_macula'] = float(output[0][5].item())

            # Derive labels
            if result['prob_central'] > result['prob_peripheral']:
                result['label'].append(labels[0])
            else:
                result['label'].append(labels[1])

            if result['prob_left'] > result['prob_right']:
                result['label'].append(labels[2])
            else:
                result['label'].append(labels[3])

            if (result['prob_od'] > 0.5) or (result['prob_macula'] > 0.5):
                if (result['prob_od'] > result['prob_macula']):
                    result['label'].append(labels[4])
                else:
                    result['label'].append(labels[5])
            else:
                pass

        return result
