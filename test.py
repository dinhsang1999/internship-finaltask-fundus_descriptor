import torch
from src.utils import preprocess
from src.trainer import CustomTrainer
import json
import os


def test():
    """
    Test the model with configuration loaded from json file
    Return:
        Metric of each class & total metric of train dataset and test dataset
    """

    # Load testing parameters
    params = json.load(open('config/config.json', 'r'))

    # Create CustomTrainer instance with loaded testing parameters
    trainer = CustomTrainer(**params)

    # Set up DataLoader
    train_dataloader, test_dataloader = trainer.setup_training_data()

    # Set up training details
    model, _, loss_fn, device = trainer.setup_training()

    # Load trained model
    model.load_state_dict(torch.load(os.path.join(
        trainer.MODEL_DIR, "trial-" + trainer.TRIAL +
        "-model-" + trainer.ARCHITECTURE +
        "-optim-" + trainer.OPTIMIZER +
        "-lr-" + str(trainer.LEARNING_RATE) +
        ".pth")))

    # Calculate test loss and accuracy
    trainer.epoch_evaluate(train_dataloader, model, loss_fn,
                           device, train_or_validation="validation")

    # Calculate test loss and accuracy
    trainer.epoch_evaluate(test_dataloader, model, loss_fn,
                           device, train_or_validation="validation")


if __name__ == '__main__':
    test()
