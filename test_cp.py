import torch
from src.utils_cp import preprocess, blockPrint, enablePrint
from src.trainer_cp import CustomTrainer
from src.dataset_cp import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
import json
import os
import sys
import pandas as pd
import numpy as np


def test():
    """
    Test the model with configuration loaded from json file
    """
    blockPrint()

    # Load testing parameters
    params = json.load(open('config/config.json', 'r'))

    # Create CustomTrainer instance with loaded testing parameters
    trainer = CustomTrainer(**params)

    # FIXME:
    path_csv_test = 'csvConvert/val_cp.csv'
    data_val = pd.read_csv('csvConvert/val_cp.csv')

    # Set up DataLoader
    # _, test_dataloader = trainer.setup_training_data()
    df_test = preprocess(
        trainer.DATA_DIR, csv_dir=path_csv_test, train_val_split_status=False)

    test_dataset = CustomImageDataset(
        df_test,
        trainer.DATA_DIR,
        transform=trainer.TRANSFORM_IMAGE,
        type_transform=True,
        target_transform=Lambda(
            lambda y: torch.tensor(y).type(
                torch.long)))

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=round(len(data_val)/10),  # FIXME:
        shuffle=True)
    # Set up training details
    model, _, loss_fn, device = trainer.setup_training()

    # Load trained model
    model.load_state_dict(torch.load(os.path.join(
        trainer.MODEL_DIR, "trial-" + trainer.TRIAL + ".pth")))
    enablePrint()
    # Calculate test loss and accuracy
    trainer.epoch_evaluate(test_dataloader, model,
                           loss_fn, device, use_checkpoint=False)


if __name__ == '__main__':
    test()
