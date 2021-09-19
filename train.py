import torch
from torch.nn import parameter
from src.trainer import CustomTrainer
from src.keyboardinterrupt import keyboardInterruptHandler
import json


def train():
    """
    Train the model with configuration loaded from json file
    FIXME:
    1.  TRAIN_VAL_SPLIT_STATUS at config.json if need 'train_test_split' function.
    2.  use_checkpoint: apply early stoping
    3.  best_model: save the best model in training or validation
    """

    # Load training parameters
    params = json.load(open('config/config.json', 'r'))

    # params_string = json.dumps(params, indent=4, separators=", ")

    # Create CustomTrainer instance with loaded training parameters
    trainer = CustomTrainer(**params)

    # Set up DataLoader
    if trainer.TRAIN_VAL_SPLIT_STATUS or trainer.VAL_MODE:
        train_dataloader, test_dataloader = trainer.setup_training_data()
    else:
        train_dataloader = trainer.setup_training_data()
    # Set up training details
    model, optimizer, loss_fn, device = trainer.setup_training()

    # Loop through epochs
    for epoch in range(trainer.EPOCHS):
        # Train model
        if trainer.VAL_MODE or trainer.TRAIN_VAL_SPLIT_STATUS:
            trainer.epoch_train(
                train_dataloader,
                model,
                loss_fn,
                optimizer,
                device,
                epoch,
                use_checkpoint=False,
                best_model=False
            )

            # Calculate validation loss and accuracy score
            trainer.epoch_evaluate(
                test_dataloader,
                model,
                loss_fn,
                device,
                epoch,
                use_checkpoint=False,
                best_model=True
            )
        else:
            trainer.epoch_train(
                train_dataloader,
                model,
                loss_fn,
                optimizer,
                device,
                epoch,
                use_checkpoint=False,
                best_model=True
            )

        if trainer.early_stop():
            print("EARLY STOPING!!!")
            break

        trainer.writer.flush()
        trainer.writer.close()

    print('TRAINING DONE')


if __name__ == '__main__':
    train()
