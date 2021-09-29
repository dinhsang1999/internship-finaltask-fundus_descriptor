import torch
from src.trainer import CustomTrainer
import json
import os
from src.keyboard_interrupt import keyboardInterruptHandler
import timeit


def train():
    """
    Train the model with configuration loaded from json file
    """

    # Load training parameters
    params = json.load(open('config/config.json', 'r'))

    # params_string = json.dumps(params, indent=4, separators=", ")
    # print(params_string)

    # Create CustomTrainer instance with loaded training parameters
    trainer = CustomTrainer(**params)

    # print(trainer.__dict__)

    # Set up DataLoader
    train_dataloader, test_dataloader = trainer.setup_training_data()

    # Set up training details
    model, optimizer, loss_fn, device = trainer.setup_training()

    # Load pretrained model
    pretrained_model_path = os.path.join(
        trainer.MODEL_DIR, "trial-" + trainer.TRIAL + ".pth")
    if os.path.exists(pretrained_model_path):
        # Load trained model
        model.load_state_dict(torch.load(pretrained_model_path))

    # print(model)
    # Loop through epochs
    for epoch in range(trainer.EPOCHS):  # epochs
        # Train model
        trainer.epoch_train(
            train_dataloader,
            model,
            loss_fn,
            optimizer,
            device,
            epoch)

        # Calculate validation loss and accuracy score
        trainer.epoch_evaluate(
            test_dataloader,
            model,
            loss_fn,
            device,
            epoch,
            use_checkpoint=True)
        if trainer.early_stop():
            print("EARLY STOPING!!!")
            break

        trainer.writer.flush()
        trainer.writer.close()

    print('TRAINING DONE')


if __name__ == '__main__':
    start = timeit.default_timer()
    train()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    # try:
    #     train()
    # except KeyboardInterrupt:
    #     print("something's wrong")
