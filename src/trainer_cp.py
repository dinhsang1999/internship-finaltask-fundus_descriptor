from sys import path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from src.utils_cp import preprocess, EarlyStopping, calculate_metrics
from src.dataset_cp import CustomImageDataset
from src.model import NeuralNetwork
from torch.utils.tensorboard import SummaryWriter
import os


class Trainer(object):
    """
    Create parent object for passing keywords arguments from json file
    """

    def __init__(self, **args):
        for key in args:
            setattr(self, key.upper(), args[key])


class CustomTrainer(Trainer):
    """
    Class to control the training process
    """

    def __init__(self, **args):
        """
        Create CustomTrainer object
        """
        super(CustomTrainer, self).__init__(**args)
        print('Trainer class constructed')

        # Get train and test dataframe from data directory
        if self.TRAIN_VAL_SPLIT_STATUS:
            self.df_train, self.df_test = preprocess(
                self.DATA_DIR, self.CSV_DIR, train_val_split=self.TRAIN_VAL_SPLIT, train_val_split_status=self.TRAIN_VAL_SPLIT_STATUS)
        else:
            self.df_train = preprocess(self.DATA_DIR, self.CSV_DIR, train_val_split=self.TRAIN_VAL_SPLIT,
                                       train_val_split_status=self.TRAIN_VAL_SPLIT_STATUS)
        # Save best model
        self.best_acc_train = 0
        self.best_loss_train = 0
        self.best_acc_val = 0
        self.best_loss_val = 0
        self.path = os.path.join(
            self.MODEL_DIR, "trial-" + self.TRIAL + ".pth")
        # Initiate early stoppping object
        self.early_stopping = EarlyStopping(
            verbose=True, patience=self.PATIENCE, path=self.path, monitor=self.EARLY_STOPING)
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                "log",
                "trial-" + self.TRIAL +
                "-model-" + self.ARCHITECTURE +
                "-optim-" + self.OPTIMIZER +
                "-lr-" + str(self.LEARNING_RATE)))

    def epoch_train(self, dataloader, model, loss_fn, optimizer, device, epoch, use_checkpoint=False, best_model=False):
        """
        Train the model one time (forward propagation, loss calculation,
            back propagation, update parameters)

        Args:
            dataloader (torch.utils.data.dataloader.DataLoader):
                Dataset holder, load training data in batch.
            model (src_Tran.model.NeuralNetwork): Model architecture
            loss_fn (torch.nn.modules.loss.CrossEntropyLoss): Loss function
            optimizer (torch.optim.sgd.SGD): Optimization algorithm
            device (str): Device for training (GPU or CPU)
            epochs (int): Number of epochs
        """
        # Calculate number of samples
        size = len(dataloader.dataset)
        # Convert model to training state
        model.train()
        train_loss = 0
        correct = 0
        total_train = 0
        avr_acc = 0

        out_pred = torch.FloatTensor().to(device)
        out_gt = torch.FloatTensor().to(device)

        print("Epoch:", epoch)
        # Loop through samples in data loader
        for batch, (X, y) in enumerate(dataloader):
            # Load data and label batches to GPU/ CPU
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X.float())

            # Update groundtruth values
            out_gt = torch.cat((out_gt, y), 0)

            # Update prediction values
            out_pred = torch.cat((out_pred, pred), 0)

            # Compute loss
            y = y.type(torch.long)
            loss = loss_fn(pred, y)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()

            # Update variables
            optimizer.step()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # Acurancy
            total_train += y.nelement()
            train_accuracy = 100. * correct / total_train
            avr_acc += train_accuracy
            print('\tTraining batch {} Loss: {:.6f}, Accurancy:{:.3f}%'.format(
                batch + 1, loss.item(), train_accuracy))

        out_pred = out_pred.argmax(1)
        accuracy, precision, recall, f1_score, sensitivity, specificity = calculate_metrics(
            out_gt, out_pred)
        print('Training set: Average loss: {:.6f}, Average accuracy: {:.0f}/{} ({:.3f}%)\n'.format(
            train_loss / (batch + 1), correct, len(dataloader.dataset), 100 * accuracy))

        if use_checkpoint:
            # Earlystoping
            if self.EARLY_STOPING == 'val_loss':
                value = train_loss / (batch + 1)
            elif self.EARLY_STOPING == 'val_accuracy':
                value = correct / len(dataloader.dataset)
            else:
                value = None

            if value is not None:
                self.early_stopping.__call__(value, model)
            else:
                print(f'Save model to {self.path}')
                torch.save(model.state_dict(), self.path)

        if best_model:
            if (accuracy > self.best_acc_train):
                print(f'Training accuracy increased ({self.best_acc_train:.3f} --> {accuracy:.3f}).  Saving model to {self.path}')
                self.best_acc_train = accuracy
                torch.save(model.state_dict(), self.path)
                self.best_loss_train = (train_loss / (batch + 1))
            elif((train_loss / (batch + 1)) < self.best_loss_train) and (accuracy > self.best_acc_train-0.002):
                print(f'Training loss increased ({self.best_loss_train:.6f} --> {(train_loss / (batch + 1)):.6f}).  Saving model to {self.path}')
                self.best_loss_train = (train_loss / (batch + 1))
                torch.save(model.state_dict(), self.path)
            else:
                print(
                    'Accuracy_train and Loss_train not improve! the model will not save')

        self.writer.add_scalar("Loss/train", train_loss / (batch + 1), epoch)
        self.writer.add_scalar("Accuracy/train", accuracy, epoch)
        self.writer.add_scalar("Precision/train", precision, epoch)
        self.writer.add_scalar("Recall/train", recall, epoch)
        self.writer.add_scalar("F1_score/train", f1_score, epoch)
        self.writer.add_scalar("Sensitivity/train", sensitivity, epoch)
        self.writer.add_scalar("Specificity/train", specificity, epoch)

    def epoch_evaluate(self, dataloader, model, loss_fn, device, epoch=0, use_checkpoint=False, best_model=False):
        """
        Evaluate the model one time (forward propagation, loss calculation)

        Args:
            dataloader (torch.utils.data.dataloader.DataLoader):
                Dataset holder, load training and testing data in batch.
            model (src_Tran.model.NeuralNetwork): Model architecture
            loss_fn (torch.nn.modules.loss.CrossEntropyLoss): Loss function
            device (str): Device for training (GPU or CPU)
            epoch (int): Number of epochs
        """
        # Calculate number of samples
        size = len(dataloader.dataset)
        # Calculate number of batches
        num_batches = len(dataloader)

        # Initialize loss and correct prediction variables
        test_loss, correct = 0, 0

        # Total torch tensor of all prediction and groundtruth
        out_pred = torch.FloatTensor().to(device)
        out_gt = torch.FloatTensor().to(device)

        # Convert model to evaluation state
        model.eval()

        # Not compute gradient
        with torch.no_grad():
            batch_count = 0
            # Loop thorugh samples in data loader
            for X, y in dataloader:
                batch_count += 1

                # Load data and label batches to GPU/ CPU
                X, y = X.to(device), y.to(device)

                # Compute prediction
                pred = model(X.float())

                # Update groundtruth values
                out_gt = torch.cat((out_gt, y), 0)

                # Update prediction values
                out_pred = torch.cat((out_pred, pred), 0)

                # Accumulate loss and true prediction
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Calculate average loss and accuracy
        avg_loss = test_loss / batch_count

        out_pred = out_pred.argmax(1)
        #print("Ground truth", out_gt)
        # print("Prediction",out_pred.argmax(1))

        accuracy, precision, recall, f1_score, sensitivity, specificity = calculate_metrics(
            out_gt, out_pred)
        #test_loss /= num_batches
        print(
            'Validation set: Average loss: {:.6f}, Average accuracy: {:.0f}/{} ({:.3f}%)\n'.format(
                avg_loss,
                correct,
                len(dataloader.dataset),
                100. * accuracy))

        if use_checkpoint:
            # Earlystoping
            if self.EARLY_STOPING == 'val_loss':
                value = avg_loss
            elif self.EARLY_STOPING == 'val_accuracy':
                value = correct / len(dataloader.dataset)
            else:
                value = None

            if value is not None:
                self.early_stopping.__call__(value, model)
            else:
                print(f'Save model to {self.path}')
                torch.save(model.state_dict(), self.path)

        if best_model:
            if (accuracy > self.best_acc_val):
                print(f'Validate accuracy increased ({self.best_acc_val:.3f} --> {accuracy:.3f}).  Saving model to {self.path}')
                self.best_acc_val = accuracy
                torch.save(model.state_dict(), self.path)
                self.best_loss_val = avg_loss
            elif(avg_loss < self.best_loss_val) and (accuracy > self.best_acc_val-0.002):
                print(f'Validate loss increased ({self.best_loss_val:.6f} --> {avg_loss:.6f}).  Saving model to {self.path}')
                self.best_loss_train = avg_loss
                torch.save(model.state_dict(), self.path)
            else:
                print('Accuracy_val and Loss_val not improve! the model will not save')

        # TensorBoard
        self.writer.add_scalar("Loss/test", test_loss, epoch)
        self.writer.add_scalar("Accuracy/test", accuracy, epoch)
        self.writer.add_scalar("Precision/test", precision, epoch)
        self.writer.add_scalar("Recall/test", recall, epoch)
        self.writer.add_scalar("F1_score/test", f1_score, epoch)
        self.writer.add_scalar("Sensitivity/test", sensitivity, epoch)
        self.writer.add_scalar("Specificity/test", specificity, epoch)

    def early_stop(self):
        return self.early_stopping.early_stop

    def setup_training_data(self, type_transform=True):
        """
        Setup training data. Return data loaders of test and train set
        for training preparation.

        Returns:
            train_dataloader (torch.utils.data.dataloader.DataLoader):
                data loader for train set
            test_dataloader (torch.utils.data.dataloader.DataLoader):
                data loader for test set
        """
        # Create CustomImageDataset objects for train and test set
        if self.TRAIN_VAL_SPLIT_STATUS:
            train_dataset = CustomImageDataset(
                self.df_train,
                self.DATA_DIR,
                transform=self.TRANSFORM_IMAGE,
                type_transform=type_transform,
                target_transform=None)
            test_dataset = CustomImageDataset(
                self.df_test,
                self.DATA_DIR,
                transform=self.TRANSFORM_IMAGE,
                target_transform=Lambda(
                    lambda y: torch.tensor(y).type(
                        torch.long)))

            # Create data loader objects for train and test dataset objects
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True)
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True)

            return train_dataloader, test_dataloader
        else:
            train_dataset = CustomImageDataset(
                self.df_train,
                self.DATA_DIR,
                transform=self.TRANSFORM_IMAGE,
                type_transform=type_transform,
                target_transform=None)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True)
            return train_dataloader

    def setup_training(self):
        """
        Create loss function, optimizers, and model. Then bring model to GPU.

        Returns:
            model (src_Tran.model.NeuralNetwork): Model architecture
            optimizer (torch.optim.sgd.SGD): Optimization algorithm
            loss_fn (torch.nn.modules.loss.CrossEntropyLoss): Loss function
            device (str): Device for training (GPU or CPU)
        """
        # Load model
        model = NeuralNetwork(self.ARCHITECTURE)

        # Loss function
        loss_fn = nn.CrossEntropyLoss()

        # Optimizers
        if self.OPTIMIZER.lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.LEARNING_RATE)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.LEARNING_RATE)

        # Choose device and bring model to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        print("Using architecture: " + self.ARCHITECTURE + " with optimizer: " +
              self.OPTIMIZER + " and learning rate: " + str(self.LEARNING_RATE))
        print('Setup training successful')

        return model, optimizer, loss_fn, device
