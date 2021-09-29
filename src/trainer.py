import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from src.utils import calculate_metrics_multilabel_full, preprocess, EarlyStopping, calculate_metrics_multilabel
from src.dataset import CustomImageDataset
from src.model import NeuralNetwork
from torch.utils.tensorboard import SummaryWriter
from src.generator import Generator
import os
import timeit


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
        self.df_train, self.df_test = preprocess(
            self.DATA_DIR, self.CSV_DIR, self.DATA_SOURCE)

        # Initiate early stoppping object
        self.early_stopping = EarlyStopping(
            verbose=True,
            patience=self.PATIENCE,
            path=os.path.join(
                self.MODEL_DIR,
                "trial-" + self.TRIAL + ".pth"),
            monitor=self.EARLY_STOPING)

        # Initiate Tensorboard Summary Writer object
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                "log",
                "trial-" + self.TRIAL +
                "-model-" + self.ARCHITECTURE +
                "-optim-" + self.OPTIMIZER +
                "-lr-" + str(self.LEARNING_RATE)))

        self.train_imgs = 0
        self.train_labels = 0
        self.test_imgs = 0
        self.test_labels = 0

    def epoch_train(
            self,
            dataloader,
            model,
            loss_fn,
            optimizer,
            device,
            epoch):
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

        # Initialize overall prediction and groundtruth tensor
        out_pred = torch.FloatTensor().to(device)
        out_gt = torch.FloatTensor().to(device)

        print("Epoch:", epoch)
        # Loop through samples in data loader
        for batch, (X, y) in enumerate(dataloader):
            # Load data and label batches to GPU/ CPU
            X, y = X.to(device), y.to(device)

            # Clear previous gradient
            optimizer.zero_grad()

            # Feed forward
            pred = model(X)

            # Update groundtruth values
            out_gt = torch.cat((out_gt, y), 0)

            # Compute loss
            loss = loss_fn(pred, y)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()

            # Update variables
            optimizer.step()

            # Update prediction values
            out_pred = torch.cat((out_pred, pred), 0)

            # Print batch index
            print('\tTraining batch {} Loss: {:.6f}'.format(
                batch + 1, loss.item()))

        # Calculate metrics
        accuracy, precision, recall, micro_f1_score, macro_f1_score, sensitivity, specificity, auc_score = calculate_metrics_multilabel(
            out_gt, out_pred)

        print('Training set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
            train_loss / (batch + 1), 100 * accuracy, auc_score))

        # Record performance
        self.writer.add_scalar("Loss/train", train_loss / (batch + 1), epoch)
        self.writer.add_scalar("Accuracy/train", accuracy, epoch)
        self.writer.add_scalar("Precision/train", precision, epoch)
        self.writer.add_scalar("Recall/train", recall, epoch)
        self.writer.add_scalar("Micro_F1_score/train", micro_f1_score, epoch)
        self.writer.add_scalar("Macro_F1_score/train", macro_f1_score, epoch)
        self.writer.add_scalar("Sensitivity/train", sensitivity, epoch)
        self.writer.add_scalar("Specificity/train", specificity, epoch)
        self.writer.add_scalar("AUROC/train", auc_score, epoch)

    def epoch_evaluate(
            self,
            dataloader,
            model,
            loss_fn,
            device,
            epoch=0,
            use_checkpoint=True,
            train_or_validation="train"):
        """
        Evaluate the model one time (forward propagation, loss calculation)

        Args:
            dataloader (torch.utils.data.dataloader.DataLoader):
                Dataset holder, load training and testing data in batch.
            model (src_Tran.model.NeuralNetwork): Model architecture
            loss_fn (torch.nn.modules.loss.CrossEntropyLoss): Loss function
            device (str): Device for training (GPU or CPU)
            epoch (int): Number of epochs
            use_checkpoint (bool): enable model checkpoint or not
            train_or_validation (String): specify that the epoch is used for training or validation purposes. ("train" or "validation")
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
                pred = model(X)

                # Update groundtruth values
                out_gt = torch.cat((out_gt, y), 0)

                # Accumulate loss and true prediction
                test_loss += loss_fn(pred, y).item()

                # Update prediction values
                out_pred = torch.cat((out_pred, pred), 0)

        # Calculate average loss and other metrics
        avg_loss = test_loss / batch_count

        accuracy, precision, recall, micro_f1_score, macro_f1_score, sensitivity, specificity, auc_score = calculate_metrics_multilabel(
            out_gt, out_pred)
        test_loss /= num_batches

        print(
            'Validation set: Average loss: {:.6f}, Average accuracy: {:.3f}%, AUC score: {:.3f}\n'.format(
                avg_loss,
                100. *
                accuracy,
                auc_score))

        if train_or_validation == "train":
            # Use model checkpoint and early stopping
            if use_checkpoint:
                # Earlystoping
                if self.EARLY_STOPING == 'val_loss':
                    value = avg_loss
                elif self.EARLY_STOPING == 'val_accuracy':
                    value = accuracy
                else:
                    value = None

                if value is not None:
                    self.early_stopping.__call__(value, model)

            # Record performance on TensorBoard
            # TensorBoard
            self.writer.add_scalar("Loss/test", test_loss, epoch)
            self.writer.add_scalar("Accuracy/test", accuracy, epoch)
            self.writer.add_scalar("Precision/test", precision, epoch)
            self.writer.add_scalar("Recall/test", recall, epoch)
            self.writer.add_scalar(
                "Micro_F1_score/test", micro_f1_score, epoch)
            self.writer.add_scalar(
                "Macro_F1_score/test", macro_f1_score, epoch)
            self.writer.add_scalar("Sensitivity/test", sensitivity, epoch)
            self.writer.add_scalar("Specificity/test", specificity, epoch)
            self.writer.add_scalar("AUROC/test", auc_score, epoch)

        if train_or_validation == "validation":
            # Calculate perfomances for each labels
            accuracy_list, precision_list, recall_list, f1_list, sensitivity_list, specificity_list, auc_score_list = calculate_metrics_multilabel_full(
                out_gt, out_pred)
            print("accuracy_list\t", accuracy_list)
            print("precision_list\t", precision_list)
            print("recall_list\t", recall_list)
            print("f1_list\t", f1_list)
            print("sensitivity_list\t", sensitivity_list)
            print("specificity_list\t", specificity_list)
            print("auc_score_list\t", auc_score_list)

            # Calculate means performances
            print("accuracy mean\t", accuracy)
            print("precision mean\t", precision)
            print("recall mean\t", recall)
            print("f1_score micro\t", micro_f1_score)
            print("f1_score macro\t", macro_f1_score)
            print("sensitivity mean\t", sensitivity)
            print("specitivity mean\t", specificity)
            print("auc_score mean\t", auc_score)

    def early_stop(self):
        return self.early_stopping.early_stop

    def setup_training_data(self):
        """
        Setup training data. Return data loaders of test and train set
        for training preparation.

        Returns:
            train_dataloader (torch.utils.data.dataloader.DataLoader):
                data loader for train set
            test_dataloader (torch.utils.data.dataloader.DataLoader):
                data loader for test set
        """

        if not self.USE_TRANSFOMED_DATA:
            # First time training
            # Basic preprocessing with generator
            train_generator = Generator(
                df=self.df_train, data_dir=self.DATA_DIR)
            test_generator = Generator(df=self.df_test, data_dir=self.DATA_DIR)

            # Create training and testing tensor, and save those file in ".pt"
            # format
            self.train_imgs, self.train_labels = train_generator(train=True)
            self.test_imgs, self.test_labels = test_generator(train=False)
        else:
            # Later training - load the preprocessed ".pt" file
            self.train_imgs, self.train_labels = torch.load(
                "tensor/train_imgs_array.pt"), torch.load("tensor/train_labels_array.pt")
            self.test_imgs, self.test_labels = torch.load(
                "tensor/test_imgs_array.pt"), torch.load("tensor/test_labels_array.pt")

        # Further augmemntation in dataset objects
        train_dataset = CustomImageDataset(
            self.train_imgs,
            self.train_labels,
            augment=self.USE_AUGMENTATION,
            rotation=self.ROTATE,
            translation=self.TRANSLATE,
            scaling=self.SCALE)  # augment = True
        test_dataset = CustomImageDataset(
            self.test_imgs, self.test_labels, augment=False)

        # Create data loader objects for train and test dataset objects
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True)

        print("setup training data complete")
        return train_dataloader, test_dataloader

    def setup_training(self):
        """
        Create loss function, optimizers, and model. Then bring model to GPU.

        Returns:
            model (src_Tran.model.NeuralNetwork): Model architecture
            optimizer (torch.optim.sgd.SGD): Optimization algorithm
            loss_fn (torch.nn.modules.loss.CrossEntropyLoss): Loss function
            device (str): Device for training (GPU or CPU)
        """
        # Model
        model = NeuralNetwork(self.ARCHITECTURE, self.DROPOUT_RATE)

        # Loss function
        loss_fn = nn.BCELoss(reduction="sum")

        # Optimizers
        if self.OPTIMIZER.lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.LEARNING_RATE,
                weight_decay=self.WEIGHT_DECAY)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.LEARNING_RATE,
                weight_decay=self.WEIGHT_DECAY)

        # Choose device and bring model to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        print('Setup training successful')

        return model, optimizer, loss_fn, device
