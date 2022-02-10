import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from src.utils import calculate_metrics_multilabel_full, preprocess, EarlyStopping, calculate_metrics, calculate_metrics_multilabel
from src.dataset import CustomImageDataset
from src.dataGenerator import DataGenerator
from src.model import NeuralNetwork
from torch.utils.tensorboard import SummaryWriter
import os
import sys
import time


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
        if self.PREDICT_MODE == False:
            print('Trainer class constructed')
            print("Update initial dataset")

            # Get train and test dataframe from data directory
            if self.TRAIN_VAL_SPLIT_STATUS:
                # Save resized image and label of train and test to decrease training time, also agument dataset
                if self.TRANSFORM_IMAGE:
                    start_time = time.time()

                    self.df_train, self.df_test = preprocess(
                        self.DATA_DIR, self.CSV_DIR_TRAIN, train_val_split=self.TRAIN_VAL_SPLIT, train_val_split_status=self.TRAIN_VAL_SPLIT_STATUS)
                    initial_data = DataGenerator(
                        self.df_train, self.DATA_DIR, train=True)
                    self.initial_img_train, self.initial_label_train = initial_data()

                    torch.save(self.initial_img_train,
                               'img/initial_img_train.pt')
                    torch.save(self.initial_label_train,
                               'img/initial_label_train.pt')

                    initial_data = DataGenerator(
                        self.df_test, self.DATA_DIR, train=False)
                    self.initial_img_test, self.initial_label_test = initial_data()

                    torch.save(self.initial_img_test,
                               'img/initial_img_test.pt')
                    torch.save(self.initial_label_test,
                               'img/initial_label_test.pt')

                    print('DONE!!! SAVED TRANSFORMED IMAGE')
                    print(
                        "--- {:.0f} seconds ---".format(time.time() - start_time))
                    print('PLZ!! TURN OFF TRASFORM_IMAGE mode in config.json')
                    exit()
                else:
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    self.initial_img_train = torch.FloatTensor().to(self.device)
                    self.initial_img_test = torch.FloatTensor().to(self.device)
                    self.initial_label_train = torch.FloatTensor().to(self.device)
                    self.initial_label_test = torch.FloatTensor().to(self.device)

                    self.initial_img_train = torch.load(
                        'img/initial_img_train.pt')
                    self.initial_img_test = torch.load(
                        'img/initial_img_test.pt')
                    self.initial_label_train = torch.load(
                        'img/initial_label_train.pt')
                    self.initial_label_test = torch.load(
                        'img/initial_label_test.pt')

            # Check metrics of one .csv file
            elif self.VAL_MODE:
                self.df_train = preprocess(self.DATA_DIR, self.CSV_DIR_TRAIN)
                initial_data = DataGenerator(self.df_train, self.DATA_DIR)
                self.initial_img_train, self.initial_label_train = initial_data()
                self.df_test = preprocess(self.DATA_DIR, self.CSV_DIR_TEST)
                initial_data = DataGenerator(self.df_test, self.DATA_DIR)
                self.initial_img_test, self.initial_label_test = initial_data()

            else:
                if self.TRANSFORM_IMAGE:
                    self.df_train = preprocess(
                        self.DATA_DIR, self.CSV_DIR_TRAIN, train_val_split=self.TRAIN_VAL_SPLIT, train_val_split_status=self.TRAIN_VAL_SPLIT_STATUS)
                    initial_data = DataGenerator(self.df_train, self.DATA_DIR)
                    self.initial_img, self.initial_label = initial_data()
                    torch.save(self.initial_img, 'img/initial_img.pt')
                    torch.save(self.initial_label, 'img/initial_label.pt')
                    print('DONE!!! SAVED TRANSFORMED IMAGE')
                    print('PLZ!! TURN ON TRAIN_MODE')
                    exit()
                else:
                    self.initial_img = torch.load('img/initial_img.pt')
                    self.initial_label = torch.load('img/initial_label.pt')

            print('Update complete')

            # Save best model
            self.best_acc_train = 0
            self.best_loss_train = 0
            self.best_acc_val = 0
            self.best_loss_val = 0
            self.start_time = 0
            self.path = os.path.join(self.MODEL_DIR, "trial-" + self.TRIAL +
                                     "-model-" + self.ARCHITECTURE +
                                     "-optim-" + self.OPTIMIZER +
                                     "-lr-" + str(self.LEARNING_RATE) +
                                     "-wc-" + str(self.WEIGHDECAY) +
                                     ".pth")
            # Initiate early stoppping object
        self.early_stopping = EarlyStopping(
            verbose=True, patience=self.PATIENCE, path=self.path, monitor=self.EARLY_STOPING)
        self.writer = SummaryWriter(
            log_dir=os.path.join(
                "log",
                "trial-" + self.TRIAL +
                "-model-" + self.ARCHITECTURE +
                "-optim-" + self.OPTIMIZER +
                "-lr-" + str(self.LEARNING_RATE) +
                "-wc-" + str(self.WEIGHDECAY)))

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
            use_chekpoint(bool): Early Stoping mode
            best_model(bool): Save best accurancy 
        """
        # Calculate number of samples
        size = len(dataloader.dataset)

        # Convert model to training state
        model.train()
        train_loss = 0
        correct = 0
        total_train = 0
        avr_acc = 0
        total_acc = 0

        out_pred = torch.FloatTensor().to(device)
        out_gt = torch.FloatTensor().to(device)

        print("Epoch:", epoch)
        # Loop through samples in data loader
        for batch, (X, y) in enumerate(dataloader):
            correct_per_batch = 0
            sample_per_batch = y.nelement()/7

            # Load data and label batches to GPU/ CPU
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X.float())

            # Update groundtruth values
            out_gt = torch.cat((out_gt, y), 0)

            # Compute loss
            loss = loss_fn(pred, y)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()

            # Update variables
            optimizer.step()

            out_pred = torch.cat((out_pred, pred), 0)

            # fillter prediction
            for i in range(len(pred)):
                index_max_cp = torch.argmax(pred[i][0:2])
                index_max_lr = torch.argmax(pred[i][2:4])
                index_max_odm = torch.argmax(pred[i][4:7])
                pred[i] = torch.zeros(7)
                pred[i][index_max_cp] = 1
                pred[i][index_max_lr+2] = 1
                pred[i][index_max_odm+4] = 1

            # Acurancy
            for i in range(len(pred)):  # len(pred) = batchsize
                if (pred[i][0] == y[i][0]) and (pred[i][1] == y[i][1]) and (pred[i][2] == y[i][2]) and (pred[i][3] == y[i][3]) and (pred[i][4] == y[i][4]) and (pred[i][5] == y[i][5]) and (pred[i][6] == y[i][6]):
                    correct += 1
                    correct_per_batch += 1

            total_train += sample_per_batch
            train_accuracy = correct_per_batch / sample_per_batch
            total_acc += train_accuracy
            print('\tTraining batch {} Loss: {:.6f}, Accurancy:{:.3f}%'.format(
                batch + 1, loss.item(), 100. * train_accuracy))

        avr_acc = correct / total_train
        accuracy = avr_acc

        accuracy_sklearn, precision, recall, micro_f1_score, macro_f1_score, sensitivity, specificity, auc_score = calculate_metrics_multilabel(
            out_gt, out_pred)

        print('Training set: Average loss: {:.6f}, Average accuracy: {:.0f}/{} ({:.3f}%), AUC score: {:.3f}\n'.format(
            train_loss / (batch + 1), correct, len(dataloader.dataset), 100. * accuracy_sklearn, auc_score))

        if use_checkpoint:
            # Earlystoping
            if self.EARLY_STOPING == 'val_loss':
                value = train_loss / (batch + 1)
            elif self.EARLY_STOPING == 'val_accuracy':
                value = correct / len(dataloader.dataset)
            else:
                value = None

            if value is not None:
                self.early_stopping(value, model)
            else:
                print(f'Save model to {self.path}')
                torch.save(model.state_dict(), self.path)

        if best_model:
            if (accuracy_sklearn > self.best_acc_train):
                print(
                    f'Training accuracy increased ({self.best_acc_train:.3f} --> {accuracy_sklearn:.3f}).  Saving model to {self.path}')
                self.best_acc_train = accuracy_sklearn
                torch.save(model.state_dict(), self.path)
                self.best_loss_train = (train_loss / (batch + 1))
            elif((train_loss / (batch + 1)) < self.best_loss_train) and (accuracy_sklearn > self.best_acc_train-0.001):
                print(
                    f'Training loss increased ({self.best_loss_train:.6f} --> {(train_loss / (batch + 1)):.6f}).  Saving model to {self.path}')
                self.best_loss_train = (train_loss / (batch + 1))
                torch.save(model.state_dict(), self.path)
            else:
                print(
                    'Accuracy_train and Loss_train not improve! the model will not save')

        self.writer.add_scalar("Loss/train", train_loss / (batch + 1), epoch)
        self.writer.add_scalar("Accuracy/train", accuracy, epoch)
        self.writer.add_scalar("Accuracy_sklern/train",
                               accuracy_sklearn, epoch)
        self.writer.add_scalar("Precision/train", precision, epoch)
        self.writer.add_scalar("Recall/train", recall, epoch)
        self.writer.add_scalar("Micro_F1_score/train", micro_f1_score, epoch)
        self.writer.add_scalar("Macro_F1_score/train", macro_f1_score, epoch)
        self.writer.add_scalar("Sensitivity/train", sensitivity, epoch)
        self.writer.add_scalar("Specificity/train", specificity, epoch)
        self.writer.add_scalar("AUROC/train", auc_score, epoch)

    def epoch_evaluate(self, dataloader, model, loss_fn, device, epoch=0, use_checkpoint=False, best_model=False, train_or_validation='train'):
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
        correct = 0
        test_loss = 0
        avg_acc = 0

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

                # Accumulate loss and true prediction
                test_loss += loss_fn(pred, y).item()

                # Update prediction values
                out_pred = torch.cat((out_pred, pred), 0)

                for i in range(len(pred)):
                    index_max_cp = torch.argmax(pred[i][0:2])
                    index_max_lr = torch.argmax(pred[i][2:4])
                    index_max_odm = torch.argmax(pred[i][4:7])
                    pred[i] = torch.zeros(7)
                    pred[i][index_max_cp] = 1
                    pred[i][index_max_lr+2] = 1
                    pred[i][index_max_odm+4] = 1

                for i in range(len(pred)):  # len(pred) = batchsize
                    if (pred[i][0] == y[i][0]) and (pred[i][1] == y[i][1]) and (pred[i][2] == y[i][2]) and (pred[i][3] == y[i][3]) and (pred[i][4] == y[i][4]) and (pred[i][5] == y[i][5]) and (pred[i][6] == y[i][6]):
                        correct += 1

        # Calculate average loss and accuracy
        avg_loss = test_loss / batch_count

        accuracy_sklearn, precision, recall, micro_f1_score, macro_f1_score, sensitivity, specificity, auc_score = calculate_metrics_multilabel(
            out_gt, out_pred)

        accuracy = correct / len(dataloader.dataset)
        print(
            'Validation set: Average loss: {:.6f}, Average accuracy: {:.0f}/{} ({:.3f}%), AUC score: {:.3f}\n'.format(
                avg_loss,
                correct,
                len(dataloader.dataset),
                100. * accuracy_sklearn,
                auc_score))

        if train_or_validation == 'train':
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
                if (accuracy_sklearn > self.best_acc_val):
                    print(
                        f'Validate accuracy increased ({self.best_acc_val:.3f} --> {accuracy_sklearn:.3f}).  Saving model to {self.path}')
                    self.best_acc_val = accuracy_sklearn
                    torch.save(model.state_dict(), self.path)
                    self.best_loss_val = avg_loss
                elif(avg_loss < self.best_loss_val) and (accuracy_sklearn > self.best_acc_val-0.001):
                    print(
                        f'Validate loss increased ({self.best_loss_val:.6f} --> {avg_loss:.6f}).  Saving model to {self.path}')
                    self.best_loss_train = avg_loss
                    torch.save(model.state_dict(), self.path)
                else:
                    print(
                        'Accuracy_val and Loss_val not improve! the model will not save')

            # TensorBoard
            self.writer.add_scalar("Loss/test", test_loss, epoch)
            self.writer.add_scalar("Accuracy/test", accuracy, epoch)
            self.writer.add_scalar(
                "Accuracy_sklern/test", accuracy_sklearn, epoch)
            self.writer.add_scalar("Precision/test", precision, epoch)
            self.writer.add_scalar("Recall/test", recall, epoch)
            self.writer.add_scalar("Micro_F1_score/test",
                                   micro_f1_score, epoch)
            self.writer.add_scalar("Macro_F1_score/test",
                                   macro_f1_score, epoch)
            self.writer.add_scalar("Sensitivity/test", sensitivity, epoch)
            self.writer.add_scalar("Specificity/test", specificity, epoch)
            self.writer.add_scalar("AUROC/test", auc_score, epoch)

        if train_or_validation == "validation":
            accuracy_list, precision_list, recall_list, f1_list, sensitivity_list, specificity_list, auc_score_list = calculate_metrics_multilabel_full(
                out_gt, out_pred)
            print("accuracy_list\t", accuracy_list)
            print("precision_list\t", precision_list)
            print("recall_list\t", recall_list)
            print("f1_list\t", f1_list)
            print("sensitivity_list\t", sensitivity_list)
            print("specificity_list\t", specificity_list)
            print("auc_score_list\t", auc_score_list)

            print("accuracy mean\t", accuracy_sklearn)
            print("precision mean\t", precision)
            print("recall mean\t", recall)
            print("f1_score micro\t", micro_f1_score)
            print("f1_score macro\t", macro_f1_score)
            print("sensitivity mean\t", sensitivity)
            print("specitivity mean\t", specificity)
            print("auc_score mean\t", auc_score)

    def early_stop(self):
        return self.early_stopping.early_stop

    def setup_training_data(self, diff=False):
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
        if self.VAL_MODE or self.TRAIN_VAL_SPLIT_STATUS:

            train_dataset = CustomImageDataset(
                self.initial_img_train,
                self.initial_label_train,
                transform=None,
                target_transform=None,
                diff=True)

            test_dataset = CustomImageDataset(
                self.initial_img_test,
                self.initial_label_test,
                transform=None,
                target_transform=None, diff=False
            )

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
                self.initial_img,
                self.initial_label,
                transform=None,
                target_transform=None,
                diff=0)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True)
            return train_dataloader

    def setup_training(self):
        """
        Create loss function, optimizers, and model. Then bring model to GPU.

        Returns:
            model (src.model.NeuralNetwork): Model architecture
            optimizer (torch.optim.sgd.SGD): Optimization algorithm
            loss_fn (torch.nn.modules.loss.BCELoss): Loss function
            device (str): Device for training (GPU or CPU)
        """
        # Load model
        model = NeuralNetwork(self.ARCHITECTURE)

        # Loss function
        loss_fn = nn.BCELoss(reduction="sum")

        # Optimizers
        if self.OPTIMIZER.lower() == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=self.LEARNING_RATE, weight_decay=self.WEIGHDECAY)
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
