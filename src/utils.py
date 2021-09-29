import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import timeit


class EarlyStopping:
    """Early stops the training if validation loss and validation accuracy don't improve after a given patience."""

    def __init__(
            self,
            patience=7,
            verbose=False,
            delta=0,
            path='models/checkpoint.pt',
            trace_func=print,
            monitor='val_loss'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            monitor (Mode): If val_loss, stop at maximum mode, else val_accuracy, stop at minimum mode
                            Default: val_loss
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_acc_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = monitor

    def __call__(self, values, model):
        """ Check for early stopping when validation loss not decrease or validation accuracy not increase

        Args:
            values (numpy.float64): value to be used for early stopping
            model (src.model.NeuralNetwork): model to be used for early stopping
        """
        if self.mode == 'val_loss':
            score = -values

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(values, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(values, model)
                self.counter = 0
        else:
            score = values
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(values, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(values, model)
                self.counter = 0

    def save_checkpoint(self, values, model):
        """ Saves model when validation loss decrease or validation accuracy increase

        Args:
            values (numpy.float64): value to be saved for checkpointing
            model (src.model.NeuralNetwork): model to be saved in checkpointing
        """
        if self.mode == 'val_loss':
            if self.verbose:
                self.trace_func(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {values:.6f}).   Saving model to {self.path}')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = values
        elif self.mode == 'val_accuracy':
            if self.verbose:
                self.trace_func(
                    f'Validation accuracy increased ({self.val_acc_max:.3f} --> {values:.3f}).  Saving model to {self.path}')
            torch.save(model.state_dict(), self.path)
            self.val_acc_max = values


def preprocess(data_dir, csv_dir, data_source=None):  # CTEH - EYESCAN - BAIDU
    """
    Get training dataframe and testing dataframe from image directory and
    csv description file.

    Args:
        data_dir (String): Directory of image data
        csv_dir (String): Directory of csv description file
        data_source (String, optional): Source of images ("CTEH", "BAIDU", or "EYESCAN"). Defaults to None.

    Returns:
        df_train (pandas.DataFrame): Data frame of training set
        df_test (pandas.DataFrame):  Data frame of test set
    """

    # Read in dataframe of dataset information
    url_dataframe = pd.read_csv(csv_dir, index_col=0)

    # Filter the data of a certain source
    if data_source:
        filt = (url_dataframe["src"] == data_source)
        url_dataframe = url_dataframe[filt]

    # Convert the image ID to image file name
    url_dataframe["ID"] = [str(x) + ".png" for x in url_dataframe["ID"]]

    id_list = (url_dataframe["ID"]).to_list()
    label_central_list = []
    label_peripheral_list = []
    label_left_list = []
    label_right_list = []
    label_od_list = []
    label_macula_list = []
    index_list = url_dataframe.index.to_list()

    # Convert string labels to number label
    for i in index_list:
        if (url_dataframe.loc[i, "label_cp"] == "central"):
            label_central_list.append(1)
            label_peripheral_list.append(0)
        else:
            label_central_list.append(0)
            label_peripheral_list.append(1)

        if (url_dataframe.loc[i, "label_lr"] == "left"):
            label_left_list.append(1)
            label_right_list.append(0)
        else:
            label_left_list.append(0)
            label_right_list.append(1)

        if (url_dataframe.loc[i, "label_odm"] == "od"):
            label_od_list.append(1)
            label_macula_list.append(0)
        elif (url_dataframe.loc[i, "label_odm"] == "macula"):
            label_od_list.append(0)
            label_macula_list.append(1)
        else:
            label_od_list.append(0)
            label_macula_list.append(0)

    label_total_list = []

    for i in range(len(id_list)):
        # print(i)
        item_list = [
            label_central_list[i],
            label_peripheral_list[i],
            label_left_list[i],
            label_right_list[i],
            label_od_list[i],
            label_macula_list[i]]
        label_total_list.append(item_list)

    # Split the original dataset to training set and test set
    name_train, name_test, label_train, label_test = train_test_split(
        id_list, label_total_list, test_size=0.3, random_state=42)

    data_train = {'Name': name_train,
                  'Label': label_train}

    data_test = {'Name': name_test,
                 'Label': label_test}

    # Create traing dataframe and test dataframe
    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)

    # print("Sample training dataframe")
    # print(df_train.head())
    # print()

    # print("Training class distribution")
    # print(df_train["Label"].value_counts())
    # print()

    # print("Number of training samples")
    # print(len(df_train))
    # print()

    # print("Sample testing dataframe")
    # print(df_test.head())
    # print()

    # print("Testing class distribution")
    # print(df_test["Label"].value_counts())
    # print()

    # print("Number of testing samples")
    # print(len(df_test))
    # print()

    # print("Number of total samples")
    # print(len(df_train) + len(df_test))
    # print()

    print("preprocessing complete")
    return df_train, df_test


def calculate_metrics(out_gt, out_pred):
    """
    Calculate methics for model evaluation

    Args:
        out_gt (torch.Tensor)   : Grouth truth array
        out_pred (torch.Tensor) : Prediction array

    Returns:
        accuracy (float)    : Accuracy
        precision (float)   : Precision
        recall (float)      : Recall
        f1_score (float)    : F1 Score
        sensitivity (float) : Sensitivity
        specificity (float) : Specificity
        auc score(float)    : Area under receiver operating characteristics
    """
    # Calculate AUROC score
    try:
        auc_score = roc_auc_score(out_gt, out_pred)
    except BaseException:
        auc_score = 0

    # Calculate true_positives, true_negatives, false_positives,
    # false_negatives
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0

    for i in range(len(out_gt)):
        if ((out_gt[i] == 1) and (out_pred[i] == 1)):
            true_positives += 1
        if ((out_gt[i] == 0) and (out_pred[i] == 0)):
            true_negatives += 1
        if ((out_gt[i] == 0) and (out_pred[i] == 1)):
            false_positives += 1
        if ((out_gt[i] == 1) and (out_pred[i] == 0)):
            false_negatives += 1

    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / (true_positives +
                                                    true_negatives + false_positives + false_negatives)

    # Calculate precision
    precision = true_positives / \
        (true_positives + false_positives + np.finfo(float).eps)

    # Calculate recall
    recall = true_positives / \
        (true_positives + false_negatives + np.finfo(float).eps)

    # Calculate F1 score
    f1_score = 2 * precision * recall / \
        (precision + recall + np.finfo(float).eps)

    # Calculate sensitivity and specificity
    sensitivity = recall
    specificity = true_negatives / \
        (true_negatives + false_positives + np.finfo(float).eps)

    return accuracy, precision, recall, f1_score, sensitivity, specificity, auc_score


def calculate_metrics_multilabel(out_gt, out_pred):
    """
    Calculate methics for model evaluation (Multilabel version)

    Args:
        out_gt (torch.Tensor)   : Grouth truth array of shape [len_dataset, 6]
        out_pred (torch.Tensor) : Prediction array of shape [len_dataset, 6]

    Returns:
        accuracy (float)    : Accuracy (average = "samples")
        precision (float)   : Precision (average = "macro")
        recall (float)      : Recall (average = "macro")
        micro_f1 (float)    : F1 Score (average = "micro")
        macro_f1 (float)    : F1 Score (average = "macro")
        sensitivity (float) : Sensitivity (average = "macro")
        specificity (float) : Specificity (average = "macro")
        auc score(float)    : Area under receiver operating characteristics (average = "macro")
    """
    # Convert torch tensor to numpy array
    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()

    # Calculate AUROC score
    try:
        auc_score = roc_auc_score(
            out_gt, out_pred, average="macro")  # average=None
    except BaseException:
        auc_score = 0

    # Convert continuous values to discrete labels
    for i in range(len(out_pred)):
        for j in range(len(out_pred[i])):
            if (out_pred[i][j] < 0.5):
                out_pred[i][j] = 0
            else:
                out_pred[i][j] = 1

    # Calculate accuracy
    accuracy = jaccard_score(out_gt, out_pred, average="samples")

    # Calculate precision
    precision = precision_score(
        out_gt,
        out_pred,
        average='macro',
        zero_division=1)

    # Calculate recall
    recall = recall_score(out_gt, out_pred, average='macro', zero_division=1)

    # Calculate micro f1 score
    micro_f1 = f1_score(out_gt, out_pred, average='micro', zero_division=1)

    # Calculate macro f1 score
    macro_f1 = f1_score(out_gt, out_pred, average='macro', zero_division=1)

    # # Calculate sensitivity and specificity
    out_gt_transpose = np.transpose(out_gt)
    out_pred_transpose = np.transpose(out_pred)

    sensitivity_list = []
    specificity_list = []

    for i in range(len(out_gt_transpose)):
        _, _, _, _, sensitivity, specificity, _ = calculate_metrics(
            out_gt_transpose[i], out_pred_transpose[i])
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)

    sensitivity = sensitivity_list.mean()
    specificity = specificity_list.mean()

    return accuracy, precision, recall, micro_f1, macro_f1, sensitivity, specificity, auc_score


def calculate_metrics_multilabel_full(out_gt, out_pred):
    """
    Calculate methics for model evaluation (Multilabel version for each label)

    Args:
        out_gt (torch.Tensor)   : Grouth truth array of shape [len_dataset, 6]
        out_pred (torch.Tensor) : Prediction array of shape [len_dataset, 6]

    Returns:
        accuracy (numpy.ndarray)    : Accuracy (average = None) of shape (6,)
        precision (numpy.ndarray)   : Precision (average = None) of shape (6,)
        recall (numpy.ndarray)      : Recall (average = None) of shape (6,)
        f1_score (numpy.ndarray)    : F1 Score (average = None) of shape (6,)
        sensitivity (numpy.ndarray) : Sensitivity (average = None) of shape (6,)
        specificity (numpy.ndarray) : Specificity (average = None) of shape (6,)
        auc score(numpy.ndarray)    : Area under receiver operating characteristics of shape (6,)
    """
    # Convert torch tensor to numpy array
    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()

    # Calculate AUROC score
    try:
        auc_score_list = roc_auc_score(out_gt, out_pred, average=None)
    except BaseException:
        auc_score_list = [0, 0, 0, 0, 0, 0]

    # Convert continuous values to discrete labels
    for i in range(len(out_pred)):
        for j in range(len(out_pred[i])):
            if (out_pred[i][j] < 0.5):
                out_pred[i][j] = 0
            else:
                out_pred[i][j] = 1

    # Calculate accuracy
    accuracy_list = jaccard_score(out_gt, out_pred, average=None)

    # Calculate precision
    precision_list = precision_score(
        out_gt, out_pred, average=None, zero_division=1)

    # Calculate recall
    recall_list = recall_score(out_gt, out_pred, average=None, zero_division=1)

    # Calculate f1 score
    f1_list = f1_score(out_gt, out_pred, average=None, zero_division=1)

    # Transpose out_gt and out_pred
    out_gt_transpose = np.transpose(out_gt)
    out_pred_transpose = np.transpose(out_pred)

    # Calculate sensitivity and specificity
    # accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    for i in range(len(out_gt_transpose)):
        _, _, _, _, sensitivity, specificity, _ = calculate_metrics(
            out_gt_transpose[i], out_pred_transpose[i])
        # accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    # accuracy_list = np.array(accuracy_list)
    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)

    return accuracy_list, precision_list, recall_list, f1_list, sensitivity_list, specificity_list, auc_score_list


if __name__ == '__main__':
    preprocess(
        '../data/data',
        '../pseudolabel_done.csv',
        data_source="EYESCAN")

    # out_y = np.array([[1., 0., 0., 1., 0., 1.],
    #     [1., 0., 0., 1., 1., 0.],
    #     [1., 0., 0., 1., 0., 1.],
    #     [0., 1., 1., 0., 0., 0.],
    #     [1., 0., 0., 1., 0., 1.],
    #     [1., 0., 1., 0., 0., 1.],
    #     [1., 0., 0., 1., 1., 0.]])

    # out_pred = np.array([[1., 0., 0., 1., 0., 1.],
    #     [1., 0., 0., 1., 0., 0.],
    #     [1., 0., 0., 1., 0., 1.],
    #     [0., 1., 0., 1., 0., 0.],
    #     [0., 1., 0., 1., 0., 1.],
    #     [1., 0., 1., 0., 0., 1.],
    #     [1., 0., 0., 1., 1., 0.]])

    # # out_y = out_y.transpose()
    # # out_pred = out_pred.transpose()
    # accuracy_list, precision_list, recall_list, f1_list, sensitivity_list, specificity_list, auc_score_list = calculate_metrics_multilabel_full(out_y, out_pred)
    # print("accuracy",accuracy_list)
    # print("precision_list",precision_list)
    # print("recall_list",recall_list)
    # print("f1_list",f1_list)
    # print("sensitivity_list",sensitivity_list)
    # print("specificity_list",specificity_list)
    # print("auc_score_list",auc_score_list)
