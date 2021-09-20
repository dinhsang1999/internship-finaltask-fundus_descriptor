import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import timeit

class EarlyStopping:
    """Early stops the training if validation loss and validation accuracy don't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='models/checkpoint.pt', trace_func=print, monitor='val_loss'):
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
        '''Saves model when validation loss decrease.'''
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



def preprocess(data_dir, csv_dir, data_source=None): #CTEH - EYESCAN - BAIDU
    """
    Get training dataframe and testing dataframe from image directory and
    csv description file.

    Args:
        data_dir (String): Directory of image data
        csv_dir (String): Directory of csv description file

    Returns:
        df_train (pandas.DataFrame): Data frame of training set
        df_test (pandas.DataFrame):  Data frame of test set
    """
    print("data_source",data_source)
    # data_name = os.listdir(data_dir)
    # print(len(data_name))

    url_dataframe = pd.read_csv(csv_dir, index_col =0)
    
    # print(url_dataframe["src"].value_counts())
    if data_source:
        filt = (url_dataframe["src"] == data_source)
        url_dataframe = url_dataframe[filt]
    # print(len(url_dataframe))
    # print(url_dataframe.tail(10))
    
    # url_dataframe = url_dataframe.iloc[:20] ### delete this line for full dataset of 2068 samples
    
    # # filt = url_dataframe["train_or_val"].isnotnull()
    
    # # url_dataframe = pd.notnull(url_dataframe["train_or_val"])
    
    # filt = ((url_dataframe["train_or_val"] == "train") | (url_dataframe["train_or_val"] == "val"))
    # url_dataframe = url_dataframe[filt]

    url_dataframe["ID"] = [str(x) + ".png" for x in url_dataframe["ID"]]
    # print(url_dataframe.head(10))
    # url_dataframe["label_cp"] = [
    #     0 if x == "central" else 1 for x in url_dataframe["label_cp"]] #not true with unlabeled data

    id_list = (url_dataframe["ID"]).to_list()
    # print(id_list[:5])

    label_central_list = []
    label_peripheral_list = []
    label_left_list = []
    label_right_list = []
    label_od_list = []
    label_macula_list = []

    index_list = url_dataframe.index.to_list()
    # print(index_list)

    # print(id_list)
    # python(id_list)
    # for i in range(len(id_list)):
    for i in index_list:
        # print(url_dataframe.iloc[i])

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

    # print(label_central_list[:10])
    # print(label_peripheral_list[:10])
    # print(label_left_list[:10])
    # print(label_right_list[:10])
    # print(label_od_list[:10])
    # print(label_macula_list[:10])

    label_total_list = []

    # print(len(id_list))
    # print(len(index_list))
    # print(len(label_central_list))
    # print(len(label_peripheral_list))
    # print(len(label_left_list))
    # print(len(label_right_list))
    # print(len(label_od_list))
    # print(len(label_macula_list))
    
    for i in range(len(id_list)):
        # print(i)
        item_list = [label_central_list[i], label_peripheral_list[i], label_left_list[i], label_right_list[i], label_od_list[i], label_macula_list[i]]
        label_total_list.append(item_list)

    # for i in range(len(label_total_list)): 
    #     print(label_total_list[i])
    #     break

    # replace this with code that device train/val 
    name_train, name_test, label_train, label_test = train_test_split(
         id_list, label_total_list, test_size=0.3, random_state=42)

    data_train = {'Name': name_train,
                  'Label': label_train}

    data_test = {'Name': name_test,
                 'Label': label_test}

    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)
    
    print("Sample training dataframe")
    print(df_train.head())
    print()

    print("Training class distribution")
    print(df_train["Label"].value_counts())
    print()

    print("Number of training samples")
    print(len(df_train))
    print()
    
    print("Sample testing dataframe")
    print(df_test.head())
    print()

    print("Testing class distribution")
    print(df_test["Label"].value_counts())
    print()

    print("Number of testing samples")
    print(len(df_test))
    print()
    
    print("Number of total samples")
    print(len(df_train) + len(df_test))
    print()
    
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

    """
    # auc_score = roc_auc_score(out_gt.cpu(), out_pred.cpu())
    try:
        auc_score = roc_auc_score(out_gt, out_pred)
    except:
        auc_score = 0
    # for i in range(len(out_pred)):
    #     for j in range(len(out_pred[i])):
    #         if (out_pred[i][j] < 0.5):
    #             out_pred[i][j] = 0
    #         else:
    #             out_pred[i][j] = 1

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

    # print("True positives", true_positives)
    # print("True negatives", true_negatives)
    # print("False positives", false_positives)
    # print("False negatives", false_negatives)
    # print("Support", true_positives + true_negatives + false_positives + false_negatives)

    accuracy = (true_positives + true_negatives) / (true_positives + \
                true_negatives + false_positives + false_negatives)
    #print("Accuracy", accuracy)

    precision = true_positives / \
        (true_positives + false_positives + np.finfo(float).eps)
    recall = true_positives / \
        (true_positives + false_negatives + np.finfo(float).eps)
    # print("Precision", precision)
    # print("Recall", recall)

    f1_score = 2 * precision * recall / \
        (precision + recall + np.finfo(float).eps)
    # print("F1_score", f1_score)

    sensitivity = recall
    specificity = true_negatives / \
        (true_negatives + false_positives + np.finfo(float).eps)
    # print("Sensitivity", sensitivity)
    # print("Specificity", specificity)

    
    return accuracy, precision, recall, f1_score, sensitivity, specificity, auc_score

def calculate_metrics_multilabel(out_gt, out_pred): #calculate_metrics_multilabel_average(out_gt, out_pred)
    # print(out_gt)
    # print(out_pred)

    # start = timeit.default_timer()
    # print(out_gt)
    # print(len(out_gt))

    # print(out_pred)
    # print(len(out_pred))
    # accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()

    try:
        auc_score = roc_auc_score(out_gt, out_pred, average="macro") #average=None
    except:
        auc_score = 0

    for i in range(len(out_pred)):
        for j in range(len(out_pred[i])):
            if (out_pred[i][j] < 0.5):
                out_pred[i][j] = 0
            else:
                out_pred[i][j] = 1

    # print(out_pred)
    
    # print(out_gt)
    # print(out_pred)
    # accuracy = accuracy_score(out_gt, out_pred)
    accuracy = jaccard_score(out_gt, out_pred, average="samples")

    # print(accuracy)

    precision = precision_score(out_gt, out_pred, average='macro',zero_division=1)  #average=None
    recall = recall_score(out_gt, out_pred, average='macro',zero_division=1)  #average=None
    micro_f1 = f1_score(out_gt, out_pred, average='micro',zero_division=1)  #average=None
    macro_f1 = f1_score(out_gt, out_pred, average='macro',zero_division=1)  #average=None
    # auc_score_full = roc_auc_score(out_gt, out_pred,average=None) #average=None
    # print(auc_score_full)

    
    # print(auc_score)

    
    # print(accuracy)
    # print(precision)
    # print(recall)
    # print(f1)
    # print(auc_score)


    out_gt_transpose = np.transpose(out_gt)
    out_pred_transpose = np.transpose(out_pred)

    # print(out_gt_transpose)
    # print(out_pred_transpose)

    sensitivity_list = []
    specificity_list = []
    for i in range(len(out_gt_transpose)):
        _, _, _, _, sensitivity, specificity, _ = calculate_metrics(out_gt_transpose[i], out_pred_transpose[i])
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)
    # print(sensitivity_list.mean())
    # print(specificity_list.mean())
    sensitivity = sensitivity_list.mean()
    specificity = specificity_list.mean()
    # stop = timeit.default_timer()
    # print('Time: ', stop - start)

    
    return accuracy, precision, recall, micro_f1, macro_f1, sensitivity, specificity, auc_score


def calculate_metrics_multilabel_full(out_gt, out_pred):
    # start = timeit.default_timer()
    
    # print(out_gt)
    # print(len(out_gt))

    # print(out_pred)
    # print(len(out_pred))
    # accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()

    try:
        auc_score_list = roc_auc_score(out_gt, out_pred, average=None)
    except:
        auc_score_list = [0, 0, 0, 0, 0, 0]
    # print(out_gt)
    # print(out_pred)
    # accuracy = accuracy_score(out_gt, out_pred)

    # print(accuracy)

    for i in range(len(out_pred)):
        for j in range(len(out_pred[i])):
            if (out_pred[i][j] < 0.5):
                out_pred[i][j] = 0
            else:
                out_pred[i][j] = 1

    precision_list = precision_score(out_gt, out_pred, average=None,zero_division=1)  #average=None, "samples"
    recall_list = recall_score(out_gt, out_pred, average=None,zero_division=1)  #average=None
    f1_list = f1_score(out_gt, out_pred, average=None,zero_division=1)  #average=None
    accuracy_list = jaccard_score(out_gt, out_pred, average=None)
    # auc_score_full = roc_auc_score(out_gt, out_pred,average=None) #average=None
    # print(auc_score_full)

    
    # print(accuracy)
    # print(precision)
    # print(recall)
    # print(f1)
    # print(auc_score)


    out_gt_transpose = np.transpose(out_gt)
    out_pred_transpose = np.transpose(out_pred)

    # print(out_gt_transpose)
    # print(out_pred_transpose)
    # accuracy_list = []

    sensitivity_list = []
    specificity_list = []
    for i in range(len(out_gt_transpose)):
        _, _, _, _, sensitivity, specificity, _ = calculate_metrics(out_gt_transpose[i], out_pred_transpose[i])
        # accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    # accuracy_list = np.array(accuracy_list)
    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)

    # stop = timeit.default_timer()
    # print('Time: ', stop - start) 
    # print(sensitivity_list.mean())
    # print(specificity_list.mean())
    return accuracy_list, precision_list, recall_list, f1_list, sensitivity_list, specificity_list, auc_score_list

if __name__ == '__main__':
    preprocess('../data/data', '../pseudolabel_done.csv', data_source="EYESCAN")

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