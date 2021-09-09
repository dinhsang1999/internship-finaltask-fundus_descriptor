import pandas as pd
import os,sys
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

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
            elif score <= self.best_score + self.delta:
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
            elif score <= self.best_score + self.delta:
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


def preprocess(data_dir, csv_dir,train_val_split=0.3,train_val_split_status=False):
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
    data_name = os.listdir(data_dir)
    init_dataframe = pd.read_csv(csv_dir)

    list_id = [str(x) + ".png" for x in init_dataframe["ID"]]
    list_src = init_dataframe["src"].to_list()

    list_cp = init_dataframe['label_cp']
    list_lr = init_dataframe['label_lr']
    list_odm = init_dataframe['label_odm']

    total_list_label = []
    #central-peri-left-right-od-macula-null

    for idx in range(len(list_id)):
        list_label = []

        if list_cp[idx] == 'central':
            list_label.append(1)
        else:
            list_label.append(0)

        if list_cp[idx] == 'peripheral':
            list_label.append(1)
        else:
            list_label.append(0)

        if list_lr[idx] == 'left':
            list_label.append(1)
        else:
            list_label.append(0)

        if list_lr[idx] == 'right':
            list_label.append(1)
        else:
            list_label.append(0)

        if list_odm[idx] == 'od':
            list_label.append(1)
        else:
            list_label.append(0)
        
        if list_odm[idx] == 'macula':
            list_label.append(1)
        else:
            list_label.append(0)

        if list_odm[idx] == 'null-centered':
            list_label.append(1)
        else:
            list_label.append(0)
        
        total_list_label.append(list_label)

    # if train_val_split_status:
    #     name_train, name_test, label_train, label_test,src_train,src_test = train_test_split(
    #         total_name, total_label,total_src, test_size=train_val_split, random_state=42)

    #     data_train = {'Name': name_train,
    #                 'Label': label_train,
    #                 'Source':src_train}

    #     data_test = {'Name': name_test,
    #                 'Label': label_test,
    #                 'Source':src_test}

    #     df_train = pd.DataFrame(data_train)
    #     df_test = pd.DataFrame(data_test)

    #     return df_train, df_test
    if train_val_split_status == False:
        data = {'Name': list_id,
                    'Label': total_list_label,
                        'src': list_src} 
        df_data = pd.DataFrame(data)
        return df_data
        

    

# Block
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

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

    accuracy = (true_positives + true_negatives) / (true_positives +
                                                    true_negatives + false_positives + false_negatives)

    precision = true_positives / \
        (true_positives + false_positives + np.finfo(float).eps)
    recall = true_positives / \
        (true_positives + false_negatives + np.finfo(float).eps)

    f1_score = 2 * precision * recall / \
        (precision + recall + np.finfo(float).eps)

    sensitivity = recall
    specificity = true_negatives / \
        (true_negatives + false_positives + np.finfo(float).eps)

    return accuracy, precision, recall, f1_score, sensitivity, specificity

def calculate_metrics_multilabel(out_gt, out_pred): #calculate_metrics_multilabel_average(out_gt, out_pred)
    # print(out_gt)
    # print(len(out_gt))

    # print(out_pred)
    # print(len(out_pred))
    # accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

    out_gt = out_gt.cpu().detach().numpy()
    out_pred = out_pred.cpu().detach().numpy()
    # print(out_gt)
    # print(out_pred)
    accuracy = accuracy_score(out_gt, out_pred)

    # print(accuracy)

    precision = precision_score(out_gt, out_pred, average='samples',zero_division=1)  #average=None
    recall = recall_score(out_gt, out_pred, average='samples',zero_division=1)  #average=None
    f1 = f1_score(out_gt, out_pred, average='samples',zero_division=1)  #average=None
    
    # auc_score_full = roc_auc_score(out_gt, out_pred,average=None) #average=None
    # print(auc_score_full)

    auc_score = roc_auc_score(out_gt, out_pred) #average=None
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
        _, _, _, sensitivity, specificity, _ = calculate_metrics(out_gt_transpose[i], out_pred_transpose[i])
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)
    # print(sensitivity_list.mean())
    # print(specificity_list.mean())
    return accuracy, precision, recall, f1, sensitivity, specificity, auc_score

def calculate_metrics_multilabel_full(out_gt, out_pred):
    # print(out_gt)
    # print(len(out_gt))

    # print(out_pred)
    # print(len(out_pred))
    # accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))

    out_gt = out_gt
    out_pred = out_pred

    # print(out_gt)
    # print(out_pred)
    # accuracy = accuracy_score(out_gt, out_pred)

    # print(accuracy)

    precision_list = precision_score(out_gt, out_pred, average=None,zero_division=1)  #average=None, "samples"
    recall_list = recall_score(out_gt, out_pred, average=None,zero_division=1)  #average=None
    f1_list = f1_score(out_gt, out_pred, average=None,zero_division=1)  #average=None
    
    # auc_score_full = roc_auc_score(out_gt, out_pred,average=None) #average=None
    # print(auc_score_full)

    auc_score_list = roc_auc_score(out_gt, out_pred, average=None)
    # print(accuracy)
    # print(precision)
    # print(recall)
    # print(f1)
    # print(auc_score)


    out_gt_transpose = np.transpose(out_gt)
    out_pred_transpose = np.transpose(out_pred)

    # print(out_gt_transpose)
    # print(out_pred_transpose)
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    for i in range(len(out_gt_transpose)):
        accuracy, _, _, _, sensitivity, specificity, _ = calculate_metrics(out_gt_transpose[i], out_pred_transpose[i])
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    accuracy_list = np.array(accuracy_list)
    sensitivity_list = np.array(sensitivity_list)
    specificity_list = np.array(specificity_list)

    # print(sensitivity_list.mean())
    # print(specificity_list.mean())
    return accuracy_list, precision_list, recall_list, f1_list, sensitivity_list, specificity_list, auc_score_list

if __name__ == '__main__':
    # test = EarlyStopping()
    # preprocess('../data', '../csvConvert/pseudolabel_done.csv')
    out_y = np.array([[1., 0., 0., 1., 0., 1., 0.],
        [1., 0., 0., 1., 1., 0., 0.],
        [1., 0., 0., 1., 0., 1., 0.],
        [0., 1., 1., 0., 0., 0., 0.],
        [1., 0., 0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 0., 1., 0.],
        [1., 0., 0., 1., 1., 0., 0.]])

    out_pred = np.array([[1., 0., 0., 1., 0., 1., 0.],
        [1., 0., 0., 1., 0., 0., 1.],
        [1., 0., 0., 1., 0., 1., 0.],
        [0., 1., 0., 1., 0., 0., 0.],
        [0., 1., 0., 1., 0., 1., 1.],
        [1., 0., 1., 0., 0., 1., 0.],
        [1., 0., 0., 1., 1., 0., 1.]])
    
    accuracy_list, precision_list, recall_list, f1_list, sensitivity_list, specificity_list, auc_score_list = calculate_metrics_multilabel_full(out_y, out_pred)
    print("accuracy",accuracy_list)
    print("precision_list",precision_list)
    print("recall_list",recall_list)
    print("f1_list",f1_list)
    print("sensitivity_list",sensitivity_list)
    print("specificity_list",specificity_list)
    print("auc_score_list",auc_score_list)
