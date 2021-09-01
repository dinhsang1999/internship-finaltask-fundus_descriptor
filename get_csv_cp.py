import json
import csv
import pandas as pd
from pprint import pprint

path_json_file = 'csvConvert/cp_train.json'
data = json.load(open(path_json_file, 'r'))

image_id_list = []
label_list = []
src_list = []
url_list = []
train_val_predict_list = []

for item in data:
    image_id = item["id"]
    image_url_F = item["image"]

    if "BAIDU" in image_url_F:
        image_url = "http://192.168.0.21:8080" + image_url_F
        src = "BAIDU"
    elif "EYESCAN" in image_url_F:
        image_url = "http://192.168.0.21:8080" + image_url_F
        src = "EYESCAN"
    elif "CTEH" in image_url_F:
        image_url = "http://192.168.0.21:8080" + image_url_F
        src = "CTEH"
    else:
        image_url = image_url_F
        src = "CTEH"

    if "choices" in item["choice"]:
        list_choices = item["choice"]["choices"]
        if "label-central-peri" in list_choices:
            if 'central' in list_choices:
                label = 'central'
            elif 'peripheral' in list_choices:
                label = 'peripheral'
            else:
                label = ''
            
            if 'train-cp' in list_choices:
                train_val_predict = 'train'
            elif 'val-cp' in list_choices:
                train_val_predict = 'val'
            else:
                train_val_predict = 'predict_need'
        else:
            label = ''
            train_val_predict = 'predict_need'
    
    else:
        label = ''
        train_val_predict = 'predict_need'

    image_id_list.append(image_id)
    url_list.append(image_url)
    label_list.append(label)
    train_val_predict_list.append(train_val_predict)
    src_list.append(src)

data_for_pandas = {'ID': image_id_list,
                   'label_cp': label_list,
                   'train_val_predict': train_val_predict_list,
                   'src': src_list,
                   'URL': url_list}

df = pd.DataFrame(data_for_pandas)
print(df.head())
df.to_csv('./csvConvert/cp_train.csv')
print("save csv successfull")

