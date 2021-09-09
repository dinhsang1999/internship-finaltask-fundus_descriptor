from os import path
from src.predicter_lr import Predicter
import torch
import pandas as pd
import  os,sys
# read DataFrame
data = pd.read_csv("csvConvert/predict_lr.csv")
no_predict = len(data)
list_ID = data['ID']
list_src = data['src']

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

# print(no_predict) #=1767
for i in range(no_predict):
    blockPrint()
    path = os.path.join("data_cp/",str(list_ID[i])+".png")
    prediction = Predicter(path_predict=path,src=list_src[i])
    result = prediction.predict()
    data['label_lr'][i] = result['label']
    enablePrint()
    print('{:.0f}/{:.0f}'.format(i+1,no_predict))

print('pseudo label!Done')
data.drop(columns=['Unnamed: 0','train_val_predict','URL'],inplace=True)
data.to_csv('csvConvert/labeled_lr.csv',index=True)
# print(list_ID[0])
# print(data['label_cp'][0])

# prediction = Predicter()
# result = prediction.predict()

# print(result)

# path = os.path.join("data/",str(list_ID[0])+".png")
# print(path)
# prediction = Predicter(path_predict=path)
# result = prediction.predict()
# print(result)