import pandas as pd

# read DataFrame
data = pd.read_csv("csvConvert/cp_train.csv")

# for (train_val_predict),group in data.groupby(['train_val_cp']):
#     group.to_csv(f'{train_val_predict}_cp.csv',index=True)

# print(pd.read_csv("train.csv").head())
# # print(pd.read_csv("predict_cp.csv").head())

train = data[data['train_val_predict'] == 'train']
val = data[data['train_val_predict'] == 'val']
train = train.append(val)
predict = data[data['train_val_predict'] == 'predict_need']

train.to_csv('csvConvert/train_cp.csv',index=False)
val.to_csv('csvConvert/val_cp.csv',index=False)
predict.to_csv('csvConvert/predict_cp.csv',index=False)

print('complete split train,val,predict!')




