import pandas as pd

# read DataFrame
data = pd.read_csv("odm_train.csv")

train = data[data['train_val_predict'] == 'train']
val = data[data['train_val_predict'] == 'val']
predict = data[data['train_val_predict'] == 'predict_need']

train.to_csv('train_odm.csv',index=False)
val.to_csv('val_odm.csv',index=False)
predict.to_csv('predict_odm.csv',index=False)

print('complete slit train,val,predict!')