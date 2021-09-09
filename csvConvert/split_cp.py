import pandas as pd

# read DataFrame
data = pd.read_csv("cp_train.csv")

train = data[data['train_val_predict'] == 'train']
val = data[data['train_val_predict'] == 'val']
train = train.append(val)
predict = data[data['train_val_predict'] == 'predict_need']

train.to_csv('train_cp.csv',index=False)
val.to_csv('val_cp.csv',index=False)
predict.to_csv('predict_cp.csv',index=False)

print('complete slit train,val,predict!')




