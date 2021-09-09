import pandas as pd
df = pd.concat(
    map(pd.read_csv, ['train_lr.csv','val_lr.csv','labeled_lr.csv']), ignore_index=True)
df.to_csv('complete_lr.csv',index=False)
print(df)