import pandas as pd

# read DataFrame
data = pd.read_csv("only_val.csv")
df = data[data['train_val_cp'] == 'val']
df.drop(columns=['Unnamed: 0.1','Unnamed: 0'],inplace=True)
df.to_csv('only_val.csv')
