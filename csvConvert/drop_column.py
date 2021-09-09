import pandas as pd

# read DataFrame
data = pd.read_csv("complete_odm.csv")

data.drop(columns=['Unnamed: 0','train_val_predict',"URL"],inplace=True)

data.to_csv("complete_odm_drop.csv",index=False)