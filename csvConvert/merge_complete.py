from numpy import nan
import pandas as pd
import sys

df_cp = pd.read_csv('complete_cp_drop.csv')
df_lr = pd.read_csv('complete_lr_drop.csv')
df_odm = pd.read_csv('complete_odm_drop.csv')
df_odm['label_odm'] = ["macula" if x == "macula" else "od" if x == "od" else "null-centered" for x in df_odm["label_odm"]]
df = pd.merge(df_cp, df_lr,on='ID',how='inner')
df = pd.merge(df,df_odm,on='ID',how='inner')
df = df.nsmallest(2068,"ID")
df.to_csv('pseudolabel_done.csv',index=False)
df = pd.read_csv('pseudolabel_done.csv')
# print(df["label_odm"][720])
# sys.exit()
df.to_csv('pseudolabel_done.csv',index=True)
print(df)
