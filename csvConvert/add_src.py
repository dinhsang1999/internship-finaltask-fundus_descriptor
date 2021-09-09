import pandas as pd
import sys

df_1 = pd.read_csv('pseudolabel_done.csv')
df_2 = pd.read_csv('complete_cp.csv')

df = pd.merge(df_1, df_1,on='ID',how='inner')