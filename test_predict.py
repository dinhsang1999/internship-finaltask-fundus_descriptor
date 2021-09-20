from os import path
from src.predicter import Predicter
from src.utils import blockPrint,enablePrint
import torch
import pandas as pd
import  os,sys

blockPrint()
path = "data/11805.png"

prediction = Predicter(path_predict=path,src = 'EYESCAN')
result = prediction.predict()


p1 = torch.argmax(result[0][0:2])
p2 = torch.argmax(result[0][2:4])
p3 = torch.argmax(result[0][4:7])

if p1 == 0:
    p_cp = 'central'
else: p_cp = 'peripheral'

if p2 == 0:
    p_lr = 'left'
else: p_lr = 'right'

if p3 == 0:
    p_odm = 'od'
elif p3 == 1:
    p_odm = 'macula'
else:
    p_odm = "null-centered"

enablePrint()
print(p_cp,p_lr,p_odm)

