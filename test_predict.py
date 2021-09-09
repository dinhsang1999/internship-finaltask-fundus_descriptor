from os import path
from src.predicter_odm import Predicter
from src.utils_odm import blockPrint,enablePrint
import torch
import pandas as pd
import  os,sys

blockPrint()
path = "data/11805.png"
prediction = Predicter(path_predict=path,src = 'EYESCAN')
result = prediction.predict()
enablePrint()
print(result)