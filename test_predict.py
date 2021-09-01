from os import path
from src.predicter_cp import Predicter
from src.utils_cp import blockPrint,enablePrint
import torch
import pandas as pd
import  os,sys

blockPrint()
path = "data_cp/12453.png"
prediction = Predicter(path_predict=path)
result = prediction.predict()
enablePrint()
print(result)