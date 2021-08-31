import pandas as pd
import requests
from PIL import Image
import io
import os

url_dataframe = pd.read_csv('csvConvert/cp_train.csv')
url_list = url_dataframe["URL"]
id_list = url_dataframe["ID"]

headers = {"Authorization": "token 0b4e6bf83b76636c40b93f6d6da30a928f73a1bc",
           "Cookie": "sessionid=.eJxVj82OwjAMhN_FZ1oRhyYOx73zDJVjG9pllaz6cwHx7hTEhcscZr4Zae6wjgpHcEaMGHKTXNDmgJ4bkrhvyAmJksNNYAd1unAZb7yMtfT_Vzi6HfS8LkO_zjb17ymELy-zXK28Av3lcqmt1LJMY25fSPtJ5_ZU1f5-PuzXwMDzsLW7ZBqTxOTV79V3Zzww6dmEIwdCSl0W0e1HNsNM0dC7EC2wZJ8oZ3g8AbuHS3w:1mIHCO:xyZspCdAQZBfPIhwMogioTNxfX0XTY3n6Qg9Wun4vLU"}

for index in range(len(url_list)):
    image_id = id_list[index]
    response = requests.get(url_list[index], headers=headers, stream=True)
    image_data = response.content
    image = Image.open(io.BytesIO(image_data))
    image.save(os.path.join("data_cp", str(image_id) + ".png"))
    print("loading " + str(index + 1) + " / " + str(len(url_list)) + " images")
print('DONE!!!')
