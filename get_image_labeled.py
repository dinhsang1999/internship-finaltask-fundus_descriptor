import pandas as pd
import requests
from PIL import Image
import io
import os

url_dataframe = pd.read_csv('get_csv_all.csv')
url_list = url_dataframe["URL"]
id_list = url_dataframe["ID"]
split_list = url_dataframe["train_or_val"]

# print(url_list)
# print(id_list)


for index in range(len(url_list)):
    # print("index: ", index)
    image_id = id_list[index]
    if (os.path.exists(os.path.join("data", str(image_id) + ".png"))) or (os.path.exists(os.path.join("samples", str(image_id) + ".png"))):
        continue
    
    
    
    if ("BAIDU" in url_list[index])  or ("EYESCAN" in url_list[index]) or ("CTEH" in url_list[index]) :
        headers = {"Authorization": "token 0b4e6bf83b76636c40b93f6d6da30a928f73a1bc",
           "Cookie": "sessionid=.eJxVj82OwjAMhN_FZ1oRhyYOx73zDJVjG9pllaz6cwHx7hTEhcscZr4Zae6wjgpHcEaMGHKTXNDmgJ4bkrhvyAmJksNNYAd1unAZb7yMtfT_Vzi6HfS8LkO_zjb17ymELy-zXK28Av3lcqmt1LJMY25fSPtJ5_ZU1f5-PuzXwMDzsLW7ZBqTxOTV79V3Zzww6dmEIwdCSl0W0e1HNsNM0dC7EC2wZJ8oZ3g8AbuHS3w:1mIHCO:xyZspCdAQZBfPIhwMogioTNxfX0XTY3n6Qg9Wun4vLU"}
        response = requests.get(
            url_list[index], headers=headers, stream=True)

        # print(response.status_code)

        response.raw.decode_content = True
        image = Image.open(response.raw)
    else:
        # print('getting image from: ', url_list[index])
        response = requests.get(url_list[index])
        # print('response: ',response)
        
        image_data = response.content
        image = Image.open(io.BytesIO(image_data))
        # print(image)
    
    image_split = split_list[index] 
    destination = ""
    if image_split == "train" or image_split == "val":
        image.save(os.path.join("data", str(image_id) + ".png"))
        destination = "data"
    else:
        image.save(os.path.join("samples", str(image_id) + ".png"))
        destination = "samples"
        
    print("loading " + str(index + 1) + " / " + str(len(url_list)) + " images" + ", image id: " + str(id_list[index]) + ", destination: " + destination)





# http://192.168.0.21:8080/data/local-files/?d=CTEH-NIDEK/CTEH-001426.jpg
# http://192.168.0.21:8080/data/local-files/?d=CTEH-NIDEK/CTEH-001426.jpg