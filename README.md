# FUNDUS DESCRIPTOR
***
## Requirements
- Python >= 3.9

## Setup environment
Run this script to create a virtual environment and install dependency libraries
```bash
pip install -r requirements.txt
```
Set up: .csv,folder,tensorboard and download data
```bash
bash download_data.sh
```
*Note:* Don't worry about geting error because the code downloading data from label_studio was not work.
I changed data path from file config, it will be work if you get clone from the sever.
***
## *<p style='color:cyan'>Edit training configuration in file config/config.json.</p>*
***
## Train
```bash
python train.py
```
*It will save images and labels into "img" folder*  
***
*To train, you need change training mode in config by replace "TRANSFORM_IMAGE": false --> true*
*Then, run train.py again*
```bash
python train.py
```
## Test
```bash
python test.py
```

