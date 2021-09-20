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
*Note:* Don't be worried if you receive an error since the code for downloading data from label studio did not function.
I updated the data path in the file configuration; it should work if you obtain a clone from the server.
***
## *<p style='color:cyan'>Edit training configuration in file config/config.json.</p>*
***
## Train
```bash
python train.py
```
*Images and labels will be saved in the "img" folder.*  
***
*To train, alter the training mode in the configuration by replacing "TRANSFORM IMAGE": true --> false*
*Then, run train.py again*
```bash
python train.py
```
## Test
```bash
python test.py
```

