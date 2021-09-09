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
## *<p style='color:cyan'>Edit training configuration in file config/config.json.</p>*
## Train
```bash
python train.py
```
***