mkdir data
mkdir models
mkdir log
pip install tensorboard

python get_csv_odm.py
python get_data_odm.py
python csvConvert/split_odm.py