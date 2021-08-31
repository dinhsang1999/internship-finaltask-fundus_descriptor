mkdir data_cp
mkdir models
mkdir log
pip install tensorboard

python get_csv_cp.py
python get_data_cp.py
python csvConvert/split_cp.py