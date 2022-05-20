
import tensorflow as tf
import keras
import pandas as pd
import os
import datetime
import pickle
from validation_module import test_model


katse_nr = 1
order = 1
path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/'

model_path = path + 'models/'
model_name = 'lstm_katse_01_02_10000_epoch_05_03_2022_21_04_50'

lstm_model = tf.keras.models.load_model(model_path + model_name)

data_name = 'katse_1_2_testing_data_05_03_2022_23_06_55'
data_file_name = path + 'processed_datasets/' + data_name + '.pkl'

with open(data_file_name, 'rb') as f:
    initial_data_valid, target_data_valid, valid_index = pickle.load(f)

test_model(initial_data_valid, target_data_valid, lstm_model, valid_index, model_path, data_name, model_name)

print("That's all folks!!!")