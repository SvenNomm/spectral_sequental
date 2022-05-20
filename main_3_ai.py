# this is the the main file for  lstm case the spectral project
# NB case of the data with two polarisations

import pandas as pd
import os
import datetime

from model_building_and_training_module import lstm_wrapper_gen_2
from model_building_and_training_module import lstm_wrapper_gen_2_with_callbacks
from specific_module_v2 import *

katse_nr = 1
order = 1
#path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/'
path_initial_data = '/illukas/data/projects/airscs/wave/data_v2/' # case of ai lab
path_processed= '/illukas/data/projects/airscs/svens_experiments/wave/'
#initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = \
#    combine_katsed_ai(path_initial_data, path_processed, katse_nr, order)

initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test, \
valid_index, test_index = combine_katsed_ai(path_initial_data, path_processed, katse_nr, order)


model_path = path_processed + 'models/'
data_name = 'katse_0' + str(katse_nr) + '_0' + str(katse_nr+1)
#lstm_wrapper_gen_2(initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index,
#                   model_path, data_name, epochs_nr=10000)
lstm_wrapper_gen_2_with_callbacks(initial_data_train, initial_data_valid, initial_data_test, target_data_train,
                                  target_data_valid, target_data_test, valid_index, test_index, model_path, data_name,
                                  epochs_nr=5000)

print("That's all folks!!!")