# this is the the main file for  lstm case the spectral project
# NB case of the data with two polarisations

import pandas as pd

from preprocessing_module import initial_formatting_gen_2
from preprocessing_module import initial_formatting_gen_2_3# use this line for the third generation of the  data set
from model_building_and_training_module import lstm_wrapper_gen_2
from preprocessing_module import delete_nan_rows
from preprocessing_module import element_wise_lin_div


path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/katse_01/'
unique_id = '_hgh_order_1_winx_256_'
polarisation = 'co_'
initial_data_file = path + 'sarspec'+ unique_id + polarisation +'clean.csv' # this and the following row are for the data xx.12.2021 - xx.01.2022
target_data_file = path + 'wavespec' + unique_id + 'clean.csv'

polarisation = 'co_'
initial_data_file_1 = path + 'sarspec'+ unique_id + polarisation +'clean.csv'
initial_data_1 = pd.read_csv(initial_data_file_1, sep=',')
polarisation = 'cro_'
initial_data_file_2 = path + 'sarspec'+ unique_id + polarisation +'clean.csv'
initial_data_2 = pd.read_csv(initial_data_file_2, sep=',')
polarisation = 'cocro_'


#initial_data = pd.read_csv(initial_data_file, sep=',') # keep in mind which separator to use
target_data = pd.read_csv(target_data_file, sep=',')

initial_data_1, initial_data_2, target_data = initial_formatting_gen_2_3(initial_data_1, initial_data_2, target_data)
initial_data = element_wise_lin_div(initial_data_1, initial_data_2)
initial_data, target_data = delete_nan_rows(initial_data, target_data)

#initial_data, target_data = initial_formatting_gen_2(initial_data, target_data)
#initial_data = down_sample(initial_data, target_data)
model_path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/models/'
data_name = unique_id + polarisation
lstm_wrapper_gen_2(initial_data, target_data, model_path, data_name, epochs_nr=10000)

print("That's all folks!!!")