# this is the the main file for  lstm case the spectral project
# NB case of the data with two polarisations

import pandas as pd
import os
import datetime
import tensorboard

from model_building_and_training_module import lstm_wrapper_gen_2
from model_building_and_training_module import lstm_wrapper_gen_2_with_callbacks
from specific_module_v2 import *
from preprocessing_module import clustering_support
import matplotlib.pyplot as plt
from preprocessing_module import element_wise_lin_div

katse_nr = 8
order = 2
path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/'
# path = '/illukas/data/projects/airscs/svens_experiments/processed_by_sven/' # case of ai lab
initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test, \
valid_index, test_index = combine_katsed_simple(path, katse_nr, order)


initial_data_train_1, initial_data_train_2 = return_data_sets_simple(path, order, 384)

ffname = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_insitu/insitusarspec_hgh_order_2_winx_512_co.csv'
external_data_1 = pd.read_csv(ffname, sep=',', encoding="utf-8")

ffname = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_insitu/insitusarspec_hgh_order_2_winx_512_cro.csv'
external_data_2 = pd.read_csv(ffname, sep=',', encoding="utf-8")


del external_data_1['station_ind']
del external_data_1['time']
del external_data_1['Unnamed: 0']

del external_data_2['station_ind']
del external_data_2['time']
del external_data_2['Unnamed: 0']


selected_rows = external_data_1.loc[external_data_1.isna().any(axis=1)].index.tolist()
external_data_1 = external_data_1.drop(external_data_1.index[selected_rows])

selected_rows = external_data_2.loc[external_data_2.isna().any(axis=1)].index.tolist()
external_data_2 = external_data_2.drop(external_data_2.index[selected_rows])

#external_data = element_wise_lin_div(external_data_1, external_data_2)

#selected_rows = external_data_1.loc[external_data_1.isna().any(axis=1)].index.tolist()
#external_data = external_data_1.drop(external_data_1.index[selected_rows])

#selected_rows = external_data_2.loc[external_data_2.isna().any(axis=1)].index.tolist()
#external_data = external_data_2.drop(external_data_2.index[selected_rows])


fig1, axis = plt.subplots()
initial_data_train_2 = initial_data_train_2.to_numpy()
rows, cols = initial_data_train_2.shape

for i in range(0, rows):
    #print(i)
    plt.plot(initial_data_train_2[i, :], color='blue', linewidth=0.1)

external_data_2 = external_data_2.to_numpy(dtype='float')
rows, cols = external_data_2.shape
for i in range(0, rows):
    #print(i)
    plt.plot(external_data_2[i, :], color='orange', linewidth=0.1)
#plt.title("validation for", str(test_index.loc[i]))

plt.show()

# ffname = '/Users/svennomm/Downloads/extern_spec_predictions_co_cro_sepa_hgh_2_384_20220526T0019_katse_08_extrateston_valid_data.csv'
# external_data_2 = pd.read_csv(ffname, sep=',', encoding="utf-8")
# del external_data_2['index']
# del external_data_2['hs']
# del external_data_2['tp']
#
# selected_rows = external_data_2.loc[external_data_2.isna().any(axis=1)].index.tolist()
# external_data_2 = external_data_2.drop(external_data_2.index[selected_rows])
# external_data_2 = external_data_2.iloc[:,0:69].to_numpy(dtype='float')
# rows, cols = external_data_2.shape
#
# fig2, axis = plt.subplots()
# for i in range(0, rows):
#     plt.plot(external_data_2[i,:], color='orange', linewidth=0.1)
#
# initial_data_test = initial_data_test.to_numpy()
# rows, cols = initial_data_test.shape
# for i in range(0, rows):
#     #print(i)
#     plt.plot(initial_data_test[i, :], color='blue', linewidth=0.1)
#
# plt.show()
#
# print("That's all folks!!!")
