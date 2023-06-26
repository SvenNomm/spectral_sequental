# this is the the main file for  lstm case the spectral project
# NB case of the data with two polarisations

import pandas as pd
import os
import datetime
import tensorboard

from model_building_and_training_module import lstm_wrapper_gen_2
from model_building_and_training_module import lstm_wrapper_gen_2_with_callbacks
from model_building_and_training_module import lstm_wrapper_gen_2_with_callbacks_output

from specific_module_v2 import *
from support_module import clustering_support
from support_module import clustering_wrapper
from support_module import plot_clusters
from support_module import input_output_analysis_wrapper
from support_module import clustering_test
import preprocessing_module as ppm

katse_nr = 4
order = 2
dx = 5
number_of_clusters = 2

lim_val = 50
tail_start = 55
path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/'
#path = '/illukas/data/projects/airscs/svens_experiments/processed_by_sven/' # case of ai lab
initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test, \
    valid_index, test_index = combine_katsed_time_based(path, katse_nr, order)


initial_data_train = ppm.add_noise(initial_data_train)
initial_data_test = ppm.add_noise(initial_data_test)



#initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test, \
#    valid_index, test_index = return_datasets_dnora(path)

#clustering_test(initial_data_train, target_data_train, initial_data_test, target_data_test, number_of_clusters)
#initial_data_train, target_data_train = ppm.value_range_filter(initial_data_train, target_data_train, lim_val)
#initial_data_test, target_data_test = ppm.value_range_filter(initial_data_test, target_data_test, lim_val)
#initial_data_valid, target_data_valid = ppm.value_range_filter(initial_data_valid, target_data_valid, lim_val)

#initial_data_train = ppm.cut_tail(initial_data_train, tail_start)
#target_data_train = ppm.cut_tail(target_data_train, tail_start)

#initial_data_test = ppm.cut_tail(initial_data_test, tail_start)
#target_data_test = ppm.cut_tail(target_data_test, tail_start)

#initial_data_valid = ppm.cut_tail(initial_data_valid, tail_start)
#target_data_valid = ppm.cut_tail(target_data_valid, tail_start)



#dtw_labels, labels_k_means = clustering_wrapper(initial_data_train, target_data_train, number_of_clusters)
#plot_clusters(initial_data_train, target_data_train, dtw_labels, labels_k_means)

#initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test,\
#    valid_index, test_index = combine_katsed_with_clustering(path, katse_nr, order)

# apply clustering and see
# initial_data_cluster, target_data_cluster, largest_cluster = clustering_support(initial_data_train, target_data_train)
# initial_data_train = initial_data_cluster[largest_cluster]
# target_data_train = target_data_cluster[largest_cluster]

# input_params, output_params, difference_params, corr_matrix = input_output_analysis_wrapper(initial_data_test,
#                                                                                            target_data_test, dx=dx)


model_path = path + 'models/'
data_name = 'katse_0' + str(katse_nr) + '_0' + str(katse_nr + 1)

lstm_wrapper_gen_2_with_callbacks(initial_data_train, initial_data_valid, initial_data_test, target_data_train,
                                  target_data_valid, target_data_test, valid_index, test_index, model_path, data_name,
                                  epochs_nr=500)

print("That's all folks!!!")
