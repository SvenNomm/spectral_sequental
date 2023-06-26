# this is the the main file for  boosted regression
# NB case of the data with two polarisations

import pandas as pd
import os
import datetime
import tensorboard
from sklearn.ensemble import GradientBoostingRegressor


from specific_module_v2 import *


katse_nr = 8
order = 2
path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_v2/'
# path = '/illukas/data/projects/airscs/svens_experiments/processed_by_sven/' # case of ai lab
initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test, \
valid_index, test_index = combine_katsed_simple(path, katse_nr, order)

model_path = path + 'models/'
data_name = 'katse_0' + str(katse_nr) + '_0' + str(katse_nr + 1)

init_data = initial_data_train.to_numpy
targ_data = initial


model_gradientBoosting = GradientBoostingRegressor()
model_gradientBoosting.fit(initial_data_train, target_data_train)

print("That's all folks!!!")
