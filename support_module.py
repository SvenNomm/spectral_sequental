# this file contains support functions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.integrate import simps
#from model_building_and_training_module import dtw_measure



def return_curve_descriptors(data, dx):
    data = data.to_numpy()
    #print(data.shape)
    area_trapz = np.trapz(data, axis=0, dx=dx)
    area_simps = simps(data, axis=0, dx=dx)
    area_sum = np.sum(data)

    data_max = np.max(data)

    data_skew = skew(data, bias=True, nan_policy='propagate')

    return [area_trapz, area_simps, area_sum, data_max, data_skew]


def input_output_analysis_wrapper(input_data, output_data, dx):
    rows = len(input_data)

    input_params = np.zeros((rows, 5))
    output_params =np.zeros((rows, 5))
    difference_params = np.zeros((rows, 6))

    for i in range(0, rows):
        input_params[i, :] = np.array(return_curve_descriptors(input_data.loc[i, :], dx))
        output_params[i, :] = np.array(return_curve_descriptors(output_data.loc[i, :], dx))
        difference_params[i, 0:-1] = input_params[i, :] - output_params[i, :]
        a = dtw_measure(input_data.loc[i, :], output_data.loc[i, :])
        difference_params[i, 5] = dtw_measure(input_data.loc[i, :], output_data.loc[i, :])

    corr_matrix = np.zeros((5,5))

    for i in range(0, 5):
        for j in range(0, 5):
            a = np.corrcoef(input_params[:, i], output_params[:, j])
            #print(a)
            corr_matrix[i, j] = np.corrcoef(input_params[:, i], output_params[:, j])[0, 1]

    return input_params, output_params, difference_params, corr_matrix


class DtwLoss(tf.keras.losses.Loss):
    def __init__(self, batch_size: int = 32):
        super(DtwLoss, self).__init__()
        self.batch_size = batch_size

    def call(self, y_true, y_pred):
        tmp = []
        for item in range(self.batch_size):
            tf.print(f'Working on batch: {item}\n')
            s = y_true[item, :]
            t = y_pred[item, :]
            n, m = len(s), len(t)
            dtw_matrix = []
            for i in range(n + 1):
                line = []
                for j in range(m + 1):
                    if i == 0 and j == 0:
                        line.append(0)
                    else:
                        line.append(np.inf)
                dtw_matrix.append(line)

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = tf.abs(s[i - 1] - t[j - 1])
                    last_min = tf.reduce_min([dtw_matrix[i - 1][j], dtw_matrix[i][j - 1], dtw_matrix[i - 1][j - 1]])
                    dtw_matrix[i][j] = tf.cast(cost, dtype=tf.float32) + tf.cast(last_min, dtype=tf.float32)

            temp = []
            for i in range(len(dtw_matrix)):
                temp.append(tf.stack(dtw_matrix[i]))

            tmp.append(tf.stack(temp)[n, m])
        return tf.reduce_mean(tmp)