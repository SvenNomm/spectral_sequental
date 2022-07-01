# this file contains support functions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.integrate import simps
#from model_building_and_training_module import dtw_measure
from dtaidistance import dtw
from tslearn.clustering import TimeSeriesKMeans
import sklearn.cluster as sklc

from matplotlib import colors


def labels2colors(labels, color_list):
    color_labels = []
    for i in range(0, len(labels)):
        color_labels.append(color_list[labels[i]])

    return color_labels



def dtw_measure(y_true, y_pred):
    y_true = y_true.to_numpy()
    y_pred = y_pred.to_numpy()
    return dtw.distance(y_true, y_pred)


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
            #a = np.corrcoef(input_params[:, i], input_params[:, j])
            #print(i,j, a)
            corr_matrix[i, j] = np.corrcoef(input_params[:, i], output_params[:, j])[0, 1]

    #fig, ax = plt.subplots()
    #fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    #plt.scatter(input_params[:,2], input_params[:,3], input_params[:,4], alpha=0.5)
    #plt.show()

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


def clustering_support(initial_data, target_data):
    number_of_clusters = 5

    #DTW clustering
    km_init = TimeSeriesKMeans(n_clusters=5, metric="softdtw", max_iter=5, random_state=0).fit_predict(initial_data)
    km_target = TimeSeriesKMeans(n_clusters=5, metric="softdtw", max_iter=5, random_state=0).fit_predict(target_data)
    #fig, ax = plt.subplots()

    dx = 5
    input_params, output_params, difference_params, corr_matrix = input_output_analysis_wrapper(initial_data,
                                                                                                target_data, dx=dx)
    #cluster by input and output params
    kmeans = sklc.KMeans(n_clusters=number_of_clusters, random_state=0).fit(input_params[:, 2:4])
    labels_inp = kmeans.labels_

    kmeans = sklc.KMeans(n_clusters=number_of_clusters, random_state=0).fit(output_params[:, 2:4])
    labels_output = kmeans.labels_


    fig = plt.figure()
    plt.axis('off')
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    #color_list = ['b', 'y', 'g', 'r', 'm']
    color_list = np.array([[228, 26, 28], [55, 126, 184], [77, 175, 74], [152, 78, 163], [255, 127, 0], [255, 255, 51],
                    [166, 86, 40], [247, 129, 191], [153, 153, 153]]) / 255



    for k in range(0, np.max(km_init)+1):
        selected_rows_init = np.where(km_init == k)
        selected_rows_target = np.where(km_target == k)

        initial_data_cluster = initial_data.loc[selected_rows_init[0]]
        #target_data_cluster = target_data.loc[selected_rows_init[0]]
        #print("k is : ", k)

        for i in range(0, len(selected_rows_init[0])):
            #print("i is : ", i, "out of:", len(selected_rows_init[0]))
            #b = selected_rows_init[0][i]
            #a = initial_data.loc[selected_rows_init[i], :]
            ax1.plot(initial_data.loc[selected_rows_init[0][i], :], color=color_list[k], alpha=0.4, linewidth=0.5)
            ax2.plot(target_data.loc[selected_rows_init[0][i], :], color=color_list[k], alpha=0.4, linewidth=0.5)
            #title('Clustered initial data')
            ax1.set_xticks([])
            ax2.set_xticks([])
            #plt2.title('Corresponding target data')

        for i in range(0, len(selected_rows_target[0])):
            #print("i is : ", i, "out of:", len(selected_rows_init[0]))
            #b = selected_rows_init[0][i]
            #a = initial_data.loc[selected_rows_init[i], :]
            ax3.plot(initial_data.loc[selected_rows_target[0][i], :], color=color_list[k], alpha=0.5, linewidth=0.5)
            ax4.plot(target_data.loc[selected_rows_target[0][i], :], color=color_list[k], alpha=0.5, linewidth=0.5)
            ax3.set_xticks([])
            ax4.set_xticks([])
            #plt3.title('Corresponding initial data')
            #plt4.title('Clustered target data')

    plt.show()


    fig = plt.subplots()
    plt.axis('off')
    ax1 = plt.axes(projection='3d')
    color_labels = labels2colors(labels_output, color_list)
    ax1.scatter(input_params[:, 2], input_params[:, 3], input_params[:, 4], c=labels_inp, cmap=matplotlib.colors.ListedColormap(color_list), edgecolors=color_labels)
    plt.show()

    fig, ax = plt.subplots()
    plt.axis('off')
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    for k in range(0, np.max(labels_inp)):
        selected_rows_init = np.where(labels_inp == k)
        selected_rows_target = np.where(labels_output == k)

        for i in range(0, len(selected_rows_init[0])):
            # title('Clustered initial data')
            ax1.plot(initial_data.loc[selected_rows_init[0][i], :], color=color_list[k], alpha=0.4, linewidth=0.5)
            # plt2.title('Corresponding target data')
            ax2.plot(target_data.loc[selected_rows_init[0][i], :], color=color_list[k], alpha=0.4, linewidth=0.5)

            ax1.set_xticks([])
            ax2.set_xticks([])

        for i in range(0, len(selected_rows_target[0])):
            # title('Clustered initial data')
            ax3.plot(initial_data.loc[selected_rows_target[0][i], :], color=color_list[k], alpha=0.4, linewidth=0.5)
            # plt2.title('Corresponding target data')
            ax4.plot(target_data.loc[selected_rows_target[0][i], :], color=color_list[k], alpha=0.4, linewidth=0.5)

            ax3.set_xticks([])
            ax4.set_xticks([])

    plt.show()

    return initial_data, target_data


def rho2labels(goodness_descriptors):
    rho_labels = []
    for i in range(0, len(goodness_descriptors)):
        rl = 6
        if goodness_descriptors.loc[i, 'rho'] < 0.95:
            rl = 5

        if goodness_descriptors.loc[i, 'rho'] < 0.9:
            rl = 4

        if goodness_descriptors.loc[i, 'rho'] < 0.85:
            rl = 3

        if goodness_descriptors.loc[i, 'rho'] < 0.8:
            rl = 2

        if goodness_descriptors.loc[i, 'rho'] < 0.75:
            rl = 1

        if goodness_descriptors.loc[i, 'rho'] < 0.7:
            rl = 0

        rho_labels.append(rl)

    return rho_labels