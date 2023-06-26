import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import datetime
from support_module import clustering_support
from support_module import clustering_wrapper
from support_module import get_largest_cluster
from support_module import plot_clusters
import matplotlib.pyplot as plt
import pickle as pkl


def extract_station_observations(input_data_1, input_data_2, target_data):
    input_columns_1 = input_data_1.columns.values.tolist()
    input_frame_1 = pd.DataFrame(columns=input_columns_1)

    input_columns_2 = input_data_2.columns.values.tolist()
    input_frame_2 = pd.DataFrame(columns=input_columns_2)

    target_columns = target_data.columns.values.tolist()
    output_frame = pd.DataFrame(columns=target_columns)

    time_list = input_data_1['time'].drop_duplicates()

    for instance in time_list:
        time_frame = input_data_1.loc[input_data_1['time'] == instance]
        station_list = time_frame['station_ind'].drop_duplicates().tolist()

        for station in station_list:
            station_frame = time_frame.loc[time_frame['station_ind'] == station]
            index_list = station_frame.index.values.tolist()
            input_frame_1 = pd.concat([input_frame_1, station_frame.loc[index_list[0], :].to_frame().T])

    indexes = input_frame_1.index.values.tolist()
    input_frame_2 = input_data_2.iloc[indexes]
    output_frame = target_data.iloc[indexes]

    input_frame_1 = input_frame_1.reset_index()
    del input_frame_1['index']

    input_frame_2 = input_frame_2.reset_index()
    del input_frame_2['index']

    output_frame = output_frame.reset_index()
    del output_frame['index']

    return input_frame_1, input_frame_2, output_frame


def initial_formatting_gen_2(initial_data_1, initial_data_2, target_data):
    print("Aloha! Performing initial formatting")

    # the following lines are very local
    del initial_data_1['time']
    del initial_data_2['time']
    del target_data['time']
    del initial_data_1['station_ind']
    del initial_data_2['station_ind']
    del target_data['station_ind']
    # del initial_data_1['Unnamed: 0']
    # del initial_data_2['Unnamed: 0']
    # del target_data['Unnamed: 0']

    # target_data.drop(target_data.index[0], inplace=True)
    # target_data = target_data.reset_index()
    # del target_data['index']

    print("Initial  formatting has been completed!")
    return initial_data_1, initial_data_2, target_data


def delete_nan_rows(df1, df2):
    selected_rows = df1.loc[df1.isna().any(axis=1)].index.tolist()
    df1 = df1.drop(df1.index[selected_rows])
    df2 = df2.drop(df2.index[selected_rows])

    selected_rows = df2.loc[df2.isna().any(axis=1)].index.tolist()
    df1 = df1.drop(df1.index[selected_rows])
    df2 = df2.drop(df2.index[selected_rows])

    return df1, df2


def delete_nan_rows_index(df1, df2, indexing_list):
    selected_rows = df1.loc[df1.isna().any(axis=1)].index.tolist()
    df1 = df1.drop(df1.index[selected_rows])
    df2 = df2.drop(df2.index[selected_rows])
    for i in reversed(selected_rows):
        del indexing_list[i]

    selected_rows = df2.loc[df2.isna().any(axis=1)].index.tolist()
    df1 = df1.drop(df1.index[selected_rows])
    df2 = df2.drop(df2.index[selected_rows])
    for i in reversed(selected_rows):
        del indexing_list[i]

    return df1, df2, indexing_list


def splitting_wrapper(initial_data, target_data):
    print("Hello, this is splitting wrapper!")
    _, cols = initial_data.shape

    target_columns = target_data.columns
    initial_columns = initial_data.columns
    merged_df = pd.concat([initial_data, target_data], axis=1)
    train_df, test_df = train_test_split(merged_df, test_size=0.3)
    initial_data_train = train_df.iloc[:, 0: cols]
    initial_data_test = test_df.iloc[:, 0: cols]
    target_data_train = train_df.iloc[:, cols: cols * 2]
    target_data_test = test_df.iloc[:, cols: cols * 2]

    initial_data_train = initial_data_train.reset_index()
    del initial_data_train['index']

    initial_data_test = initial_data_test.reset_index()
    test_index = initial_data_test['index']
    del initial_data_test['index']

    target_data_train = target_data_train.reset_index()
    del target_data_train['index']

    target_data_test = target_data_test.reset_index()
    del target_data_test['index']
    print("Splitting has been completed.")
    return initial_data_train, initial_data_test, target_data_train, target_data_test, test_index


def apply_log(data):
    print("transfering to the log scale!")
    columns = data.columns

    data_1 = tf.convert_to_tensor(data)
    data_1 = tf.math.log(data_1)

    data_1 = data_1.numpy()
    data_1 = pd.DataFrame(data_1, columns=columns)
    # rows = len(data)
    # for column in data.columns:
    #    for i in range(0, rows):
    #        data.loc[i, column] = np.log(data.loc[i, column])
    print("log scaling has been completed.")
    return data_1


def apply_normalization(data):
    print("Normalizing the data!")
    columns = data.columns

    data_1 = tf.convert_to_tensor(data)
    data_1 = tf.keras.utils.normalize(data_1, axis=-1, order=2)
    data_1 = data_1.numpy()
    data_1 = pd.DataFrame(data_1, columns=columns)
    # rows, cols = data.shape
    # for j in range(0, cols):
    #     col_min = np.min(data[:, j])
    #     col_max = np.max(data[:, j])
    #     col_ampl = col_max - col_min
    #     print(col_ampl)
    #     for i in range(0, rows):
    #         data[i,j] = ( data[i,j] - col_min) / col_ampl
    print("Data has been normalized!")
    return data_1


def element_wise_lin_div(data_1, data_2):
    columns = data_1.columns
    data_1_t = tf.convert_to_tensor(data_1)
    data_2_t = tf.convert_to_tensor(data_2)
    data_t = tf.divide(data_1_t, data_2_t)
    data_t = data_t.numpy()
    data_t = pd.DataFrame(data_t, columns=columns)
    return data_t


def return_data_sets(path, order, winx):
    initial_data_file_1 = path + 'sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_co_clean.csv'
    initial_data_file_2 = path + 'sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_cro_clean.csv'
    target_data_file = path + 'wavespec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_clean.csv'

    initial_data_1 = pd.read_csv(initial_data_file_1, sep=',')
    initial_data_2 = pd.read_csv(initial_data_file_2, sep=',')
    target_data = pd.read_csv(target_data_file, sep=',')

    initial_data_1, initial_data_2, target_data = time_based_splitter(initial_data_1, initial_data_2, target_data)
    # initial_data_1, initial_data_2, target_data = delete_nan_rows(initial_data_1, initial_data_2, target_data)
    initial_data_1, initial_data_2, target_data = initial_formatting_gen_2(initial_data_1, initial_data_2, target_data)

    # initial_data = initial_data_1
    initial_data = element_wise_lin_div(initial_data_1, initial_data_2)
    # initial_data = element_wise_lin_div(initial_data_2, initial_data_1)
    initial_data, target_data = delete_nan_rows(initial_data, target_data)

    # initial_data = apply_log(initial_data)
    # target_data = apply_log(target_data)
    # initial_data = apply_normalization(initial_data)
    # target_data = apply_normalization(target_data)

    return initial_data, target_data


# def return_data_sets_parallel(path, order, winx):
#     initial_data_file_1 = path + 'sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_co_clean.csv'
#     initial_data_file_2 = path + 'sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_cro_clean.csv'
#     target_data_file = path + 'wavespec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_clean.csv'
#
#     initial_data_1 = pd.read_csv(initial_data_file_1, sep=',')
#     initial_data_2 = pd.read_csv(initial_data_file_2, sep=',')
#     target_data = pd.read_csv(target_data_file, sep=',')
#     # initial_data_1, initial_data_2, target_data = delete_nan_rows(initial_data_1, initial_data_2, target_data)
#     initial_data_1, initial_data_2, target_data = initial_formatting_gen_2(initial_data_1, initial_data_2, target_data)
#
#     #initial_data = initial_data_2
#     #initial_data = element_wise_lin_div(initial_data_1, initial_data_2)
#     initial_data, target_data = delete_nan_rows(initial_data, target_data)

# initial_data = apply_log(initial_data)
# target_data = apply_log(target_data)
# initial_data = apply_normalization(initial_data)
# target_data = apply_normalization(target_data)
#
#
# return initial_data, target_data


def combine_katsed(path, katse_nr, order):
    path_1 = path + "katse_0" + str(katse_nr) + "/"
    path_2 = path + "katse_0" + str(katse_nr + 1) + "/"

    initial_data, target_data = return_data_sets(path_1, order=order, winx=256)
    # initial_data_2, target_data_2 = return_data_sets(path_2, order=order, winx=512)

    # initial_data = pd.concat([initial_data_1, initial_data_2], ignore_index=True)
    # target_data = pd.concat([target_data_1, target_data_2], ignore_index=True)

    # initial_data, target_data, _ = clustering_support(initial_data, target_data)

    initial_data_train, initial_data_test, target_data_train, target_data_test, test_index = splitting_wrapper(
        initial_data, target_data)

    start_time = datetime.datetime.now()
    start_time.strftime("%m_%d_%Y_%H_%M_%S")

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_testing_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_test, target_data_test, test_index], f)

    initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(
        initial_data_train, target_data_train)

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_training_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index], f)

    return initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test, valid_index, test_index


def combine_katsed_with_clustering(path, katse_nr, order, number_of_clusters):
    path_1 = path + "katse_0" + str(katse_nr) + "/"
    path_2 = path + "katse_0" + str(katse_nr + 1) + "/"

    initial_data_1, target_data_1 = return_data_sets(path_1, order=order, winx=256)
    initial_data_2, target_data_2 = return_data_sets(path_2, order=order, winx=512)

    initial_data = pd.concat([initial_data_1, initial_data_2], ignore_index=True)
    target_data = pd.concat([target_data_1, target_data_2], ignore_index=True)

    # apply clustering and see
    # dtw_labels, labels_k_means = clustering_wrapper(initial_data, target_data, number_of_clusters)
    # initial_data_cluster, target_data_cluster, largest_cluster = get_largest_cluster(initial_data, target_data, dtw_labels)
    # plot
    # fig, axs = plt.subplots(nrows=1,
    #                        ncols=2)
    # for j in range(0, len(dtw_labels)):
    #    axs[0, 0].plot(initial_data.loc[j, :], alpha=0.5, linewidth=0.5, color='blue')
    #    axs[0, 1].plot(target_data.loc[j, :], alpha=0.5, linewidth=0.5, color='green')
    # axs[k, 0].set_ylim([0, max_elt_input * 1.05])
    # axs[k, 1].set_ylim([0, max_elt_target * 1.05])
    # axs[0, 0].set_xticks([])
    # axs[0, 1].set_xticks([])
    # plt.show()

    # initial_data = initial_data_cluster
    # target_data = target_data_cluster
    # initial_data, target_data = clustering_support(initial_data, target_data)

    initial_data_train, initial_data_test, target_data_train, target_data_test, test_index = splitting_wrapper(
        initial_data, target_data)

    start_time = datetime.datetime.now()
    start_time.strftime("%m_%d_%Y_%H_%M_%S")

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_testing_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_test, target_data_test, test_index], f)

    initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(
        initial_data_train, target_data_train)

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_training_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index], f)

    return initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test, valid_index, test_index


def combine_katsed_ai(path_initial_data, path_processed, katse_nr, order):
    # this part is tailored particularly for the case of merging different katsed

    path_1 = path_initial_data + "katse_0" + str(katse_nr) + "/"
    path_2 = path_initial_data + "katse_0" + str(katse_nr + 1) + "/"

    initial_data_1, target_data_1 = return_data_sets(path_1, order=order, winx=256)
    initial_data_2, target_data_2 = return_data_sets(path_2, order=order, winx=512)

    initial_data = pd.concat([initial_data_1, initial_data_2], ignore_index=True)
    target_data = pd.concat([target_data_1, target_data_2], ignore_index=True)

    initial_data_train, initial_data_test, target_data_train, target_data_test, test_index = splitting_wrapper(
        initial_data, target_data)

    start_time = datetime.datetime.now()
    start_time.strftime("%m_%d_%Y_%H_%M_%S")

    file_name = path_processed + 'data/' + 'katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_testing_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_test, target_data_test, test_index], f)

    initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(
        initial_data_train, target_data_train)

    file_name = path_processed + 'katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_training_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index], f)

    return initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, \
        target_data_test, valid_index, test_index


def combine_katsed_simple(path, katse_nr, order):
    path_1 = path + "katse_0" + str(katse_nr) + "/"

    initial_data, target_data = return_data_sets(path_1, order=order, winx=512)

    # initial_data, target_data = clustering_support(initial_data, target_data)

    initial_data_train, initial_data_test, target_data_train, target_data_test, test_index = splitting_wrapper(
        initial_data, target_data)

    start_time = datetime.datetime.now()
    start_time.strftime("%m_%d_%Y_%H_%M_%S")

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_testing_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_test, target_data_test, test_index], f)

    initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(
        initial_data_train, target_data_train)

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_training_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index], f)

    return initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test, valid_index, test_index


def combine_katsed_simple_2(path, katse_nr, order):
    path_1 = path + "katse_0" + str(katse_nr) + "/"

    initial_data, target_data = return_data_sets(path_1, order=order, winx=384)

    # initial_data, target_data = clustering_support(initial_data, target_data)

    initial_data_train, initial_data_test, target_data_train, target_data_test, test_index = splitting_wrapper(
        initial_data, target_data)

    start_time = datetime.datetime.now()
    start_time.strftime("%m_%d_%Y_%H_%M_%S")

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_testing_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_test, target_data_test, test_index], f)

    initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(
        initial_data_train, target_data_train)

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_training_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index], f)

    return initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test, valid_index, test_index


def return_data_sets_simple(path, order, winx):
    initial_data_file_1 = path + 'katse_08/' + 'sarspec_hgh_order_' + str(order) + '_winx_' + str(
        winx) + '_co_clean.csv'
    initial_data_file_2 = path + 'katse_08/' + 'sarspec_hgh_order_' + str(order) + '_winx_' + str(
        winx) + '_cro_clean.csv'
    target_data_file = path + 'katse_08/' + 'wavespec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_clean.csv'

    initial_data_1 = pd.read_csv(initial_data_file_1, sep=',')
    initial_data_2 = pd.read_csv(initial_data_file_2, sep=',')
    target_data = pd.read_csv(target_data_file, sep=',')
    # initial_data_1, initial_data_2, target_data = delete_nan_rows(initial_data_1, initial_data_2, target_data)
    initial_data_1, initial_data_2, target_data = initial_formatting_gen_2(initial_data_1, initial_data_2, target_data)

    # initial_data = initial_data_2
    # initial_data = element_wise_lin_div(initial_data_1, initial_data_2)
    # initial_data = element_wise_lin_div(initial_data_2, initial_data_1)
    # initial_data, target_data = delete_nan_rows(initial_data, target_data)

    # initial_data = apply_log(initial_data)
    # target_data = apply_log(target_data)
    # initial_data = apply_normalization(initial_data)
    # target_data = apply_normalization(target_data)

    return initial_data_1, initial_data_2


def combine_katsed_with_clustering_xxx(path, katse_nr, order, number_of_clusters):
    path_1 = path + "katse_0" + str(katse_nr) + "/"
    path_2 = path + "katse_0" + str(katse_nr + 1) + "/"

    initial_data_1, target_data_1 = return_data_sets(path_1, order=order, winx=256)
    initial_data_2, target_data_2 = return_data_sets(path_2, order=order, winx=512)

    initial_data = initial_data_1
    target_data = target_data_1

    # apply clustering and see
    dtw_labels, labels_k_means = clustering_wrapper(initial_data, target_data, number_of_clusters)
    initial_data_cluster, target_data_cluster, largest_cluster = get_largest_cluster(initial_data, target_data,
                                                                                     dtw_labels)
    # plot
    # fig, axs = plt.subplots(nrows=1,
    #                        ncols=2)
    # for j in range(0, len(dtw_labels)):
    #    axs[0, 0].plot(initial_data.loc[j, :], alpha=0.5, linewidth=0.5, color='blue')
    #    axs[0, 1].plot(target_data.loc[j, :], alpha=0.5, linewidth=0.5, color='green')
    # axs[k, 0].set_ylim([0, max_elt_input * 1.05])
    # axs[k, 1].set_ylim([0, max_elt_target * 1.05])
    # axs[0, 0].set_xticks([])
    # axs[0, 1].set_xticks([])
    # plt.show()

    initial_data = initial_data_cluster
    target_data = target_data_cluster
    # initial_data, target_data = clustering_support(initial_data, target_data)

    initial_data_train, initial_data_test, target_data_train, target_data_test, test_index = splitting_wrapper(
        initial_data, target_data)

    start_time = datetime.datetime.now()
    start_time.strftime("%m_%d_%Y_%H_%M_%S")

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_testing_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_test, target_data_test, test_index], f)

    initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(
        initial_data_train, target_data_train)

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_training_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index], f)

    return initial_data_train, initial_data_valid, initial_data_test, target_data_train, target_data_valid, target_data_test, valid_index, test_index


###########################################################


def return_data_sets_ng(path, order, winx):
    initial_data_file_1 = path + 'sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_co_clean.csv'
    initial_data_file_2 = path + 'sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_cro_clean.csv'
    target_data_file = path + 'wavespec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_clean.csv'

    initial_data_1 = pd.read_csv(initial_data_file_1, sep=',')
    initial_data_2 = pd.read_csv(initial_data_file_2, sep=',')
    target_data = pd.read_csv(target_data_file, sep=',')

    input_data_1_train, input_data_2_train, target_data_train, input_data_1_test, input_data_2_test, target_data_test, \
        test_data_indexes = time_based_splitter(initial_data_1, initial_data_2, target_data)
    # initial_data_1, initial_data_2, target_data = delete_nan_rows(initial_data_1, initial_data_2, target_data)
    input_data_1_train, input_data_2_train, target_data_train = initial_formatting_gen_2(input_data_1_train,
                                                                                         input_data_2_train,
                                                                                         target_data_train)
    input_data_1_test, input_data_2_test, target_data_test = initial_formatting_gen_2(input_data_1_test,
                                                                                      input_data_2_test,
                                                                                      target_data_test)

    # initial_data = initial_data_1
    input_data_train = element_wise_lin_div(input_data_1_train, input_data_2_train)
    input_data_test = element_wise_lin_div(input_data_1_test, input_data_2_test)
    input_data_train, target_data_train = delete_nan_rows(input_data_train, target_data_train)
    input_data_test, target_data_test, test_data_indexes = delete_nan_rows_index(input_data_test, target_data_test,
                                                                                 test_data_indexes)

    # initial_data = apply_log(initial_data)
    # target_data = apply_log(target_data)
    input_data_train = apply_normalization(input_data_train)
    target_data_train = apply_normalization(target_data_train)
    input_data_test = apply_normalization(input_data_test)
    target_data_test = apply_normalization(target_data_test)

    return input_data_train, input_data_test, target_data_train, target_data_test, test_data_indexes


def time_based_splitter(input_data_1, input_data_2, target_data):
    input_columns_1 = input_data_1.columns.values.tolist()
    input_frame_1 = pd.DataFrame(columns=input_columns_1)

    input_columns_2 = input_data_2.columns.values.tolist()
    input_frame_2 = pd.DataFrame(columns=input_columns_2)

    target_columns = target_data.columns.values.tolist()
    output_frame = pd.DataFrame(columns=target_columns)

    time_list = input_data_1['time'].drop_duplicates()
    time_list_train, time_list_test = train_test_split(time_list, train_size=0.7, test_size=0.3)

    input_data_1_train = input_data_1.loc[input_data_1['time'].isin(time_list_train)]
    input_data_2_train = input_data_2.loc[input_data_1['time'].isin(time_list_train)]
    target_data_train = target_data.loc[input_data_1['time'].isin(time_list_train)]

    input_data_1_test = input_data_1.loc[input_data_1['time'].isin(time_list_test)]
    input_data_2_test = input_data_2.loc[input_data_2['time'].isin(time_list_test)]
    target_data_test = target_data.loc[input_data_1['time'].isin(time_list_test)]

    for instance in time_list_test:
        time_frame = input_data_1_test.loc[input_data_1_test['time'] == instance]
        time_frame_2 = input_data_2_test.loc[input_data_2_test['time'] == instance]
        station_list = time_frame['station_ind'].drop_duplicates().tolist()
        station_list_2 = time_frame['station_ind'].drop_duplicates().tolist()

        for station in station_list:
            station_frame = time_frame.loc[time_frame['station_ind'] == station]
            index_list = station_frame.index.values.tolist()
            #input_frame_1 = pd.concat([input_frame_1, station_frame.loc[index_list[0], :].to_frame().T])

            # for using random selection of the area
            idx = np.random.randint(0, high=len(index_list), dtype=int)
            input_frame_1 = pd.concat([input_frame_1, station_frame.loc[index_list[idx], :].to_frame().T])

    test_indexes = input_frame_1.index.values.tolist()

    input_data_2_test = input_data_2_test.loc[test_indexes]
    target_data_test = target_data_test.loc[test_indexes]
    input_data_1_test = input_frame_1

    input_data_1_train = input_data_1_train.reset_index()
    del input_data_1_train['index']

    input_data_2_train = input_data_2_train.reset_index()
    del input_data_2_train['index']

    target_data_train = target_data_train.reset_index()
    del target_data_train['index']

    input_data_1_test = input_data_1_test.reset_index()
    del input_data_1_test['index']

    input_data_2_test = input_data_2_test.reset_index()
    del input_data_2_test['index']

    target_data_test = target_data_test.reset_index()
    del target_data_test['index']

    return input_data_1_train, input_data_2_train, target_data_train, input_data_1_test, input_data_2_test, target_data_test, test_indexes


def combine_katsed_time_based(path, katse_nr, order):
    path_1 = path + "katse_0" + str(katse_nr) + "/"

    input_data_train, input_data_test, target_data_train, target_data_test, test_data_indexes = return_data_sets_ng(
        path_1, order=order, winx=512)

    start_time = datetime.datetime.now()
    start_time.strftime("%m_%d_%Y_%H_%M_%S")

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_testing_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([input_data_test, target_data_test, test_data_indexes], f)

    initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(
        input_data_train, target_data_train)

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_training_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index], f)

    return initial_data_train, initial_data_valid, input_data_test, target_data_train, target_data_valid, target_data_test, valid_index, test_data_indexes


def combine_katsed_time_based_2(path, katse_nr, order):
    path_1 = path + "katse_0" + str(katse_nr) + "/"

    initial_data_train, input_data_test, initial_data_valid, target_data_train, target_data_test, target_data_valid, \
        test_data_indexes, valid_index = return_data_sets_ng_2(
        path_1, order=order, winx=512)

    start_time = datetime.datetime.now()
    start_time.strftime("%m_%d_%Y_%H_%M_%S")

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_testing_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump([input_data_test, target_data_test, test_data_indexes], f)

    #initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(
    #    input_data_train, target_data_train)

    file_name = path + 'processed_datasets/katse_' + str(katse_nr) + '_' + str(katse_nr + 1) + '_training_data_' \
                + start_time.strftime("%m_%d_%Y_%H_%M_%S") + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump([initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index], f)

    return initial_data_train, initial_data_valid, input_data_test, target_data_train, target_data_valid, target_data_test, valid_index, test_data_indexes


def return_data_sets_ng_2(path, order, winx):
    initial_data_file_1 = path + 'sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_co_clean.csv'
    initial_data_file_2 = path + 'sarspec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_cro_clean.csv'
    target_data_file = path + 'wavespec_hgh_order_' + str(order) + '_winx_' + str(winx) + '_clean.csv'

    initial_data_1 = pd.read_csv(initial_data_file_1, sep=',')
    initial_data_2 = pd.read_csv(initial_data_file_2, sep=',')
    target_data = pd.read_csv(target_data_file, sep=',')

    input_data_1_train, input_data_2_train, target_data_train, input_data_1_test, input_data_2_test, target_data_test, \
        test_data_indexes = time_based_splitter(initial_data_1, initial_data_2, target_data)

    input_data_1_train, input_data_2_train, target_data_train, input_data_1_valid, input_data_2_valid, target_data_valid, \
        valid_data_indexes = time_based_splitter(input_data_1_train, input_data_2_train, target_data_train)

    # initial_data_1, initial_data_2, target_data = delete_nan_rows(initial_data_1, initial_data_2, target_data)
    input_data_1_train, input_data_2_train, target_data_train = initial_formatting_gen_2(input_data_1_train,
                                                                                         input_data_2_train,
                                                                                         target_data_train)
    input_data_1_test, input_data_2_test, target_data_test = initial_formatting_gen_2(input_data_1_test,
                                                                                      input_data_2_test,
                                                                                      target_data_test)

    input_data_1_valid, input_data_2_valid, target_data_valid = initial_formatting_gen_2(input_data_1_valid,
                                                                                         input_data_2_valid,
                                                                                         target_data_valid)

    # initial_data = initial_data_1
    input_data_train = element_wise_lin_div(input_data_1_train, input_data_2_train)
    input_data_test = element_wise_lin_div(input_data_1_test, input_data_2_test)
    input_data_valid = element_wise_lin_div(input_data_1_valid, input_data_2_valid)
    input_data_train, target_data_train = delete_nan_rows(input_data_train, target_data_train)
    input_data_test, target_data_test, test_data_indexes = delete_nan_rows_index(input_data_test, target_data_test,
                                                                                 test_data_indexes)
    input_data_valid, target_data_valid, valid_data_indexes = delete_nan_rows_index(input_data_valid, target_data_valid,
                                                                                    valid_data_indexes)

    # initial_data = apply_log(initial_data)
    # target_data = apply_log(target_data)
    input_data_train = apply_normalization(input_data_train)
    target_data_train = apply_normalization(target_data_train)
    input_data_test = apply_normalization(input_data_test)
    target_data_test = apply_normalization(target_data_test)
    input_data_valid = apply_normalization(input_data_valid)
    target_data_valid = apply_normalization(target_data_valid)

    return input_data_train, input_data_test, input_data_valid, target_data_train, target_data_test, target_data_valid, test_data_indexes, valid_data_indexes


def return_datasets_dnora(path):
    path = '/Users/svennomm/kohalikTree/Data/AIRSCS/wave/data_dnora/'
    fname_co_spec = path + 'co_spec.pkl'
    fname_cro_spec = path + 'cro_spec.pkl'
    # fname_co_spec_log = path + 'co_spec.pkl'
    # fname_cro_spec_log = path + 'cro_spec.pkl'
    fname_div_spec = path + 'dive_spec.pkl'
    fname_model_spec = path + 'model_spec.pkl'

    with open(fname_co_spec, 'rb') as f:
        input_data_1 = pkl.load(f)

    with open(fname_cro_spec, 'rb') as f:
        input_data_2 = pkl.load(f)

    with open(fname_model_spec, 'rb') as f:
        target_data = pkl.load(f)

    with open(fname_div_spec, 'rb') as f:
        div_spec = pkl.load(f)

    #div_spec = div_spec.values
    #fig_1 = plt.figure()
    #for i in range(0, len(div_spec)):
    #    if i % 100 == 0:
    #        plt.plot(div_spec[i, :])
    #plt.show()

    input_data = element_wise_lin_div(input_data_1, input_data_2)
    input_data_train, target_data_train = delete_nan_rows(input_data, target_data)

    input_data_train, input_data_test, target_data_train, target_data_test, test_index = splitting_wrapper(input_data, target_data)
    input_data_train, input_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(input_data_train,
                                                                                                           target_data_train)

    input_data_train = apply_normalization(input_data_train)
    target_data_train = apply_normalization(target_data_train)
    input_data_test = apply_normalization(input_data_test)
    target_data_test = apply_normalization(target_data_test)
    input_data_valid = apply_normalization(input_data_valid)
    target_data_valid = apply_normalization(target_data_valid)

    return input_data_train, input_data_valid, input_data_test, target_data_train, target_data_valid, target_data_test, valid_index, test_index
