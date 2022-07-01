import pandas as pd
import numpy as np
import tensorflow as tf
#import tensorflow_datasets as tfds




from scipy.signal import resample
from sklearn.model_selection import train_test_split


def initial_formatting_gen_2(initial_data, target_data):
    print("Aloha! Performing initial formatting")

    # the following lines are very local
    del initial_data['time']
    del target_data['time']
    del initial_data['station_ind']
    del target_data['station_ind']

    #target_data.drop(target_data.index[0], inplace=True)
    #target_data = target_data.reset_index()
    #del target_data['index']

    print("Initial  formatting has been completed!")
    return initial_data, target_data


def initial_formatting_gen_2_3(initial_data_1, initial_data_2, target_data):
    print("Aloha! Performing initial formatting")

    # the following lines are very local
    del initial_data_1['time']
    del initial_data_2['time']
    del target_data['time']
    del initial_data_1['station_ind']
    del initial_data_2['station_ind']
    del target_data['station_ind']

    #target_data.drop(target_data.index[0], inplace=True)
    #target_data = target_data.reset_index()
    #del target_data['index']

    print("Initial  formatting has been completed!")
    return initial_data_1,initial_data_2, target_data


def delete_nan_rows(df1, df2):
    selected_rows = df1.loc[df1.isna().any(axis=1)].index.tolist()
    df1 = df1.drop(df1.index[selected_rows])
    df2 = df2.drop(df2.index[selected_rows])
    return df1, df2


def delete_nan_rows_3(df1, df2, df3):
    selected_rows = df1.loc[df1.isna().any(axis=1)].index.tolist()
    df1 = df1.drop(df1.index[selected_rows])
    df2 = df2.drop(df2.index[selected_rows])
    df3 = df3.drop(df3.index[selected_rows])

    selected_rows_2 = df2.loc[df2.isna().any(axis=1)].index.tolist()
    df1 = df1.drop(df1.index[selected_rows_2])
    df2 = df2.drop(df2.index[selected_rows_2])
    df3 = df3.drop(df3.index[selected_rows_2])
    return df1, df2, df3


def splitting_wrapper(initial_data, target_data):
    print("Hello, this is splitting wrapper!")
    _, cols = initial_data.shape

    target_columns = target_data.columns
    initial_columns = initial_data.columns
    merged_df = pd.concat([initial_data, target_data], axis=1)
    train_df, test_df = train_test_split(merged_df, test_size=0.2)
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
    #rows = len(data)
    #for column in data.columns:
    #    for i in range(0, rows):
    #        data.loc[i, column] = np.log(data.loc[i, column])
    print("log scaling has been completed.")
    return data_1


def apply_normalization(data):
    print("Normalizing the data!")
    rows, cols = data.shape
    for j in range(0, cols):
        col_min = np.min(data[:, j])
        col_max = np.max(data[:, j])
        col_ampl = col_max - col_min
        print(col_ampl)
        for i in range(0, rows):
            data[i,j] = ( data[i,j] - col_min) / col_ampl
    print("Data has been normalized!")
    return data


def element_wise_lin_div(data_1, data_2):
    columns = data_1.columns
    data_1_t = tf.convert_to_tensor(data_1)
    data_2_t = tf.convert_to_tensor(data_2)
    data_t = tf.divide(data_1_t,data_2_t)
    data_t = data_t.numpy()
    data_t = pd.DataFrame(data_t, columns=columns)

    return data_t
