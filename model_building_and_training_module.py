import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from preprocessing_module import apply_log
from preprocessing_module import splitting_wrapper
from validation_module import test_model

import datetime
import time
from keras.layers.core import Dense, Activation, Dropout
#from keras.layers.recurrent import LSTM
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model


def build_lstm_model(train_x, train_y):
    print("Building LSTM model!")
    rows, cols = train_x.shape
    model = Sequential()
    model.add(LSTM(46, input_shape=(cols, 1), return_sequences=True))   # 64, 64, 64
    model.add(LSTM(46))
    model.add(Dense(46))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print("LSTM model is ready")
    model.summary()
    #
    return model


def train_model_gen_2(train_x, train_y, epochs_nr):
    print("Training LSTM model!")
    start_time = time.time()
    model = build_lstm_model(train_x, train_y)
    train_x = apply_log(train_x)
    train_y = apply_log(train_y)

    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()

    #train_x = apply_normalization(train_x)
    #train_y = apply_normalization(train_y)

    rows, cols = train_x.shape
    train_x = train_x.reshape(rows, cols, 1)
    print("training...")
    model.fit(train_x, train_y,
        epochs=epochs_nr,
        batch_size=2800,
        verbose=2)
    #for i in range(0, rows):
    #    x = train_x[i, :].reshape(1, cols, 1)
    #    model.fit(
    #        x, train_y[i, :],
    #        epochs=1,
    #        batch_size=1,
    #        verbose=2)
            #nb_epoch=LstmSettings.EPOCHS,
            #validation_split=0.05)
        #print('prediction duration (s) : ', time.time() - start_time)
        #print('saving model: %s' % model_path)
        #model.save(model_path)
    print("LSTM model has been trained!")
    return model

def lstm_wrapper_gen_2(initial_data, target_data, path, data_name, epochs_nr):
    print("Aloha! this is lstm_wrapper!!!")

    np.set_printoptions(precision=3, suppress=True)
    print(tf.__version__)
    initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index = splitting_wrapper(initial_data, target_data)
    initial_data_train, initial_data_test, target_data_train, target_data_test, no_need_index = splitting_wrapper(initial_data_train, target_data_train)
    #initial_data_test.to_csv('test_x_08_04_2022_1.csv', sep='\t', encoding='utf-8')
    #target_data_test.to_csv('test_y_08_04_2022_1.csv', sep='\t', encoding='utf-8')
    #valid_index.to_csv('test_index_08_04_2022_1.csv', sep='\t', encoding='utf-8')

    #data_train, data_test, dict_train, dict_test = data_formatter(initial_data, target_data, model_order)

    #train_features = data_train[:, 0:model_order - 1] #inputs
    #train_labels = data_train[:, model_order] #outputs

    #train_features = data_train[:, 0:model_order]  # inputs
    #train_labels = data_train[:, model_order]  # outputs

    start_time = datetime.datetime.now()
    #model_path = return_model_path()
    model_name = 'lstm' + data_name + start_time.strftime("%m_%d_%Y_%H_%M_%S")
    full_model_name = path + model_name

    lstm_model = train_model_gen_2(initial_data_train, target_data_train, epochs_nr)
    lstm_model.save(full_model_name)

    test_model(initial_data_valid, target_data_valid, lstm_model, valid_index, path, data_name, model_name)
    print("Uhhh managed the task")