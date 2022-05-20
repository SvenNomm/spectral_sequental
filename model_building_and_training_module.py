import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from validation_module import test_model
from keras.callbacks import EarlyStopping

import datetime
import time
from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import tensorflow_probability as tfp


def rho(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tfp.stats.correlation(y_true, y_pred, sample_axis=0, event_axis=None)


def build_lstm_model(train_x, train_y):
    print("Building LSTM model!")
    print(tf.version.VERSION)
    #opt = keras.optimizers.Adam(learning_rate=0.005)
    opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    rows, cols = train_x.shape
    y_rows, y_cols = train_y.shape
    model = Sequential()
    model.add(LSTM(69, input_shape=(cols, 1), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(y_cols))
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[rho])

    # Martin's model
    #batch = 512
    #x_dim = 1
    #model = Sequential()
    #model.add(LSTM(69, input_shape=(cols, x_dim), return_sequences=True))
    #model.add(Dropout(0.2))
    #model.add(LSTM(128))
    # model.add(Dropout(0.2))
    #model.add(Dense(y_cols))
    #model.compile(loss='mean_squared_error', optimizer=opt)


    print("LSTM model is ready")
    #plot_model(model, to_file='model_plot.pdf', show_shapes=True, show_layer_names=True)
    model.summary()
    #
    return model


def train_model_gen_2(train_x, train_y, epochs_nr):
    print("Training LSTM model!")
    start_time = time.time()
    model = build_lstm_model(train_x, train_y)
    # train_x = apply_log(train_x)
    # train_y = apply_log(train_y)

    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()

    rows, cols = train_x.shape
    train_x = train_x.reshape(rows, cols, 1)
    print("training...")
    model.fit(train_x, train_y,
              epochs=epochs_nr,
              batch_size=512,
              verbose=2)

    print("LSTM model has been trained!")
    return model


def train_model_with_with_callbacks(train_x, train_y, test_x, test_y, epochs_nr, path):
    print("Training LSTM model!")
    start_time = time.time()
    model = build_lstm_model(train_x, train_y)
    # train_x = apply_log(train_x)
    # train_y = apply_log(train_y)

    train_x = train_x.to_numpy()
    train_y = train_y.to_numpy()
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    rows, cols = train_x.shape
    train_x = train_x.reshape(rows, cols, 1)
    print("training...")
    log_dir = path + 'logs/fit/' + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,
    write_images=True)
    #callbacks2 = EarlyStopping(monitor='val_loss', patience=50, mode='min')

    writer = tf.summary.create_file_writer(log_dir)
    tf.summary.trace_on(graph=True, profiler=True)

    tensorboard_callback.set_model(model)
    model.fit(train_x, train_y,
              epochs=epochs_nr,
              batch_size=512,
              verbose=2,
              validation_data=(test_x, test_y),
              callbacks=[tensorboard_callback])

    print("LSTM model has been trained!")
    #z = model(train_x[0,:], train_y[0,:])
    #with writer.as_default():
    #    tf.summary.trace_export(
    #        name="my_func_trace",
    #        step=0,
    #        profiler_outdir=log_dir)
    return model


def lstm_wrapper_gen_2(initial_data_train, initial_data_valid, target_data_train, target_data_valid, valid_index, path,
                       data_name, epochs_nr):
    print("Aloha! this is lstm_wrapper!!!")

    np.set_printoptions(precision=3, suppress=True)
    print(tf.__version__)

    start_time = datetime.datetime.now()
    model_name = 'lstm_' + data_name + '_' + str(epochs_nr) + '_epoch_' + start_time.strftime("%m_%d_%Y_%H_%M_%S")
    full_model_name = path + model_name

    lstm_model = train_model_gen_2(initial_data_train, target_data_train, epochs_nr)
    lstm_model.save(full_model_name)

    test_model(initial_data_valid, target_data_valid, lstm_model, valid_index, path, data_name, model_name)
    print("Uhhh managed the task")


def lstm_wrapper_gen_2_with_callbacks(initial_data_train, initial_data_valid, initial_data_test, target_data_train,
                                      target_data_valid, target_data_test, valid_index, test_index, path, data_name,
                                      epochs_nr):

    print("Aloha! this is lstm_wrapper!!!")

    np.set_printoptions(precision=3, suppress=True)
    print(tf.__version__)

    start_time = datetime.datetime.now()
    model_name = 'lstm_' + data_name + '_' + str(epochs_nr) + '_epoch_' + start_time.strftime("%m_%d_%Y_%H_%M_%S")
    full_model_name = path + model_name

    lstm_model = train_model_with_with_callbacks(initial_data_train, target_data_train, initial_data_valid,
                                                 target_data_valid, epochs_nr, path)
    lstm_model.save(full_model_name)

    test_model(initial_data_test, target_data_test, lstm_model, test_index, path, data_name, model_name)
    print("Uhhh managed the task")