import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
# tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers
from validation_module import test_model
from validation_module import test_model_with_output
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.layers import GaussianNoise

import datetime
import time
from keras.layers.core import Dense, Activation, Dropout
# from keras.layers.recurrent import LSTM
from keras.layers import LSTM
from keras.models import Sequential
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import tensorflow_probability as tfp
from dtaidistance import dtw
from support_module import DtwLoss

import tf2onnx
import onnx


# import keras2onnx


def rho(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    # print(y_true.shape)
    # print(type(y_true))
    return tfp.stats.correlation(y_true, y_pred, sample_axis=0, event_axis=None)


def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)


def dtw_measure(y_true, y_pred):
    y_true = y_true.to_numpy()
    y_pred = y_pred.to_numpy()
    return dtw.distance(y_true, y_pred)


def tf_dtw_measure(y_true, y_pred):
    tf.config.run_functions_eagerly(True)
    print(tf.__version__)
    y_true = y_true.numpy()
    # print(y_true.shape)
    # print(type(y_true))
    y_pred = y_pred.numpy()
    d = dtw.distance(y_true, y_pred)
    return dtw.distance(y_true, y_pred)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32), tf.TensorSpec(None, tf.float32)])
def tf_dtw(input_1, input_2):
    d = tf.numpy_function(dtw.distance, input_1, input_2, tf.float32)
    return d


def build_lstm_model(train_x, train_y):
    print("Building LSTM model!")
    # print(tf.version.VERSION)
    opt = keras.optimizers.Adam(learning_rate=0.005)
    # opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    try:
        xrows, xcols, xdim = train_x.shape
        yrows, ycols, ydim = train_y.shape
    except ValueError:
        xrows, xcols = train_x.shape
        yrows, ycols = train_y.shape
        xdim = 1

    rows, cols = train_x.shape
    y_rows, y_cols = train_y.shape
    #model = Sequential()
    #model.add(LSTM(64, input_shape=(cols, xdim), return_sequences=True))  #69, 128
    #model.add(GaussianNoise(0.5))
    #model.add(LSTM(64, return_sequences=True))
    #model.add(LSTM(64))
    #model.add(Dense(y_cols))

    model = keras.Sequential(
        [
            layers.LSTM(64, input_shape=(cols, xdim), return_sequences=True),  #64 -> 4
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(64),
            layers.Dense(y_cols, name="output")

        ]
    )

    # model = keras.Sequential(
    #     [
    #         layers.Input(shape=(cols, xdim)),
    #         layers.LSTM(46, return_sequences=True),
    #         layers.LSTM(92),
    #         layers.Dense(y_cols, name="output")
    #
    #     ]
    # )

    # model.compile(loss='mean_squared_error', optimizer=opt, metrics=[rho])
    # tf.config.run_functions_eagerly(True)
    # loss = DtwLoss(batch_size=1)
    # model.compile(loss=correlation_coefficient_loss, optimizer='adam',  metrics=[rho])
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=tf.keras.metrics.MeanSquaredError())
    save_format = 'h5'
    model.save('LSTM_model_1_1.h5')
    # model.compile(loss=DtwLoss(batch_size=1), optimizer=opt, metrics=[rho])

    # Martin's model
    # batch = 512
    # x_dim = 1
    # model = Sequential()
    # model.add(LSTM(69, input_shape=(cols, x_dim), return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(128))
    # model.add(Dropout(0.2))
    # model.add(Dense(y_cols))
    # model.compile(loss='mean_squared_error', optimizer=opt)

    print("LSTM model is ready")
    # plot_model(model, to_file='model_plot.pdf', show_shapes=True, show_layer_names=True)
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
    # callbacks2 = EarlyStopping(monitor='val_loss', patience=50, mode='min')

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
    # z = model(train_x[0,:], train_y[0,:])
    # with writer.as_default():
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
    save_format = 'h5'
    lstm_model.save(full_model_name + '.h5')

    # input_signature = [tf.TensorSpec([None, 46, 1], tf.float32, name='x')]
    # onnx_model, _ = tf2onnx.convert.from_keras(lstm_model, input_signature, opset=13)
    # onnx.save(onnx_model, path + model_name + '.onnx')

    goodness_descriptors, y_hat = test_model_with_output(initial_data_test, target_data_test, lstm_model, test_index, path, data_name, model_name)
    #fig_1 = plt.figure()
    #ax_1 = fig_1.add_subplot()
    #ax_1 = goodness_descriptors.plot.hist(column=["rho"])
    #plt.show()

    goodness_analysis(goodness_descriptors, initial_data_test, target_data_test, y_hat)
    print("Uhhh managed the task")


def lstm_wrapper_gen_2_with_callbacks_output(initial_data_train, initial_data_valid, initial_data_test,
                                             target_data_train, target_data_valid, target_data_test, valid_index,
                                             test_index, path, data_name, epochs_nr):
    print("Aloha! this is lstm_wrapper!!!")

    np.set_printoptions(precision=3, suppress=True)
    print(tf.__version__)

    start_time = datetime.datetime.now()
    model_name = 'lstm_' + data_name + '_' + str(epochs_nr) + '_epoch_' + start_time.strftime("%m_%d_%Y_%H_%M_%S")
    full_model_name = path + model_name

    lstm_model = train_model_with_with_callbacks(initial_data_train, target_data_train, initial_data_valid,
                                                 target_data_valid, epochs_nr, path)
    lstm_model.save(full_model_name)

    input_signature = [tf.TensorSpec([3, 3], tf.float32, name='x')]
    onnx_model, _ = tf2onnx.convert.from_keras(lstm_model, input_signature, opset=13)
    onnx.save(onnx_model, path + model_name + '.onnx')

    test_model_with_output(initial_data_test, target_data_test, lstm_model, test_index, path, data_name, model_name)
    print("Uhhh managed the task")
    return lstm_model


def goodness_analysis(goodness_descriptors, input_data_test, target_data_test, y_hat):
    input_data_test = input_data_test.reset_index()
    target_data_test = target_data_test.reset_index()
    del input_data_test['index']
    del target_data_test['index']
    drop_indexes = goodness_descriptors.index[goodness_descriptors['rho'] < 0.8].tolist()
    keep_indexes = goodness_descriptors.index[goodness_descriptors['rho'] >= 0.8].tolist()
    n = len(input_data_test)
    l = len(drop_indexes)
    print("Data length ", n, " observation points.")
    print("Low quality prediction is made for :", l, " observation points.")
    y_hat = pd.DataFrame(y_hat)


    input_data_test_keep = input_data_test.loc[keep_indexes, :]
    target_data_test_keep = target_data_test.loc[keep_indexes, :]
    input_data_test_drop = input_data_test.loc[drop_indexes, :]
    target_data_test_drop = target_data_test.loc[drop_indexes, :]
    y_hat_keep = y_hat.loc[keep_indexes, :]
    y_hat_drop = y_hat.loc[drop_indexes, :]

    input_data_test_keep = input_data_test_keep.values
    target_data_test_keep = target_data_test_keep.values
    input_data_test_drop = input_data_test_drop.values
    target_data_test_drop = target_data_test_drop.values
    n_keep = len(y_hat_keep)
    n_drop = len(y_hat_drop)
    y_hat_keep = y_hat_keep.values
    y_hat_drop = y_hat_drop.values


    fig_2 = plt.figure()
    ax_1 = fig_2.add_subplot(2, 3, 1)
    ax_2 = fig_2.add_subplot(2, 3, 2)
    ax_3 = fig_2.add_subplot(2, 3, 3)
    ax_4 = fig_2.add_subplot(2, 3, 4)
    ax_5 = fig_2.add_subplot(2, 3, 5)
    ax_6 = fig_2.add_subplot(2, 3, 6)

    for i in range(0, n_keep):

        ax_1.plot(input_data_test_keep[i, :], color='blue', linewidth=0.2)
        ax_2.plot(target_data_test_keep[i, :], color='gold', linewidth=0.2)
        ax_3.plot(y_hat_keep[i, :], color='darkviolet', linewidth=0.2)

    for i in range(0, n_drop):
        ax_4.plot(input_data_test_drop[i, :], color='blue', linewidth=0.2)
        ax_5.plot(target_data_test_drop[i, :], color='gold', linewidth=0.2)
        ax_6.plot(y_hat_drop[i, :], color='darkviolet', linewidth=0.2)

    plt.show()
