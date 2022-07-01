
import datetime
from preprocessing_module import apply_log
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
from support_module import rho2labels
import matplotlib.pyplot as plt

def goodness_descriptor(test_y, hat_y):
    mse = mean_squared_error(test_y, hat_y)
    rho = np.corrcoef(test_y, hat_y)[0, 1]
    max_test = test_y.argmax(axis=0)
    max_hat = hat_y.argmax(axis=0)
    delta_max_val = np.max(test_y) - np.max(hat_y)
    delta_max_loc = max_test - max_hat

    return mse, rho, max_test, max_hat, delta_max_val, delta_max_loc


def test_model(test_x, test_y, model, test_index, path, data_name, model_name):
    print("Testing LSTM model!")
    #test_x = apply_log(test_x)
    #test_y = apply_log(test_y)
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    rows, cols = test_y.shape
    test_x = test_x.reshape(rows, cols, 1)

    y_hat = model.predict(test_x, batch_size=2, verbose=0)
    goodness_descriptors = []
    print(rows)
    for i in range(0, rows):
        mse, rho, max_test, max_hat, delta_max_val, delta_max_loc = goodness_descriptor(test_y[i, :], y_hat[i, :])
        goodness_descriptors.append([int(test_index[i]), mse, rho, max_test, max_hat, delta_max_val, delta_max_loc])

        #print("Testing for datapoint", test_index.loc[i])
        #y_ampl = np.abs(np.max(test_y[i, :]) - np.min(test_y[i, :]))
        #residuals_nn = (test_y[i, :] - y_hat[i, :]) / y_ampl

        #fig2, axis = plt.subplots()
        #plt.plot(test_y[i, :], color='blue')
        #plt.plot(y_hat[i, :], color='orange')
        #plt.title("validation for", str(test_index.loc[i]))
        #plt.show()

        #fig3, axis = plt.subplots()
        #plt.plot(residuals_nn, color='green')
        #plt.title("residuals for a small set")
        #plt.show()
    goodness_descriptors = np.array(goodness_descriptors)
    columns = ['index', 'mse', 'rho', 'max_test', 'max_hat', 'delta_max_val', 'delta_max_loc']
    print(columns)
    print("Average values are: ", np.average(goodness_descriptors, axis=0))

    goodness_descriptors = pd.DataFrame(goodness_descriptors, columns=columns)

    time = datetime.datetime.now()
    #path = return_model_path()
    goodness_file_name = path + 'validation_of_' + model_name + '_on' + data_name + '.csv'
    goodness_descriptors.to_csv(goodness_file_name, index=False)
    print("LSTM model has been tested! ")

def test_model_with_output(test_x, test_y, model, test_index, path, data_name, model_name):
    print("Testing LSTM model!")
    #test_x = apply_log(test_x)
    #test_y = apply_log(test_y)
    test_x = test_x.to_numpy()
    test_y = test_y.to_numpy()

    rows, cols = test_y.shape
    test_x = test_x.reshape(rows, cols, 1)

    y_hat = model.predict(test_x, batch_size=2, verbose=0)
    goodness_descriptors = []
    print(rows)
    for i in range(0, rows):
        mse, rho, max_test, max_hat, delta_max_val, delta_max_loc = goodness_descriptor(test_y[i, :], y_hat[i, :])
        goodness_descriptors.append([int(test_index[i]), mse, rho, max_test, max_hat, delta_max_val, delta_max_loc])

        #print("Testing for datapoint", test_index.loc[i])
        #y_ampl = np.abs(np.max(test_y[i, :]) - np.min(test_y[i, :]))
        #residuals_nn = (test_y[i, :] - y_hat[i, :]) / y_ampl

        #fig2, axis = plt.subplots()
        #plt.plot(test_y[i, :], color='blue')
        #plt.plot(y_hat[i, :], color='orange')
        #plt.title("validation for", str(test_index.loc[i]))
        #plt.show()

        #fig3, axis = plt.subplots()
        #plt.plot(residuals_nn, color='green')
        #plt.title("residuals for a small set")
        #plt.show()
    goodness_descriptors = np.array(goodness_descriptors)
    columns = ['index', 'mse', 'rho', 'max_test', 'max_hat', 'delta_max_val', 'delta_max_loc']
    print(columns)
    print("Average values are: ", np.average(goodness_descriptors, axis=0))

    goodness_descriptors = pd.DataFrame(goodness_descriptors, columns=columns)
    rho_labels = rho2labels(goodness_descriptors)
    rho_labels = np.array(rho_labels)
    #color_list = np.array([[213,62,79], [244,109,67], [253,174,97], [254,224,139], [255,255,191], [230,245,152],
    #                      [171,221,164], [102,194,165], [50,136,189]]) / 255

    color_list = np.array([[166,206,227], [31,120,180], [178,223,138], [51,160,44], [251,154,153], [227,26,28],
                           [253,191,111], [255,127,0], [202,178,214]]) / 255

    fig, ax = plt.subplots()
    plt.axis('off')
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    for k in range(0, np.max(rho_labels)):
        #kk = np.max(rho_labels) - (k + 1)
        selected_rows = np.where(rho_labels == k)

        for i in range(0, len(selected_rows[0])):
            # title('Clustered initial data')
            ax1.plot(test_x[selected_rows[0][i], :], color=color_list[k], alpha=0.4, linewidth=0.5)
            # plt2.title('Corresponding target data')
            ax2.plot(test_y[selected_rows[0][i], :], color=color_list[k], alpha=0.4, linewidth=0.5)

            ax1.set_xticks([])
            ax2.set_xticks([])
    plt.show()





    time = datetime.datetime.now()
    #path = return_model_path()
    goodness_file_name = path + 'validation_of_' + model_name + '_on' + data_name + '.csv'
    goodness_descriptors.to_csv(goodness_file_name, index=False)
    print("LSTM model has been tested! ")
    return  goodness_descriptors