import numpy as np
import os
import re
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error, r2_score

def normalize(input, reverse = False):
    input_min, input_max = np.min(input), np.max(input)
    rate =  (input - input_min) / (input_max - input_min)
    assert ((rate>=0).all() and (rate <= 1).all())
    if reverse:
        rata = 1 - rate
    return rate

def get_multi(error):
    error[error <= 1.0] = 1
    error[error > 1.0] = 2
    error[error > 2.0] = 3
    error[error > 3.0] = 4
    error[error > 4.0] = 5
    return error


if __name__ == '__main__':
    file_name = './log.txt'
    with open(file_name) as fp:
        input_lines = fp.readlines()

    confi_r = []
    confi_b = []
    error = []
    file_names = []
    for line in input_lines:
        if 'cofindece_r' in line:
            confi_r.append(float(line.split(' ')[-1][:-1]))
        elif 'cofindece_b' in line:
            confi_b.append(float(line.split(' ')[-1][:-1]))
        elif 'error is' in line:
            error.append(float(line.split(' ')[-1][:-1]))
        elif '.png' in line:
            file_names.append(line.split(' ')[1])
    error = np.array(error)
    # accuracy = normalize(error, reverse = True)
    accuracy = get_multi(error)
    confi_r = np.array(confi_r)
    confi_r = normalize(confi_r)
    confi_b = np.array(confi_b)
    confi_b = normalize(confi_b)

    confi_dif = np.abs(confi_r - confi_b)
    confi_r_squre = confi_r * confi_r
    confi_b_squre = confi_b * confi_b
    confi_dif_squre = confi_dif * confi_dif
    data_x = np.stack([confi_r, confi_b, confi_dif, confi_r_squre, confi_b_squre, confi_dif_squre], axis = 1)
    data_y = accuracy.copy()

    data_x_train = data_x[:90]
    data_y_train = data_y[:90]
    data_x_val = data_x[90:]
    data_y_val = data_y[90:]

    regr = lm.LinearRegression()
    regr.fit(data_x_train, data_y_train)
    # regr.fit(data_x,  data_y)

    # y_pred = regr.predict(data_x_train)
    y_pred = regr.predict(data_x_val)
    print ('y_pred: ', y_pred.shape)
    print (y_pred)
    print ('y_gt: ', data_y_val)
    print ('y_gt: ', data_y_train.shape)
    print (data_y_val)
    # print (data_y_train)
    # print("Mean squared error: %.2f"
    #       % mean_squared_error(data_y_val, y_pred))
    # print('Variance score: %.2f' % r2_score(data_y_val, y_pred))
