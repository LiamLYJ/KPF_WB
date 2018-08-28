import numpy as np
import os
import re
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures

def normalize_rate(input, reverse = False):
    input_min, input_max = np.min(input), np.max(input)
    rate =  (input - input_min) / (input_max - input_min)
    assert ((rate>=0).all() and (rate <= 1).all())
    if reverse:
        rata = 1 - rate
    return rate

def normalize(input):
    mean = np.mean(input)
    var = np.var(input)
    return (input-mean)/var

def get_multi(error):
    tmp = np.zeros_like(error)
    tmp[error <= 1.0] = 0
    tmp[error > 1.0] = 5
    tmp[error > 1.5] = 10
    tmp[error > 2.0] = 15
    tmp[error > 2.5] = 20
    tmp[error > 3.0] = 30
    return tmp

def get_filt(input):
    ref_list = [0, 5, 10, 15, 20, 30]
    ref_box = []
    abs_box = []
    for item in ref_list:
        ref_box.append(np.ones_like(input) * item)
        abs_box.append(np.abs(input - np.ones_like(input) * item))
    ref_box = np.stack(ref_box, axis=-1)
    abs_box = np.stack(abs_box, axis=-1)
    check = np.argmin(abs_box, axis = -1)
    after = ref_box[..., check]
    return after[0]

def mx_MLP(train_x,train_y):
    mx = MLPRegressor(hidden_layer_sizes=(1000,100,50,), max_iter=100000000, activation="logistic")
    mx.fit(train_x,train_y)
    return mx

def get_data(file_name):
    with open(file_name) as fp:
        input_lines = fp.readlines()

    confi_r = []
    confi_b = []
    error = []
    file_names = []
    for line in input_lines:
        if 'ce_r' in line:
            confi_r.append(float(line.split(' ')[-1][:-1]))
        elif 'ce_b' in line:
            confi_b.append(float(line.split(' ')[-1][:-1]))
        elif 'error is' in line:
            error.append(float(line.split(' ')[-1][:-1]))
        elif '.png' in line:
            file_names.append(line.split(' ')[1])
    error = np.reshape(np.array(error), [-1, 1])
    confi_r = np.reshape(np.array(confi_r), [-1, 1])
    confi_b = np.reshape(np.array(confi_b), [-1, 1])
    return error, confi_r, confi_b, file_names


def learning_based(error, confi_r, confi_b):
    # accuracy = get_multi(error)
    accuracy = error
    confi_dif = np.abs(confi_r - confi_b)
    # confi_r = normalize(confi_r)
    # confi_b = normalize(confi_b)
    # confi_dif = normalize(confi_dif)
    confi_r = preprocessing.scale(confi_r)
    confi_b = preprocessing.scale(confi_b)
    confi_dif = preprocessing.scale(confi_dif)
    # confi_r = preprocessing.scale(confi_r)
    # confi_b = preprocessing.scale(confi_b)
    # confi_dif = preprocessing.scale(confi_dif)

    min_max_scaler = preprocessing.MinMaxScaler()
    data_x = np.concatenate([confi_r, confi_b, confi_dif], axis = -1)
    data_x = min_max_scaler.fit_transform(data_x)
    poly = PolynomialFeatures(2)
    data_x = poly.fit_transform(data_x)
    data_y = accuracy.copy()

    data_len = data_x.shape[0]
    print ('data_x shape: ', data_x.shape)
    print ('data_y shape: ', data_y.shape)

    data_x_train = np.squeeze(data_x[:(data_len-10),:])
    data_y_train = np.squeeze(data_y[:(data_len-10),:])
    data_x_val = np.squeeze(data_x[(data_len-10):,:])
    data_y_val = np.squeeze(data_y[(data_len-10):,:])
    data_x_train = data_x[1,:]
    data_y_train = data_y[1,:]
    regr = mx_MLP(data_x_train, data_y_train)
    y_pred = regr.predict(data_x_train)
    # filt_y_pred = get_filt(y_pred)
    print ('y_pred: ', y_pred)
    print ('y_gt: ', data_y_train)
    # print ('y_filt: ', filt_y_pred)

    y_test_pred = regr.predict(data_x_val)
    # filt_y_test_pred = get_filt(y_test_pred)
    print ('y_test_pred: ', y_test_pred)
    # print ('y_test_pred_filt:', filt_y_test_pred)
    print ('y_gt_test:', data_y_val)

if __name__ == '__main__':
    # file_name = './log_tmp.txt'
    file_name = './log.txt'
    error, confi_r, confi_b, file_name = get_data(file_name)
    confi = np.concatenate([confi_r, confi_b], axis = -1)
    confi_norm = np.expand_dims(np.linalg.norm(confi, axis = -1), -1)
    print (confi_norm.shape)
    results = (confi_b + confi_r) /2 / np.abs(confi_b - confi_r) / confi_norm
    print (results.shape)
    num_all = len(file_name)
    for i in range(num_all):
        print (file_name[i])
        print (results[i])
    # learning_based(error, confi_r, confi_b)
