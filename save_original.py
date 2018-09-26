import os
import json
import sys
import utils
import numpy as np
import cv2
from skimage.io import imread
from skimage import transform
from scipy.misc import imsave


if __name__ == '__main__':

    data_dir = './data/nus_1024'
    # dir = './dump_nus_train_ms'
    # fn = 'SonyA57_0132_Canon1DsMkIII_0175'
    # fn = 'Canon600D_0136_SamsungNX2000_0028'

    # fn = 'SonyA57_0031_FujifilmXM1_0033'
    # fn = 'OlympusEPL6_0092_SamsungNX2000_0051'
    # fn = 'OlympusEPL6_0037_NikonD5200_0001'
    fn = 'SonyA57_0048_SonyA57_0033'
    dir = './dump_nus_test_ms'

    try:
        file_np = os.path.join(dir, fn + '.npy')
        tmp = np.load(file_np)

        size = tmp.shape[1]

        img0 = tmp[:size,:,:]
        img2 = tmp[size:,:,:]
    except:
        img_all = imread(os.path.join(dir, fn + '.png'))
        size = img_all.shape[0]
        img0 = img_all[:,:size,:]
        img2 = img_all[:, 2*size:3*size,: ]
    img2 = np.clip(img2, 0,500)
    print ('min ref:', np.min(img2))
    print ('max ref:', np.max(img2))

    json_file = os.path.join(dir, fn + '.json')

    try:
        concat_original, scale_h, scale_w = utils.get_original(data_dir= data_dir, json_file = json_file)
    except:
        concat_original = imread(os.path.join(data_dir, fn +'.png'))
        scale_h = scale_w = int(1024 / 128)

    gain_box_pure = utils.gain_fitting(img0, img2, is_pure = True)
    pure = utils.apply_gain_box(concat_original, gain_box_pure, scale_h, scale_w)
    imsave(fn+'_big_pure.png', pure)

    # gain_box, clus_img, clus_labels = utils.gain_fitting(img0, img2, is_local = True, n_clusters =2, gamma = 2.0, with_clus = True)
    # imsave(fn+'_clus.png', clus_img)
    # big = utils.apply_gain_box(concat_original, gain_box, scale_h, scale_w)
    # imsave(fn + '_big.png', big)

    global_gain = utils.gain_fitting(img0, img2, is_local = False)
    global_big = utils.apply_gain(concat_original, global_gain)
    global_big = np.power(global_big, 1.0/ 2.2)
    imsave(fn + '_global_big.png', global_big)
