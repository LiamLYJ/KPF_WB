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

    # data_dir = '/home/lyj/Downloads/Sony/preprocessed'
    # dir = './tmp_save_ex'
    # fn = '111MSDCF_DSC07730_108MSDCF_DSC07506'
    # fn = '108MSDCF_DSC07087_111MSDCF_DSC07977'
    # fn = '111MSDCF_DSC07750_109MSDCF_DSC06798'
    # fn = '112MSDCF_DSC08800_111MSDCF_DSC07916'
    # fn = '105MSDCF_DSC03577_112MSDCF_DSC09082'
    # fn = '106MSDCF_DSC04946_106MSDCF_DSC04946'


    # dir = './tmp_save_64'
    # fn = '105MSDCF_DSC03577_111MSDCF_DSC07998'
    # fn = '106MSDCF_DSC05012_108MSDCF_DSC07138'
    # fn = '107MSDCF_DSC04083_109MSDCF_DSC06778'
    # fn = '107MSDCF_DSC04100_111MSDCF_DSC07918'

    # data_dir = '../gogogo/cube_1024'
    #
    # dir = './dump_cube_train_ms'
    # fn = '39_910'

    # dir = './dump_cube_test_ms'
    # fn = '533_1162'

    data_dir = '../gogogo/sony_1024'
    # dir = './dump_sony_64_train_ms'
    dir = './dump_sony_train_ms'
    # fn = '106MSDCF_DSC05108_109MSDCF_DSC06733'
    # fn = '107MSDCF_DSC04083_109MSDCF_DSC06562'
    # fn = '108MSDCF_DSC07506_111MSDCF_DSC07918'
    # fn = '111MSDCF_DSC07977_111MSDCF_DSC07977'
    # fn = '109MSDCF_DSC06534_111MSDCF_DSC07851'
    # fn = '105MSDCF_DSC03577_103MSDCF_DSC00361'

    # dir = './dump_sony_train'
    # fn = '100MSDCF_DSC09972'


    data_dir = '../gogogo/gehler_1024'
    dir = './dump_gehler_train_ms'
    fn = 'IMG_0603_IMG_0303'

    # data_dir = '../gogogo/cube_1024'
    # dir = './dump_cube_train_ms'
    # # fn = '613_465'
    # fn = '597_1132'

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

    gain_box, clus_img, clus_labels = utils.gain_fitting(img0, img2, is_local = True, n_clusters =2, gamma = 2.0, with_clus = True)
    imsave(fn+'_clus.png', clus_img)
    big = utils.apply_gain_box(concat_original, gain_box, scale_h, scale_w)
    imsave(fn + '_big.png', big)

    global_gain = utils.gain_fitting(img0, img2, is_local = False)
    global_big = utils.apply_gain(concat_original, global_gain)
    imsave(fn + '_global_big.png', global_big)
